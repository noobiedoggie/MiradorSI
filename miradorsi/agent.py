"""L'agent Mirador SI : relie l'identité, le modèle, internet et l'analyse."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Callable

from .analysis.lexicon import load_lexicon
from .analysis.researcher import AnalysisReport, DeepAnalyzer
from .analysis.text import content_terms, normalize
from .config import Config
from .identity import Identity
from .model.brain import Brain
from .web.http import WebClient, WebError
from .web.search import Fetcher, SearchEngine

HELP = """Commandes :
  /analyse <question>        analyse approfondie sur internet (ajoute -p 1|2|3 pour la profondeur)
  /cherche <requête>         résultats de recherche bruts
  /lire <url>                lire et résumer une page
  /identite                  qui suis-je ?
  /aide                      cette aide
  /quitter                   quitter
Tu peux aussi écrire « analyse … » ou « recherche … » en langage naturel."""

_ANALYSIS_INTENT = re.compile(r"^(?:analyse|analyser|analyze|recherche|enquête|investigue|étudie)\s*[:\-]?\s+(.+)",
                              re.I)

# Messages de conversation : une recherche sur internet n'aurait aucun sens (« yo », « merci »…).
_GREETINGS = {
    "yo", "salut", "slt", "bonjour", "bjr", "bonsoir", "coucou", "cc", "hello", "hi", "hey", "wesh",
    "re", "rebonjour", "allo", "hola", "bonne nuit", "bonne journee", "bonne soiree",
}
_SMALL_TALK = _GREETINGS | {
    "merci", "mrc", "thanks", "thx", "ok", "okay", "d accord", "dac", "cool", "super", "top", "parfait",
    "genial", "bravo", "lol", "mdr", "oui", "non", "bye", "au revoir", "a plus", "a+", "ciao",
    "ca va", "cv", "tu vas bien", "comment ca va", "comment vas tu", "quoi de neuf", "ca dit quoi",
    "quoi de beau", "ca va bien", "la forme",
    "qui es tu", "t es qui", "tu es qui", "c est quoi ton nom", "quel est ton nom", "comment tu t appelles",
    "presente toi", "que sais tu faire", "tu sais faire quoi", "qui t a cree", "aide",
}


_HOW_ARE_YOU = ("ca va", "cv", "tu vas bien", "comment vas tu", "quoi de neuf", "ca dit quoi", "quoi de beau",
                "la forme")
_GOODBYES = {"bye", "au revoir", "a plus", "a+", "ciao", "bonne nuit", "bonne journee", "bonne soiree"}
_INSULTS = {
    "connard", "connasse", "con", "conne", "debile", "idiot", "idiote", "abruti", "abrutie", "cretin", "imbecile",
    "nul", "nulle", "naze", "tg", "fdp", "pute", "salope", "encule", "batard", "merde", "bouffon", "stupide",
    "ta gueule", "ferme la", "fils de pute", "va te faire",
}
_WRITE_INTENT = re.compile(r"^\s*(?:écris|ecris|invente|imagine|continue)\s*(?:la suite de|une histoire sur)?\s*:?\s+(.+)",
                           re.I)
OFFLINE_LIMIT = (
    "Je ne peux pas te répondre de façon fiable hors ligne. Mon modèle embarqué est tout petit "
    "(quelques millions de paramètres, des milliers de fois moins qu'un grand assistant) : il ne "
    "comprend pas encore les questions libres, et je préfère te le dire plutôt que d'inventer.\n\n"
    "Passe en mode Analyse (la loupe) : je cherche la réponse sur internet, je lis les sources et "
    "je te rédige une réponse sourcée. Hors ligne, je sais faire les calculs et parler de moi."
)


def is_insult(message: str) -> bool:
    words = re.sub(r"[^a-z]+", " ", normalize(message)).split()
    if not words or len(words) > 6:
        return False
    pairs = {" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    return bool(pairs & _INSULTS)


_CALC_PREFIX = re.compile(r"^(combien (font|fait|ca fait)|calcule[rz]?|calcul|que vaut|ca fait|resultat de|"
                          r"what is|how much is)\s*:?\s*")
_CALC_WORDS = [(r"\bdivise par\b", "/"), (r"\bsur\b", "/"), (r"\bmultiplie par\b", "*"), (r"\bfois\b", "*"),
               (r"\bplus\b", "+"), (r"\bmoins\b", "-"), (r"(?<=\d)\s*[x×]\s*(?=[\d(])", "*"), ("÷", "/"),
               (r"\^", "**"), (r"(?<=\d),(?=\d)", ".")]


def calculate(message: str) -> str | None:
    """Calcule « 1+1 », « combien font 12 x 7 ? », « 2^10 »… (arithmétique seulement, sans eval)."""
    import ast
    import operator

    text = _CALC_PREFIX.sub("", normalize(message).strip(" ?!=."))
    for pattern, op in _CALC_WORDS:
        text = re.sub(pattern, op, text)
    expr = text.replace(" ", "")
    if not re.fullmatch(r"[\d.+\-*/%()]+", expr) or not re.search(r"\d\)?(\*\*|[+\-*/%])\(?[\d.]", expr):
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise OverflowError
            return ops[type(node.op)](left, right)
        raise ValueError

    try:
        result = value(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return "Division par zéro : ce calcul n'a pas de résultat."
    except (SyntaxError, ValueError, OverflowError, TypeError, RecursionError):
        return None
    if isinstance(result, float):
        result = int(result) if result.is_integer() and abs(result) < 1e15 else float(f"{result:.10g}")
    shown = re.sub(r"\s*(\*\*|[+\-*/%])\s*", lambda m: f" {m.group(1)} ", expr)
    shown = shown.replace(" ** ", "^").replace("*", "×").replace("/", "÷").replace(".", ",")
    shown = re.sub(r"(^|\()\s*- ", r"\1-", shown.strip())  # signe moins en tête : « -5 + 2 »
    pretty = f"{result:,}".replace(",", " ").replace(".", ",") if isinstance(result, int) else \
        str(result).replace(".", ",")
    return f"{shown.strip()} = {pretty}"


_FILLER = {"mirador", "si", "toi", "vous", "a", "tout", "le", "monde", "les", "gars", "l", "ami"}


def is_small_talk(message: str) -> bool:
    """Vrai pour une salutation, un remerciement ou une question sur Mirador SI lui-même."""
    words = re.sub(r"[^a-z0-9+]+", " ", normalize(message)).split()
    if not words:
        return True
    text = " ".join(words)
    if text in _SMALL_TALK or text.replace("mirador si", "").replace("mirador", "").strip() in _SMALL_TALK:
        return True
    # « salut ça va », « yo mirador », « bonjour toi »… : salutation suivie de peu de mots
    # (les mots composés restent entiers : « yo-yo » n'est pas une salutation)
    tokens = re.sub(r"[^a-z0-9+-]+", " ", normalize(message)).split()
    for size in (2, 1):
        if " ".join(tokens[:size]) in _GREETINGS:
            rest = [w for w in words[size:] if w not in _FILLER]
            return len(rest) <= 1 or " ".join(rest) in _SMALL_TALK
    return False


class MiradorAgent:
    def __init__(self, cfg: Config, identity: Identity, analyzer: DeepAnalyzer, search: SearchEngine,
                 fetcher: Fetcher, brain: Brain | None = None,
                 on_progress: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.identity = identity
        self.analyzer = analyzer
        self.search = search
        self.fetcher = fetcher
        self.brain = brain
        self.history: list[tuple[str, str]] = []
        self.progress = on_progress or (lambda message: None)
        self.last_report: AnalysisReport | None = None

    @classmethod
    def create(cls, cfg: Config, on_progress: Callable[[str], None] | None = None,
               load_brain: bool = True) -> "MiradorAgent":
        identity = Identity.load(cfg.path("identity"))
        cfg.web.user_agent = cfg.web.user_agent or f"{identity.short_name}/{identity.version}"
        client = WebClient(cfg.web)
        search = SearchEngine.from_config(cfg.web, client)
        fetcher = Fetcher(client)
        lexicon = load_lexicon(str(cfg.path("lexicon")))
        analyzer = DeepAnalyzer(search, fetcher, cfg.analysis, on_progress, analyst=identity.name,
                                lexicon=lexicon)
        brain = Brain.load(cfg.path("checkpoints"), cfg.generation) if load_brain else None
        return cls(cfg, identity, analyzer, search, fetcher, brain, on_progress)

    # ------------------------------------------------------------------ routage
    def handle(self, message: str) -> str:
        message = message.strip()
        if not message:
            return ""
        lowered = message.lower()
        if lowered in ("/aide", "/help", "aide", "help", "?"):
            return HELP
        if lowered in ("/identite", "/identité", "/qui"):
            return self.identity.presentation()
        if lowered.startswith(("/analyse", "/analyze")):
            question, depth = _parse_depth(message.split(" ", 1)[1] if " " in message else "")
            return self.analyze(question, depth) if question else "Usage : /analyse <question>"
        if lowered.startswith(("/cherche", "/search")):
            query = message.split(" ", 1)[1] if " " in message else ""
            return self.search_web(query) if query else "Usage : /cherche <requête>"
        if lowered.startswith(("/lire", "/read")):
            url = message.split(" ", 1)[1].strip() if " " in message else ""
            return self.read(url) if url else "Usage : /lire <url>"
        intent = _ANALYSIS_INTENT.match(message)
        if intent:
            question, depth = _parse_depth(intent.group(1))
            return self.analyze(question, depth)
        answer = self.chat(message)
        self.history.append((message, answer))
        return answer

    def quick_reply(self, message: str) -> str | None:
        """Réponse sûre sans internet ni modèle : calcul, salutation, politesse, question sur moi."""
        result = calculate(message)
        if result:
            return result
        if is_insult(message):
            return ("Je reste poli. Si ma réponse ne t'a pas aidé, dis-moi ce que tu cherchais : "
                    "en mode Analyse (la loupe), je vérifie sur internet et je cite mes sources.")
        if not is_small_talk(message):
            return None
        words = re.sub(r"[^a-z0-9+]+", " ", normalize(message)).split()
        text, first = " ".join(words), " ".join(words[:1])
        canned = dict(self.identity.training_dialogues())
        greeting = canned["Bonjour"] if first in {"bonjour", "bjr", "bonsoir", "rebonjour"} else canned["Salut"]
        if first == "bonsoir":
            greeting = greeting.replace("Bonjour", "Bonsoir").replace("aujourd'hui", "ce soir")
        if any(phrase in text for phrase in _HOW_ARE_YOU):
            hello = (greeting.split("!")[0] + "! ") if first in _GREETINGS else ""
            return f"{hello}Ça va bien, merci ! Pose-moi une question : je l'analyse sur internet, sources à l'appui."
        if set(words) & {"merci", "mrc", "thanks", "thx"}:
            return canned["Merci"]
        if text in _GOODBYES:
            return "À bientôt ! Reviens quand tu veux une analyse."
        if first in _GREETINGS:
            return greeting
        if re.search(r"\b(qui es|es qui|appel|ton nom|presente)", text):
            return canned["Qui es-tu ?"]
        remembered, score = self._identity_memory(message)
        if remembered and score >= 0.3:
            return remembered
        return "D'accord ! Dis-moi ce que tu veux que j'analyse."

    # ------------------------------------------------------------------ actions
    def chat(self, message: str) -> str:
        """Hors ligne. Le petit modèle embarqué ne comprend pas une question libre : plutôt que de
        laisser deviner une réponse fausse, on répond seulement quand c'est fiable."""
        quick = self.quick_reply(message)
        if quick:
            return quick
        remembered, score = self._identity_memory(message)
        if remembered and score >= 0.34:
            return remembered
        write = _WRITE_INTENT.match(message)
        if write and self.brain is not None:
            start = write.group(1).strip()
            return (start + self.brain.complete(start)).strip() + "\n\n(Texte inventé par mon modèle, sans vérification.)"
        if self.brain is None:
            return ("Mon réseau de neurones n'est pas encore entraîné (lance `mirador train`). "
                    "Je peux déjà analyser n'importe quel sujet : /analyse <ta question>")
        return OFFLINE_LIMIT

    def analyze(self, question: str, depth: int | None = None) -> str:
        report = self.analyzer.analyze(question, depth)
        self.last_report = report
        path = self.save_report(report)
        return summarize_report(report) + f"\n\nRapport complet : {path}"

    def search_web(self, query: str, max_results: int = 8) -> str:
        results = self.search.search(query, max_results)
        if not results:
            errors = "; ".join(self.search.errors[-3:])
            return "Aucun résultat." + (f" ({errors})" if errors else "")
        return "\n".join(f"{r.rank:>2}. [{r.provider}] {r.title}\n    {r.url}\n    {r.snippet[:200]}"
                         for r in results)

    def read(self, url: str, max_chars: int = 2500) -> str:
        try:
            doc = self.fetcher.fetch(url)
        except WebError as exc:
            return f"Lecture impossible : {exc}"
        text = doc.text[:max_chars] + ("…" if len(doc.text) > max_chars else "")
        return f"{doc.title}\n{doc.url}\n{'─' * 40}\n{text}"

    def save_report(self, report: AnalysisReport) -> Path:
        out_dir = self.cfg.path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = "-".join(content_terms(report.question, stem=False)[:6]) or "analyse"
        path = out_dir / f"{stamp}-{slug[:60]}.md"
        path.write_text(report.to_markdown(), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ interne
    def _identity_memory(self, message: str) -> tuple[str | None, float]:
        """Réponse d'identité la plus proche du message, et sa similarité (0 à 1)."""
        terms = set(content_terms(message))
        best, best_score = None, 0.0
        for question, answer in self.identity.training_dialogues():
            q_terms = set(content_terms(question))
            if not q_terms:
                q_terms = {question.lower().strip(" ?!.")}
                score = 1.0 if message.lower().strip(" ?!.") in q_terms else 0.0
            else:
                score = len(terms & q_terms) / len(terms | q_terms) if terms else 0.0
            if score > best_score:
                best, best_score = answer, score
        return best, best_score


def _parse_depth(text: str) -> tuple[str, int | None]:
    match = re.search(r"(?:^|\s)(?:-p|--profondeur|--depth|-d)\s*([123])\b", text)
    if not match:
        return text.strip(), None
    return (text[: match.start()] + text[match.end():]).strip(), int(match.group(1))


def summarize_report(report: AnalysisReport, max_findings: int = 5) -> str:
    if not report.findings:
        errors = "\n".join(f"  - {e}" for e in report.errors[:5])
        return ("Je n'ai pas pu collecter d'information exploitable." +
                (f"\nIncidents :\n{errors}" if errors else ""))
    lines = [f"Analyse ({len(report.sources)} sources, {len(report.queries)} requêtes, "
             f"{report.duration_seconds:.0f} s) :", ""]
    for i, f in enumerate(report.findings[:max_findings], 1):
        cites = "".join(f"[{c}]" for c in f.citations)
        lines.append(f"{i}. {f.text} {cites} (confiance {f.confidence_label})")
    if report.divergent:
        lines += ["", f"⚠ {len(report.divergent)} divergence(s) de chiffres entre sources — voir le rapport."]
    lines += ["", "Sources :"]
    cited = sorted({c for f in report.findings[:max_findings] for c in f.citations})
    for sid in cited:
        s = report.source(sid)
        if s:
            lines.append(f"  [{s.id}] {s.title[:80]} — {s.url}")
    return "\n".join(lines)
