"""L'agent Mirador SI : relie l'identité, le modèle, internet et l'analyse."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Callable

from .analysis.researcher import AnalysisReport, DeepAnalyzer
from .analysis.text import content_terms
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
        analyzer = DeepAnalyzer(search, fetcher, cfg.analysis, on_progress, analyst=identity.name)
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
        return self.chat(message)

    # ------------------------------------------------------------------ actions
    def chat(self, message: str) -> str:
        remembered, score = self._identity_memory(message)
        if self.brain is not None:
            answer = self.brain.reply(message, self.history)
            if answer:
                self.history.append((message, answer))
                if score < 0.3 and message.rstrip().endswith("?"):
                    answer += "\n(Réponse générée sans vérification. Pour une réponse sourcée : /analyse …)"
                return answer
        if remembered and score >= 0.5:
            return remembered
        if self.brain is None:
            return ("Mon réseau de neurones n'est pas encore entraîné (lance `mirador train`). "
                    "Je peux déjà analyser n'importe quel sujet : /analyse <ta question>")
        return "Je n'ai pas de réponse fiable. Pour une réponse sourcée : /analyse <ta question>"

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
