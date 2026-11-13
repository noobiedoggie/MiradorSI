"""Cerveau « lecteur » : Vigie apprend à rédiger une réponse à partir de sources.

Un petit modèle ne peut pas tout savoir par cœur, mais il peut apprendre à LIRE : on lui
donne des sources numérotées et une question, il rédige la réponse en citant sa source.
Les exercices sont fabriqués automatiquement à partir de la bibliothèque hors ligne
(introductions et fiches des articles de Wikipédia) ; aucune autre IA n'intervient.

  <|system|>[1] Australie : L'Australie est un pays d'Océanie… Australie — Capitale : Canberra.
  [2] Japon : Le Japon est un pays insulaire… Japon — Capitale : Tokyo.<|end|>
  <|user|>Quelle est la capitale du Japon ?<|end|>
  <|assistant|>La capitale du Japon est Tokyo. [2]<|end|>

Seuls les jetons de la réponse comptent dans la perte : le modèle apprend à répondre, pas
à recopier la question. Des exercices sans réponse dans les sources lui apprennent à dire
« je ne trouve pas » plutôt qu'à inventer.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..analysis.text import content_terms, split_sentences

NOT_FOUND = "Je ne trouve pas la réponse dans ces sources."
_SOURCE_CHARS = 260  # longueur maximale d'une source dans l'exercice


@dataclass
class ReaderExample:
    question: str
    sources: list[tuple[str, str]]  # (titre, texte), dans l'ordre de leurs numéros
    answer: str                     # réponse rédigée, avec renvoi [n]
    kind: str                       # « fiche:Capitale », « definition », « absent »


def format_prompt(question: str, sources: list[tuple[str, str]]) -> str:
    lines = [f"[{i}] {title} : {text}" for i, (title, text) in enumerate(sources, 1)]
    return f"<|system|>{chr(10).join(lines)}<|end|><|user|>{question}<|end|><|assistant|>"


def encode_example(tokenizer, example: ReaderExample, max_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(entrées, cibles) ; les cibles valent -1 hors de la réponse. None si trop long."""
    prompt = tokenizer.encode(format_prompt(example.question, example.sources))
    answer = tokenizer.encode(example.answer) + [tokenizer.end_id]
    ids = prompt + answer
    if len(ids) - 1 > max_len:
        return None
    x = np.array(ids[:-1], dtype=np.int64)
    y = np.array(ids[1:], dtype=np.int64)
    y[: len(prompt) - 1] = -1
    return x, y


# ---------------------------------------------------------------- formes du sujet en français
def subject_forms(title: str, lead: str) -> tuple[str, str]:
    """(« Le Japon », « du Japon ») d'après l'introduction ; (« Lyon », « de Lyon ») sinon."""
    base = re.sub(r"\s*\(.*\)$", "", title)
    titled = re.match(r"^(?:(Les|Le|La) |(L'|L’))(.+)$", base)  # « Les Misérables » → « des Misérables »
    if titled:
        article, rest = (titled.group(1) or titled.group(2)).replace("’", "'"), titled.group(3)
        return base, {"Le": f"du {rest}", "La": f"de la {rest}", "Les": f"des {rest}", "L'": f"de l'{rest}"}[article]
    match = re.match(rf"^(?:(Les|Le|La) |(L'|L’)){re.escape(base)}\b", lead)
    if match:
        article = (match.group(1) or match.group(2)).replace("’", "'")
        subject = f"{article.lower()}{base}" if article == "L'" else f"{article.lower()} {base}"
        of = {"Le": f"du {base}", "La": f"de la {base}", "Les": f"des {base}", "L'": f"de l'{base}"}[article]
        return subject, of  # « le Japon » : en milieu de phrase ; capitalisé en début de réponse
    of = f"d'{base}" if base[:1].lower() in "aeiouyéèêàâîôûh" else f"de {base}"
    return base, of


def in_phrase(of: str, subject: str) -> str:
    """« du Japon » → « au Japon », « de la France » → « en France », « de Lyon » → « à Lyon »."""
    for prefix, preposition in (("du ", "au "), ("de la ", "en "), ("de l'", "en "), ("des ", "aux ")):
        if of.startswith(prefix):
            return preposition + of[len(prefix):]
    return f"à {subject}"


_DATE_PLACE = re.compile(r"^((?:\d{1,2}(?:er)? )?\w+ \d{3,4})\s+(.+)$")


def _when_where(value: str) -> str:
    """« 7 novembre 1867 Varsovie (Pologne) » → « le 7 novembre 1867 à Varsovie (Pologne) »."""
    match = _DATE_PLACE.match(value)
    if match and re.search(r"\d", match.group(1)):
        return f"le {match.group(1)} à {match.group(2)}"
    return f"le {value}" if value[:1].isdigit() else value


# question(s) possibles et réponse rédigée, par rubrique de fiche
_FICHE_TEMPLATES = {
    "Capitale": (["Quelle est la capitale {of} ?", "C'est quoi la capitale {of} ?", "Capitale {of} ?"],
                 "La capitale {of} est {v}."),
    "Population municipale": (["Quelle est la population {of} ?", "Combien d'habitants compte {S} ?"],
                              "La population municipale {of} est de {v}."),
    "Population": (["Quelle est la population {of} ?", "Combien d'habitants compte {S} ?"],
                   "La population {of} est de {v}."),
    "Population totale": (["Quelle est la population {of} ?", "Combien d'habitants compte {S} ?"],
                          "La population totale {of} est de {v}."),
    "Superficie": (["Quelle est la superficie {of} ?", "Quelle est la taille {of} ?"], "La superficie {of} est de {v}."),
    "Densité": (["Quelle est la densité de population {of} ?"], "La densité {of} est de {v}."),
    "Altitude": (["Quelle est l'altitude {of} ?"], "L'altitude {of} est de {v}."),
    "Hauteur": (["Quelle est la hauteur {of} ?", "Combien mesure {S} ?"], "La hauteur {of} est de {v}."),
    "Langues officielles": (["Quelle est la langue officielle {of} ?", "Quelles langues parle-t-on {in_} ?"],
                            "Les langues officielles {of} sont : {v}."),
    "Langue officielle": (["Quelle est la langue officielle {of} ?"], "La langue officielle {of} est {v}."),
    "Monnaie": (["Quelle est la monnaie {of} ?"], "La monnaie {of} est {v}."),
    "Président": (["Qui est le président {of} ?"], "Le président {of} est {v}."),
    "Premier ministre": (["Qui est le Premier ministre {of} ?"], "Le Premier ministre {of} est {v}."),
    "Maire": (["Qui est le maire {of} ?"], "Le maire {of} est {v}."),
    "Naissance": (["Quand est né {S} ?", "Où est né {S} ?", "Quelle est la date de naissance {of} ?"],
                  "{S} est {born} {wv}."),
    "Décès": (["Quand est mort {S} ?", "Quand est décédé {S} ?"], "{S} est {died} {wv}."),
    "Auteur": (["Qui a écrit {S} ?", "Qui est l'auteur {of} ?"], "L'auteur {of} est {v}."),
    "Réalisation": (["Qui a réalisé {S} ?", "Qui est le réalisateur {of} ?"], "{S} a été réalisé par {v}."),
    "Fondateur": (["Qui a fondé {S} ?"], "Le fondateur {of} est {v}."),
    "Fondation": (["Quand a été fondé {S} ?"], "La fondation {of} date {wv}."),
    "Architecte": (["Qui est l'architecte {of} ?"], "L'architecte {of} est {v}."),
    "Nationalité": (["Quelle est la nationalité {of} ?"], "La nationalité {of} est : {v}."),
    "Siège": (["Où se trouve le siège {of} ?"], "Le siège {of} se trouve à {v}."),
    "Genre": (["Quel est le genre {of} ?"], "Le genre {of} est : {v}."),
    "Club actuel": (["Dans quel club joue {S} ?"], "{S} joue au club {v}."),
    "Conjoint": (["Avec qui était marié {S} ?", "Qui est le conjoint {of} ?"], "Le conjoint {of} est {v}."),
    "Date de sortie": (["Quand est sorti {S} ?"], "{S} est sorti {wv}."),
    "Date de parution": (["Quand a été publié {S} ?"], "{S} a été publié en {v}."),
}

_DEFINITION_QUESTIONS = ["C'est quoi {S} ?", "Qu'est-ce que {S} ?", "Parle-moi {of}.", "{S} ?", "Explique-moi {S}."]
_PERSON_QUESTIONS = ["Qui est {S} ?", "Qui était {S} ?", "Parle-moi {of}."]


_NOISE = re.compile(r"\s*\((?:[^()]*/[^()]*/[^()]*|[^()]*[\u0370-\u03FF\u0400-\u04FF\u0590-\u06FF"
                    r"\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF][^()]*|prononcé[^()]*|écouter)\)")


def tidy(text: str) -> str:
    """Retire prononciations (« /pa.ʁi/ ») et écritures étrangères entre parenthèses."""
    return " ".join(_NOISE.sub("", text).split())


def _first_sentences(text: str, limit: int) -> str:
    """Phrases entières tenant dans `limit` caractères ; une phrase trop longue est coupée
    proprement à une virgule ou à un espace."""
    out = ""
    for sentence in split_sentences(text):
        if len(out) + len(sentence) + 1 <= limit:
            out = f"{out} {sentence}".strip()
            continue
        if not out:  # première phrase trop longue
            cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
            out = sentence[:cut].rstrip(",;: ") + "…"
        break
    return out


def _split_article(title: str, text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """(introduction, [(ligne, rubrique, valeur)]) d'un article de la bibliothèque."""
    prefix = f"{title} — "
    lead, fiche = [], []
    for line in text.split("\n"):
        if line.startswith(prefix) and " : " in line:
            label, value = line[len(prefix):].split(" : ", 1)
            fiche.append((line, label, value.rstrip(".")))
        elif line.strip():
            lead.append(line.strip())
    return tidy(" ".join(lead)), fiche


def _passage(title: str, lead: str, fiche_lines: list[str], limit: int = _SOURCE_CHARS) -> str:
    head = _first_sentences(lead, max(80, limit - sum(len(f) + 1 for f in fiche_lines)))
    return " ".join([head] + fiche_lines).strip()


def make_examples(library, doc_id: int, rng: random.Random, distractor_ids: list[int],
                  absent_ratio: float = 0.1) -> list[ReaderExample]:
    """Exercices tirés d'un article : une question de fiche (s'il y en a) et une définition."""
    title, text = library.article(doc_id)
    lead, fiche = _split_article(title, text)
    if len(lead) < 40:
        return []
    subject, of = subject_forms(title, lead)
    in_ = in_phrase(of, subject)
    born = "née" if re.search(r"\bnée\b", lead[:300]) else "né"
    died = "morte" if born == "née" else "mort"

    distractors = []
    for other in distractor_ids:
        if other == doc_id:
            continue
        o_title, o_text = library.article(other)
        o_lead, o_fiche = _split_article(o_title, o_text)
        if len(o_lead) >= 40:
            distractors.append((o_title, o_lead, o_fiche))
        if len(distractors) == 2:
            break
    if not distractors:
        return []

    examples = []
    usable = [(line, label, value) for line, label, value in fiche if label in _FICHE_TEMPLATES and value]
    if usable:
        line, label, value = rng.choice(usable)
        questions, answer_template = _FICHE_TEMPLATES[label]
        fields = {"S": subject, "of": of, "v": value, "wv": _when_where(value), "born": born, "died": died,
                  "in_": in_}
        question = rng.choice(questions).format(**fields)
        others = [f for f, lab, _ in fiche if lab != label]
        own = rng.sample(others, min(2, len(others))) + [line]
        rng.shuffle(own)
        target = (title, _passage(title, lead, own))
        # piège : la même rubrique pour un autre sujet (« Australie — Capitale : Canberra »)
        traps = []
        for o_title, o_lead, o_fiche in distractors:
            same = [f for f, lab, _ in o_fiche if lab == label]
            traps.append((o_title, _passage(o_title, o_lead, same[:1])))
        answer = answer_template.format(**fields)
        examples.append(_assemble(question, target, traps, answer[:1].upper() + answer[1:], f"fiche:{label}", rng,
                                  absent_ratio))
    person = bool(re.search(r"\bnée?\b", lead[:200]))
    question = rng.choice(_PERSON_QUESTIONS if person else _DEFINITION_QUESTIONS).format(S=subject, of=of)
    target = (title, _passage(title, lead, []))
    answer = _first_sentences(target[1], _SOURCE_CHARS)  # rien qui ne soit dans la source
    traps = [(o_title, _passage(o_title, o_lead, [])) for o_title, o_lead, _ in distractors]
    examples.append(_assemble(question, target, traps, answer, "definition", rng, absent_ratio))
    return examples


def _assemble(question: str, target: tuple[str, str], traps: list[tuple[str, str]], answer: str, kind: str,
              rng: random.Random, absent_ratio: float) -> ReaderExample:
    if rng.random() < absent_ratio:  # la réponse n'est pas dans les sources : il faut le dire
        return ReaderExample(question, traps, NOT_FOUND, "absent")
    sources = traps + [target]
    rng.shuffle(sources)
    n = sources.index(target) + 1
    return ReaderExample(question, sources, f"{answer} [{n}]", kind)


def build_dataset(library, out_dir: str | Path, max_articles: int | None = None, val_ratio: float = 0.02,
                  seed: int = 7, log=print) -> dict:
    """Fabrique les exercices (JSON par ligne) ; la validation porte sur des articles jamais vus."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_docs = library.size if max_articles is None else min(library.size, max_articles)
    counts = {"train": 0, "val": 0}
    kinds: dict[str, int] = {}
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as train, \
            open(out_dir / "val.jsonl", "w", encoding="utf-8") as val:
        for doc_id in range(n_docs):
            if doc_id and doc_id % 5000 == 0:
                log(f"  {doc_id:,} articles, {counts['train'] + counts['val']:,} exercices".replace(",", " "))
            title, text = library.article(doc_id)
            lead, _ = _split_article(title, text)
            query = f"{title} {' '.join(content_terms(lead)[:6])}"
            hits = [h.id for h in library.search(query, 6)]
            random_ids = [rng.randrange(library.size) for _ in range(2)]
            distractors = [h for h in hits if h != doc_id][:1] + random_ids  # un proche, un au hasard
            split = "val" if rng.random() < val_ratio else "train"
            for example in make_examples(library, doc_id, rng, distractors):
                (val if split == "val" else train).write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
                counts[split] += 1
                kinds[example.kind.split(":")[0]] = kinds.get(example.kind.split(":")[0], 0) + 1
    return {**counts, "types": kinds}


def load_examples(path: str | Path) -> list[ReaderExample]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            out.append(ReaderExample(data["question"], [tuple(s) for s in data["sources"]], data["answer"],
                                     data["kind"]))
    return out


# ---------------------------------------------------------------- entraînement
def _batch(encoded: list[tuple[np.ndarray, np.ndarray]], idx: np.ndarray, pad_id: int):
    length = max(len(encoded[i][0]) for i in idx)
    x = np.full((len(idx), length), pad_id, dtype=np.int64)
    y = np.full((len(idx), length), -1, dtype=np.int64)
    for row, i in enumerate(idx):
        xi, yi = encoded[i]
        x[row, : len(xi)] = xi
        y[row, : len(yi)] = yi
    return x, y


def train_reader(base_dir: str | Path, data_dir: str | Path, out_dir: str | Path, steps: int = 2000,
                 batch_size: int = 16, lr: float = 3e-4, max_len: int = 384, window: int = 256,
                 eval_every: int = 200, seed: int = 0, log=print) -> dict:
    """Apprend à Vigie (repris de `base_dir`) à répondre à partir de sources. Reprend tout seul
    après une interruption ; garde le meilleur modèle (perte de validation sur des articles
    jamais vus)."""
    import os
    import shutil

    from ..core import no_grad
    from ..core.optim import AdamW, clip_grad_norm, cosine_lr
    from .checkpoint import build_model, load_model, model_arch, save_model
    from .tokenizer import BPETokenizer

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.load(base_dir / "tokenizer.json")
    shutil.copy(base_dir / "tokenizer.json", out_dir / "tokenizer.json")

    def encode_all(path):
        pairs = [encode_example(tokenizer, ex, max_len) for ex in load_examples(path)]
        return [p for p in pairs if p is not None]

    train_set, val_set = encode_all(Path(data_dir) / "train.jsonl"), encode_all(Path(data_dir) / "val.jsonl")
    answer_tokens = sum(int((y >= 0).sum()) for _, y in train_set)
    log(f"Exercices : {len(train_set):,} d'entraînement ({answer_tokens:,} jetons de réponse), "
        f"{len(val_set):,} de validation (articles jamais vus)".replace(",", " "))

    resume = out_dir / "reprise.npz"
    start, best = 0, float("inf")
    if resume.exists():
        model, extra = load_model(resume)
        start, best = int(extra.get("step", 0)), float(extra.get("best_val", best))
        log(f"Reprise de l'entraînement du lecteur : étape {start}/{steps}")
    else:
        base, _ = load_model(base_dir / "model.npz")
        config = dict(vars(base.cfg))
        config.update(context_length=max_len, window=window)  # le lecteur voit toutes ses sources
        model = build_model(model_arch(base), **config)
        model.load_state_dict(base.state_dict())
        log(f"Départ : {base_dir / 'model.npz'} ({model.num_parameters():,} paramètres), "
            f"fenêtre {window}, contexte {max_len}".replace(",", " "))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if resume.exists() and (out_dir / "reprise-optim.npz").exists():
        with np.load(out_dir / "reprise-optim.npz") as archive:
            optimizer.load_state_dict({k: archive[k] for k in archive.files})

    rng = np.random.default_rng(seed + start)
    val_rng = np.random.default_rng(1234)
    val_batches = [val_rng.choice(len(val_set), size=min(batch_size, len(val_set)), replace=False)
                   for _ in range(8)]

    def validate() -> float:
        with no_grad():
            return float(np.mean([model.loss(*_batch(val_set, b, tokenizer.end_id)).item() for b in val_batches]))

    import time
    t0, seen = time.time(), 0
    for step in range(start, steps):
        idx = rng.choice(len(train_set), size=batch_size, replace=False)
        x, y = _batch(train_set, idx, tokenizer.end_id)
        loss = model.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        grad = clip_grad_norm(model.parameters(), 1.0)
        optimizer.step(cosine_lr(step, steps, lr, 100, 0.1))
        seen += x.size
        done = step + 1
        if done % 10 == 0:
            log(f"étape {done:5d}/{steps} | perte {loss.item():.3f} | grad {grad:.2f} | "
                f"{seen / (time.time() - t0):,.0f} jetons/s".replace(",", " "))
        if done % eval_every == 0 or done == steps:
            val = validate()
            star = ""
            if val < best:
                best, star = val, "  ★ meilleur"
                save_model(model, out_dir / "model.npz", extra={"step": done, "val_loss": val, "role": "lecteur"})
            log(f"  ↳ validation (articles inédits) : perte {val:.3f}{star}")
        if done % 50 == 0 or done == steps:
            save_model(model, out_dir / "reprise.tmp.npz", extra={"step": done, "best_val": best})
            np.savez(out_dir / "reprise-optim.tmp.npz", **optimizer.state_dict())
            os.replace(out_dir / "reprise.tmp.npz", resume)
            os.replace(out_dir / "reprise-optim.tmp.npz", out_dir / "reprise-optim.npz")
    best_model, extra = load_model(out_dir / "model.npz")
    save_model(best_model, out_dir / "model-int8.npz", extra=extra, quantize=True)
    return {"best_val": best, "steps": steps}


# ---------------------------------------------------------------- lecture et évaluation
def read(model, tokenizer, question: str, sources: list[tuple[str, str]], max_new_tokens: int = 90) -> str:
    """Réponse rédigée par le lecteur (décodage glouton : pas d'invention par hasard)."""
    from .streaming import generate

    ids = tokenizer.encode(format_prompt(question, sources))
    stop = {tokenizer.end_id} | {tokenizer.special_id(t) for t in ("<|user|>", "<|system|>")}
    out = generate(model, ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1, top_p=None,
                   stop_ids=stop, repetition_penalty=1.0)
    return tokenizer.decode(out).strip()


_FACT = re.compile(r"\d[\d  ,.]*\d|\d|\b[A-ZÉÈÀÂÎÔ][\w’'-]{2,}")


def unsupported(answer: str, question: str, sources: list[tuple[str, str]]) -> list[str]:
    """Chiffres et noms propres de la réponse absents des sources : des inventions."""
    context = " ".join([question] + [f"{t} {x}" for t, x in sources])
    flat = context.replace(" ", " ").replace(" ", " ")
    body = re.sub(r"\[\d+\]", "", answer)
    return [f for f in _FACT.findall(body) if f.replace(" ", " ").replace(" ", " ") not in flat
            and f not in ("Je", "La", "Le", "Les", "L'", "Il", "Elle")]


def evaluate_reader(model, tokenizer, examples: list[ReaderExample], limit: int = 100, log=print) -> dict:
    """Sur des articles jamais vus : bonne réponse, bonne source citée, aucune invention."""
    stats = {"exemples": 0, "bonne_reponse": 0, "bonne_source": 0, "sans_invention": 0}
    by_kind: dict[str, list[int]] = {}
    for ex in examples[:limit]:
        got = read(model, tokenizer, ex.question, ex.sources)
        ref_cite = re.findall(r"\[(\d+)\]", ex.answer)
        got_cite = re.findall(r"\[(\d+)\]", got)
        if ex.kind == "absent":
            ok = got.startswith("Je ne trouve pas")
            cite_ok = not got_cite
        else:  # mots de la réponse attendue retrouvés (presque tous pour une fiche)
            ref = set(content_terms(re.sub(r"\[\d+\]", "", ex.answer)))
            need = 0.8 if ex.kind.startswith("fiche") else 0.6
            ok = bool(ref) and len(ref & set(content_terms(got))) / len(ref) >= need
            cite_ok = got_cite[:1] == ref_cite[:1]
        faithful = not unsupported(got, ex.question, ex.sources)
        stats["exemples"] += 1
        stats["bonne_reponse"] += ok
        stats["bonne_source"] += cite_ok
        stats["sans_invention"] += faithful
        by_kind.setdefault(ex.kind.split(":")[0], []).append(int(ok))
    n = max(1, stats["exemples"])
    summary = {k: (v if k == "exemples" else round(100 * v / n, 1)) for k, v in stats.items()}
    summary["par_type"] = {k: round(100 * sum(v) / len(v), 1) for k, v in by_kind.items()}
    return summary


def normalize_text(text: str) -> str:
    from ..analysis.text import normalize

    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize(text)).split())
