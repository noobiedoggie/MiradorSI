"""Lexique de fond : la rareté de chaque mot dans la langue courante.

Dans les documents d'une analyse, le sujet est partout (« solaire » dans des articles
sur le solaire) : la rareté *locale* ne dit pas quels mots de la question sont décisifs.
La rareté mesurée sur un corpus général, si : « production » ou « électricité » sont
courants, « solaire » l'est beaucoup moins. Une phrase qui contient les mots rares de
la question répond plus probablement à la question.

Le lexique ne contient que des statistiques de mots (aucune phrase des documents).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .text import content_terms


def build_idf(documents: list[str], min_df: int = 3) -> dict[str, float]:
    """IDF calculé par paragraphe (unités plus fines et plus nombreuses que les documents)."""
    paragraphs = [p for doc in documents for p in re.split(r"\n\s*\n|\n", doc) if len(p) > 80]
    df: Counter[str] = Counter()
    for paragraph in paragraphs:
        df.update(set(content_terms(paragraph)))
    n = max(1, len(paragraphs))
    return {t: round(math.log((n + 1) / (f + 1)) + 1.0, 2) for t, f in df.items() if f >= min_df}


def save_lexicon(idf: dict[str, float], path: Path, source: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"type": "mirador-lexique", "version": 1, "source": source, "terms": len(idf),
            "max_idf": max(idf.values(), default=1.0), "idf": dict(sorted(idf.items()))}
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


@lru_cache(maxsize=4)
def load_lexicon(path: str) -> "Lexicon | None":
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Lexicon(data["idf"], data.get("max_idf", 1.0))


class Lexicon:
    def __init__(self, idf: dict[str, float], max_idf: float):
        self.idf = idf
        self.max_idf = max_idf

    def weight(self, term: str) -> float:
        """Rareté d'un terme ; un mot inconnu du lexique est considéré comme rare."""
        return self.idf.get(term, self.max_idf)

    def question_weights(self, terms: set[str]) -> dict[str, float]:
        """Poids relatifs des termes d'une question (moyenne = 1)."""
        if not terms:
            return {}
        raw = {t: self.weight(t) for t in terms}
        mean = sum(raw.values()) / len(raw)
        return {t: v / mean for t, v in raw.items()}
