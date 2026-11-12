"""Classement et similarité : BM25, TF-IDF, sélection diversifiée (MMR), TextRank."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    """Okapi BM25 : score de pertinence d'un passage pour une requête."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in documents]
        self.lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avg_len = float(self.lengths.mean()) if len(documents) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], weights: dict[str, float] | None = None) -> np.ndarray:
        out = np.zeros(len(self.docs))
        if not self.docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_len, 1e-9))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            w = (weights or {}).get(term, 1.0)
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=np.float64)
            out += w * idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class TfidfIndex:
    """Vecteurs TF-IDF normalisés pour mesurer la similarité cosinus entre textes."""

    def __init__(self, documents: list[list[str]]):
        df: Counter[str] = Counter()
        for d in documents:
            df.update(set(d))
        n = max(len(documents), 1)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1.0 for t, f in df.items()}
        self.vectors = [self.vectorize(d) for d in documents]

    def vectorize(self, terms: list[str]) -> dict[str, float]:
        tf = Counter(terms)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def top_terms(self, k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """Termes les plus caractéristiques de l'ensemble du corpus."""
        exclude = exclude or set()
        totals: Counter[str] = Counter()
        for vec in self.vectors:
            for t, v in vec.items():
                if t not in exclude and len(t) > 3 and not t.isdigit():
                    totals[t] += v
        return totals.most_common(k)


def mmr_select(scores: list[float], similarity, k: int, diversity: float = 0.3) -> list[int]:
    """Maximal Marginal Relevance : pertinent ET non redondant."""
    candidates = list(range(len(scores)))
    if not candidates:
        return []
    max_score = max(scores) or 1.0
    chosen: list[int] = []
    while candidates and len(chosen) < k:
        def value(i):
            redundancy = max((similarity(i, j) for j in chosen), default=0.0)
            return (1 - diversity) * scores[i] / max_score - diversity * redundancy
        best = max(candidates, key=value)
        chosen.append(best)
        candidates.remove(best)
    return chosen


def textrank(similarity_matrix: np.ndarray, damping: float = 0.85, iterations: int = 50) -> np.ndarray:
    """Centralité des phrases (PageRank sur le graphe de similarité)."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    m = similarity_matrix.copy()
    np.fill_diagonal(m, 0.0)
    row_sums = m.sum(axis=1, keepdims=True)
    m = np.divide(m, row_sums, out=np.full_like(m, 1.0 / n), where=row_sums > 0)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        rank = (1 - damping) / n + damping * (m.T @ rank)
    return rank
