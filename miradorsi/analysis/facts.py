"""Repérage des chiffres clés et comparaison entre sources (accords / divergences)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import content_terms

_FACT_RE = re.compile(r"""
    (?<![\w.,/-])
    (?P<cur>[$€£])?[\s ]?
    (?P<int>\d{1,3}(?:[   ]\d{3})+(?!\d)|\d{1,3}(?:,\d{3})+(?![\d,])|\d+)
    (?:[.,](?P<dec>\d+))?
    [\s  ]?
    (?P<scale>milliards?|billions?|millions?|mille|thousand|mds?\b|md\b|m(?=€|\$)|k(?=€|\$))?
    [\s ]?(?:d['’]|de\s|of\s)?[\s ]?
    (?P<unit>%|pour\s?cent|percent|€|\$|£|euros?\b|dollars?\b|km²|km\b|kg\b|tonnes?\b|°c|
             ans\b|années\b|years\b|habitants\b|personnes\b|people\b|twh\b|gwh\b|mwh\b|kwh\b|gw\b|mw\b)?
""", re.X | re.I)

_SCALES = {"milliard": 1e9, "billion": 1e9, "md": 1e9, "mds": 1e9, "million": 1e6, "m": 1e6,
           "mille": 1e3, "thousand": 1e3, "k": 1e3}
_UNITS = {"pour cent": "%", "pourcent": "%", "percent": "%", "euro": "€", "euros": "€",
          "dollar": "$", "dollars": "$", "tonne": "t", "tonnes": "t", "années": "ans", "years": "ans",
          "personnes": "habitants", "people": "habitants", "°c": "°C", "gw": "GW", "mw": "MW",
          "twh": "TWh", "gwh": "GWh", "mwh": "MWh", "kwh": "kWh"}


@dataclass
class NumericFact:
    value: float
    unit: str
    raw: str
    sentence: str
    context: frozenset[str]
    source_id: int

    def display(self) -> str:
        return format_value(self.value, self.unit)


@dataclass
class FactGroup:
    facts: list[NumericFact] = field(default_factory=list)

    @property
    def source_ids(self) -> list[int]:
        return sorted({f.source_id for f in self.facts})

    @property
    def unit(self) -> str:
        return self.facts[0].unit

    @property
    def context(self) -> list[str]:
        common = frozenset.intersection(*(f.context for f in self.facts)) if self.facts else frozenset()
        return sorted(common or self.facts[0].context)[:6]


def format_value(value: float, unit: str) -> str:
    if abs(value) >= 1e9:
        text = f"{value / 1e9:.3g} milliard(s)"
    elif abs(value) >= 1e6:
        text = f"{value / 1e6:.3g} million(s)"
    elif float(value).is_integer():
        text = f"{int(value):,}".replace(",", " ")
    else:
        text = f"{value:.4g}".replace(".", ",")
    return f"{text} {unit}".strip()


def extract_facts(sentence: str, source_id: int) -> list[NumericFact]:
    facts = []
    words_all = sentence.split()
    for match in _FACT_RE.finditer(sentence):
        integer = re.sub(r"[   ,]", "", match.group("int"))
        if not integer:
            continue
        value = float(integer + ("." + match.group("dec") if match.group("dec") else ""))
        scale = (match.group("scale") or "").lower().rstrip("s") if match.group("scale") else ""
        if scale:
            value *= _SCALES.get(scale, _SCALES.get(scale + "s", 1.0))
        unit = (match.group("unit") or match.group("cur") or "").lower()
        unit = _UNITS.get(unit, unit)
        if unit in ("€", "$", "£") or match.group("cur"):
            unit = match.group("cur") or unit
        is_year = not unit and not scale and not match.group("dec") and 1800 <= value <= 2100
        if is_year or (not unit and not scale and value < 100):
            continue  # dates et petits nombres isolés : trop ambigus pour être comparés
        # contexte : les mots autour du nombre
        prefix_words = len(sentence[: match.start()].split())
        window = " ".join(words_all[max(0, prefix_words - 8): prefix_words + 5])
        context = frozenset(t for t in content_terms(window) if not any(c.isdigit() for c in t))
        facts.append(NumericFact(value, unit, match.group(0).strip(), sentence, context, source_id))
    return facts


# Mots qui décrivent l'unité plutôt que le sujet : ils ne suffisent pas à dire que deux
# chiffres parlent de la même chose.
_UNIT_WORDS = frozenset({"twh", "gwh", "mwh", "kwh", "gw", "mw", "kw", "km", "kg", "euro", "dollar",
                         "pourcent", "percent", "an", "annee", "year", "million", "milliard", "billion",
                         "environ", "pres", "plus", "moins", "total", "soit", "contre"})


def _subject(context: frozenset) -> frozenset:
    return context - _UNIT_WORDS


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_facts(facts: list[NumericFact], context_threshold: float = 0.25,
                  tolerance: float = 0.03, divergence_threshold: float = 0.4,
                  divergence_shared: int = 3, sentence_threshold: float = 0.35
                  ) -> tuple[list[FactGroup], list[tuple[FactGroup, FactGroup]]]:
    """Regroupe les chiffres qui parlent de la même chose.

    Renvoie (chiffres corroborés par plusieurs sources, paires de groupes divergents).
    Une divergence exige un contexte beaucoup plus proche qu'un accord : deux valeurs
    égales avec un contexte voisin se confirment, mais deux valeurs différentes ne se
    contredisent que si elles décrivent vraiment le même sujet.
    """
    n = len(facts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sentence_terms = [_subject(frozenset(t for t in content_terms(f.sentence) if not any(c.isdigit() for c in t)))
                      for f in facts]
    related: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = facts[i], facts[j]
            if a.unit != b.unit:
                continue
            sa, sb = _subject(a.context), _subject(b.context)
            similarity = _jaccard(sa, sb)
            if similarity < context_threshold:
                continue
            biggest = max(abs(a.value), abs(b.value)) or 1.0
            if abs(a.value - b.value) / biggest <= tolerance:
                parent[find(i)] = find(j)
            elif (similarity >= divergence_threshold and len(sa & sb) >= divergence_shared
                  and _jaccard(sentence_terms[i], sentence_terms[j]) >= sentence_threshold
                  and biggest / (min(abs(a.value), abs(b.value)) or 1e-9) < 10):
                related.append((i, j))  # même affirmation, chiffres différents

    groups: dict[int, FactGroup] = {}
    for i in range(n):
        groups.setdefault(find(i), FactGroup()).facts.append(facts[i])

    corroborated = [g for g in groups.values() if len(g.source_ids) >= 2]
    corroborated.sort(key=lambda g: -len(g.source_ids))

    divergent, seen = [], set()
    for i, j in related:
        gi, gj = find(i), find(j)
        if gi == gj or facts[i].source_id == facts[j].source_id:
            continue
        key = (min(gi, gj), max(gi, gj))
        if key not in seen:
            seen.add(key)
            divergent.append((groups[gi], groups[gj]))
    return corroborated, divergent
