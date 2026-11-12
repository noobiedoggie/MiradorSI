"""Estimation de la fiabilité d'une source (heuristiques explicables et modifiables)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Listes volontairement courtes : à enrichir selon tes domaines d'analyse.
INSTITUTIONAL_SUFFIXES = (".gouv.fr", ".gov", ".gov.uk", ".edu", ".ac.uk", ".int", ".europa.eu",
                          ".gc.ca", ".admin.ch", ".gouv.qc.ca", ".mil")
REFERENCE_DOMAINS = {
    "insee.fr": 0.95, "who.int": 0.95, "un.org": 0.9, "oecd.org": 0.9, "worldbank.org": 0.9,
    "imf.org": 0.9, "ec.europa.eu": 0.9, "eurostat.ec.europa.eu": 0.95, "ipcc.ch": 0.95,
    "nature.com": 0.9, "science.org": 0.9, "thelancet.com": 0.9, "nejm.org": 0.9,
    "ncbi.nlm.nih.gov": 0.9, "pubmed.ncbi.nlm.nih.gov": 0.9, "cnrs.fr": 0.9, "inserm.fr": 0.9,
    "inrae.fr": 0.9, "ademe.fr": 0.85, "legifrance.gouv.fr": 0.95, "service-public.fr": 0.9,
    "arxiv.org": 0.75, "hal.science": 0.8, "scholar.archive.org": 0.75,
    "wikipedia.org": 0.7, "britannica.com": 0.8, "larousse.fr": 0.75,
    "reuters.com": 0.85, "apnews.com": 0.85, "afp.com": 0.85, "bbc.com": 0.8, "bbc.co.uk": 0.8,
    "lemonde.fr": 0.8, "lefigaro.fr": 0.75, "liberation.fr": 0.75, "francetvinfo.fr": 0.75,
    "radiofrance.fr": 0.8, "lesechos.fr": 0.8, "ft.com": 0.8, "economist.com": 0.8,
    "nytimes.com": 0.8, "theguardian.com": 0.8, "washingtonpost.com": 0.8,
}
LOW_TRUST_DOMAINS = {
    "facebook.com": 0.25, "twitter.com": 0.25, "x.com": 0.25, "tiktok.com": 0.2, "instagram.com": 0.2,
    "pinterest.com": 0.15, "quora.com": 0.3, "reddit.com": 0.35, "medium.com": 0.45,
    "youtube.com": 0.35, "blogspot.com": 0.35, "wordpress.com": 0.35, "doctissimo.fr": 0.35,
}


@dataclass
class CredibilityScore:
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score >= 0.8:
            return "élevée"
        if self.score >= 0.6:
            return "bonne"
        if self.score >= 0.4:
            return "moyenne"
        return "faible"


def _lookup(domain: str, table: dict[str, float]) -> tuple[str, float] | None:
    for known, value in table.items():
        if domain == known or domain.endswith("." + known):
            return known, value
    return None


def assess(url: str, domain: str, text: str = "", published: str = "") -> CredibilityScore:
    reasons: list[str] = []
    reference = _lookup(domain, REFERENCE_DOMAINS)
    low = _lookup(domain, LOW_TRUST_DOMAINS)
    if reference:
        score = reference[1]
        reasons.append(f"source de référence connue ({reference[0]})")
    elif any(domain.endswith(s) for s in INSTITUTIONAL_SUFFIXES):
        score = 0.9
        reasons.append("site institutionnel ou académique")
    elif low:
        score = low[1]
        reasons.append(f"plateforme à contenu non vérifié ({low[0]})")
    else:
        score = 0.5
        reasons.append("source non répertoriée")

    if url.startswith("https://"):
        score += 0.02
    else:
        score -= 0.05
        reasons.append("connexion non chiffrée")
    if published:
        score += 0.03
        reasons.append("date de publication indiquée")
    length = len(text)
    if length > 4000:
        score += 0.03
        reasons.append("contenu développé")
    elif length < 800:
        score -= 0.05
        reasons.append("contenu très court")
    return CredibilityScore(round(min(max(score, 0.05), 0.99), 2), reasons)
