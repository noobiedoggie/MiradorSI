"""Rédaction en français : transformer les faits d'une fiche en phrases naturelles.

« Japon — Capitale : Tokyo » devient « La capitale du Japon est Tokyo. » dans une réponse
directe, et « Sa capitale est Tokyo. » dans un petit résumé. Utilisé par les réponses de
l'analyse et par les exercices du cerveau lecteur (qui apprend ainsi à écrire de cette façon).
"""

from __future__ import annotations

import re

_VOWELS = "aeiouyéèêàâîôûh"


def subject_forms(title: str, lead: str) -> tuple[str, str]:
    """(« le Japon », « du Japon ») d'après l'introduction ; (« Lyon », « de Lyon ») sinon."""
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
    of = f"d'{base}" if base[:1].lower() in _VOWELS else f"de {base}"
    return base, of


def in_phrase(of: str, subject: str) -> str:
    """« du Japon » → « au Japon », « de la France » → « en France », « de Lyon » → « à Lyon »."""
    for prefix, preposition in (("du ", "au "), ("de la ", "en "), ("de l'", "en "), ("des ", "aux ")):
        if of.startswith(prefix):
            return preposition + of[len(prefix):]
    return f"à {subject}"


_DATE_PLACE = re.compile(r"^((?:\d{1,2}(?:er)? )?\w+ \d{3,4})\s+(.+)$")


def when_where(value: str) -> str:
    """« 7 novembre 1867 Varsovie (Pologne) » → « le 7 novembre 1867 à Varsovie (Pologne) »."""
    value = re.sub(r"\s*\((?:à )?\d+ ans\)", "", value).strip()
    match = _DATE_PLACE.match(value)
    if match and re.search(r"\d", match.group(1)):
        return f"le {match.group(1)} à {match.group(2)}"
    return f"le {value}" if value[:1].isdigit() else value


def clean_value(value: str) -> str:
    """« (1er mai 2025) 123 340 000 hab. (classé 11e) » → « 123 340 000 habitants (1er mai 2025) »."""
    value = value.strip().rstrip(".")
    dated = re.match(r"^\(([^()]*\d{4}[^()]*)\)\s*(.+)$", value)
    if dated:
        value = f"{dated.group(2)} ({dated.group(1)})"
    value = re.sub(r"\s*\(class[ée][^()]*\)", "", value)
    value = re.split(r"\s+(?:Min|Max)\.\s", value)[0]           # « 237 m Min. 162 m Max. 305 m »
    value = re.sub(r"\bhab\.(?=\s|$|\))", "habitants", value)
    value = re.sub(r"\bhab\./km2", "habitants par km²", value).replace("km2", "km²")
    return " ".join(value.split())


def capitalize(sentence: str) -> str:
    return sentence[:1].upper() + sentence[1:]


# rubrique → (phrase d'une réponse directe, phrase d'un résumé avec pronom)
_SENTENCES = {
    "Capitale": ("La capitale {of} est {v}.", "Sa capitale est {v}."),
    "Population municipale": ("La population {of} est de {v}.", "Sa population est de {v}."),
    "Population totale": ("La population {of} est de {v}.", "Sa population est de {v}."),
    "Population": ("La population {of} est de {v}.", "Sa population est de {v}."),
    "Superficie": ("La superficie {of} est de {v}.", "Sa superficie est de {v}."),
    "Superficie totale": ("La superficie {of} est de {v}.", "Sa superficie est de {v}."),
    "Densité": ("La densité {of} est de {v}.", "Sa densité est de {v}."),
    "Altitude": ("L'altitude {of} est de {v}.", "Son altitude est de {v}."),
    "Hauteur": ("La hauteur {of} est de {v}.", "Sa hauteur est de {v}."),
    "Langues officielles": ("Les langues officielles {of} sont : {v}.", "Ses langues officielles : {v}."),
    "Langue officielle": ("La langue officielle {of} est {v}.", "Sa langue officielle est {v}."),
    "Monnaie": ("La monnaie {of} est {v}.", "Sa monnaie est {v}."),
    "Président": ("Le président {of} est {v}.", None),
    "Premier ministre": ("Le Premier ministre {of} est {v}.", None),
    "Maire": ("Le maire {of} est {v}.", None),
    "Naissance": ("{S} est {born} {wv}.", "{He} est {born} {wv}."),
    "Date de naissance": ("{S} est {born} {wv}.", "{He} est {born} {wv}."),
    "Décès": ("{S} est {died} {wv}.", "{He} est {died} {wv}."),
    "Date de décès": ("{S} est {died} {wv}.", "{He} est {died} {wv}."),
    "Auteur": ("L'auteur {of} est {v}.", "Son auteur est {v}."),
    "Réalisation": ("{S} a été réalisé par {v}.", "Il a été réalisé par {v}."),
    "Fondateur": ("Le fondateur {of} est {v}.", "Son fondateur est {v}."),
    "Fondation": ("La fondation {of} date {wv}.", "Sa fondation date {wv}."),
    "Architecte": ("L'architecte {of} est {v}.", "Son architecte est {v}."),
    "Siège": ("Le siège {of} se trouve à {v}.", "Son siège se trouve à {v}."),
    "Club actuel": ("{S} joue au club {v}.", "{He} joue au club {v}."),
    "Conjoint": ("Le conjoint {of} est {v}.", None),
    "Date de sortie": ("{S} est sorti {wv}.", None),
    "Date de parution": ("{S} a été publié en {v}.", "Il a été publié en {v}."),
    "Gentilé": ("Les habitants {of} s'appellent les {v}.", "Ses habitants s'appellent les {v}."),
}

# faits retenus pour un petit résumé, selon le genre de sujet
_SUMMARY_PRIORITY = ["Naissance", "Date de naissance", "Décès", "Date de décès", "Capitale", "Population totale",
                     "Population municipale", "Population", "Langues officielles", "Langue officielle", "Auteur",
                     "Réalisation", "Date de parution", "Fondation", "Fondateur", "Siège", "Hauteur", "Altitude",
                     "Superficie", "Gentilé", "Club actuel"]


class FactPhraser:
    """Transforme les faits d'une fiche en phrases pour un sujet donné."""

    def __init__(self, title: str, lead: str):
        self.title = title
        self.subject, self.of = subject_forms(title, lead)
        feminine = bool(re.search(r"\bnée\b", lead[:400]))
        self.born, self.died, self.he = ("née", "morte", "Elle") if feminine else ("né", "mort", "Il")

    def sentence(self, label: str, value: str, summary: bool = False) -> str | None:
        templates = _SENTENCES.get(label) or _SENTENCES.get(re.sub(r"\s*\(.*\)$", "", label))
        if not templates:
            return None
        template = templates[1] if summary else templates[0]
        if template is None:
            return None
        value = clean_value(value)
        if not value:
            return None
        fields = {"S": self.subject, "of": self.of, "v": value, "wv": when_where(value), "born": self.born,
                  "died": self.died, "He": self.he}
        return capitalize(template.format(**fields))

    def summary(self, fiche: dict[str, str], max_facts: int = 2, already: str = "") -> list[str]:
        """Deux faits clés en phrases (« Sa capitale est Tokyo. Sa population est de… »), sans
        répéter ce que dit déjà l'introduction (`already`) ni reprendre une valeur confuse."""
        out, used = [], set()
        for label in _SUMMARY_PRIORITY:
            group = label.split(" ")[0]  # une seule population, une seule naissance…
            value = fiche.get(label, "")
            if not value or len(clean_value(value)) > 70 or re.search(r"\b(?:Codes?|ISO|IETF)\b", value):
                continue
            key = re.search(r"\d{3,4}|\d[\d  ]{3,}", value)  # date ou nombre déjà dans l'introduction ?
            if key and key.group(0).replace("\u202f", " ") in already.replace("\u202f", " "):
                continue
            if clean_value(value)[:20].lower() in already.lower():
                continue  # « Son auteur est Victor Hugo » après « … un roman de Victor Hugo »
            if group not in used:
                sentence = self.sentence(label, value, summary=True)
                if sentence:
                    out.append(sentence)
                    used.add(group)
            if len(out) == max_facts:
                break
        return out


def fiche_of(title: str, text: str) -> dict[str, str]:
    """Rubriques de la fiche d'un article (« Titre — Rubrique : valeur. »)."""
    prefix = f"{title} — "
    out: dict[str, str] = {}
    for line in text.split("\n"):
        if line.startswith(prefix) and " : " in line:
            label, value = line[len(prefix):].split(" : ", 1)
            out.setdefault(re.sub(r"\s*\(\d{4}\)$", "", label), value.rstrip("."))
    return out
