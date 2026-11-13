"""Outils de traitement du texte : langue, phrases, mots, mots vides."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS_FR = set("""
a à ai aie aient aies ait alors après as au aucun aucune aujourd auprès aussi autre autres aux avaient
avais avait avant avec avoir avons ayant c ça car ce ceci cela celle celles celui cependant ces cet cette
ceux chaque chez ci comme comment d dans de des depuis donc dont du elle elles en encore entre es est et
étaient étais était étant été être eu eux fait faire fois font hors il ils j je jusqu l la le les leur
leurs lors lorsque lui m ma mais me même mêmes mes moi mon n ne ni nos notre nous on ont or ou où par
parce pas pendant peu peut peuvent plus plusieurs pour pourquoi qu quand que quel quelle quelles quels
qui quoi s sa sans se selon ses si sien soit son sont sous suis sur t ta tandis te tes toi ton tous tout
toute toutes très tu un une unes uns v va vers voici voilà vos votre vous y quelque quelques ainsi
dit dire doit être aussi bien sera seront été ceux cela leurs autant tel telle tels telles
""".split())

STOPWORDS_EN = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours yourself also
may might must shall one two new however many much use used using within without
""".split())

STOPWORDS = STOPWORDS_FR | STOPWORDS_EN

QUESTION_WORDS = {"quoi", "quel", "quelle", "quels", "quelles", "comment", "pourquoi", "combien",
                  "est-ce", "what", "why", "how", "which", "who", "when", "where", "explique",
                  "expliquer", "analyse", "analyser", "donne", "dis", "moi", "peux", "pourrais",
                  "please", "explain", "tell", "describe", "décris", "décrire"}

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_ABBREVIATIONS = {"m", "mm", "mme", "mlle", "dr", "pr", "st", "etc", "ex", "cf", "p", "pp", "vol",
                  "no", "n", "fig", "mr", "mrs", "ms", "prof", "inc", "ltd", "jr", "sr", "vs",
                  "env", "av", "apr", "j.-c", "e.g", "i.e", "u.s", "art", "hab"}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def detect_language(text: str) -> str:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    fr = sum(w in STOPWORDS_FR for w in words) + 2 * sum(c in "éèêàçùûôî" for c in text.lower())
    en = sum(w in STOPWORDS_EN for w in words)
    return "fr" if fr >= en else "en"


def light_stem(word: str) -> str:
    """Racinisation très légère (pluriels) pour mieux faire correspondre les termes."""
    if len(word) > 4 and word.endswith(("s", "x")) and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


_IDIOMS = re.compile(r"\b(?:a|ont|avait|avaient|aura|auront|aurait|avoir|va avoir)\s+(?:eu\s+)?lieu\b|"
                     r"\bse (?:passe|passait|trouve|situe)\b", re.I)


def strip_idioms(question: str) -> str:
    """Retire les tournures dont les mots ne sont pas le sujet : « a eu lieu » ne parle pas d'un lieu."""
    return _IDIOMS.sub(" ", question)


def content_terms(text: str, stem: bool = True) -> list[str]:
    """Mots porteurs de sens : minuscules, sans accents, sans mots vides."""
    out = []
    for w in _WORD_RE.findall(text):
        low = w.lower()
        if low in STOPWORDS or low in QUESTION_WORDS or len(low) < 2:
            continue
        if low.isdigit() and len(low) < 4:
            continue
        n = strip_accents(low)
        if "'" in n:  # l'énergie → energie
            n = n.split("'")[-1]
        out.append(light_stem(n) if stem else n)
    return out


def keywords(question: str) -> list[str]:
    """Mots-clés d'une question en conservant la forme d'origine (pour les requêtes web)."""
    out = []
    for w in _WORD_RE.findall(question):
        low = w.lower()
        if "'" in low:
            low = low.split("'")[-1]
            w = w.split("'")[-1]
        if low in STOPWORDS or low in QUESTION_WORDS or len(low) < 2:
            continue
        out.append(w)
    return out


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        start = 0
        for match in re.finditer(r"[.!?…]+[\"»”')\]]*\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"“(])", paragraph):
            candidate = paragraph[start:match.end()].strip()
            last_word = candidate.rstrip(".!?…\"»”')] ").split(" ")[-1].lower().strip("(")
            if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                continue
            sentences.append(candidate)
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def chunk_words(text: str, size: int = 90, overlap: int = 20) -> list[str]:
    """Découpe un texte en passages d'environ `size` mots, alignés sur les phrases."""
    # Les titres de section (lignes courtes sans ponctuation finale) sont écartés : recollés au
    # passage, ils se fondraient dans la phrase suivante (« Histoire Lyon est fondée… »).
    sentences = [s for s in split_sentences(text)
                 if s.rstrip()[-1:] in ".!?…:;»" or len(s.split()) > 12]
    chunks, current, count = [], [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > size:
            chunks.append(" ".join(current))
            keep, kept = [], 0  # chevauchement : on garde les dernières phrases
            for s in reversed(current):
                kept += len(s.split())
                if kept > overlap:
                    break
                keep.insert(0, s)
            current, count = keep, sum(len(s.split()) for s in keep)
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks
