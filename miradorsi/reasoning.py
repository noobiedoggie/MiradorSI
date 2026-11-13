"""Raisonnement en étapes : Mirador enchaîne des recherches et des calculs simples.

Un petit modèle ne « raisonne » pas tout seul ; mais une méthode explicite, oui :
  • suivi de conversation : « Parle-moi du Japon. » puis « Et sa capitale ? »
  • âges : « Quel âge a Kylian Mbappé ? » → fiche (naissance) + date du jour
  • comparaisons : « Qui est le plus peuplé, Lyon ou Marseille ? » → deux fiches, un calcul
  • questions en chaîne : « Quelle est la capitale du pays où est né Chopin ? »
      né à Żelazowa Wola → en Pologne → capitale Varsovie
  • dates et conversions : « Combien de jours avant Noël ? », « 10 km en miles »
Chaque réponse montre ses étapes et ses sources : on peut vérifier le raisonnement.
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.parse
from dataclasses import dataclass, field

from .analysis.text import keywords, normalize, strip_idioms

MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTH_RE = "|".join(MONTHS + ["fevrier", "aout", "decembre"])


def french_date(day: dt.date) -> str:
    return f"{DAYS[day.weekday()]} {day.day}{'er' if day.day == 1 else ''} {MONTHS[day.month - 1]} {day.year}"


def parse_date(text: str) -> dt.date | None:
    """« 20 décembre 1998 », « 1er janvier 2000 » → date ; None sinon."""
    match = re.search(rf"\b(\d{{1,2}})(?:er)?\s+({_MONTH_RE})\s+(\d{{3,4}})", normalize_month(text))
    if not match:
        return None
    month = [normalize_month(m) for m in MONTHS].index(match.group(2)) + 1
    try:
        return dt.date(int(match.group(3)), month, int(match.group(1)))
    except ValueError:
        return None


def normalize_month(text: str) -> str:
    return text.lower().replace("février", "fevrier").replace("août", "aout").replace("décembre", "decembre")


def parse_number(text: str) -> float | None:
    """« 519 127 hab. » → 519127 ; « 68,6 millions » → 68 600 000 ; « 47,87 km2 » → 47.87."""
    text = text.replace(" ", " ").replace(" ", " ")
    match = re.search(r"\d{1,3}(?: \d{3})+(?:,\d+)?|\d+(?:,\d+)?", text)
    if not match:
        return None
    value = float(match.group(0).replace(" ", "").replace(",", "."))
    after = text[match.end(): match.end() + 12].lower()
    if after.strip().startswith("milliard"):
        value *= 1e9
    elif after.strip().startswith("million"):
        value *= 1e6
    return value


def years_between(start: dt.date, end: dt.date) -> int:
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


def _number(n: float) -> str:
    if n == int(n):
        return f"{int(n):,}".replace(",", " ")
    return f"{n:,.2f}".replace(",", " ").replace(".", ",").rstrip("0").rstrip(",")


# ---------------------------------------------------------------- outils sans données
_HOLIDAYS = {"noel": (12, 25, "Noël"), "nouvel an": (1, 1, "le Nouvel An"), "jour de l'an": (1, 1, "le jour de l'An"),
             "14 juillet": (7, 14, "le 14 Juillet"), "fete nationale": (7, 14, "la fête nationale"),
             "halloween": (10, 31, "Halloween"), "saint-valentin": (2, 14, "la Saint-Valentin"),
             "saint valentin": (2, 14, "la Saint-Valentin"), "toussaint": (11, 1, "la Toussaint"),
             "1er mai": (5, 1, "le 1er Mai"), "fete du travail": (5, 1, "la fête du Travail")}


def date_tool(message: str, today: dt.date | None = None) -> str | None:
    """Date du jour, jours restants avant une fête ou une date, jour de la semaine d'une date."""
    today = today or dt.date.today()
    q = normalize(message).replace("’", "'")
    if re.search(r"\b(quel jour (on est|sommes[- ]nous|est[- ]on|nous sommes)|quelle (est la )?date|"
                 r"on est (le )?(quel|combien)|c'est quel jour|date d'aujourd'hui)\b", q):
        return f"Nous sommes le {french_date(today)}."
    label, target, holiday = None, None, False
    found = parse_date(message)  # une date complète (« 14 juillet 1789 ») passe avant les fêtes
    if found:
        target, label = found, f"le {_day(found)}"
    else:
        for name, (month, day, shown) in _HOLIDAYS.items():
            if name in q:
                label, target, holiday = shown, dt.date(today.year, month, day), True
                break
    if target is None:
        found = _day_month(message, today)
        if found:
            target, label = found, f"le {_day(found)}"
    if target is None:
        return None
    if re.search(r"combien de (jours|semaines)|dans combien|il reste combien|jusqu'a", q):
        if target < today and holiday:
            target = target.replace(year=today.year + 1)
        days = (target - today).days
        if days == 0:
            return f"C'est aujourd'hui : {label} !"
        if days < 0:
            return f"C'était il y a {-days} jour{'s' if -days > 1 else ''} ({label})."
        rest = f" et {days % 7} jour{'s' if days % 7 > 1 else ''}" if days % 7 else ""
        weeks = f" (soit {days // 7} semaine{'s' if days // 7 > 1 else ''}{rest})" if days >= 14 else ""
        return f"Il reste {days} jour{'s' if days > 1 else ''} avant {label}{weeks}."
    if re.search(r"quel jour (de la semaine )?(etait|sera|tombe|c'etait|ce sera)", q):
        return f"Le {target.day}{'er' if target.day == 1 else ''} {MONTHS[target.month - 1]} {target.year} " \
               f"{'était' if target < today else 'tombe'} un {DAYS[target.weekday()]}."
    return None


def _day_month(message: str, today: dt.date) -> dt.date | None:
    match = re.search(rf"\b(\d{{1,2}})(?:er)?\s+({_MONTH_RE})\b(?!\s+\d)", normalize_month(message))
    if not match:
        return None
    month = [normalize_month(m) for m in MONTHS].index(match.group(2)) + 1
    try:
        day = dt.date(today.year, month, int(match.group(1)))
    except ValueError:
        return None
    return day if day >= today else day.replace(year=today.year + 1)


# unité → (grandeur, facteur vers l'unité de base)
_UNITS = {
    "km": ("longueur", 1000), "kilometre": ("longueur", 1000), "kilometres": ("longueur", 1000),
    "m": ("longueur", 1), "metre": ("longueur", 1), "metres": ("longueur", 1), "cm": ("longueur", 0.01),
    "mm": ("longueur", 0.001), "mile": ("longueur", 1609.344), "miles": ("longueur", 1609.344),
    "pied": ("longueur", 0.3048), "pieds": ("longueur", 0.3048), "ft": ("longueur", 0.3048),
    "pouce": ("longueur", 0.0254), "pouces": ("longueur", 0.0254), "inch": ("longueur", 0.0254),
    "yard": ("longueur", 0.9144), "yards": ("longueur", 0.9144),
    "kg": ("masse", 1), "kilo": ("masse", 1), "kilos": ("masse", 1), "g": ("masse", 0.001),
    "gramme": ("masse", 0.001), "grammes": ("masse", 0.001), "tonne": ("masse", 1000), "tonnes": ("masse", 1000),
    "livre": ("masse", 0.45359237), "livres": ("masse", 0.45359237), "lb": ("masse", 0.45359237),
    "lbs": ("masse", 0.45359237), "once": ("masse", 0.028349523125), "onces": ("masse", 0.028349523125),
    "l": ("volume", 1), "litre": ("volume", 1), "litres": ("volume", 1), "ml": ("volume", 0.001),
    "cl": ("volume", 0.01), "gallon": ("volume", 3.785411784), "gallons": ("volume", 3.785411784),
    "km/h": ("vitesse", 1 / 3.6), "mph": ("vitesse", 0.44704), "m/s": ("vitesse", 1), "noeud": ("vitesse", 0.514444),
    "noeuds": ("vitesse", 0.514444),
    "ha": ("surface", 10_000), "hectare": ("surface", 10_000), "hectares": ("surface", 10_000),
    "km2": ("surface", 1e6), "m2": ("surface", 1), "acre": ("surface", 4046.8564224), "acres": ("surface", 4046.8564224),
    "heure": ("duree", 3600), "heures": ("duree", 3600), "h": ("duree", 3600), "minute": ("duree", 60),
    "minutes": ("duree", 60), "min": ("duree", 60), "seconde": ("duree", 1), "secondes": ("duree", 1),
    "s": ("duree", 1), "jour": ("duree", 86400), "jours": ("duree", 86400), "semaine": ("duree", 604800),
    "semaines": ("duree", 604800), "an": ("duree", 31_557_600), "ans": ("duree", 31_557_600),
    "annee": ("duree", 31_557_600), "annees": ("duree", 31_557_600),
}
_TEMPERATURE = {"c": "C", "°c": "C", "celsius": "C", "degres celsius": "C", "f": "F", "°f": "F",
                "fahrenheit": "F", "degres fahrenheit": "F", "k": "K", "kelvin": "K", "kelvins": "K"}
_UNIT_RE = "|".join(sorted(map(re.escape, list(_UNITS) + list(_TEMPERATURE)), key=len, reverse=True))


def convert_tool(message: str) -> str | None:
    """« 10 km en miles », « 30 °C en Fahrenheit », « combien font 5 pieds en cm »."""
    q = normalize(message).replace("’", "'").replace("²", "2").replace("degres ", "degres ")
    match = re.search(rf"(-?\d+(?:[.,]\d+)?)\s*({_UNIT_RE})\s*(?:en|=|->|vers|dans|a combien de)\s*({_UNIT_RE})\b", q)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    src, dst = match.group(2), match.group(3)
    if src in _TEMPERATURE and dst in _TEMPERATURE:
        a, b = _TEMPERATURE[src], _TEMPERATURE[dst]
        kelvin = {"C": value + 273.15, "F": (value - 32) * 5 / 9 + 273.15, "K": value}[a]
        result = {"C": kelvin - 273.15, "F": (kelvin - 273.15) * 9 / 5 + 32, "K": kelvin}[b]
        names = {"C": "°C", "F": "°F", "K": "K"}
        return f"{_number(round(value, 2))} {names[a]} = {_number(round(result, 2))} {names[b]}"
    if src not in _UNITS or dst not in _UNITS or _UNITS[src][0] != _UNITS[dst][0]:
        return None
    result = value * _UNITS[src][1] / _UNITS[dst][1]
    shown = round(result, 4) if abs(result) < 100 else round(result, 1)
    return f"{_number(round(value, 4))} {match.group(2)} = {_number(shown)} {match.group(3)}"


# ---------------------------------------------------------------- faits tirés des fiches
@dataclass
class Fact:
    subject: str          # titre de l'article
    label: str            # rubrique de la fiche
    value: str
    url: str


@dataclass
class Reasoning:
    text: str
    sources: list[tuple[str, str]] = field(default_factory=list)  # (titre, url), dans l'ordre des renvois
    rewrite: str | None = None  # dernière étape à confier à l'analyse (« capitale de la Pologne ? »)


class Facts:
    """Fiches des articles : bibliothèque hors ligne d'abord, Wikipédia en ligne ensuite."""

    def __init__(self, library=None, fetcher=None):
        self.library = library
        self.fetcher = fetcher
        self._cache: dict[str, tuple[str, str] | None] = {}

    def article(self, name: str) -> tuple[str, str] | None:
        key = name.lower()
        if key in self._cache:
            return self._cache[key]
        found = None
        bare = re.sub(r"(?i)^(?:le |la |les |l'|l’)", "", name).strip()
        variants = list(dict.fromkeys([name, bare, bare[:1].upper() + bare[1:]]))  # « la tour Eiffel »
        if self.library is not None:
            for variant in variants:
                doc_id = self.library.find(variant)
                if doc_id is not None:
                    found = self.library.article(doc_id)
                    break
        name = variants[-1]
        if found is None and self.fetcher is not None:
            try:
                doc = self.fetcher.fetch(f"https://fr.wikipedia.org/wiki/{urllib.parse.quote(name.replace(' ', '_'))}")
                found = (doc.title, doc.text)
            except Exception:  # hors ligne ou article introuvable
                found = None
        self._cache[key] = found
        return found

    def fiche(self, name: str) -> tuple[str, dict[str, str]] | None:
        found = self.article(name)
        if not found:
            return None
        title, text = found
        prefix = f"{title} — "
        values = {}
        for line in text.split("\n"):
            if line.startswith(prefix) and " : " in line:
                label, value = line[len(prefix):].split(" : ", 1)
                value = value.rstrip(".")
                # « Date : de naissance 21 décembre 1977 » → rubrique « Naissance »
                split = re.match(r"(?i)^de (naissance|décès|deces|mort)\s+(.+)$", value)
                if label in ("Date", "Lieu") and split:
                    kind = "Naissance" if split.group(1).lower() == "naissance" else "Décès"
                    if label == "Date":
                        values.setdefault(kind, split.group(2))
                    else:
                        values.setdefault(f"Lieu {kind.lower()}", split.group(2))
                    continue
                values.setdefault(label, value)
        # fiches au format « Date de naissance / Lieu de naissance / Date de décès »
        for label, kind in (("Date de naissance", "Naissance"), ("Date de décès", "Décès")):
            if values.get(label):
                values.setdefault(kind, values[label])
        for label, kind in (("Lieu de naissance", "Lieu naissance"), ("Lieu de décès", "Lieu décès")):
            if values.get(label):
                values.setdefault(kind, values[label])
        if "Lieu naissance" in values and "Naissance" in values and not re.search(r"\d{3,4}\s+\S", values["Naissance"]):
            values["Naissance"] = f"{values['Naissance']} {values['Lieu naissance']}"
        return title, values

    def born_word(self, name: str) -> tuple[str, str]:
        """(« né », « mort ») ou (« née », « morte ») d'après l'introduction de l'article."""
        found = self.article(name)
        feminine = bool(found and re.search(r"\bnée\b", found[1][:600]))
        return ("née", "morte") if feminine else ("né", "mort")

    def fact(self, name: str, labels: list[str]) -> Fact | None:
        found = self.fiche(name)
        if not found:
            return None
        title, values = found
        for label in labels:
            if values.get(label):
                return Fact(title, label, values[label], wiki_url(title))
        return None


def wiki_url(title: str) -> str:
    return f"https://fr.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def _born_place(value: str) -> str:
    """« 1er mars 1810 Żelazowa Wola (Pologne) » → « Żelazowa Wola »."""
    rest = re.sub(rf"^.*?\b\d{{3,4}}\b\s*", "", value) if re.search(r"\d{3,4}", value) else value
    rest = re.sub(r"\s*\(.*", "", rest).strip(" ,")
    return rest


def _in_parentheses(value: str) -> str | None:
    match = re.search(r"\(([^()]+)\)", value)
    return match.group(1).split(",")[-1].strip() if match else None


_POP = ["Population municipale", "Population totale", "Population"]
_COMPARE = {  # adjectif → (rubriques, le plus grand gagne ?, unité affichée)
    "peuple": (_POP, True, "habitants"), "peuplee": (_POP, True, "habitants"),
    "grand": (["Superficie", "Superficie totale", "Hauteur", "Taille"], True, ""),
    "grande": (["Superficie", "Superficie totale", "Hauteur", "Taille"], True, ""),
    "vaste": (["Superficie", "Superficie totale"], True, ""), "etendu": (["Superficie", "Superficie totale"], True, ""),
    "haut": (["Hauteur", "Altitude", "Point culminant"], True, ""), "haute": (["Hauteur", "Altitude"], True, ""),
    "petit": (["Superficie", "Superficie totale", "Hauteur", "Taille"], False, ""),
    "petite": (["Superficie", "Superficie totale", "Hauteur", "Taille"], False, ""),
    "dense": (["Densité"], True, ""),
    "vieux": (["Naissance", "Fondation", "Construction", "Date"], False, "date"),
    "vieille": (["Naissance", "Fondation", "Construction", "Date"], False, "date"),
    "age": (["Naissance"], False, "date"), "agee": (["Naissance"], False, "date"),
    "ancien": (["Fondation", "Construction", "Date", "Naissance"], False, "date"),
    "ancienne": (["Fondation", "Construction", "Date", "Naissance"], False, "date"),
    "jeune": (["Naissance"], True, "date"), "recent": (["Fondation", "Construction", "Date"], True, "date"),
    "recente": (["Fondation", "Construction", "Date"], True, "date"),
}


class Reasoner:
    def __init__(self, facts: Facts, today: dt.date | None = None):
        self.facts = facts
        self.today = today or dt.date.today()

    # -- point d'entrée
    def answer(self, question: str) -> Reasoning | None:
        for method in (self.age, self.compare, self.chain):
            try:
                result = method(question)
            except Exception:  # un raisonnement qui échoue laisse la place à l'analyse normale
                result = None
            if result:
                return result
        return None

    # -- âges
    def age(self, question: str) -> Reasoning | None:
        q = question.strip()
        match = re.search(r"(?i)\bquel (?:âge|age) (?:a|avait|aurait|as) ([^?]+?)\s*(?:\?|$)", q) or \
            re.search(r"(?i)\b(?:l'âge|l'age|l’âge) (?:de|d'|d’)\s*([^?]+?)\s*(?:\?|$)", q) or \
            re.search(r"(?i)^([^?]+?) a quel (?:âge|age)\s*\??$", q)
        if not match:
            return None
        name = re.sub(r"(?i)\s*(?:à|a) sa mort$", "", match.group(1)).strip()
        born = self.facts.fact(name, ["Naissance"])
        if not born or not parse_date(born.value):
            return None
        birth = parse_date(born.value)
        died = self.facts.fact(name, ["Décès"])
        death = parse_date(died.value) if died else None
        sources = [(born.subject, born.url)]
        ne, mort = self.facts.born_word(name)
        if death:
            return Reasoning(f"{born.subject} est {mort} à {years_between(birth, death)} ans [1].\n\n"
                             f"Étapes : {ne} le {_day(birth)}, {mort} le {_day(death)} (fiche de l'article).", sources)
        return Reasoning(f"{born.subject} a {years_between(birth, self.today)} ans [1].\n\n"
                         f"Étapes : {ne} le {_day(birth)} (fiche de l'article) ; calcul au {_day(self.today)}.", sources)

    # -- comparaisons
    def compare(self, question: str) -> Reasoning | None:
        q = normalize(question).replace("’", "'")
        adjective = re.search(r"\bplus (\w+)", q)
        if not adjective or adjective.group(1) not in _COMPARE:
            return None
        text = question.strip()
        pair = re.search(r"\bentre\s+([^,:?]+?)\s+(?:et|ou)\s+([^,:?]+?)\s*(?:\?|$|,)", text) \
            or re.search(r"[,:]\s*([^,:?]+?)\s+(?:ou|et)\s+([^,:?]+?)\s*(?:\?|$)", text) \
            or re.search(r"^([A-ZÉÈ][^,:?]+?)\s+(?:ou|et)\s+([A-ZÉÈ][^,:?]+?)\s*[:,]", text)
        if not pair:
            return None
        a, b = (_clean_name(pair.group(1)), _clean_name(pair.group(2)))
        labels, bigger_wins, unit = _COMPARE[adjective.group(1)]
        if adjective.group(1) in ("grand", "grande", "petit", "petite") and re.search(r"\bvilles?\b", q):
            labels, unit = _POP, "habitants"  # « la plus grande ville » : la plus peuplée
        fa, fb = self.facts.fact(a, labels), self.facts.fact(b, labels)
        if not fa or not fb:
            return None
        if unit == "date":
            va, vb = parse_date(fa.value) or _year(fa.value), parse_date(fb.value) or _year(fb.value)
        else:
            families = [set(_POP), {"Hauteur", "Altitude", "Point culminant", "Taille"},
                        {"Superficie", "Superficie totale"}]
            if fa.label != fb.label and not any({fa.label, fb.label} <= f for f in families):
                return None  # pas la même grandeur (superficie contre hauteur)
            va, vb = parse_number(fa.value), parse_number(fb.value)
        if va is None or vb is None:
            return None
        if va == vb:
            winner = None
        else:
            winner = fa if (va > vb) == bigger_wins else fb
        sources = [(fa.subject, fa.url), (fb.subject, fb.url)]
        original = re.search(r"(?i)\bplus\s+[\wéèêàâîôûç]+", question)
        word = original.group(0).lower() if original else adjective.group(0)
        if winner is None:
            text = f"{fa.subject} et {fb.subject} sont à égalité ({fa.label.lower()})."
        else:
            loser = fb if winner is fa else fa
            text = f"{winner.subject} est {word} que {loser.subject}."
        steps = f"Étapes : {fa.label.lower()} {of(fa.subject)} : {fa.value} [1] ; " \
                f"{fb.label.lower()} {of(fb.subject)} : {fb.value} [2]."
        return Reasoning(f"{text}\n\n{steps}", sources)

    # -- questions en chaîne
    def chain(self, question: str) -> Reasoning | None:
        q = question.strip().rstrip("?").strip()
        match = re.search(r"(?i)\b(?:la |le |l'|l’)?(\w[\w ]*?) (?:du|de la) (pays|ville) (?:où est (?:né|née)|natal(?:e)? de|"
                          r"d'origine de|de naissance de)\s+(.+)$", q)
        if not match:
            return None
        attribute, kind, person = match.group(1).lower(), match.group(2), _clean_name(match.group(3))
        born = self.facts.fact(person, ["Naissance"])
        if not born:
            return None
        place = _born_place(born.value)
        steps = [f"{born.subject} est {self.facts.born_word(person)[0]} à {place} [1]"]
        sources = [(born.subject, born.url)]
        target = place
        if kind == "pays":
            country = self.facts.fact(place, ["Pays"])
            target = country.value if country else _in_parentheses(born.value)
            if not target:
                return None
            target = re.sub(r"\s*\(.*", "", target).split(",")[0].strip()
            if country:
                sources.append((country.subject, country.url))
                steps.append(f"{country.subject} est en {target} [{len(sources)}]")
            else:
                steps.append(f"{place} est en {target} (fiche de naissance)")
        from .analysis.answer import _LABEL_HINTS  # mêmes correspondances que les réponses directes

        labels = [label for word in re.findall(r"\w+", normalize(attribute))
                  for label in _LABEL_HINTS.get(word, "").split("|") if label]
        if not labels:
            return None
        final = self.facts.fact(target, labels)
        if not final:  # la fiche ne le dit pas : l'analyse cherchera dans les textes
            prefix = re.sub(r"(?i)\s*(?:du|de la)\s*$", "", q[: match.start(2)]).strip()
            return Reasoning("Étapes : " + " ; ".join(steps) + ".", sources, rewrite=f"{prefix} de {target} ?")
        sources.append((final.subject, final.url))
        steps.append(f"{final.label.lower()} {of(final.subject)} : {final.value} [{len(sources)}]")
        return Reasoning(f"{final.value} [{len(sources)}].\n\nÉtapes : " + " ; ".join(steps) + ".", sources)


def of(name: str) -> str:
    """« de Lyon », « d'Emmanuel Macron »."""
    return f"d'{name}" if name[:1].lower() in "aeiouyéèêàâîôûh" else f"de {name}"


def _day(day: dt.date) -> str:
    return f"{day.day}{'er' if day.day == 1 else ''} {MONTHS[day.month - 1]} {day.year}"


def _year(value: str) -> dt.date | None:
    match = re.search(r"\b(\d{3,4})\b", value)
    return dt.date(int(match.group(1)), 1, 1) if match else None


def _clean_name(name: str) -> str:
    name = re.sub(r"(?i)^(?:qui est|quel est|quelle est|lequel|laquelle|est|entre)\s+", "", name.strip())
    name = re.sub(r"(?i)^(?:le |la |les |l'|l’)(?=[A-ZÉÈ])", "", name)
    return name.strip(" .?!")


# ---------------------------------------------------------------- suivi de conversation
_FOLLOW_UP = re.compile(r"(?i)^(?:et |mais |alors |du coup )|\b(?:sa|son|ses|leur|leurs|il|elle|ils|elles|"
                        r"lui|y|là-bas|celui-ci|celle-ci|ce pays|cette ville|cet homme|cette femme)\b")


def last_subject(history: list[tuple[str, str]]) -> str | None:
    """Le sujet de l'échange précédent : noms propres de la dernière question, sinon ses mots-clés."""
    for question, _ in reversed(history):
        names = _proper_names(question)
        if names:
            return names[0]
        if _FOLLOW_UP.search(question):
            continue  # « Et sa capitale ? » : le sujet est plus haut dans la conversation
        words = keywords(strip_idioms(question))
        if words:
            return " ".join(words)
    return None


def _proper_names(text: str) -> list[str]:
    tokens = re.findall(r"[\w’'-]+", text)
    names, current = [], []
    for i, token in enumerate(tokens):
        token = token.replace("’", "'").split("'")[-1]
        if i > 0 and token[:1].isupper():
            current.append(token)
        else:
            if current:
                names.append(" ".join(current))
            current = []
    if current:
        names.append(" ".join(current))
    return names


def resolve_follow_up(question: str, history: list[tuple[str, str]]) -> str:
    """« Et sa capitale ? » après une question sur le Japon → « Et sa capitale ? (Japon) »."""
    if not history or _proper_names(question) or not _FOLLOW_UP.search(question):
        return question
    subject = last_subject(history)
    return f"{question.rstrip()} ({subject})" if subject else question
