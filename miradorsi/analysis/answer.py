"""Rédige une réponse lisible à partir d'un rapport d'analyse.

Le rapport contient des phrases extraites et classées (les « constats »). Une liste brute
de constats se lit mal : on construit plutôt une réponse dans l'ordre où un humain
l'écrirait. D'abord l'idée principale, tirée de l'introduction de l'article qui correspond
le mieux à la question (les introductions de Wikipédia sont écrites pour définir le sujet),
puis quelques points clés qui complètent sans répéter, puis les chiffres recoupés.
"""

from __future__ import annotations

import re

from .researcher import DATE_RE, WHEN_RE, AnalysisReport, Finding, Source, is_off_topic_work
from .text import content_terms, normalize, split_sentences, strip_idioms

_DEFINITION = re.compile(
    r"^(qu'?est[- ]ce (que|qu'|qui)|c'?est quoi|ca veut dire|que veut dire|qui (est|etait|sont|etaient)|"
    r"definition|definis|explique|parle[- ]moi|presente|resume|raconte|what is|who (is|was)|tell me about)",
)
_INTERROGATIVE = re.compile(r"\b(combien|quand|ou|pourquoi|comment|quel|quelle|quels|quelles|lequel|"
                            r"est-ce que|how|when|where|why|which)\b")
_NOISE = re.compile(r"\s*\((?:prononcé[^()]*|écouter|API[^()]*|\s*[,;]?\s*)\)")


def question_kind(question: str) -> str:
    """« definition » (de quoi s'agit-il ?) ou « precise » (un fait particulier)."""
    q = normalize(question).replace("’", "'").strip(" ?!.")
    if re.match(r"(qui|quel|quelle) (est|etait|sont|etaient) (le|la|les|l') ", q):
        return "precise"  # « qui est le président de la France ? » : une personne, pas une définition
    if _DEFINITION.match(q):
        return "definition"
    if not _INTERROGATIVE.search(q) and len(content_terms(question)) <= 4:
        return "definition"  # sujet seul : « Napoléon », « fusion nucléaire »
    return "precise"


def _clean(sentence: str, limit: int = 320) -> str:
    sentence = " ".join(_NOISE.sub("", sentence).split())
    fiche = re.match(r"^(.{2,80}?) — (.{2,60}?) : (.+)$", sentence)
    if fiche:
        sentence = f"{fiche.group(2)} ({fiche.group(1)}) : {fiche.group(3)}"
    if len(sentence) > limit:
        cut = sentence.rfind(" ", 0, limit)
        sentence = sentence[: cut if cut > limit // 2 else limit].rstrip(",;:") + "…"
    return sentence


def _similar(a: str, b: str) -> bool:
    ta, tb = set(content_terms(a)), set(content_terms(b))
    return bool(ta and tb) and len(ta & tb) / min(len(ta), len(tb)) >= 0.6


def main_source(report: AnalysisReport) -> Source | None:
    """La source dont le titre correspond le mieux à la question (à égalité : la mieux classée)."""
    wanted = set(content_terms(strip_idioms(report.question)))
    best, best_score = None, 0.0
    for source in report.sources:
        title = set(content_terms(source.title))
        if not title or not wanted:
            continue
        score = len(title & wanted) / len(title | wanted)
        if is_off_topic_work(source.title, source.text[:300], report.question):
            score *= 0.5
        score -= len(source.title) / 1000  # à égalité, le titre le plus simple (« Trou noir »)
        if score > best_score:
            best, best_score = source, score
    return best


def lead_sentences(source: Source, max_chars: int = 420) -> list[str]:
    """Les premières phrases de l'introduction (avant le premier titre ou la fiche résumé)."""
    out: list[str] = []
    for line in source.text.split("\n"):
        line = line.strip()
        heading = line[-1:] not in ".!?…:;»" and len(line.split()) <= 12
        if line.startswith(f"{source.title} — ") or (heading and out):
            break  # fin de l'introduction
        for sentence in split_sentences(line):
            sentence = _clean(sentence)
            if len(sentence) < 25 or sentence.endswith(":"):
                continue
            if out and sum(len(s) for s in out) + len(sentence) > max_chars:
                return out
            out.append(sentence)
            if len(out) == 3:
                return out
    return out


# Mot de la question → rubriques de la fiche (infobox) qui y répondent
_LABEL_HINTS = {
    "ecrit": "Auteur", "ecrivain": "Auteur", "auteur": "Auteur", "redige": "Auteur",
    "realise": "Réalisation", "realisateur": "Réalisation", "tourne": "Réalisation",
    "fonde": "Fondation|Fondateur|Fondateurs", "fondateur": "Fondateur|Fondateurs", "fondation": "Fondation",
    "ne": "Naissance", "nee": "Naissance", "naissance": "Naissance", "mort": "Décès", "morte": "Décès",
    "decede": "Décès", "decedee": "Décès", "deces": "Décès", "meurt": "Décès", "capitale": "Capitale",
    "langue": "Langues officielles|Langue officielle|Langue", "parle": "Langues officielles|Langue officielle|Langue",
    "monnaie": "Monnaie", "president": "Président|Présidente|Chef de l'État", "ministre": "Premier ministre",
    "maire": "Maire", "habitant": "Population municipale|Population totale|Population",
    "population": "Population municipale|Population totale|Population", "superficie": "Superficie|Superficie totale",
    "hauteur": "Hauteur", "haut": "Hauteur|Altitude|Point culminant", "altitude": "Altitude|Point culminant",
    "architecte": "Architecte", "construit": "Construction|Architecte", "construction": "Construction",
    "pere": "Père", "mere": "Mère", "epouse": "Conjoint", "mari": "Conjoint", "conjoint": "Conjoint",
    "enfant": "Enfants", "nationalite": "Nationalité", "religion": "Religion", "siege": "Siège",
    "club": "Club actuel", "poste": "Poste", "parti": "Parti politique", "editeur": "Éditeur",
    "publie": "Date de parution|Éditeur", "devise": "Devise", "hymne": "Hymne national", "longueur": "Longueur",
    "profondeur": "Profondeur", "diametre": "Diamètre", "masse": "Masse", "distance": "Distance",
    "temperature": "Température", "densite": "Densité", "gentile": "Gentilé", "sort": "Date de sortie",
    "sortie": "Date de sortie", "inaugure": "Inauguration", "decouvert": "Découvreur|Date de découverte",
    "quand": "Date|Naissance|Fondation|Inauguration|Date de sortie|Date de parution|Construction",
}


def _fiche_answer(report: AnalysisReport, subject: Source) -> str | None:
    """La ligne de fiche qui répond : « Australie — Capitale : Canberra. » pour la capitale."""
    words = set(re.sub(r"[^a-z]+", " ", normalize(strip_idioms(report.question))).split())
    words -= set(re.sub(r"[^a-z]+", " ", normalize(subject.title)).split())  # « Marie » Curie ≠ marié
    wanted_terms = set(content_terms(strip_idioms(report.question))) - set(content_terms(subject.title))
    hinted: dict[str, int] = {}
    for word in words:
        for stem in {word, word.rstrip("s")}:
            for label in filter(None, _LABEL_HINTS.get(stem, "").split("|")):
                hinted[label] = max(hinted.get(label, 0), 1 if stem == "quand" else 2)
    prefix = f"{subject.title} — "
    best, best_score = None, 0
    for line in subject.text.split("\n"):
        if not line.startswith(prefix) or " : " not in line:
            continue
        label = line[len(prefix):].split(" : ", 1)[0]
        score = hinted.get(label, 0) + len(set(content_terms(label)) & wanted_terms)
        if score > best_score:
            best, best_score = line, score
    return best


def _direct_answer(report: AnalysisReport, subject: Source | None) -> tuple[str, int] | None:
    """Réponse directe dans l'article sujet : sa fiche (« Capitale : Canberra », « Population
    municipale : 519 127 hab. »), ou sa phrase d'introduction datée pour « quand »."""
    if subject is None:
        return None
    line = _fiche_answer(report, subject)
    if line:
        return line, subject.id
    if WHEN_RE.search(strip_idioms(report.question)):
        lead = lead_sentences(subject)
        if lead and DATE_RE.search(lead[0]):
            return lead[0], subject.id
    return None


def _cite(ids: list[int]) -> str:
    return "".join(f"[{i}]" for i in ids)


def compose_answer(report: AnalysisReport, max_points: int = 4, offline: bool = False) -> str:
    """Réponse rédigée : idée principale, points clés, chiffres recoupés, réserves."""
    if not report.findings:
        return ""
    ranked: list[Finding] = sorted(report.findings, key=lambda f: -(f.relevance * 0.6 + f.confidence * 0.4))
    used: list[str] = []
    paragraphs: list[str] = []

    kind = question_kind(report.question)
    asked = set(content_terms(strip_idioms(report.question)))

    def on_subject(source_id: int) -> bool:  # l'article parle-t-il de ce qui est demandé ?
        src = report.source(source_id)
        return bool(src and asked & set(content_terms(src.title)))

    # l'ouverture vient d'abord des articles qui portent sur le sujet (« Couleur du ciel » plutôt
    # que « Drapeau des Pays-Bas » pour « pourquoi le ciel est bleu ? »)
    opening = sorted(ranked, key=lambda f: not on_subject(f.source_id))
    source = main_source(report)
    if kind == "precise" and opening:  # le contexte doit venir de l'article qui répond
        source = report.source(opening[0].source_id) or source
    lead = lead_sentences(source) if source else []
    if kind == "definition" and lead:
        paragraphs.append(" ".join(lead) + f" {_cite([source.id])}")
        used += lead
    else:
        direct = _direct_answer(report, main_source(report))
        top = [f for f in opening[:2] if f.relevance >= 0.8 * opening[0].relevance]
        if direct:
            sentence, sid = direct
            source = report.source(sid) or source  # le contexte vient alors de l'article sujet
            lead = lead_sentences(source)
            top =[f for f in top[:1] if not _similar(f.text, sentence)]
            paragraphs.append(" ".join([f"{_clean(sentence)} {_cite([sid])}"] +
                                       [f"{_clean(f.text)} {_cite(f.citations)}" for f in top]))
            used.append(sentence)
        else:
            paragraphs.append(" ".join(f"{_clean(f.text)} {_cite(f.citations)}" for f in top))
        used += [f.text for f in top]
        if lead and source and on_subject(source.id) and not any(_similar(lead[0], u) for u in used):
            paragraphs.append(f"Pour situer : {lead[0]} {_cite([source.id])}")
            used.append(lead[0])

    points = []
    floor = 0.5 * max(f.relevance for f in ranked)
    for finding in ranked:
        if len(points) == max_points:
            break
        if finding.relevance < floor:
            continue  # trop loin de la question
        if any(_similar(finding.text, u) for u in used):
            continue
        points.append(f"• {_clean(finding.text, 260)} {_cite(finding.citations)}")
        used.append(finding.text)
    if points:
        paragraphs.append("## Points clés\n" + "\n".join(points))

    if report.corroborated:
        lines = []
        for group in report.corroborated[:3]:
            values = " / ".join(sorted({f.display() for f in group.facts}))
            lines.append(f"• {values} ({', '.join(group.context[:3])}) {_cite(group.source_ids)}")
        paragraphs.append("## Chiffres confirmés par plusieurs sources\n" + "\n".join(lines))
    count = len(report.sources)
    plural = "s" if count > 1 else ""
    if offline:
        paragraphs.append(f"D'après {count} article{plural} de la bibliothèque hors ligne (Wikipédia).")
    else:
        paragraphs.append(f"D'après {count} source{plural} lue{plural} en {report.duration_seconds:.0f} s.")
    return "\n\n".join(paragraphs)
