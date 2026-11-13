"""Rédige une réponse lisible à partir d'un rapport d'analyse.

Le rapport contient des phrases extraites et classées (les « constats »). Une liste brute
de constats se lit mal : on construit plutôt une réponse dans l'ordre où un humain
l'écrirait. D'abord l'idée principale, tirée de l'introduction de l'article qui correspond
le mieux à la question (les introductions de Wikipédia sont écrites pour définir le sujet),
puis quelques points clés qui complètent sans répéter, puis les chiffres recoupés.
"""

from __future__ import annotations

import re

from .researcher import (DATE_RE, QUANTITATIVE_RE, WHEN_RE, AnalysisReport, Finding, Source,
                         is_off_topic_work)
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


def _direct_answer(report: AnalysisReport, subject: Source | None) -> tuple[str, int] | None:
    """Réponse directe dans l'article sujet : sa fiche (« Population municipale : 519 127 hab. »)
    pour une question chiffrée, sa phrase d'introduction datée pour « quand »."""
    if subject is None:
        return None
    question = strip_idioms(report.question)
    if QUANTITATIVE_RE.search(question):
        wanted = set(content_terms(question)) - set(content_terms(subject.title))
        prefix = f"{subject.title} — "
        best, best_score = None, 0
        for line in subject.text.split("\n"):
            if line.startswith(prefix) and " : " in line and any(c.isdigit() for c in line):
                label = set(content_terms(line[len(prefix):].split(" : ", 1)[0]))
                score = len(label & wanted)
                if score > best_score:
                    best, best_score = line, score
        if best:
            return best, subject.id
    if WHEN_RE.search(question):
        lead = lead_sentences(subject)
        if lead and DATE_RE.search(lead[0]):
            return lead[0], subject.id
    return None


def _cite(ids: list[int]) -> str:
    return "".join(f"[{i}]" for i in ids)


def compose_answer(report: AnalysisReport, max_points: int = 4) -> str:
    """Réponse rédigée : idée principale, points clés, chiffres recoupés, réserves."""
    if not report.findings:
        return ""
    ranked: list[Finding] = sorted(report.findings, key=lambda f: -(f.relevance * 0.6 + f.confidence * 0.4))
    used: list[str] = []
    paragraphs: list[str] = []

    kind = question_kind(report.question)
    source = main_source(report)
    if kind == "precise" and ranked:  # le contexte doit venir de l'article qui répond
        source = report.source(ranked[0].source_id) or source
    lead = lead_sentences(source) if source else []
    if kind == "definition" and lead:
        paragraphs.append(" ".join(lead) + f" {_cite([source.id])}")
        used += lead
    else:
        direct = _direct_answer(report, main_source(report))
        top = [f for f in ranked[:2] if f.relevance >= 0.8 * ranked[0].relevance]
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
        if lead and not any(_similar(lead[0], u) for u in used):
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
    paragraphs.append(f"D'après {count} source{'s' if count > 1 else ''} lue{'s' if count > 1 else ''} "
                      f"en {report.duration_seconds:.0f} s.")
    return "\n\n".join(paragraphs)
