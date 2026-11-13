"""Consignes pour les modèles de langage ouverts (Gemma, Qwen, Ministral…) qui rédigent les réponses.

Mirador SI trouve et classe les sources (internet ou bibliothèque hors ligne) ; le modèle ouvert,
exécuté sur le téléphone, rédige à partir des passages retenus et cite leurs numéros [n].
"""

from __future__ import annotations

import re

from .answer import direct_answer, lead_sentences, main_source
from .phrasing import fiche_of
from .researcher import AnalysisReport

SYSTEM = (
    "Tu es Mirador SI, une intelligence artificielle française qui aide à comprendre le monde. "
    "Tu réponds toujours en français, de façon claire, naturelle et précise, en quelques phrases "
    "(un court paragraphe, ou une petite liste si c'est plus clair), sans émojis. "
    "Quand des sources numérotées te sont données, appuie-toi d'abord sur elles et cite leur numéro "
    "entre crochets, par exemple [1]. Si elles ne suffisent pas, tu peux compléter avec tes "
    "connaissances en le disant. N'invente jamais de chiffres, de dates ou de citations."
)

CHAT_SYSTEM = (
    "Tu es Mirador SI, une intelligence artificielle française qui aide à comprendre le monde : "
    "tu cherches l'information sur internet ou dans ta bibliothèque hors ligne (Wikipédia), tu lis "
    "les sources et tu en tires des réponses sourcées. Tu fonctionnes directement sur le téléphone. "
    "Tu réponds en français, de façon chaleureuse, naturelle et brève, sans émojis."
)


def _passages(report: AnalysisReport, max_sources: int = 4, per_source: int = 650, total: int = 2400) -> list[tuple[int, str, str]]:
    """(numéro, titre, passage) des meilleures sources : introduction de l'article principal,
    puis les phrases qui répondent le mieux à la question, dans leur ordre d'importance."""
    best: dict[int, float] = {}
    for f in report.findings:
        best[f.source_id] = max(best.get(f.source_id, 0.0), f.relevance * 0.6 + f.confidence * 0.4)
    main = main_source(report)
    if main is not None:
        best[main.id] = max(best.values(), default=1.0) + 1.0  # l'article du sujet passe en premier
    out, used = [], 0
    for source_id in sorted(best, key=lambda i: -best[i])[:max_sources]:
        source = report.source(source_id)
        if source is None:
            continue
        parts: list[str] = []
        if main is not None and source_id == main.id:
            parts.extend(lead_sentences(source, max_chars=per_source // 2))
            fiche = fiche_of(source.title, source.text)
            facts = [f"{label} : {value}" for label, value in list(fiche.items())[:6]]
            if facts:
                parts.append("Fiche : " + " ; ".join(facts) + ".")
        findings = sorted((f for f in report.findings if f.source_id == source_id), key=lambda f: -f.relevance)
        for f in findings:
            if f.text not in " ".join(parts):
                parts.append(f.text)
        text = ""
        for part in parts:
            if len(text) + len(part) > per_source and text:
                break
            text = f"{text} {part}".strip()
        if not text or used + len(text) > total:
            continue
        used += len(text)
        out.append((source_id, source.title, text))
    return out


def sourced_messages(question: str, report: AnalysisReport, history: list[tuple[str, str]] = ()) -> list[dict]:
    """Conversation à donner au modèle : consignes, échanges récents, puis sources + question."""
    passages = _passages(report)
    lines = [f"[{n}] {title} — {text}" for n, title, text in passages]
    hint = direct_answer(report)
    user = "Sources :\n" + "\n".join(lines) + f"\n\nQuestion : {question}"
    if hint:
        user += f"\n(Réponse repérée dans les sources : {hint[0]} [{hint[1]}])"
    return _with_history(SYSTEM, history, user)


def chat_messages(message: str, history: list[tuple[str, str]] = ()) -> list[dict]:
    """Discussion libre (salutations, avis, écriture…) : pas de sources à citer."""
    return _with_history(CHAT_SYSTEM, history, message)


def free_messages(question: str, history: list[tuple[str, str]] = ()) -> list[dict]:
    """Question de connaissance sans source trouvée : le modèle répond de mémoire, prudemment."""
    system = SYSTEM + (" Aucune source n'a pu être consultée pour cette question : réponds avec tes "
                       "connaissances, brièvement, et signale ce qui serait à vérifier.")
    return _with_history(system, history, question)


def _with_history(system: str, history, user: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for question, answer in list(history)[-3:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer[:600]})
    messages.append({"role": "user", "content": user})
    return messages


def provided_source_ids(messages: list[dict]) -> list[int]:
    """Numéros des sources données au modèle (pour les afficher sous sa réponse)."""
    return [int(n) for n in re.findall(r"^\[(\d+)\] ", messages[-1]["content"], re.M)]
