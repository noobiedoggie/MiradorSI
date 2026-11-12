"""Rendu Markdown des rapports d'analyse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .facts import FactGroup

if TYPE_CHECKING:  # pragma: no cover
    from .researcher import AnalysisReport

DEPTH_LABELS = {1: "rapide", 2: "standard", 3: "approfondie"}


def _cite(ids: list[int]) -> str:
    return "".join(f"[{i}]" for i in ids)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _group_line(group: FactGroup) -> str:
    values = sorted({f.display() for f in group.facts})
    context = ", ".join(group.context) or "—"
    return f"**{' / '.join(values)}** — contexte : _{context}_ {_cite(group.source_ids)}"


def render_markdown(report: "AnalysisReport") -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Rapport d'analyse — {report.analyst}")
    add("")
    add(f"> **Question :** {report.question}")
    add("")
    add(f"- **Profondeur :** {report.depth} ({DEPTH_LABELS.get(report.depth, '?')})")
    add(f"- **Sources retenues :** {len(report.sources)} sur {report.documents_seen} documents consultés")
    add(f"- **Requêtes lancées :** {len(report.queries)}")
    add(f"- **Date :** {report.started_at} · **Durée :** {report.duration_seconds:.1f} s")
    add("")

    if not report.findings:
        add("## Résultat")
        add("")
        add("Aucune information exploitable n'a pu être collectée. Vérifie l'accès à internet "
            "ou reformule la question (voir la section Méthodologie pour le détail des erreurs).")
        add("")
    else:
        add("## Synthèse")
        add("")
        top = sorted(report.findings, key=lambda f: -f.confidence)[:3]
        add(" ".join(f"{f.text} {_cite(f.citations)}" for f in top))
        add("")
        add("## Constats détaillés")
        add("")
        for i, finding in enumerate(report.findings, 1):
            support = len(finding.citations) - 1
            corroboration = (f"corroboré par {support} autre(s) source(s)" if support
                             else "source unique — à vérifier")
            add(f"{i}. {finding.text} {_cite(finding.citations)}  ")
            add(f"   _Confiance : {finding.confidence_label} ({finding.confidence:.2f}) · {corroboration}_")
        add("")

    if report.corroborated or report.divergent:
        add("## Chiffres clés")
        add("")
        if report.corroborated:
            add("### Confirmés par plusieurs sources")
            add("")
            for group in report.corroborated:
                add(f"- {_group_line(group)}")
            add("")
        if report.divergent:
            add("### Divergences à vérifier")
            add("")
            for a, b in report.divergent:
                add(f"- {_group_line(a)}  **≠**  {_group_line(b)}")
            add("")

    if report.leads:
        add("## Pistes à approfondir")
        add("")
        add("Notions récurrentes dans les sources : " + ", ".join(f"`{t}`" for t in report.leads))
        add("")
        if report.suggested_queries:
            add("Analyses de suivi suggérées :")
            add("")
            for q in report.suggested_queries:
                add(f"- `mirador analyze \"{q}\"`")
            add("")

    if report.sources:
        add("## Sources")
        add("")
        add("| # | Source | Domaine | Fiabilité | Pourquoi |")
        add("|---|--------|---------|-----------|----------|")
        for s in report.sources:
            title = _escape_cell(s.title[:90] or s.url)
            reasons = _escape_cell(", ".join(s.credibility.reasons))
            add(f"| {s.id} | [{title}]({s.url}) | {s.domain} | {s.credibility.label} "
                f"({s.credibility.score:.2f}) | {reasons} |")
        add("")

    add("## Méthodologie")
    add("")
    add("Requêtes : " + "; ".join(f"`{q}`" for q in report.queries))
    add("")
    add("Les constats sont des **extraits fidèles** des sources, sélectionnés par pertinence (BM25), "
        "recoupement entre sources (similarité TF-IDF), centralité (TextRank) et fiabilité du site. "
        "Les niveaux de confiance sont indicatifs : vérifie les sources citées avant toute décision.")
    if report.errors:
        add("")
        add(f"<details><summary>{len(report.errors)} incident(s) pendant la collecte</summary>")
        add("")
        for err in report.errors[:30]:
            add(f"- {err}")
        add("")
        add("</details>")
    add("")
    return "\n".join(lines)
