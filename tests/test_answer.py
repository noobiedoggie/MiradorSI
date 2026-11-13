"""Réponse rédigée à partir d'un rapport d'analyse."""

from miradorsi.analysis.answer import compose_answer, question_kind
from miradorsi.analysis.credibility import assess
from miradorsi.analysis.researcher import AnalysisReport, Finding, Source


def _source(sid, title, text):
    url = f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return Source(sid, url, title, "fr.wikipedia.org", text, "q", "wikipedia", assess(url, "fr.wikipedia.org"))


def test_question_kind():
    assert question_kind("C'est quoi un trou noir ?") == "definition"
    assert question_kind("Qu’est-ce que la fusion nucléaire ?") == "definition"
    assert question_kind("Napoléon") == "definition"
    assert question_kind("Quelle est la population de Lyon ?") == "precise"
    assert question_kind("Pourquoi le ciel est bleu ?") == "precise"


def test_definition_answer_starts_with_the_article_lead():
    report = AnalysisReport("C'est quoi un trou noir ?", "fr", 1, "Mirador SI", "now", duration_seconds=9)
    report.sources = [
        _source(1, "Le Trou noir", "Le Trou noir (The Black Hole) est un film de science-fiction américain sorti en 1979."),
        _source(2, "Trou noir", "En astrophysique, un trou noir est un objet céleste si compact que rien ne s'en échappe. "
                                "Il est totalement invisible pour nos télescopes.\nHistoire\nLe terme apparaît en 1967 dans un article de John Wheeler."),
    ]
    report.findings = [
        Finding("Le terme trou noir apparaît en 1967 dans un article de John Wheeler.", 2, [], 0.8, 0.7),
        Finding("Le budget du film Le Trou noir dépasse vingt millions de dollars.", 1, [], 0.3, 0.6),
    ]
    text = compose_answer(report)
    assert text.startswith("En astrophysique, un trou noir est un objet céleste si compact que rien ne s'en échappe. "
                           "Il est totalement invisible pour nos télescopes. [2]")
    assert "## Points clés\n• Le terme trou noir apparaît en 1967" in text
    assert "budget du film" not in text  # trop loin de la question
    assert text.endswith("D'après 2 sources lues en 9 s.")


def test_precise_answer_uses_the_fiche_of_the_subject():
    report = AnalysisReport("Quelle est la population de Lyon ?", "fr", 1, "Mirador SI", "now", duration_seconds=5)
    report.sources = [
        _source(1, "Métropole de Lyon", "La métropole de Lyon est une collectivité.\nMétropole de Lyon — Population : 1 436 354 hab."),
        _source(2, "Lyon", "Lyon est une commune du sud-est de la France.\nLyon — Pays : France.\n"
                           "Lyon — Population municipale : 519 127 hab. (2023).\nHistoire\nLyon est fondée en 43 av. J.-C."),
    ]
    report.findings = [Finding("La commune de Lyon concentre 37 % de la population de la métropole.", 1, [], 0.9, 0.7)]
    text = compose_answer(report)
    assert text.startswith("Population municipale (Lyon) : 519 127 hab. (2023). [2]")
    assert "Pour situer : Lyon est une commune du sud-est de la France. [2]" in text
