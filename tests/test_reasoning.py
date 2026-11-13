"""Raisonnement en étapes : outils, âges, comparaisons, chaînes, suivi de conversation."""

import datetime as dt

from miradorsi.library import Library, LibraryBuilder
from miradorsi.reasoning import (Facts, Reasoner, convert_tool, date_tool, parse_date, parse_number,
                                 resolve_follow_up)

TODAY = dt.date(2026, 11, 13)


def _facts(tmp_path):
    builder = LibraryBuilder(tmp_path / "f.db", "fr")
    articles = {
        "Lyon": "Lyon est une commune française.\nLyon — Population municipale : 519 127 hab. (2023).\nLyon — Pays : France.",
        "Marseille": "Marseille est une commune française.\nMarseille — Population municipale : 886 040 hab. (2023).",
        "Marie Curie": "Marie Curie, née le 7 novembre 1867 à Varsovie, est une physicienne.\n"
                       "Marie Curie — Naissance : 7 novembre 1867 Varsovie (Royaume du Congrès, actuelle Pologne).\n"
                       "Marie Curie — Décès : 4 juillet 1934 Passy.",
        "Varsovie": "Varsovie est la capitale de la Pologne.\nVarsovie — Pays : Pologne.",
        "Pologne": "La Pologne est un État d'Europe centrale.\nPologne — Capitale : Varsovie.",
        "Emmanuel Macron": "Emmanuel Macron, né le 21 décembre 1977 à Amiens, est un homme d'État.\n"
                           "Emmanuel Macron — Date : de naissance 21 décembre 1977 (48 ans).",
        "Kylian Mbappé": "Kylian Mbappé, né le 20 décembre 1998 à Paris, est un footballeur.\n"
                         "Kylian Mbappé — Date de naissance : 20 décembre 1998 (27 ans).",
    }
    for i, (title, text) in enumerate(articles.items()):
        builder.add(title, text, 1e-5 / (i + 1))
    builder.close()
    return Facts(library=Library(tmp_path / "f.db"))


def test_tools():
    assert date_tool("Quel jour on est ?", TODAY) == "Nous sommes le vendredi 13 novembre 2026."
    assert date_tool("Combien de jours avant Noël ?", TODAY) == "Il reste 42 jours avant Noël (soit 6 semaines)."
    assert date_tool("Quel jour de la semaine était le 14 juillet 1789 ?", TODAY) == "Le 14 juillet 1789 était un mardi."
    assert convert_tool("10 km en miles") == "10 km = 6,21 miles"
    assert convert_tool("30 °C en Fahrenheit") == "30 °C = 86 °F"
    assert convert_tool("5 kg en km") is None  # grandeurs différentes
    assert parse_number("519 127 hab. (2023)") == 519127 and parse_number("68,6 millions") == 68_600_000
    assert parse_date("né le 1er janvier 2000 à Paris") == dt.date(2000, 1, 1)


def test_age_comparison_and_chain(tmp_path):
    reasoner = Reasoner(_facts(tmp_path), TODAY)
    age = reasoner.answer("Quel âge a Emmanuel Macron ?")
    assert age.text.startswith("Emmanuel Macron a 48 ans [1].")
    dead = reasoner.answer("Quel âge avait Marie Curie à sa mort ?")
    assert dead.text.startswith("Marie Curie est morte à 66 ans [1].") and "née le 7 novembre 1867" in dead.text
    older = reasoner.answer("Qui est le plus âgé entre Emmanuel Macron et Kylian Mbappé ?")
    assert older.text.startswith("Emmanuel Macron est plus âgé que Kylian Mbappé.")
    compare = reasoner.answer("Qui est le plus peuplé, Lyon ou Marseille ?")
    assert compare.text.startswith("Marseille est plus peuplé que Lyon.")
    assert [title for title, _ in compare.sources] == ["Lyon", "Marseille"]
    chain = reasoner.answer("Quelle est la capitale du pays où est née Marie Curie ?")
    assert chain.text.startswith("Varsovie [3].")
    assert "Marie Curie est née à Varsovie [1]" in chain.text and "Varsovie est en Pologne [2]" in chain.text
    assert reasoner.answer("C'est quoi un trou noir ?") is None  # pas une question de raisonnement


def test_follow_up_keeps_the_subject():
    history = [("Parle-moi du Japon", "…"), ("Et sa capitale ?", "…")]
    assert resolve_follow_up("Et sa population ?", history) == "Et sa population ? (Japon)"
    assert resolve_follow_up("Et la France ?", history) == "Et la France ?"  # nouveau sujet nommé
    assert resolve_follow_up("C'est quoi un volcan ?", history) == "C'est quoi un volcan ?"
