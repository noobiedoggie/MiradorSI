"""Bibliothèque hors ligne : construction, recherche, réponses sans internet."""

import json

import pytest

from miradorsi.library import (Library, LibraryBuilder, LibraryFetcher, build_from_cirrus, clean_opening_text,
                               open_library, parse_fiche)
from miradorsi.web.http import WebError

ARTICLES = {
    "Trou noir": "En astrophysique, un trou noir est un objet céleste si compact que l'intensité de son champ "
                 "gravitationnel empêche toute forme de matière ou de rayonnement de s'en échapper.",
    "Le Trou noir": "Le Trou noir (The Black Hole) est un film de science-fiction américain sorti en 1979.",
    "Lyon": "Lyon est une commune française située au confluent du Rhône et de la Saône.\n"
            "Lyon — Population municipale : 519 127 hab. (2023).",
    "Tokyo": "Tokyo est la capitale du Japon et la plus grande agglomération du monde.",
}


def _build(path):
    builder = LibraryBuilder(path, "fr", "essai")
    for popularity, (title, text) in enumerate(ARTICLES.items()):
        builder.add(title, text, importance=(10 - popularity) * 1e-6,
                    aliases=("Tōkyō",) if title == "Tokyo" else ())
    return builder.close("2025-12-29")


def test_build_search_and_read(tmp_path):
    stats = _build(tmp_path / "b.db")
    assert stats["articles"] == 4 and stats["octets"] > 0
    library = Library(tmp_path / "b.db")
    assert library.describe() == "4 articles de Wikipédia (fr, 2025-12-29)"
    hits = library.search("C'est quoi un trou noir ?", 3)
    assert hits[0].title == "Trou noir"  # le titre exact passe devant le film homonyme
    assert library.search("capitale du Japon")[0].title == "Tokyo"
    assert library.find("tōkyō") == library.find("Tokyo")  # redirection
    title, text = library.article(library.find("Lyon"))
    assert title == "Lyon" and "519 127" in text
    assert library.search("zorglub") == []

    fetcher = LibraryFetcher(library)
    doc = fetcher.fetch("https://fr.wikipedia.org/wiki/Trou_noir")
    assert doc.title == "Trou noir" and doc.text.startswith("En astrophysique")
    with pytest.raises(WebError, match="hors ligne"):
        fetcher.fetch("https://exemple.fr/page")
    assert open_library(tmp_path / "absente.db") is None


def test_agent_answers_offline_from_the_library(tmp_path):
    from miradorsi.agent import NOT_IN_LIBRARY
    from test_agent import make_agent

    _build(tmp_path / "b.db")
    agent = make_agent(tmp_path)
    agent.use_library(Library(tmp_path / "b.db"))
    text, report = agent.offline_reply("C'est quoi un trou noir ?")
    assert text.startswith("En astrophysique, un trou noir est un objet céleste")
    assert text.endswith("de la bibliothèque hors ligne (Wikipédia).") and report.sources
    text, _ = agent.offline_reply("Quelle est la population de Lyon ?")
    assert text.startswith("Population municipale (Lyon) : 519 127 hab. (2023).")
    assert agent.offline_reply("Qui a inventé le zorglub ?")[0] == NOT_IN_LIBRARY


def test_cleaning_of_real_wikipedia_openings():
    assert clean_opening_text("Antoine Meillet", "Antoine Meillet Antoine Meillet, né le 11 novembre 1866 à "
                              "Moulins, est un philologue français.") == \
        "Antoine Meillet, né le 11 novembre 1866 à Moulins, est un philologue français."
    assert clean_opening_text("Algèbre linéaire", "Pour les articles homonymes, voir Algèbre (homonymie). Algèbre "
                              "linéaire ℝ³ est un espace vectoriel de dimension 3. Les droites et plans qui passent "
                              "par l'origine sont des sous-espaces vectoriels. L’algèbre linéaire est la branche "
                              "des mathématiques qui s'intéresse aux espaces vectoriels.") == \
        "L’algèbre linéaire est la branche des mathématiques qui s'intéresse aux espaces vectoriels."
    assert clean_opening_text("Abréviation", "« Abr. » redirige ici. Ne doit pas être confondu avec Sigle. Cet "
                              "article ne cite pas suffisamment ses sources (septembre 2009). Si vous disposez "
                              "d'ouvrages ou d'articles de référence, merci de compléter l'article. En pratique : "
                              "Quelles sources sont attendues ? Comment ajouter mes sources ? Une abréviation est "
                              "le raccourcissement d'un mot.") == "Une abréviation est le raccourcissement d'un mot."
    assert parse_fiche("Lyon", ["Légende de la photo du pont.",
                                "Administration Pays France Maire Grégory Doucet Démographie Population municipale "
                                "519 127 hab. (2023) Géographie Superficie 47,87 km2"]) == [
        "Lyon — Pays : France.", "Lyon — Maire : Grégory Doucet.",
        "Lyon — Population municipale : 519 127 hab. (2023).", "Lyon — Superficie : 47,87 km2."]


def test_build_from_cirrus_keeps_the_most_popular(tmp_path):
    def page(title, popularity, opening, **extra):
        doc = {"title": title, "namespace": 0, "popularity_score": popularity, "opening_text": opening,
               "auxiliary_text": [], "redirect": [], "template": []}
        doc.update(extra)
        return [json.dumps({"index": {"_id": "1"}}).encode(), json.dumps(doc, ensure_ascii=False).encode()]

    lines = (page("Paris", 9e-5, "Paris est la capitale de la France et la plus grande ville du pays.",
                  redirect=[{"namespace": 0, "title": "Ville lumière"}])
             + page("Obscur", 1e-9, "Obscur est un village très peu connu, perdu dans les montagnes du nord.")
             + page("Mercure", 5e-5, "Mercure peut désigner plusieurs choses : une planète, un dieu romain…",
                    template=["Modèle:Homonymie"])
             + page("Lune", 7e-5, "La Lune est l'unique satellite naturel permanent de la planète Terre."))
    stats = build_from_cirrus(iter(lines), tmp_path / "c.db", articles=2, log=lambda m: None)
    library = Library(tmp_path / "c.db")
    assert stats["articles"] == 2
    assert [library.article(i)[0] for i in range(2)] == ["Paris", "Lune"]  # par popularité, sans homonymie
    assert library.find("ville lumière") == library.find("Paris")
