from miradorsi.analysis.credibility import assess
from miradorsi.analysis.facts import compare_facts, extract_facts
from miradorsi.analysis.rank import BM25
from miradorsi.analysis.researcher import DeepAnalyzer, question_entities
from miradorsi.analysis.text import (chunk_words, content_terms, detect_language, keywords,
                                     split_sentences)
from miradorsi.config import AnalysisConfig
from miradorsi.web.extract import Document
from miradorsi.web.http import WebError
from miradorsi.web.search import SearchResult

FILLER = (" Les installations se multiplient sur les toitures des maisons et des entrepôts. "
          "Les collectivités locales accompagnent les projets avec des aides dédiées. ")

PAGES = {
    "https://www.ecologie.gouv.fr/solaire": (
        "Le photovoltaïque en France",
        "La puissance solaire photovoltaïque installée en France atteint 20 GW fin 2024. "
        "Le solaire photovoltaïque représente environ 5 % de la production électrique française. "
        "Le gouvernement vise 100 GW de solaire installé à l'horizon 2050." + FILLER * 3),
    "https://fr.wikipedia.org/wiki/Energie_solaire_en_France": (
        "Énergie solaire en France",
        "Fin 2024, la puissance photovoltaïque installée en France atteint 20 GW selon les chiffres officiels. "
        "Le photovoltaïque fournit environ 5 % de la production électrique du pays. "
        "Les premières centrales solaires françaises datent des années 1980." + FILLER * 3),
    "https://blog-solaire.example/avis": (
        "Mon avis sur le solaire",
        "D'après notre estimation, la puissance photovoltaïque installée en France atteint 18 GW fin 2024. "
        "Le prix des panneaux solaires a fortement baissé depuis dix ans." + FILLER * 3),
    "https://copie.example/plagiat": (
        "Copie",
        "La puissance solaire photovoltaïque installée en France atteint 20 GW fin 2024. "
        "Le solaire photovoltaïque représente environ 5 % de la production électrique française. "
        "Le gouvernement vise 100 GW de solaire installé à l'horizon 2050." + FILLER * 3),
    "https://court.example/x": ("Court", "Trop court."),
}


class FakeSearch:
    def __init__(self):
        self.errors = []
        self.queries = []

    def search(self, query, max_results=8, lang="fr"):
        self.queries.append(query)
        return [SearchResult(title, url, "", "fake", rank)
                for rank, (url, (title, _)) in enumerate(PAGES.items(), 1)] + \
            [SearchResult("Panne", "https://panne.example/", "", "fake", 9)]


class FakeFetcher:
    def fetch(self, url):
        if url not in PAGES:
            raise WebError(f"HTTP 500 sur {url}")
        title, text = PAGES[url]
        return Document(url=url, title=title, text=text, published="2025-01-01")


def test_text_utils():
    assert detect_language("Quelle est la production d'électricité en France ?") == "fr"
    assert detect_language("What is the electricity production of France?") == "en"
    assert keywords("Quelle est la puissance solaire installée en France ?") == \
        ["puissance", "solaire", "installée", "France"]
    assert "energie" in content_terms("L'énergie solaire")
    assert split_sentences("M. Dupont arrive. Il parle du solaire ! Ensuite ? Fin.") == \
        ["M. Dupont arrive.", "Il parle du solaire !", "Ensuite ?", "Fin."]
    chunks = chunk_words(" ".join(f"Phrase numéro {i} assez courte." for i in range(40)), size=30)
    assert len(chunks) > 3 and all(len(c.split()) <= 40 for c in chunks)


def test_bm25_prefers_relevant_document():
    docs = [content_terms("le solaire photovoltaïque en France"),
            content_terms("la recette de la tarte aux pommes")]
    scores = BM25(docs).scores(content_terms("puissance solaire"))
    assert scores[0] > scores[1]


def test_numeric_facts_and_comparison():
    facts = extract_facts("La puissance installée atteint 20 GW et 1,5 milliard d'euros en 2024.", 1)
    values = {(f.value, f.unit) for f in facts}
    assert (20.0, "GW") in values and (1.5e9, "€") in values
    assert all(f.value != 2024 for f in facts)  # les années ne sont pas des quantités
    a = extract_facts("La puissance solaire installée atteint 20 GW en France.", 1)
    b = extract_facts("En France la puissance solaire installée atteint 20 GW.", 2)
    c = extract_facts("La puissance solaire installée en France atteint 18 GW.", 3)
    corroborated, divergent = compare_facts(a + b + c)
    assert corroborated and corroborated[0].source_ids == [1, 2]
    assert divergent


def test_credibility():
    assert assess("https://www.insee.fr/x", "insee.fr").score > 0.9
    assert assess("https://data.gouv.fr/x", "data.gouv.fr").label == "élevée"
    assert assess("http://facebook.com/x", "facebook.com").label == "faible"


def test_deep_analysis_end_to_end():
    search = FakeSearch()
    messages = []
    cfg = AnalysisConfig(max_per_domain=2, min_document_chars=200)
    analyzer = DeepAnalyzer(search, FakeFetcher(), cfg, on_progress=messages.append)
    report = analyzer.analyze("Quelle est la puissance solaire installée en France ?", depth=2)

    domains = [s.domain for s in report.sources]
    assert "copie.example" not in domains          # doublon détecté
    assert "court.example" not in domains          # trop court
    assert len(report.sources) == 3
    assert any("doublon" in e for e in report.errors)
    assert any("lecture impossible" in e for e in report.errors)
    assert len(search.queries) >= 4                 # question déclinée en plusieurs angles

    assert report.findings
    best = report.findings[0]
    assert "GW" in best.text or "%" in best.text
    assert any(f.supporting for f in report.findings)  # recoupement entre sources
    texts = [f.text for f in report.findings]
    assert not any("toitures" in t for t in texts)  # phrases génériques hors sujet écartées
    assert sum("20 GW" in t for t in texts) == 1     # reformulations fusionnées
    low = next(f for f in report.findings if "18 GW" in f.text)
    assert low.supporting == [] and low.confidence_label == "faible"  # 18 ≠ 20 : pas de recoupement
    gov = next(s for s in report.sources if s.domain == "ecologie.gouv.fr")
    blog = next(s for s in report.sources if s.domain == "blog-solaire.example")
    assert gov.credibility.score > blog.credibility.score

    units = {g.unit for g in report.corroborated}
    assert "GW" in units or "%" in units
    assert report.divergent                         # 20 GW ≠ 18 GW

    md = report.to_markdown()
    for section in ("## Synthèse", "## Constats détaillés", "## Chiffres clés", "## Sources", "## Méthodologie"):
        assert section in md
    assert "[1]" in md
    data = report.to_dict()
    assert data["sources"] and "text" not in data["sources"][0]
    assert any(m.startswith("[recherche]") for m in messages)


def test_analysis_without_internet_is_graceful():
    class Offline:
        errors = ["duckduckgo : pas de réseau"]

        def search(self, *a, **k):
            return []

    report = DeepAnalyzer(Offline(), FakeFetcher()).analyze("Question sans réseau", depth=1)
    assert not report.findings and report.errors
    assert "Aucune information exploitable" in report.to_markdown()


def test_same_site_pages_do_not_corroborate_each_other():
    shared = ("La puissance photovoltaïque installée en France atteint 20 GW fin 2024 selon le gestionnaire. "
              "Cette puissance progresse chaque année grâce aux nouvelles centrales. ")
    pages = {
        "https://fr.wikipedia.org/wiki/A": ("Solaire en France", shared * 4 + FILLER * 2),
        "https://fr.wikipedia.org/wiki/B": ("Électricité en France", shared + (
            "Le réseau électrique français relie les centrales aux consommateurs sur tout le territoire. "
            "Les échanges avec les pays voisins équilibrent l'offre et la demande en hiver. "
            "La consommation dépend fortement de la température et du chauffage électrique. ") * 5),
    }

    class Search:
        errors = []

        def search(self, query, max_results=8, lang="fr"):
            return [SearchResult(t, u, "", "fake", i) for i, (u, (t, _)) in enumerate(pages.items(), 1)]

    class Fetch:
        def fetch(self, url):
            return Document(url=url, title=pages[url][0], text=pages[url][1])

    report = DeepAnalyzer(Search(), Fetch(), AnalysisConfig(max_per_domain=5, min_document_chars=100)) \
        .analyze("Quelle est la puissance photovoltaïque installée en France ?", depth=1)
    assert len(report.sources) == 2 and report.findings
    assert all(f.supporting == [] for f in report.findings)   # même site : pas de corroboration
    assert report.corroborated == []


def test_question_entities():
    assert question_entities("Quelle est la part du solaire en France ?") == {"france"}
    assert question_entities("Quel est le rôle de l'ONU et de l'Union européenne ?") == {"onu", "union"}
    assert question_entities("Comment fonctionne la photosynthèse ?") == set()


def test_lexicon_makes_rare_question_words_decisive():
    from miradorsi.analysis.lexicon import Lexicon, build_idf

    background = ["La production d'électricité en France augmente chaque année dans les régions. " * 3,
                  "La part de la production nationale dépend de la France et de ses centrales. " * 3,
                  "L'électricité est produite en France par plusieurs filières industrielles. " * 3,
                  "Les panneaux solaires transforment la lumière en courant pour les maisons. " * 3] * 3
    idf = build_idf(background, min_df=1)
    assert idf["solaire"] > idf["production"]
    weights = Lexicon(idf, max(idf.values())).question_weights({"solaire", "production", "france"})
    assert weights["solaire"] > weights["production"] and abs(sum(weights.values()) / 3 - 1) < 1e-9
