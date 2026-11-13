import gzip
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from miradorsi.config import WebConfig
from miradorsi.web.extract import extract_html
from miradorsi.web.http import WebClient, WebError
from miradorsi.web.search import Fetcher, normalize_url, parse_duckduckgo_html

PAGE = """<!doctype html><html lang="fr"><head><title>Énergie solaire — Dossier</title>
<meta name="description" content="Tout sur le solaire">
<meta property="article:published_time" content="2025-03-01">
<script>var tracking = 1;</script><style>body{}</style></head>
<body><nav><a href="/menu">Menu</a> Accueil</nav>
<div class="cookie-banner">Acceptez les cookies</div>
<article><h1>Le solaire en France</h1>
<p>La puissance solaire installée atteint 20 GW en 2024 selon les données publiques.</p>
<p>Les panneaux photovoltaïques convertissent la lumière en électricité. <a href="https://exemple.org/source">Source</a></p>
</article><footer>Mentions légales</footer></body></html>"""

DDG = """<div class="result results_links web-result"><h2><a class="result__a"
href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexemple.fr%2Fsolaire&rut=abc">Le <b>solaire</b></a></h2>
<a class="result__snippet" href="#">Un extrait <b>utile</b>.</a></div>
<div class="result results_links result--ad"><a class="result__a" href="https://pub.example/ad">Pub</a></div>
<div class="result results_links web-result"><a class="result__a" href="https://second.org/page">Deuxième</a>
<div class="result__snippet">Autre extrait</div></div>"""


def test_extract_html_keeps_content_and_drops_noise():
    doc = extract_html(PAGE, "https://site.fr/dossier")
    assert doc.title == "Énergie solaire — Dossier"
    assert "20 GW" in doc.text and "photovoltaïques" in doc.text
    for noise in ("tracking", "cookies", "Menu", "Mentions légales"):
        assert noise not in doc.text
    assert doc.published == "2025-03-01" and doc.language == "fr"
    assert "https://exemple.org/source" in doc.links


def test_parse_duckduckgo_html():
    results = parse_duckduckgo_html(DDG)
    assert [r.url for r in results] == ["https://exemple.fr/solaire", "https://second.org/page"]
    assert results[0].title == "Le solaire" and results[0].snippet == "Un extrait utile."
    assert results[1].snippet == "Autre extrait"


def test_normalize_url():
    assert normalize_url("http://www.Site.fr/a/?utm_source=x&id=3#top") == "https://site.fr/a?id=3"


class _Handler(BaseHTTPRequestHandler):
    hits = 0
    limited = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        _Handler.hits += 1
        if self.path == "/robots.txt":
            body = b"User-agent: *\nDisallow: /prive\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
        elif self.path.startswith("/limite"):
            _Handler.limited += 1
            if _Handler.limited == 1:  # première fois : « trop de requêtes, réessaie dans 0 s »
                body = b"trop de requetes"
                self.send_response(429)
                self.send_header("Retry-After", "0")
            else:
                body = b"ok enfin"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
        elif self.path.startswith("/attente"):
            body = b"reviens plus tard"
            self.send_response(429)
            self.send_header("Retry-After", "3600")
        elif self.path.startswith("/page"):
            body = gzip.compress(PAGE.encode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
        else:
            body = b"absent"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_client_fetch_gzip_cache_and_robots(server, tmp_path):
    client = WebClient(WebConfig(min_delay_seconds=0, cache_dir=str(tmp_path), retries=0))
    doc = Fetcher(client).fetch(server + "/page")
    assert "20 GW" in doc.text
    hits = _Handler.hits
    again = client.get(server + "/page")
    assert again.from_cache and _Handler.hits == hits
    with pytest.raises(WebError, match="robots"):
        client.get(server + "/prive/x")
    with pytest.raises(WebError, match="404"):
        client.get(server + "/manquant")


def test_client_honors_retry_after(server, tmp_path):
    client = WebClient(WebConfig(min_delay_seconds=0, cache_dir="", retries=2, respect_robots_txt=False))
    assert client.get(server + "/limite").text == "ok enfin"      # attend puis réessaie
    host = server.split("//")[1]
    slowed = client._delay_for(host)
    assert slowed >= 1.6                                          # et ralentit pour ce site
    client.get(server + "/limite?encore", use_cache=False)        # nouveau succès…
    assert client._delay_for(host) < slowed                       # … le délai redescend
    fresh = WebClient(WebConfig(min_delay_seconds=0, cache_dir="", retries=2, respect_robots_txt=False))
    start = time.monotonic()
    with pytest.raises(WebError, match="429"):
        fresh.get(server + "/attente")                            # 1 h demandée : abandon immédiat
    assert time.monotonic() - start < 2


def test_wikitext_to_text():
    from miradorsi.model.corpus import wikitext_to_text

    wikitext = """{{Infobox Pays|nom=France|capitale=[[Paris]]}}
La '''France''' est un [[État]] souverain<ref>Source.</ref> d'[[Europe de l'Ouest|Europe occidentale]].
[[Fichier:Drapeau.svg|vignette|Le [[drapeau de la France]].]]
== Géographie ==
{| class="wikitable"
| a || b
|}
* Paris compte {{unité|2.1|millions}} d'habitants au {{s|XXI}}, depuis le {{date|14|juillet|1789}}.
Voir [https://example.org le site].<!-- caché -->
[[Catégorie:Pays d'Europe]]"""
    text = wikitext_to_text(wikitext)
    assert "La France est un État souverain d'Europe occidentale." in text
    assert "Paris compte 2.1 millions d'habitants au XXIe siècle, depuis le 14 juillet 1789." in text
    assert "Voir le site." in text and "\nGéographie\n" in text
    for noise in ("{{", "[[", "<ref", "wikitable", "Fichier", "Catégorie", "caché", "'''"):
        assert noise not in text


def test_bulk_wikipedia_follows_redirects(tmp_path):
    from miradorsi.model.corpus import build_wikipedia_bulk
    from miradorsi.web.http import HttpResponse

    long_text = "Le '''Soleil''' est l'étoile du [[Système solaire]]. " * 60

    class Client:
        calls = 0

        def get(self, url, params=None, **kw):
            Client.calls += 1
            data = {"query": {"redirects": [{"from": "Astre du jour", "to": "Soleil"}],
                              "pages": [{"title": "Soleil", "revisions": [{"slots": {"main": {"content": long_text}}}]},
                                        {"title": "Inexistant", "missing": True}]}}
            return HttpResponse(url, 200, {"content-type": "application/json"},
                                __import__("json").dumps(data).encode())

    files = build_wikipedia_bulk(["Astre du jour", "Inexistant"], Client(), tmp_path, log=lambda m: None)
    assert [f.name for f in files] == ["soleil.txt"] and Client.calls == 1   # une seule requête
    assert files[0].read_text(encoding="utf-8").startswith("Soleil\n\nLe Soleil est l'étoile du Système solaire.")


def test_wikipedia_search_survives_one_language_failing():
    from miradorsi.web.http import HttpResponse
    from miradorsi.web.search import WikipediaSearch

    class Client:
        def get(self, url, **kw):
            if url.startswith("https://en."):
                raise WebError("HTTP 403 sur en.wikipedia.org")
            data = {"query": {"search": [{"title": "Tour Eiffel", "snippet": "324 <b>mètres</b>"}]}}
            return HttpResponse(url, 200, {}, __import__("json").dumps(data).encode())

    results = WikipediaSearch(Client(), ["fr", "en"]).search("tour Eiffel", 4, "fr")
    assert [r.title for r in results] == ["Tour Eiffel"] and results[0].snippet == "324 mètres"


_SEARCH_PAGE = """<ul class="mw-search-results"><li class="mw-search-result mw-search-result-ns-0">
<div class="searchResultImage-text"><div class="mw-search-result-heading"><a href="/wiki/Fusion_nucl%C3%A9aire"
 title="Fusion nucléaire" data-serp-pos="0"><span class="searchmatch">Fusion</span> nucléaire</a></div>
<div class="searchresult">La <span class="searchmatch">fusion</span> d&#039;atomes légers...</div></div></li>
<li class="mw-search-result"><div class="mw-search-result-heading"><a href="/wiki/Tokamak" title="Tokamak">Tokamak</a>
</div><div class="searchresult">Un tokamak est une chambre...</div></li></ul>"""


def test_parse_wikipedia_search_page():
    from miradorsi.web.search import parse_wikipedia_search_html

    results = parse_wikipedia_search_html(_SEARCH_PAGE, "fr")
    assert [r.title for r in results] == ["Fusion nucléaire", "Tokamak"]
    assert results[0].url == "https://fr.wikipedia.org/wiki/Fusion_nucl%C3%A9aire"
    assert results[0].snippet == "La fusion d'atomes légers..." and results[1].rank == 2


def test_wikipedia_search_reads_the_search_page_before_the_api():
    from miradorsi.web.http import HttpResponse
    from miradorsi.web.search import WikipediaSearch

    calls = []

    class Client:
        def get(self, url, params=None, **kw):
            calls.append(url)
            if "api.php" in url:
                raise WebError("HTTP 429 sur l'API")  # l'API limitée ne doit pas être nécessaire
            return HttpResponse(url, 200, {"content-type": "text/html"}, _SEARCH_PAGE.encode())

    results = WikipediaSearch(Client(), ["fr"]).search("fusion nucléaire", 4, "fr")
    assert [r.title for r in results] == ["Fusion nucléaire", "Tokamak"]
    assert all("index.php" in url for url in calls)


def test_fetch_wikipedia_reads_wikitext_and_follows_redirects():
    from miradorsi.web.http import HttpResponse, HttpStatusError
    from miradorsi.web.search import Fetcher

    pages = {"Einstein": "#REDIRECTION [[Albert Einstein#Jeunesse]]",
             "Albert Einstein": "'''Albert Einstein''' est un [[physicien]].\n== Travaux ==\n"
                                "L''''énergie''' de masse.\n== Notes et références ==\n<references/>"}
    seen = []

    class Client:
        def get(self, url, params=None, **kw):
            seen.append(url)
            title = (params or {}).get("title")
            if "api.php" in url:
                data = {"query": {"pages": {"1": {"title": "Repli", "extract": "Texte de l'API."}}}}
                return HttpResponse(url, 200, {}, __import__("json").dumps(data).encode())
            if title == "Panne":
                raise WebError("Échec : HTTP 429") from HttpStatusError(429, url)
            if title not in pages:
                raise WebError("Échec : HTTP 404") from HttpStatusError(404, url)
            return HttpResponse(url, 200, {"last-modified": "Sat, 24 Oct 2026 14:23:17 GMT"},
                                pages[title].encode())

    doc = Fetcher(Client()).fetch("https://fr.wikipedia.org/wiki/Einstein")
    assert doc.title == "Albert Einstein" and doc.url == "https://fr.wikipedia.org/wiki/Albert_Einstein"
    assert doc.text.startswith("Albert Einstein est un physicien.") and "L'énergie de masse." in doc.text
    assert "références" not in doc.text and doc.published.startswith("Sat")
    assert not any("api.php" in url for url in seen)
    with pytest.raises(WebError, match="introuvable"):
        Fetcher(Client()).fetch("https://fr.wikipedia.org/wiki/Inexistant")
    assert Fetcher(Client()).fetch("https://fr.wikipedia.org/wiki/Panne").text == "Texte de l'API."


def test_wikipedia_html_to_text_keeps_numbers_and_fiche():
    from miradorsi.web.wikitext import wikipedia_html_to_text

    page = """<div class="mw-parser-output"><section><div class="bandeau-container homonymie">Pour les
    articles homonymes, voir Lyon (homonymie).</div>
    <table class="infobox_v2 infobox"><tr><th colspan="2">Lyon</th></tr>
    <tr><th>Population<br/>municipale</th><td>519&#160;127 hab. (2023)</td></tr>
    <tr><th>Superficie</th><td>47,87 km<sup>2</sup></td></tr></table>
    <p><b>Lyon</b> est une commune<sup class="mw-ref reference"><a>[1]</a></sup> fondée au I<sup>er</sup> siècle
    av. J.-C.[réf. nécessaire] Elle compte <span class="nowrap">519 127</span> habitants.</p>
    <figure><figcaption>Une légende</figcaption></figure><style>.x{}</style>
    </section><section><div class="mw-heading"><h2>Histoire</h2></div><p>Texte de l'histoire.</p>
    <table class="wikitable"><tr><td>tableau ignoré</td></tr></table></section></div>"""
    text = wikipedia_html_to_text(page, "Lyon")
    lines = text.split("\n")
    assert lines[0] == "Lyon est une commune fondée au Ier siècle av. J.-C. Elle compte 519 127 habitants."
    assert "Lyon — Population municipale : 519 127 hab. (2023)." in lines
    assert "Lyon — Superficie : 47,87 km2." in lines
    assert lines.index("Histoire") > lines.index("Lyon — Superficie : 47,87 km2.")  # fiche après l'introduction
    for noise in ("[1]", "homonymes", "légende", "tableau ignoré", ".x{", "réf."):
        assert noise not in text
