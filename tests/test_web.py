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
    assert client._delay_for(host) >= 2.0                         # et ralentit pour ce site
    fresh = WebClient(WebConfig(min_delay_seconds=0, cache_dir="", retries=2, respect_robots_txt=False))
    start = time.monotonic()
    with pytest.raises(WebError, match="429"):
        fresh.get(server + "/attente")                            # 1 h demandée : abandon immédiat
    assert time.monotonic() - start < 2
