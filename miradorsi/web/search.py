"""Recherche sur internet et lecture de pages : moteurs interchangeables."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from ..config import WebConfig
from .extract import Document, domain_of, extract_html, strip_tags
from .http import WebClient, WebError

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src)")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""
    rank: int = 0

    @property
    def domain(self) -> str:
        return domain_of(self.url)


def normalize_url(url: str) -> str:
    """Forme canonique d'une URL pour dédoublonner (sans traqueurs ni fragment)."""
    parts = urllib.parse.urlsplit(url.strip())
    query = urllib.parse.urlencode(
        [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if not TRACKING_PARAMS.match(k)]
    )
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit(("https" if parts.scheme in ("http", "https") else parts.scheme,
                                    host, path, query, ""))


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]: ...


# ---------------------------------------------------------------- DuckDuckGo
class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self.capture: str | None = None
        self.depth = 0
        self.skip_ad = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "") or ""
        if tag == "div" and "result" in cls.split():
            self.skip_ad = "result--ad" in cls
        if self.skip_ad:
            return
        if tag == "a" and "result__a" in cls:
            self.results.append({"title": "", "url": attrs.get("href", ""), "snippet": ""})
            self.capture, self.depth = "title", 0
        elif "result__snippet" in cls and self.results:
            self.capture, self.depth = "snippet", 0
        elif self.capture:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.capture:
            if self.depth == 0:
                self.capture = None
            else:
                self.depth -= 1

    def handle_data(self, data):
        if self.capture and self.results:
            self.results[-1][self.capture] += data


def _ddg_real_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return urllib.parse.unquote(target) if target else ""
    return href


def parse_duckduckgo_html(page: str) -> list[SearchResult]:
    parser = _DDGParser()
    parser.feed(page)
    results = []
    for raw in parser.results:
        url = _ddg_real_url(raw["url"])
        if not url.startswith(("http://", "https://")) or "duckduckgo.com/y.js" in url:
            continue
        results.append(SearchResult(" ".join(raw["title"].split()), url,
                                    " ".join(raw["snippet"].split()), "duckduckgo", len(results) + 1))
    return results


class DuckDuckGoSearch:
    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, client: WebClient):
        self.client = client

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        region = {"fr": "fr-fr", "en": "us-en"}.get(lang, "wt-wt")
        resp = self.client.get(self.endpoint, params={"q": query, "kl": region}, check_robots=False)
        results = parse_duckduckgo_html(resp.text)
        if not results and "anomaly" in resp.text.lower():
            raise WebError("DuckDuckGo a demandé une vérification anti-robot")
        return results[:max_results]


# ---------------------------------------------------------------- Wikipédia
class WikipediaSearch:
    name = "wikipedia"

    def __init__(self, client: WebClient, languages: list[str] | None = None):
        self.client = client
        self.languages = languages or ["fr", "en"]

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        langs = [lang] + [l for l in self.languages if l != lang] if lang in self.languages else self.languages
        results: list[SearchResult] = []
        per_lang = max(2, max_results // max(1, len(langs)))
        for code in langs:
            resp = self.client.get(f"https://{code}.wikipedia.org/w/api.php", check_robots=False, params={
                "action": "query", "list": "search", "srsearch": query, "srlimit": per_lang,
                "format": "json", "utf8": 1,
            })
            for item in resp.json().get("query", {}).get("search", []):
                title = item["title"]
                url = f"https://{code}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append(SearchResult(title, url, strip_tags(item.get("snippet", "")),
                                            "wikipedia", len(results) + 1))
        return results[:max_results]


# ---------------------------------------------------------------- SearXNG
class SearxngSearch:
    """Métamoteur auto-hébergeable (https://docs.searxng.org) : idéal pour personnaliser."""

    name = "searxng"

    def __init__(self, client: WebClient, base_url: str):
        self.client = client
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        resp = self.client.get(f"{self.base_url}/search", check_robots=False,
                               params={"q": query, "format": "json", "language": lang})
        out = []
        for item in resp.json().get("results", [])[:max_results]:
            out.append(SearchResult(item.get("title", ""), item.get("url", ""),
                                    item.get("content", ""), "searxng", len(out) + 1))
        return out


# ---------------------------------------------------------------- agrégateur
class SearchEngine:
    """Interroge plusieurs moteurs, fusionne et dédoublonne les résultats."""

    def __init__(self, providers: list[SearchProvider]):
        self.providers = providers
        self.errors: list[str] = []

    @classmethod
    def from_config(cls, cfg: WebConfig, client: WebClient) -> "SearchEngine":
        providers: list[SearchProvider] = []
        for name in cfg.search_providers:
            if name == "duckduckgo":
                providers.append(DuckDuckGoSearch(client))
            elif name == "wikipedia":
                providers.append(WikipediaSearch(client, cfg.wikipedia_languages))
            elif name == "searxng" and cfg.searxng_url:
                providers.append(SearxngSearch(client, cfg.searxng_url))
        return cls(providers)

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        batches = []
        for provider in self.providers:
            try:
                batches.append(provider.search(query, max_results, lang))
            except (WebError, ValueError, KeyError) as exc:
                self.errors.append(f"{provider.name} : {exc}")
        merged, seen = [], set()
        for i in range(max((len(b) for b in batches), default=0)):  # entrelace les moteurs
            for batch in batches:
                if i < len(batch):
                    key = normalize_url(batch[i].url)
                    if key not in seen:
                        seen.add(key)
                        merged.append(batch[i])
        return merged[: max_results * max(1, len(self.providers))]


# ---------------------------------------------------------------- lecture
class Fetcher:
    """Télécharge une URL et renvoie un Document texte propre."""

    def __init__(self, client: WebClient):
        self.client = client

    def fetch(self, url: str) -> Document:
        wiki = re.match(r"https?://(\w+)\.wikipedia\.org/wiki/(.+)", url)
        if wiki:
            return self._fetch_wikipedia(url, wiki.group(1), urllib.parse.unquote(wiki.group(2)))
        resp = self.client.get(url)
        ctype = resp.content_type
        if ctype in ("text/html", "application/xhtml+xml", "") or ctype.endswith("+xml") and "html" in ctype:
            doc = extract_html(resp.text, resp.url)
        elif ctype.startswith("text/") or ctype == "application/json":
            doc = Document(url=resp.url, title=url.rsplit("/", 1)[-1], text=resp.text)
        else:
            raise WebError(f"Type de contenu non pris en charge ({ctype}) : {url}")
        doc.url = doc.url or url
        return doc

    def _fetch_wikipedia(self, url: str, lang: str, title: str) -> Document:
        resp = self.client.get(f"https://{lang}.wikipedia.org/w/api.php", check_robots=False, params={
            "action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
            "titles": title.replace("_", " "), "format": "json", "utf8": 1, "inprop": "url",
        })
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "missing" in page or not page.get("extract"):
            raise WebError(f"Article Wikipédia introuvable : {title}")
        text = re.sub(r"\n=+ (.+?) =+\n", r"\n\1\n", page["extract"])
        return Document(url=page.get("fullurl", url), title=page.get("title", title), text=text,
                        language=lang, published=page.get("touched", ""))
