"""Recherche sur internet et lecture de pages : moteurs interchangeables."""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from ..config import WebConfig
from .extract import Document, domain_of, extract_html, strip_reference_sections, strip_tags
from .http import HttpStatusError, WebClient, WebError
from .wikitext import wikipedia_html_to_text, wikitext_to_text

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src)")
_WIKI_REDIRECT = re.compile(r"\s*#\s*REDIRECT\w*\s*:?\s*\[\[([^\]|#]+)", re.I)


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
def wikipedia_url(lang: str, title: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


_WIKI_RESULT = re.compile(
    r'class="mw-search-result-heading"[^>]*>\s*<a\b[^>]*?href="/wiki/[^"]+"[^>]*?title="([^"]+)"'
    r'(?:(?!mw-search-result-heading).)*?class="searchresult"[^>]*>(.*?)</div>', re.S)


def parse_wikipedia_search_html(page: str, lang: str) -> list[SearchResult]:
    """Résultats de la page de recherche de Wikipédia (Spécial:Recherche)."""
    results = []
    for title, snippet in _WIKI_RESULT.findall(page):
        title = html.unescape(title)
        results.append(SearchResult(title, wikipedia_url(lang, title), strip_tags(snippet),
                                    "wikipedia", len(results) + 1))
    return results


_SEARCH_PAGE = {"fr": "Spécial:Recherche", "de": "Spezial:Suche", "es": "Especial:Buscar"}  # évite une redirection


class WikipediaSearch:
    """Recherche dans Wikipédia.

    Les API de Wikimédia limitent fortement les adresses IP partagées (réseaux mobiles,
    serveurs) : « HTTP 429, trop de requêtes » dès la première requête. La page de recherche
    ordinaire du site ne l'est pas ; on la lit d'abord et l'API ne sert qu'en secours.
    """

    name = "wikipedia"

    def __init__(self, client: WebClient, languages: list[str] | None = None):
        self.client = client
        self.languages = languages or ["fr", "en"]

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        langs = [lang] + [l for l in self.languages if l != lang] if lang in self.languages else self.languages
        results: list[SearchResult] = []
        per_lang = max_results
        failures: list[str] = []
        for code in langs:
            if code != langs[0] and len(results) >= 2:
                break  # la langue de la question suffit : une réponse française reste en français
            try:  # une langue en panne ne doit pas faire perdre les résultats des autres
                found = self._search_page(code, query, per_lang)
            except WebError as page_error:
                try:
                    found = self._search_api(code, query, per_lang)
                except (WebError, ValueError) as exc:
                    failures.append(f"{code} : {page_error if isinstance(exc, ValueError) else exc}")
                    continue
            for item in found[:per_lang]:
                item.rank = len(results) + 1
                results.append(item)
        if failures and len(failures) == len(langs):
            raise WebError("; ".join(failures))
        return results[:max_results]

    def _search_page(self, code: str, query: str, limit: int) -> list[SearchResult]:
        resp = self.client.get(f"https://{code}.wikipedia.org/w/index.php", check_robots=False, params={
            "search": query, "title": _SEARCH_PAGE.get(code, "Special:Search"), "profile": "default", "fulltext": 1,
            "ns0": 1, "limit": limit,
        })
        results = parse_wikipedia_search_html(resp.text, code)
        if not results and "mw-search-nonefound" not in resp.text and "searchresult" not in resp.text:
            raise WebError(f"page de recherche de Wikipédia ({code}) illisible")
        return results

    def _search_api(self, code: str, query: str, limit: int) -> list[SearchResult]:
        resp = self.client.get(f"https://{code}.wikipedia.org/w/api.php", check_robots=False, params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": limit,
            "format": "json", "utf8": 1,
        })
        return [SearchResult(item["title"], wikipedia_url(code, item["title"]),
                             strip_tags(item.get("snippet", "")), "wikipedia")
                for item in resp.json().get("query", {}).get("search", [])]


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
        """Lit l'article rendu (index.php?action=render) : pas limité comme l'API, et complet
        (chiffres calculés par les modèles, fiche résumé). En secours : le wikitexte, puis l'API."""
        title = title.replace("_", " ")
        try:
            resp = self.client.get(f"https://{lang}.wikipedia.org/w/index.php", check_robots=False,
                                   params={"title": title, "action": "render"})
            text = strip_reference_sections(wikipedia_html_to_text(resp.text, title))
            if len(text) >= 200:
                return Document(url=wikipedia_url(lang, title), title=title, text=text, language=lang,
                                published=resp.headers.get("last-modified", ""))
        except WebError as exc:
            cause = exc.__cause__
            if isinstance(cause, HttpStatusError) and cause.status == 404:
                raise WebError(f"Article Wikipédia introuvable : {title}") from exc
        return self._fetch_wikipedia_source(url, lang, title)

    def _fetch_wikipedia_source(self, url: str, lang: str, title: str) -> Document:
        """Le wikitexte de l'article (index.php?action=raw), en suivant les redirections."""
        try:
            for _ in range(3):  # suit au plus deux redirections (#REDIRECTION [[…]])
                resp = self.client.get(f"https://{lang}.wikipedia.org/w/index.php", check_robots=False,
                                       params={"title": title, "action": "raw"})
                redirect = _WIKI_REDIRECT.match(resp.text)
                if not redirect:
                    break
                title = redirect.group(1).strip().replace("_", " ")
        except WebError as exc:
            cause = exc.__cause__
            if isinstance(cause, HttpStatusError) and cause.status == 404:
                raise WebError(f"Article Wikipédia introuvable : {title}") from exc
            return self._fetch_wikipedia_api(url, lang, title)  # en secours
        text = strip_reference_sections(wikitext_to_text(resp.text))
        if not text:
            raise WebError(f"Article Wikipédia vide : {title}")
        return Document(url=wikipedia_url(lang, title), title=title, text=text, language=lang,
                        published=resp.headers.get("last-modified", ""))

    def _fetch_wikipedia_api(self, url: str, lang: str, title: str) -> Document:
        resp = self.client.get(f"https://{lang}.wikipedia.org/w/api.php", check_robots=False, params={
            "action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
            "titles": title.replace("_", " "), "format": "json", "utf8": 1, "inprop": "url",
        })
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "missing" in page or not page.get("extract"):
            raise WebError(f"Article Wikipédia introuvable : {title}")
        text = strip_reference_sections(re.sub(r"\n=+ (.+?) =+\n", r"\n\1\n", page["extract"]))
        return Document(url=page.get("fullurl", url), title=page.get("title", title), text=text,
                        language=lang, published=page.get("touched", ""))
