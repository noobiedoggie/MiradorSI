"""Accès à internet : client HTTP, extraction de contenu, moteurs de recherche."""

from .extract import Document, domain_of, extract_html
from .http import HttpResponse, WebClient, WebError
from .search import (DuckDuckGoSearch, Fetcher, SearchEngine, SearchResult, SearxngSearch,
                     WikipediaSearch, normalize_url)

__all__ = [
    "Document", "domain_of", "extract_html", "HttpResponse", "WebClient", "WebError",
    "DuckDuckGoSearch", "Fetcher", "SearchEngine", "SearchResult", "SearxngSearch",
    "WikipediaSearch", "normalize_url",
]
