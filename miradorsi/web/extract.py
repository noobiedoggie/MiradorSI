"""Extraction du contenu utile d'une page HTML (sans dépendance externe)."""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser

SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside",
             "form", "button", "select", "iframe", "template", "canvas", "figure"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "br", "li", "ul", "ol", "h1", "h2", "h3",
              "h4", "h5", "h6", "tr", "table", "blockquote", "pre", "dd", "dt", "td", "th"}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "wbr", "area", "base", "col"}
NOISE_HINTS = re.compile(r"cookie|banner|newsletter|subscribe|share|social|comment|sidebar|"
                         r"menu|breadcrumb|related|advert|promo|popup|modal", re.I)


@dataclass
class Document:
    url: str
    title: str = ""
    text: str = ""
    description: str = ""
    published: str = ""
    author: str = ""
    language: str = ""
    links: list[str] = field(default_factory=list)

    @property
    def domain(self) -> str:
        return domain_of(self.url)


def domain_of(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


class _Extractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.skip_depth = 0
        self.stack: list[tuple[str, bool]] = []
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.meta: dict[str, str] = {}
        self.links: list[str] = []
        self.lang = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html" and attrs.get("lang"):
            self.lang = attrs["lang"].split("-")[0].lower()
        if tag == "meta":
            key = (attrs.get("name") or attrs.get("property") or attrs.get("itemprop") or "").lower()
            if key and attrs.get("content"):
                self.meta[key] = attrs["content"]
            return
        if tag == "title":
            self.in_title = True
        if tag == "a" and attrs.get("href") and not self.skip_depth:
            href = urllib.parse.urljoin(self.base_url, attrs["href"])
            if href.startswith(("http://", "https://")):
                self.links.append(href.split("#")[0])
        if tag in VOID_TAGS:
            if tag == "br":
                self.parts.append("\n")
            return
        hint = " ".join(filter(None, [attrs.get("class"), attrs.get("id"), attrs.get("role")]))
        noisy = tag in SKIP_TAGS or bool(hint and NOISE_HINTS.search(hint) and tag in {"div", "section", "ul"})
        if attrs.get("hidden") is not None or attrs.get("aria-hidden") == "true":
            noisy = True
        self.stack.append((tag, noisy))
        if noisy:
            self.skip_depth += 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in VOID_TAGS:
            return
        # dépile jusqu'à la balise correspondante (HTML souvent mal fermé)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                for _, noisy in self.stack[i:]:
                    if noisy:
                        self.skip_depth -= 1
                del self.stack[i:]
                break
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
            return
        if not self.skip_depth:
            self.parts.append(data)


def clean_text(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\r\f\v ]+", " ", line).strip()
        if line:
            lines.append(line)
    # retire les lignes très courtes répétées (menus résiduels)
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    lines = [l for l in lines if not (counts[l] > 2 and len(l) < 60)]
    return "\n".join(lines)


def extract_html(raw_html: str, url: str = "") -> Document:
    parser = _Extractor(url)
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:  # HTMLParser est tolérant, mais on ne plante jamais sur une page
        pass
    meta = parser.meta
    title = " ".join("".join(parser.title_parts).split())
    title = meta.get("og:title") or title
    text = clean_text("".join(parser.parts))
    seen, links = set(), []
    for link in parser.links:
        if link not in seen:
            seen.add(link)
            links.append(link)
    return Document(
        url=url,
        title=html.unescape(title),
        text=text,
        description=meta.get("description") or meta.get("og:description", ""),
        published=(meta.get("article:published_time") or meta.get("datepublished")
                   or meta.get("date") or meta.get("dc.date") or ""),
        author=meta.get("author", ""),
        language=parser.lang,
        links=links,
    )


def strip_tags(fragment: str) -> str:
    """Retire les balises d'un petit fragment HTML (extraits de résultats de recherche)."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


# Sections de fin d'article (Wikipédia et autres) : listes de références, pas du contenu.
_TAIL_SECTIONS = re.compile(r"\n(Notes et références|Références|Notes|Voir aussi|Bibliographie|Liens externes|"
                            r"Articles connexes|Annexes|Sources|References|See also|Further reading|"
                            r"External links)(?:\n|$)", re.I)


def strip_reference_sections(text: str) -> str:
    """Coupe le texte au début des sections de références (si elles arrivent après le contenu)."""
    match = _TAIL_SECTIONS.search(text)
    return text[: match.start()].rstrip() if match and match.start() > len(text) * 0.3 else text
