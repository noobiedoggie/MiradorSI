"""Construction d'un corpus d'entraînement à partir d'internet (Wikipédia)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

from ..analysis.text import strip_accents
from ..web.http import WebClient, WebError
from ..web.search import Fetcher, WikipediaSearch


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", strip_accents(text.lower())).strip("-")
    return slug[:60] or "document"


# Sections de fin d'article Wikipédia : listes de liens et de références, pas de la langue.
_TAIL_SECTIONS = re.compile(r"\n(Notes et références|Références|Voir aussi|Bibliographie|Liens externes|"
                            r"Articles connexes|Annexes|Sources)\n", re.I)


_LOW_VALUE_TITLE = re.compile(r"\(homonymie\)|^Liste (des|de)\b", re.I)


def clean_article(text: str, max_chars: int) -> str:
    """Retire les sections de références et tronque proprement à la fin d'un paragraphe."""
    match = _TAIL_SECTIONS.search(text)
    if match:
        text = text[: match.start()]
    if len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > max_chars // 2 else max_chars]
    return text.strip()


def build_web_corpus(topics: list[str], wiki: WikipediaSearch, fetcher: Fetcher, out_dir: Path,
                     lang: str = "fr", per_topic: int = 3, max_chars_per_doc: int = 60_000,
                     log: Callable[[str], None] = print) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for topic in topics:
        try:
            results = wiki.search(topic, max_results=per_topic, lang=lang)
        except (WebError, ValueError, KeyError) as exc:
            log(f"  ✗ recherche « {topic} » : {exc}")
            continue
        for result in results[:per_topic]:
            if _LOW_VALUE_TITLE.search(result.title):
                continue  # pages d'homonymie et listes : des liens, pas de la langue
            path = out_dir / f"{slugify(result.title)}.txt"
            if path.exists():
                continue  # déjà collecté (relance possible sans tout retélécharger)
            try:
                doc = fetcher.fetch(result.url)
            except WebError as exc:
                log(f"  ✗ {result.url} : {exc}")
                continue
            text = clean_article(doc.text, max_chars_per_doc)
            if len(text) < 1000:
                continue
            path.write_text(f"{doc.title}\n\n{text}\n", encoding="utf-8")
            written.append(path)
            size = f"{len(text):,}".replace(",", "\u202f")
            log(f"  ✓ {doc.title} ({size} caractères)")
    return written


# ---------------------------------------------------------------- collecte groupée
def _strip_nested(text: str, opening: str, closing: str) -> str:
    """Retire les blocs imbriqués {{…}} ou {|…|} (modèles, tableaux)."""
    out, depth, i, n = [], 0, 0, len(text)
    while i < n:
        if text.startswith(opening, i):
            depth += 1
            i += len(opening)
        elif depth and text.startswith(closing, i):
            depth -= 1
            i += len(closing)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


_MEDIA_LINK = re.compile(r"\[\[(?:Fichier|File|Image|Catégorie|Category|Media):", re.I)


def _strip_media_links(text: str) -> str:
    """Retire [[Fichier:…]] et [[Catégorie:…]], dont la légende peut contenir d'autres liens."""
    out, pos = [], 0
    for match in _MEDIA_LINK.finditer(text):
        if match.start() < pos:
            continue
        out.append(text[pos:match.start()])
        depth, i = 0, match.start()
        while i < len(text):
            if text.startswith("[[", i):
                depth += 1
                i += 2
            elif text.startswith("]]", i):
                depth -= 1
                i += 2
                if depth == 0:
                    break
            else:
                i += 1
        pos = i
    out.append(text[pos:])
    return "".join(out)


_INLINE_TEMPLATE = re.compile(r"\{\{\s*([^{}|]+?)\s*\|([^{}]*)\}\}")
_SUFFIX_TEMPLATE = re.compile(r"\{\{\s*(1er|1re|er|re|e|ème|nd|nde)\s*\}\}", re.I)


def _inline_template(match: re.Match) -> str:
    """Garde le contenu des modèles courants ({{unité|…}}, {{date|…}}, {{s|XIX}}…)."""
    name = match.group(1).strip().lower()
    args = [a.strip() for a in match.group(2).split("|") if a.strip() and "=" not in a]
    if not args:
        return ""
    if name in ("nb", "formatnum", "num", "dts", "nobr", "nowrap", "monnaie") or \
            name.startswith(("unité", "nombre", "date", "heure")):
        return " ".join(args)
    if name in ("s", "siècle", "sp", "s-"):
        return f"{args[0]}e siècle"
    if name in ("lang", "langue", "citation", "japonais", "lien", "ill", "lien web", "abréviation", "sigle"):
        return args[-1] if name not in ("lien", "ill") else args[0]
    return match.group(0)  # modèle inconnu : retiré ensuite


def wikitext_to_text(wikitext: str) -> str:
    """Convertit le wikitexte d'un article en texte brut lisible."""
    s = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    s = _SUFFIX_TEMPLATE.sub(lambda m: m.group(1), s)
    for _ in range(2):  # modèles imbriqués simples : de l'intérieur vers l'extérieur
        s = _INLINE_TEMPLATE.sub(_inline_template, s)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S | re.I)
    s = re.sub(r"<(gallery|math|timeline|syntaxhighlight|source|score|graph)[^>]*>.*?</\1>", "", s,
               flags=re.S | re.I)
    s = _strip_nested(s, "{{", "}}")
    s = _strip_nested(s, "{|", "|}")
    s = _strip_media_links(s)
    s = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]", r"\1", s)          # [[cible|texte]] → texte
    s = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", s)              # [url texte] → texte
    s = re.sub(r"\[https?://[^\]]+\]", "", s)
    s = re.sub(r"'{2,}", "", s)                                                  # gras, italique
    s = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\n\1\n", s, flags=re.M)             # titres de section
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"^[*#:;]+\s*", "", s, flags=re.M)                               # puces de listes
    s = re.sub(r"__[A-ZÉ]+__", "", s)
    s = html.unescape(s).replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" +([,.;:)])", r"\1", s)
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)                                        # parenthèses vidées
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def fetch_wikipedia_bulk(client: WebClient, titles: list[str], lang: str = "fr",
                         batch: int = 10) -> dict[str, tuple[str, str]]:
    """Récupère plusieurs articles par requête : {titre demandé: (titre réel, wikitexte)}.

    Une requête pour `batch` articles au lieu d'une par article : plus rapide et plus
    respectueux des serveurs de Wikipédia.
    """
    out: dict[str, tuple[str, str]] = {}
    for start in range(0, len(titles), batch):
        chunk = titles[start:start + batch]
        resp = client.get(f"https://{lang}.wikipedia.org/w/api.php", check_robots=False, max_bytes=40_000_000,
                          params={"action": "query", "prop": "revisions", "rvprop": "content",
                                  "rvslots": "main", "titles": "|".join(chunk), "redirects": 1,
                                  "format": "json", "formatversion": 2})
        query = resp.json().get("query", {})
        renamed: dict[str, str] = {}
        for item in query.get("normalized", []) + query.get("redirects", []):
            renamed[item["from"]] = item["to"]
        pages = {p["title"]: p for p in query.get("pages", []) if not p.get("missing")}
        for asked in chunk:
            final = asked
            while final in renamed and renamed[final] != final:
                final = renamed[final]
            page = pages.get(final)
            if page and page.get("revisions"):
                out[asked] = (page["title"], page["revisions"][0]["slots"]["main"]["content"])
    return out


def build_wikipedia_bulk(titles: list[str], client: WebClient, out_dir: Path, lang: str = "fr",
                         max_chars_per_doc: int = 60_000, log: Callable[[str], None] = print) -> list[Path]:
    """Collecte des articles dont on connaît le titre exact, par lots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in titles if not (out_dir / f"{slugify(t)}.txt").exists()]
    written: list[Path] = []
    try:
        pages = fetch_wikipedia_bulk(client, todo, lang)
    except (WebError, ValueError, KeyError) as exc:
        log(f"  ✗ collecte groupée : {exc}")
        return written
    for asked in todo:
        if asked not in pages:
            log(f"  ✗ introuvable : {asked}")
            continue
        title, wikitext = pages[asked]
        if _LOW_VALUE_TITLE.search(title) or wikitext.lstrip().upper().startswith("#REDIRECT"):
            continue
        text = clean_article(wikitext_to_text(wikitext), max_chars_per_doc)
        if len(text) < 1000:
            continue
        path = out_dir / f"{slugify(title)}.txt"
        if path.exists():
            continue
        path.write_text(f"{title}\n\n{text}\n", encoding="utf-8")
        written.append(path)
        log(f"  ✓ {title} ({len(text):,} caractères)".replace(",", "\u202f"))
    return written
