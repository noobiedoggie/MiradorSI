"""Construction d'un corpus d'entraînement à partir d'internet (Wikipédia)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from ..analysis.text import strip_accents
from ..web.http import WebError
from ..web.search import Fetcher, WikipediaSearch


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", strip_accents(text.lower())).strip("-")
    return slug[:60] or "document"


# Sections de fin d'article Wikipédia : listes de liens et de références, pas de la langue.
_TAIL_SECTIONS = re.compile(r"\n(Notes et références|Références|Voir aussi|Bibliographie|Liens externes|"
                            r"Articles connexes|Annexes|Sources)\n", re.I)


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
