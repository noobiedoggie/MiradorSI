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


def build_web_corpus(topics: list[str], wiki: WikipediaSearch, fetcher: Fetcher, out_dir: Path,
                     lang: str = "fr", per_topic: int = 3,
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
            try:
                doc = fetcher.fetch(result.url)
            except WebError as exc:
                log(f"  ✗ {result.url} : {exc}")
                continue
            path = out_dir / f"{slugify(doc.title)}.txt"
            path.write_text(f"{doc.title}\n\n{doc.text}\n", encoding="utf-8")
            written.append(path)
            size = f"{len(doc.text):,}".replace(",", "\u202f")
            log(f"  ✓ {doc.title} ({size} caractères)")
    return written
