"""Conversion du wikitexte (source des articles Wikipédia) en texte brut lisible."""

from __future__ import annotations

import html
import re


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
    s = re.sub(r"(?<!')'{4}(?!')", "'", s)          # L''''énergie''' : apostrophe puis gras (règle MediaWiki)
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


# ---------------------------------------------------------------- HTML rendu → texte
from html.parser import HTMLParser  # noqa: E402

_SKIP_CLASSES = re.compile(
    r"\b(mw-ref|reference|navbox|bandeau|bandeau-container|metadata|noprint|hatnote|homonymie|reflist|references|"
    r"mw-references-wrap|thumb|gallery|toc|autres-projets|printfooter|catlinks|mw-editsection|DebutCarte|"
    r"diag-climat|Z3988|cite-bracket|mw-empty-elt|mwe-math-element|sistersitebox|portail|bandeau-portail)\b")
_SKIP_TAGS = {"style", "script", "figure", "table", "math", "noscript", "audio", "video", "map", "svg"}
_VOID = {"br", "img", "link", "meta", "wbr", "hr", "input", "source", "area", "col", "embed", "param", "track"}
_BLOCKS = {"p", "li", "dd", "dt", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "div", "section", "ul", "ol",
           "dl", "pre", "caption"}


# bandeaux de maintenance en ligne : « [réf. nécessaire] », « [Quoi ?] »…
_MAINTENANCE = re.compile(r"\[(?:réf\.|ref\.|Information|Quoi|Lequel|Laquelle|Lesquels|Qui|Quand|Où|Combien|Pourquoi|"
                          r"source|style|non neutre|pas clair|Interprétation|Passage|citation)[^\[\]]{0,40}\]", re.I)


def _tidy(text: str) -> str:
    text = html.unescape(text).replace(" ", " ").replace(" ", " ")
    text = _MAINTENANCE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" +([,.;:)\]])", r"\1", text).replace("( ", "(")
    return re.sub(r"\(\s*[,;]?\s*\)", "", text).strip()


class _WikipediaHTML(HTMLParser):
    """Texte d'un article Wikipédia rendu : paragraphes, titres et fiche (infobox) en phrases."""

    def __init__(self, title: str):
        super().__init__(convert_charrefs=True)
        self.title = title
        self.stack: list[tuple[str, str]] = []
        self.skipping = 0
        self.infobox = 0
        self.paragraphs: list[str] = []
        self.buffer: list[str] = []
        self.facts: list[str] = []
        self.row: dict[str, list[str]] | None = None
        self.lead_end: int | None = None  # index du premier titre de section

    def handle_starttag(self, tag, attrs):
        if tag in _VOID:
            if tag == "br" and not self.skipping:
                self._data(" ")
            return
        attrs = dict(attrs)
        cls, typeof = attrs.get("class") or "", attrs.get("typeof") or ""
        table_ok = tag == "table" and ("infobox" in cls or self.infobox)
        if self.skipping or _SKIP_CLASSES.search(cls) or "mw:Extension/ref" in typeof or \
                (tag in _SKIP_TAGS and not table_ok):
            mode = "skip"
            self.skipping += 1
        elif "infobox" in cls and not self.infobox:
            mode = "infobox"
            self.infobox += 1
        elif self.infobox:
            mode = tag if tag in ("tr", "th", "td") else "inline"
            if tag == "tr":
                self.row = {"th": [], "td": []}
        else:
            mode = "block" if tag in _BLOCKS else "inline"
            if mode == "block":
                self._flush()
            if tag in ("h2", "h3") and self.lead_end is None:
                self.lead_end = len(self.paragraphs)
        self.stack.append((tag, mode))

    def handle_endtag(self, tag):
        if not any(t == tag for t, _ in self.stack):
            return  # balise fermante orpheline
        while self.stack:
            t, mode = self.stack.pop()
            if mode == "skip":
                self.skipping -= 1
            elif mode == "infobox":
                self.infobox -= 1
            elif mode == "tr":
                self._finish_row()
            elif mode == "block":
                self._flush()
            if t == tag:
                break

    def handle_data(self, data):
        if not self.skipping:
            self._data(data)

    def _data(self, data: str) -> None:
        if self.infobox:
            cell = next((m for _, m in reversed(self.stack) if m in ("th", "td")), None)
            if cell and self.row is not None:
                self.row[cell].append(data)
            return
        self.buffer.append(data)

    def _finish_row(self) -> None:
        if self.row is not None:
            label, value = _tidy("".join(self.row["th"])), _tidy("".join(self.row["td"]))
            if label and value and len(value) <= 160 and len(label) <= 60:
                self.facts.append(f"{self.title} — {label} : {value.rstrip('.')}.")
        self.row = None

    def _flush(self) -> None:
        text = _tidy("".join(self.buffer))
        self.buffer = []
        if text:
            self.paragraphs.append(text)

    def text(self) -> str:
        self._flush()
        cut = self.lead_end if self.lead_end is not None else len(self.paragraphs)
        return "\n".join(self.paragraphs[:cut] + self.facts + self.paragraphs[cut:])


def wikipedia_html_to_text(page: str, title: str = "") -> str:
    """Texte lisible d'un article rendu (index.php?action=render) : les chiffres issus des
    modèles et des données de Wikidata y sont, contrairement au wikitexte brut."""
    parser = _WikipediaHTML(title)
    parser.feed(page)
    parser.close()
    return parser.text()
