"""Bibliothèque hors ligne : les introductions des articles Wikipédia les plus consultés,
indexées pour répondre sans internet.

Les connaissances vivent ici, dans un fichier, et non dans les paramètres du modèle : le
petit modèle n'a pas besoin de tout savoir par cœur, il lui suffit de chercher et de lire.

Format : un seul fichier SQLite (lisible partout, y compris sur Android), avec un index
inversé maison :
  docs(id, title, body)          texte compressé (zlib) de chaque article
  titles(key, id)                titre normalisé (et redirections) → article
  terms(term, df, postings)      pour chaque mot : identifiants des articles (écarts, uint32)
                                 et fréquences (uint8), compressés
  meta / blobs                   langue, source, date, longueurs des articles, importance
La recherche (BM25) est vectorisée avec numpy : quelques millisecondes, même sur téléphone.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import urllib.parse
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis.text import content_terms, keywords, normalize, strip_idioms
from .web.extract import Document
from .web.http import WebError
from .web.search import SearchResult, wikipedia_url

FORMAT_VERSION = 1
_K1, _B = 1.2, 0.75
_TITLE_WEIGHT = 3  # un mot du titre compte comme trois mots du texte


def title_key(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize(title)).split())


@dataclass
class LibraryHit:
    id: int
    title: str
    score: float


class LibraryBuilder:
    """Construit le fichier de la bibliothèque, article par article."""

    def __init__(self, path: str | Path, lang: str = "fr", source: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        self._tmp = tmp
        self.db = sqlite3.connect(tmp)
        self.db.executescript("""
            PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;
            CREATE TABLE blobs(key TEXT PRIMARY KEY, value BLOB) WITHOUT ROWID;
            CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT NOT NULL, body BLOB NOT NULL);
            CREATE TABLE titles(key TEXT PRIMARY KEY, id INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE terms(term TEXT PRIMARY KEY, df INTEGER NOT NULL, postings BLOB NOT NULL) WITHOUT ROWID;
        """)
        self.lang, self.source = lang, source
        self.lengths = array("I")
        self.importance = array("f")
        self.postings: dict[str, tuple[array, array]] = {}
        self._titles: dict[str, int] = {}

    def add(self, title: str, text: str, importance: float = 0.0, aliases: tuple[str, ...] = ()) -> int:
        doc_id = len(self.lengths)
        counts: dict[str, int] = {}
        for term in content_terms(title):
            counts[term] = counts.get(term, 0) + _TITLE_WEIGHT
        for term in content_terms(text):
            counts[term] = counts.get(term, 0) + 1
        for term, tf in counts.items():
            ids, tfs = self.postings.setdefault(term, (array("I"), array("B")))
            ids.append(doc_id)
            tfs.append(min(tf, 255))
        self.lengths.append(sum(counts.values()))
        self.importance.append(float(importance))
        self.db.execute("INSERT INTO docs VALUES (?, ?, ?)",
                        (doc_id, title, zlib.compress(text.encode("utf-8"), 9)))
        for name in (title, *aliases):
            self._titles.setdefault(title_key(name), doc_id)  # le premier (le plus important) gagne
        return doc_id

    def close(self, date: str = "") -> dict:
        n = len(self.lengths)
        rows = []
        for term, (ids, tfs) in self.postings.items():
            gaps = np.diff(np.frombuffer(ids, dtype=np.uint32), prepend=np.uint32(0)).astype("<u4")
            blob = zlib.compress(gaps.tobytes() + np.frombuffer(tfs, dtype=np.uint8).tobytes(), 6)
            rows.append((term, len(ids), blob))
        self.db.executemany("INSERT INTO terms VALUES (?, ?, ?)", rows)
        self.db.executemany("INSERT INTO titles VALUES (?, ?)", list(self._titles.items()))
        lengths = np.frombuffer(self.lengths, dtype=np.uint32)
        meta = {"version": FORMAT_VERSION, "langue": self.lang, "source": self.source, "date": date,
                "articles": n, "longueur_moyenne": float(lengths.mean()) if n else 0.0}
        self.db.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
        self.db.executemany("INSERT INTO blobs VALUES (?, ?)", [
            ("longueurs", zlib.compress(lengths.astype("<u4").tobytes())),
            ("importance", zlib.compress(np.frombuffer(self.importance, dtype=np.float32).astype("<f4").tobytes())),
        ])
        self.db.commit()
        self.db.execute("VACUUM")
        self.db.close()
        self._tmp.replace(self.path)
        return {"articles": n, "mots": len(rows), "octets": self.path.stat().st_size}


class Library:
    """Lecture et recherche dans une bibliothèque construite par LibraryBuilder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        uri = f"file:{urllib.parse.quote(str(self.path.resolve()))}?mode=ro"
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._lock = threading.Lock()  # l'analyse lit plusieurs articles en parallèle
        self.meta = dict(self.db.execute("SELECT key, value FROM meta"))
        if int(self.meta.get("version", 0)) != FORMAT_VERSION:
            raise ValueError(f"format de bibliothèque non pris en charge : {self.meta.get('version')}")
        blobs = dict(self.db.execute("SELECT key, value FROM blobs"))
        self.lengths = np.frombuffer(zlib.decompress(blobs["longueurs"]), dtype="<u4").astype(np.float32)
        raw = np.frombuffer(zlib.decompress(blobs["importance"]), dtype="<f4").astype(np.float32)
        # importance (popularité) ramenée entre 0 et 1, en échelle logarithmique
        log = np.log1p(np.maximum(raw, 0) * 1e6)
        self.prior = log / log.max() if len(log) and log.max() > 0 else np.zeros_like(log)
        self.size = len(self.lengths)
        self.avgdl = float(self.meta.get("longueur_moyenne") or 1.0)
        self.lang = self.meta.get("langue", "fr")

    def describe(self) -> str:
        count = f"{self.size:,}".replace(",", " ")
        date = f", {self.meta['date']}" if self.meta.get("date") else ""
        return f"{count} articles de Wikipédia ({self.lang}{date})"

    def _postings(self, term: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            row = self.db.execute("SELECT df, postings FROM terms WHERE term = ?", (term,)).fetchone()
        if not row:
            return None
        df, blob = row
        data = zlib.decompress(blob)
        ids = np.cumsum(np.frombuffer(data[: 4 * df], dtype="<u4"), dtype=np.int64)
        tfs = np.frombuffer(data[4 * df:], dtype=np.uint8).astype(np.float32)
        return ids, tfs

    def search(self, query: str, k: int = 8) -> list[LibraryHit]:
        terms = list(dict.fromkeys(content_terms(strip_idioms(query))))
        if not terms or not self.size:
            return []
        scores = np.zeros(self.size, dtype=np.float32)
        matched = np.zeros(self.size, dtype=np.float32)
        for term in terms:
            found = self._postings(term)
            if found is None:
                continue
            ids, tfs = found
            idf = math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5))
            norm = tfs + _K1 * (1 - _B + _B * self.lengths[ids] / self.avgdl)
            scores[ids] += idf * tfs * (_K1 + 1) / norm
            matched[ids] += 1
        # un article qui contient tous les mots de la question passe devant ; les plus consultés aussi
        scores *= (matched / len(terms)) ** 2 * (1 + 0.3 * self.prior)
        for doc_id in self._title_matches(query):
            scores[doc_id] = scores[doc_id] * 2.0 + 1.0  # « Trou noir » pour « c'est quoi un trou noir »
        top = np.argsort(-scores)[:k]
        hits = [int(i) for i in top if scores[i] > 0]
        titles = {}
        if hits:
            with self._lock:
                titles = dict(self.db.execute(
                    f"SELECT id, title FROM docs WHERE id IN ({','.join('?' * len(hits))})", hits).fetchall())
        return [LibraryHit(i, titles.get(i, ""), float(scores[i])) for i in hits]

    def _title_matches(self, query: str) -> list[int]:
        words = keywords(strip_idioms(query))
        candidates = {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + 4) + 1)}
        keys = [title_key(c) for c in candidates if c]
        if not keys:
            return []
        with self._lock:
            rows = self.db.execute(f"SELECT key, id FROM titles WHERE key IN ({','.join('?' * len(keys))})",
                                   keys).fetchall()
        # la correspondance la plus longue d'abord (« Révolution française » plutôt que « Révolution »)
        best = sorted(rows, key=lambda row: -len(row[0]))
        return [doc_id for _, doc_id in best[:2]]

    def find(self, title: str) -> int | None:
        with self._lock:
            row = self.db.execute("SELECT id FROM titles WHERE key = ?", (title_key(title),)).fetchone()
        return int(row[0]) if row else None

    def article(self, doc_id: int) -> tuple[str, str]:
        with self._lock:
            row = self.db.execute("SELECT title, body FROM docs WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        return row[0], zlib.decompress(row[1]).decode("utf-8")

    def close(self) -> None:
        self.db.close()


class LibrarySearch:
    """Moteur de recherche « bibliothèque » : même interface que les moteurs du web."""

    name = "bibliotheque"

    def __init__(self, library: Library):
        self.library = library

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]:
        results = []
        for rank, hit in enumerate(self.library.search(query, max_results), 1):
            _, text = self.library.article(hit.id)
            results.append(SearchResult(hit.title, wikipedia_url(self.library.lang, hit.title),
                                        text[:200], self.name, rank))
        return results


class LibraryFetcher:
    """Lit les articles dans la bibliothèque ; sans réseau, le reste est inaccessible."""

    def __init__(self, library: Library, fallback=None):
        self.library = library
        self.fallback = fallback

    def fetch(self, url: str) -> Document:
        match = re.match(r"https?://(\w+)\.wikipedia\.org/wiki/(.+)", url)
        if match and match.group(1) == self.library.lang:
            doc_id = self.library.find(urllib.parse.unquote(match.group(2)).replace("_", " "))
            if doc_id is not None:
                title, text = self.library.article(doc_id)
                return Document(url=wikipedia_url(self.library.lang, title), title=title, text=text,
                                language=self.library.lang)
        if self.fallback is not None:
            return self.fallback.fetch(url)
        raise WebError(f"hors ligne : {url} n'est pas dans la bibliothèque")


def open_library(path: str | Path | None) -> Library | None:
    """Ouvre la bibliothèque si le fichier existe et est valide, sinon None."""
    if not path or not Path(path).is_file():
        return None
    try:
        return Library(path)
    except (sqlite3.Error, ValueError, KeyError, zlib.error):
        return None


# ---------------------------------------------------------------- construction depuis Wikipédia
# Export « CirrusSearch » de Wikipédia (dumps.wikimedia.org/other/cirrussearch) : une ligne JSON
# par article, avec son introduction (opening_text), sa popularité, ses redirections et le texte
# aplati de sa fiche (auxiliary_text).

_BANNERS = re.compile(
    r"(Pour les articles homonymes, voir [^.]*\.|Pour (?:les autres significations|l'article|un article|"
    r"le|la|les|l')\b[^.]{0,160}?, voir [^.]*\.|« [^»]{1,80} » redirige ici\.|Ne doit pas être confondu "
    r"avec [^.]*\.|Cet article (?:ne cite pas|est une ébauche|concerne|adopte)[^.]*\.(?:[^.]*?(?:Comment "
    r"ajouter mes sources \?|vérifiabilité[^.]*\.|portail[^.]*\.))?|Si vous disposez d'ouvrages[^?]*\?[^?]*\?|"
    r"En pratique : Quelles sources sont attendues \? Comment ajouter mes sources \?|Vous pouvez partager "
    r"vos connaissances en l’améliorant[^.]*\.|Si ce bandeau n'est plus pertinent, retirez-le\.[^.]*\.)")

# Rubriques fréquentes des fiches (infobox) de Wikipédia en français
FICHE_LABELS = [
    "Population municipale", "Population totale", "Population", "Densité", "Superficie totale", "Superficie",
    "Altitude", "Capitale", "Plus grande ville", "Langue officielle", "Langues officielles", "Monnaie",
    "Fuseau horaire", "Président", "Présidente", "Premier ministre", "Chef de l'État", "Maire", "Pays", "Région",
    "Département", "Gentilé", "Code postal", "Naissance", "Décès", "Nationalité", "Activités", "Activité",
    "Conjoint", "Enfants", "Distinctions", "Fondation", "Fondateur", "Fondateurs", "Siège", "Chiffre d'affaires",
    "Effectif", "Date", "Lieu", "Issue", "Auteur", "Réalisation", "Durée", "Date de sortie", "Genre", "Hauteur",
    "Longueur", "Largeur", "Profondeur", "Architecte", "Construction", "Inauguration", "Diamètre", "Masse",
    "Distance", "Période de révolution", "Température", "Découvreur", "Date de découverte", "Point culminant",
    "Source", "Embouchure", "Débit moyen", "Pays traversés", "Formule brute", "Masse molaire", "Classification",
    "Statut", "Type", "Site web", "Mandat", "Prédécesseur", "Successeur", "Parti politique", "Religion",
    "Éditeur", "Langue", "Âge", "Espérance de vie", "PIB nominal", "PIB (PPA)", "IDH", "Indicatif téléphonique",
]
FICHE_LABELS += ["Père", "Mère", "Fratrie", "Formation", "Coordonnées", "Œuvres principales", "Mouvement",
                 "Membre de", "Lieu de travail", "Sépulture", "Nom de naissance", "Nom de naissance"]
# Intertitres de fiche : ils séparent les rubriques mais ne sont pas des informations
_FICHE_SECTIONS = ["Démographie", "Géographie", "Administration", "Localisation", "Liens", "Biographie",
                   "Histoire", "Caractéristiques", "Identifiants", "Fonctions", "Fonction", "Signature",
                   "Archives conservées par", "Blason", "Drapeau", "Économie", "Politique"]
_FICHE_RE = re.compile(r"(?<![\w'])(" + "|".join(sorted(map(re.escape, set(FICHE_LABELS + _FICHE_SECTIONS)),
                                                         key=len, reverse=True)) + r")(?![\w'])")


def clean_opening_text(title: str, text: str, max_chars: int = 1600) -> str:
    """Introduction lisible : sans bandeaux, sans le titre de la fiche répété, sans légende d'image."""
    text = " ".join(_BANNERS.sub(" ", text or "").split())
    if title and text.startswith(title + " "):
        text = text[len(title) + 1:]  # « Antoine Meillet Antoine Meillet, né le… » : titre de la fiche
    from .analysis.text import split_sentences  # import local : module léger côté application
    sentences = split_sentences(text)
    wanted = set(content_terms(title))
    # « ℝ³ est un espace vectoriel… » (légende d'image) avant « L'algèbre linéaire est… » : la
    # vraie introduction commence à la première phrase qui parle du sujet
    for i, sentence in enumerate(sentences[:4]):
        if wanted & set(content_terms(sentence)):
            sentences = sentences[i:]
            break
    out: list[str] = []
    for sentence in sentences:
        if out and sum(len(s) + 1 for s in out) + len(sentence) > max_chars:
            break
        out.append(sentence)
    return " ".join(out).strip()


def parse_fiche(title: str, auxiliary: list[str], max_lines: int = 14) -> list[str]:
    """Lignes « Titre — Rubrique : valeur. » tirées du texte aplati de la fiche (infobox)."""
    for block in auxiliary or []:
        matches = list(_FICHE_RE.finditer(block))
        if len(matches) < 2:
            continue  # légende d'image ou bandeau, pas une fiche
        lines, seen = [], set()
        for current, following in zip(matches, matches[1:] + [None]):
            label = current.group(1)
            value = block[current.end(): following.start() if following else len(block)].strip(" :;,")
            value = " ".join(value.split())
            if label in _FICHE_SECTIONS or not value or label in seen or len(value) > 160:
                continue
            seen.add(label)
            lines.append(f"{title} — {label} : {value.rstrip('.')}.")
            if len(lines) == max_lines:
                break
        return lines
    return []


def article_from_cirrus(doc: dict) -> tuple[str, str, float, tuple[str, ...]] | None:
    """(titre, texte, popularité, redirections) d'un article de l'export CirrusSearch, ou None."""
    title = doc.get("title") or ""
    if doc.get("namespace") != 0 or not title or title.startswith("Liste ") or "(homonymie)" in title:
        return None
    if any(t.startswith("Modèle:Homonymie") for t in doc.get("template") or []):
        return None  # page d'homonymie : une liste de liens, pas un article
    lead = clean_opening_text(title, doc.get("opening_text") or "")
    if len(lead) < 60:
        return None
    fiche = parse_fiche(title, doc.get("auxiliary_text") or [])
    text = "\n".join([lead] + fiche)
    aliases = tuple(r["title"] for r in (doc.get("redirect") or [])[:12] if r.get("namespace") == 0)
    return title, text, float(doc.get("popularity_score") or 0.0), aliases


_POPULARITY = re.compile(rb'"popularity_score":\s*([0-9.eE+-]+)')


def build_from_cirrus(lines, out: str | Path, articles: int = 100_000, date: str = "",
                      log=print) -> dict:
    """Garde les `articles` pages les plus consultées d'un export CirrusSearch, puis construit
    la bibliothèque (une seule lecture du flux, mémoire bornée)."""
    import heapq
    import json

    heap: list[tuple[float, int, dict]] = []
    seen = 0
    for line in lines:
        head = line[:10] if isinstance(line, str) else line[:10].decode("ascii", "replace")
        if not head.startswith('{"') or head.startswith('{"index"'):
            continue  # lignes d'en-tête de l'export (une par article)
        seen += 1
        if seen % 200_000 == 0:
            log(f"  {seen:,} articles lus, {len(heap):,} gardés".replace(",", " "))
        # popularité lue sans décoder tout le JSON : la plupart des articles sont écartés ici
        found = _POPULARITY.search(line if isinstance(line, bytes) else line.encode("utf-8", "ignore"))
        popularity = float(found.group(1)) if found else 0.0
        if len(heap) >= articles and popularity <= heap[0][0]:
            continue
        doc = json.loads(line)
        item = article_from_cirrus(doc)
        if item is None:
            continue
        entry = (item[2], seen, {"title": item[0], "text": item[1], "aliases": item[3]})
        if len(heap) < articles:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)
    chosen = sorted(heap, key=lambda e: -e[0])  # les plus consultés d'abord : ils gagnent les titres
    log(f"{seen:,} articles lus, {len(chosen):,} gardés ; indexation…".replace(",", " "))
    builder = LibraryBuilder(out, "fr", "Wikipédia (export CirrusSearch)")
    for popularity, _, item in chosen:
        builder.add(item["title"], item["text"], popularity, item["aliases"])
    return builder.close(date)
