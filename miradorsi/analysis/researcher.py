"""Moteur d'analyse approfondie de Mirador SI.

Boucle de recherche itérative :
  1. planifier   — décliner la question en plusieurs requêtes (angles d'analyse)
  2. chercher    — interroger les moteurs, fusionner, prioriser les résultats
  3. lire        — télécharger les pages, extraire le texte, écarter les doublons
  4. relancer    — repérer les notions récurrentes et lancer des requêtes de suivi
  5. analyser    — classer les passages (BM25), recouper entre sources (TF-IDF),
                   mesurer la centralité (TextRank), sélectionner sans redondance (MMR)
  6. vérifier    — extraire les chiffres, détecter accords et divergences
  7. rapporter   — produire un rapport sourcé avec niveaux de confiance
"""

from __future__ import annotations

import datetime as dt
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..config import AnalysisConfig
from ..web.extract import Document, domain_of
from ..web.http import WebError
from ..web.search import SearchResult, normalize_url
from . import credibility
from .facts import FactGroup, NumericFact, compare_facts, extract_facts
from .rank import BM25, TfidfIndex, mmr_select, textrank
from .text import chunk_words, content_terms, detect_language, keywords, light_stem, split_sentences

FACETS = {
    "fr": ["définition", "chiffres clés statistiques", "enjeux et limites", "actualité {year}",
           "étude analyse approfondie", "histoire origine", "avantages inconvénients", "critique controverse"],
    "en": ["definition", "key statistics", "challenges limitations", "latest news {year}",
           "research study analysis", "history origin", "pros and cons", "criticism controversy"],
}
DEPTH_LABELS = {1: "rapide", 2: "standard", 3: "approfondie"}
LEAD_CHARS = 1500  # début de document considéré comme « chapeau »
QUANTITATIVE_RE = re.compile(
    r"\b(combien|part|parts|pourcentage|proportion|taux|nombre|chiffres?|montant|quantit[ée]|"
    r"statistiques?|niveau|volume|co[uû]t|prix|how (many|much)|percentage|share|rate|number|amount)\b",
    re.I)


class SearchLike(Protocol):
    errors: list[str]

    def search(self, query: str, max_results: int = 8, lang: str = "fr") -> list[SearchResult]: ...


class FetcherLike(Protocol):
    def fetch(self, url: str) -> Document: ...


@dataclass
class Source:
    id: int
    url: str
    title: str
    domain: str
    text: str
    query: str
    provider: str
    credibility: credibility.CredibilityScore
    published: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["text_chars"] = len(data.pop("text"))
        return data


@dataclass
class Finding:
    text: str
    source_id: int
    supporting: list[int]
    relevance: float
    confidence: float

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.75:
            return "élevée"
        if self.confidence >= 0.55:
            return "moyenne"
        return "faible"

    @property
    def citations(self) -> list[int]:
        return [self.source_id] + [s for s in self.supporting if s != self.source_id]


@dataclass
class AnalysisReport:
    question: str
    language: str
    depth: int
    analyst: str
    started_at: str
    duration_seconds: float = 0.0
    queries: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    corroborated: list[FactGroup] = field(default_factory=list)
    divergent: list[tuple[FactGroup, FactGroup]] = field(default_factory=list)
    leads: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    documents_seen: int = 0

    def source(self, source_id: int) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def to_markdown(self) -> str:
        from .report import render_markdown
        return render_markdown(self)

    def to_dict(self) -> dict:
        def group(g: FactGroup) -> dict:
            return {"unit": g.unit, "sources": g.source_ids, "context": g.context,
                    "values": [{"value": f.value, "raw": f.raw, "source": f.source_id,
                                "sentence": f.sentence} for f in g.facts]}
        return {
            "question": self.question, "language": self.language, "depth": self.depth,
            "analyst": self.analyst, "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 2), "queries": self.queries,
            "sources": [s.to_dict() for s in self.sources],
            "findings": [{**asdict(f), "confidence_label": f.confidence_label} for f in self.findings],
            "corroborated_facts": [group(g) for g in self.corroborated],
            "divergent_facts": [[group(a), group(b)] for a, b in self.divergent],
            "leads": self.leads, "suggested_queries": self.suggested_queries,
            "errors": self.errors, "documents_seen": self.documents_seen,
        }


@dataclass
class _Passage:
    source_id: int
    text: str
    terms: list[str]


def _shingles(text: str, size: int = 5) -> set[int]:
    tokens = text.lower().split()
    return {hash(" ".join(tokens[i:i + size])) for i in range(0, max(1, len(tokens) - size + 1), 2)}


def _normalized(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    top = values.max()
    return values / top if top > 0 else np.zeros_like(values)


class DeepAnalyzer:
    def __init__(self, search: SearchLike, fetcher: FetcherLike, cfg: AnalysisConfig | None = None,
                 on_progress: Callable[[str], None] | None = None, analyst: str = "Mirador SI",
                 max_workers: int = 4, lexicon=None):
        self.lexicon = lexicon  # rareté des mots dans la langue (optionnel, voir analysis/lexicon.py)
        self.search_engine = search
        self.fetcher = fetcher
        self.cfg = cfg or AnalysisConfig()
        self.progress = on_progress or (lambda message: None)
        self.analyst = analyst
        self.max_workers = max_workers

    # ================================================================ public
    def analyze(self, question: str, depth: int | None = None) -> AnalysisReport:
        question = " ".join(question.split())
        depth = min(max(int(depth or self.cfg.default_depth), 1), 3)
        lang = detect_language(question)
        report = AnalysisReport(question, lang, depth, self.analyst,
                                dt.datetime.now().isoformat(timespec="seconds"))
        start = time.monotonic()
        self._fingerprints: list[set[int]] = []
        self._seen_urls: set[str] = set()
        self._domain_counts: dict[str, int] = {}

        budget = self.cfg.sources_for(depth)
        first_round = budget if depth == 1 else math.ceil(budget * 0.6)
        queries = self.plan_queries(question, lang, depth)
        for round_index in range(depth):
            if round_index > 0:
                queries = self.follow_up_queries(question, report, lang, depth)
                if not queries:
                    break
                self.progress(f"[relance {round_index}] nouvelles pistes : {', '.join(queries)}")
            remaining = budget - len(report.sources)
            if remaining <= 0:
                break
            round_budget = min(remaining, first_round if round_index == 0
                               else math.ceil((budget - first_round) / max(1, depth - 1)))
            candidates = self._gather(queries, lang, report)
            self._read(candidates, round_budget, report)

        report.errors.extend(getattr(self.search_engine, "errors", []))
        if report.sources:
            self.progress(f"[analyse] {len(report.sources)} sources, recoupement en cours…")
            self._analyze_sources(question, report)
        report.duration_seconds = time.monotonic() - start
        return report

    # ============================================================ planification
    def plan_queries(self, question: str, lang: str, depth: int) -> list[str]:
        kw = " ".join(keywords(question)) or question
        facets = [f.format(year=dt.date.today().year) for f in FACETS.get(lang, FACETS["en"])]
        count = {1: 0, 2: 3, 3: 6}[depth]
        queries = [question]
        if kw.lower() != question.lower():
            queries.append(kw)
        queries += [f"{kw} {facet}" for facet in facets[:count]]
        return list(dict.fromkeys(queries))

    def follow_up_queries(self, question: str, report: AnalysisReport, lang: str, depth: int) -> list[str]:
        anchor = set(content_terms(question))
        known = anchor | set(content_terms(" ".join(report.queries)))
        terms = self._lead_terms(self._passages(report.sources), anchor, known, k=6)
        kw = " ".join(keywords(question)) or question
        fresh = [f"{kw} {t}" for t in terms if f"{kw} {t}" not in report.queries]
        return fresh[: 2 if depth == 2 else 3]

    # ================================================================ recherche
    def _gather(self, queries: list[str], lang: str, report: AnalysisReport) -> list[tuple[SearchResult, str]]:
        scored: dict[str, tuple[float, SearchResult, str]] = {}
        for query in queries:
            self.progress(f"[recherche] {query}")
            report.queries.append(query)
            for result in self.search_engine.search(query, self.cfg.results_per_query, lang):
                key = normalize_url(result.url)
                if key in self._seen_urls:
                    continue
                prior = credibility.assess(result.url, result.domain).score
                value = 1.0 / (1 + result.rank) + 0.6 * prior
                if key in scored:  # trouvé par plusieurs requêtes : signal de pertinence
                    old_value, old_result, old_query = scored[key]
                    scored[key] = (old_value + 0.25, old_result, old_query)
                else:
                    scored[key] = (value, result, query)
        ranked = sorted(scored.values(), key=lambda item: -item[0])
        return [(result, query) for _, result, query in ranked]

    def _read(self, candidates: list[tuple[SearchResult, str]], budget: int, report: AnalysisReport) -> None:
        queue = list(candidates)
        while queue and budget > 0:
            batch = []
            while queue and len(batch) < budget + 2:
                result, query = queue.pop(0)
                if self._domain_counts.get(result.domain, 0) + sum(r.domain == result.domain for r, _ in batch) \
                        >= self.cfg.max_per_domain:
                    continue
                self._seen_urls.add(normalize_url(result.url))
                batch.append((result, query))
            if not batch:
                break
            for result, _ in batch:
                self.progress(f"[lecture] {result.domain} — {result.title[:70]}")
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                documents = list(pool.map(lambda item: self._safe_fetch(item[0].url, report), batch))
            for (result, query), doc in zip(batch, documents):
                report.documents_seen += 1
                if doc is None or budget <= 0:
                    continue
                if len(doc.text) < self.cfg.min_document_chars:
                    report.errors.append(f"contenu trop court ignoré : {result.url}")
                    continue
                prints = _shingles(doc.text[:20000])
                if any(len(prints & p) / max(1, len(prints | p)) > 0.6 for p in self._fingerprints):
                    report.errors.append(f"doublon ignoré : {result.url}")
                    continue
                self._fingerprints.append(prints)
                domain = domain_of(doc.url or result.url)
                self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1
                report.sources.append(Source(
                    id=len(report.sources) + 1, url=doc.url or result.url,
                    title=doc.title or result.title, domain=domain, text=doc.text, query=query,
                    provider=result.provider, published=doc.published,
                    credibility=credibility.assess(doc.url or result.url, domain, doc.text, doc.published),
                ))
                budget -= 1

    def _safe_fetch(self, url: str, report: AnalysisReport) -> Document | None:
        try:
            return self.fetcher.fetch(url)
        except (WebError, ValueError, UnicodeError) as exc:
            report.errors.append(f"lecture impossible : {exc}")
            return None

    # ================================================================ analyse
    def _passages(self, sources: list[Source]) -> list[_Passage]:
        passages = [
            _Passage(s.id, chunk, content_terms(chunk))
            for s in sources
            for chunk in chunk_words(s.text, self.cfg.passage_words)
        ]
        return [p for p in passages if len(p.terms) >= 5]

    def _analyze_sources(self, question: str, report: AnalysisReport) -> None:
        cfg = self.cfg
        passages = self._passages(report.sources)
        if not passages:
            return
        question_terms = set(content_terms(question))
        entities = question_entities(question)
        # Poids des mots de la question : rareté dans la langue (lexique) × nom propre
        rarity = self.lexicon.question_weights(question_terms) if self.lexicon else {}
        term_weight = {t: rarity.get(t, 1.0) * (2.0 if t in entities else 1.0) for t in question_terms}
        weights = dict(term_weight)
        for t in content_terms(" ".join(report.queries)):
            weights.setdefault(t, 0.3)

        bm25 = BM25([p.terms for p in passages])
        passage_scores = bm25.scores(list(weights), weights)
        top_n = min(len(passages), cfg.findings_for(report.depth) * 6)
        top_passages = [int(i) for i in np.argsort(-passage_scores)[:top_n] if passage_scores[i] > 0]

        # --- phrases candidates : issues des meilleurs passages ET liées à la question
        candidates: list[tuple[int, str, list[str], float]] = []
        seen_sentences = set()
        for pi in top_passages:
            p = passages[pi]
            for sentence in split_sentences(p.text):
                n_words = len(sentence.split())
                key = sentence.lower()
                if n_words < 8 or n_words > 70 or key in seen_sentences:
                    continue
                letters = sum(c.isalpha() for c in sentence) / max(1, len(sentence))
                if letters < 0.6 or sentence.rstrip().endswith((":", ",")):
                    continue
                terms = content_terms(sentence)
                if question_terms and not question_terms & set(terms):
                    continue  # phrase hors sujet (menus, mentions génériques…)
                seen_sentences.add(key)
                candidates.append((p.source_id, sentence, terms, float(passage_scores[pi])))
        if not candidates:
            return

        sentence_bm25 = BM25([c[2] for c in candidates])
        # Le titre du document sert de contexte : « Elle fournit 6 % de la production… » dans
        # l'article « Énergie solaire en France » parle bien du solaire en France.
        source_by_id = {s.id: s for s in report.sources}
        title_terms = {s.id: set(content_terms(s.title)) for s in report.sources}
        total_weight = sum(term_weight.values()) or 1.0

        def coverage(found: set[str]) -> float:
            return sum(term_weight[t] for t in question_terms & found) / total_weight

        overlap = np.array([0.6 * coverage(set(terms)) + 0.4 * coverage(set(terms) | title_terms[sid])
                            for sid, _, terms, _ in candidates])
        # La couverture des mots (pondérés) de la question est le signal le plus fiable ;
        # BM25 départage, sans laisser une phrase longue gagner par simple répétition.
        relevance = (0.30 * _normalized(sentence_bm25.scores(list(weights), weights))
                     + 0.15 * _normalized(np.array([c[3] for c in candidates]))
                     + 0.55 * overlap)
        # Bonus : phrase d'introduction (les chapeaux résument), chiffre pour une question chiffrée
        quantitative = bool(QUANTITATIVE_RE.search(question))
        for i, (sid, sentence, terms, _) in enumerate(candidates):
            bonus = 1.0
            if 0 <= source_by_id[sid].text.find(sentence) < LEAD_CHARS:
                bonus += 0.12
            if quantitative and extract_facts(sentence, sid):
                bonus += 0.15
            # Un nom propre de la question (« France ») est une contrainte : une phrase sur
            # l'Allemagne, même très proche par ailleurs, répond à une autre question.
            if entities and not entities & (set(terms) | title_terms[sid]):
                bonus *= 0.55
            relevance[i] = relevance[i] * bonus  # pas de plafond : il créerait des ex æquo

        # --- recoupement : la même idée (avec les mêmes chiffres) dans d'autres sources ?
        index = TfidfIndex([p.terms for p in passages] + [c[2] for c in candidates])
        passage_vectors = index.vectors[: len(passages)]
        sentence_vectors = index.vectors[len(passages):]
        passage_numbers = [_quantities(p.text) for p in passages]
        sentence_numbers = [_quantities(c[1]) for c in candidates]
        support: list[list[int]] = []
        for (source_id, *_), vec, numbers in zip(candidates, sentence_vectors, sentence_numbers):
            best: dict[int, float] = {}
            own_domain = source_by_id[source_id].domain
            for p, pvec, pnumbers in zip(passages, passage_vectors, passage_numbers):
                if source_by_id[p.source_id].domain == own_domain:
                    continue  # deux pages d'un même site ne sont pas des sources indépendantes
                if numbers and not numbers & pnumbers:
                    continue  # un chiffre différent ne corrobore pas : il contredit
                sim = TfidfIndex.cosine(vec, pvec)
                if sim > best.get(p.source_id, 0.0):
                    best[p.source_id] = sim
            support.append(_one_per_domain(
                sorted(s for s, sim in best.items() if sim >= cfg.corroboration_threshold), source_by_id))

        n = len(candidates)
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                similarity[i, j] = similarity[j, i] = TfidfIndex.cosine(sentence_vectors[i], sentence_vectors[j])
        centrality = _normalized(textrank(similarity))

        other_sources = max(1, len(report.sources) - 1)
        scores = []
        for i, (source_id, *_rest) in enumerate(candidates):
            cred = source_by_id[source_id].credibility.score
            support_ratio = min(1.0, len(support[i]) / min(3, other_sources))
            quality = 0.35 * cred + 0.45 * support_ratio + 0.20 * centrality[i]
            scores.append(float(relevance[i] * (0.55 + 0.45 * quality)))

        # --- sélection diversifiée ; les reformulations d'un même constat sont fusionnées
        k = cfg.findings_for(report.depth)
        order = mmr_select(scores, lambda i, j: similarity[i, j], k=min(n, k * 4), diversity=0.3)
        chosen: list[int] = []
        per_source: dict[int, int] = {}
        for i in order:
            source_id, sentence, _, _ = candidates[i]
            twin = next((j for j in chosen if similarity[i, j] >= 0.55
                         and sentence_numbers[i] == sentence_numbers[j]), None)
            if twin is not None:
                finding = report.findings[chosen.index(twin)]
                domains = {source_by_id[x].domain for x in [finding.source_id, *finding.supporting]}
                if source_by_id[source_id].domain not in domains:
                    finding.supporting = sorted(finding.supporting + [source_id])
                continue
            if per_source.get(source_id, 0) >= 3 or len(report.findings) >= k:
                continue
            per_source[source_id] = per_source.get(source_id, 0) + 1
            chosen.append(i)
            report.findings.append(Finding(sentence, source_id, support[i],
                                           round(float(min(1.0, relevance[i])), 3), 0.0))
        for finding, i in zip(report.findings, chosen):
            cred = source_by_id[finding.source_id].credibility.score
            confidence = 0.5 * cred + 0.15 * min(len(finding.supporting), 2) + 0.2 * min(1.0, relevance[i])
            finding.confidence = round(float(min(0.99, confidence)), 3)

        # --- chiffres clés
        facts: list[NumericFact] = []
        for pi in top_passages:
            for sentence in split_sentences(passages[pi].text):
                facts.extend(extract_facts(sentence, passages[pi].source_id))
        unique = {(f.source_id, f.value, f.unit, f.sentence): f for f in facts}
        corroborated, report.divergent = compare_facts(list(unique.values()))
        report.corroborated = [g for g in corroborated  # confirmé = au moins deux SITES différents
                               if len({source_by_id[i].domain for i in g.source_ids}) >= 2][:8]
        report.divergent = report.divergent[:6]

        # --- pistes à creuser
        known = question_terms | set(content_terms(" ".join(report.queries)))
        report.leads = self._lead_terms(passages, question_terms, known, k=8)
        kw = " ".join(keywords(question)) or question
        report.suggested_queries = [f"{kw} {t}" for t in report.leads[:4]]

    def _lead_terms(self, passages: list[_Passage], anchor: set[str], known: set[str], k: int = 8) -> list[str]:
        """Notions qui reviennent, dans plusieurs sources, À CÔTÉ des termes de la question."""
        by_source: dict[int, list[str]] = {}
        for p in passages:
            if not anchor or anchor & set(p.terms):
                by_source.setdefault(p.source_id, []).extend(content_terms(p.text, stem=False))
        if not by_source:
            return []
        documents = list(by_source.values())
        index = TfidfIndex(documents)
        doc_freq: dict[str, int] = {}
        for d in documents:
            for t in set(d):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        min_docs = 2 if len(documents) >= 3 else 1
        leads = []
        for term, _ in index.top_terms(k * 8, exclude=known):
            if doc_freq.get(term, 0) >= min_docs and light_stem(term) not in known and term not in known:
                leads.append(term)
            if len(leads) >= k:
                break
        return leads


def question_entities(question: str) -> set[str]:
    """Noms propres de la question (mots à majuscule hors début de phrase), normalisés."""
    words = re.findall(r"[^\W\d_][\w'’-]*", question)
    out = set()
    for i, word in enumerate(words):
        word = word.split("'")[-1].split("’")[-1]
        if i > 0 and word[:1].isupper() and len(word) > 1:
            out.update(content_terms(word))
    return out


def _one_per_domain(source_ids: list[int], source_by_id: dict) -> list[int]:
    """Garde une seule source par site : la corroboration compte des sites indépendants."""
    seen, kept = set(), []
    for sid in source_ids:
        domain = source_by_id[sid].domain
        if domain not in seen:
            seen.add(domain)
            kept.append(sid)
    return kept


_QUANTITY_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _quantities(text: str) -> frozenset[str]:
    """Nombres d'un texte (hors années), pour vérifier qu'un recoupement porte sur les mêmes chiffres."""
    out = set()
    for raw in _QUANTITY_RE.findall(re.sub(r"(?<=\d)[\u00a0\u202f ](?=\d{3}\b)", "", text)):
        value = raw.replace(",", ".")
        if len(raw) == 4 and raw.isdigit() and 1800 <= int(raw) <= 2100:
            continue
        out.add(value.rstrip("0").rstrip(".") if "." in value else value)
    return frozenset(out)
