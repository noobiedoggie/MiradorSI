"""Chargement de la configuration de l'environnement (config/mirador.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "mirador.toml"


@dataclass
class PathsConfig:
    identity: str = "config/identity.toml"
    corpus: str = "data/corpus"
    checkpoints: str = "checkpoints"
    cache: str = ".cache/web"
    reports: str = "reports"
    lexicon: str = "data/lexique_fr.json"


@dataclass
class ModelSettings:
    arch: str = "vigie"            # "vigie" (compacte, v0.2) ou "transformer" (v0.1)
    vocab_size: int = 2048
    context_length: int = 256
    d_model: int = 288
    n_heads: int = 4
    seed: int = 1337
    # Vigie
    d_embed: int = 128
    pattern: str = "RA"
    n_loops: int = 2
    n_kv_heads: int = 1
    window: int = 64
    conv_kernel: int = 4
    mlp_hidden: int = 0
    # Transformer
    n_layers: int = 4
    mlp_ratio: int = 4


@dataclass
class TrainingConfig:
    steps: int = 600
    batch_size: int = 16
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 30
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 4
    identity_repeats: int = 8
    dialogue_ratio: float = 0.2          # part minimale des dialogues d'identité dans chaque lot
    tokenizer_sample_chars: int = 1_500_000
    keep_best: bool = True               # garder le modèle qui généralise le mieux
    checkpoint_every: int = 200          # point de reprise complet (poids + optimiseur)


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_new_tokens: int = 160
    repetition_penalty: float = 1.15   # > 1 : décourage les répétitions


@dataclass
class WebConfig:
    user_agent: str = "MiradorSI/0.2 (agent de recherche open source) Python-urllib"
    timeout_seconds: float = 15
    max_bytes: int = 3_000_000
    retries: int = 3
    max_retry_wait_seconds: float = 60
    min_delay_seconds: float = 1.0
    respect_robots_txt: bool = True
    cache_ttl_hours: float = 24
    search_providers: list[str] = field(default_factory=lambda: ["duckduckgo", "wikipedia"])
    searxng_url: str = ""
    wikipedia_languages: list[str] = field(default_factory=lambda: ["fr", "en"])
    cache_dir: str = ""  # rempli à partir de [paths].cache


@dataclass
class AnalysisConfig:
    default_depth: int = 2
    results_per_query: int = 8
    max_sources: dict[int, int] = field(default_factory=lambda: {1: 6, 2: 12, 3: 20})
    max_per_domain: int = 2
    max_per_encyclopedia: int = 5  # articles distincts d'une encyclopédie (Wikipédia)
    min_document_chars: int = 400
    passage_words: int = 90
    findings: dict[int, int] = field(default_factory=lambda: {1: 5, 2: 8, 3: 12})
    corroboration_threshold: float = 0.32

    def sources_for(self, depth: int) -> int:
        return self.max_sources.get(depth, max(self.max_sources.values()))

    def findings_for(self, depth: int) -> int:
        return self.findings.get(depth, max(self.findings.values()))


@dataclass
class Config:
    root: Path = PROJECT_ROOT
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    web: WebConfig = field(default_factory=WebConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else self.root / p


def _fill(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Clés inconnues dans [{cls.__name__}] : {sorted(unknown)}")
    obj = cls(**values)
    for name in ("max_sources", "findings"):
        if isinstance(getattr(obj, name, None), dict):
            setattr(obj, name, {int(k): int(v) for k, v in getattr(obj, name).items()})
    return obj


def find_config() -> Path:
    """MIRADOR_CONFIG, sinon ./config/mirador.toml, sinon celui livré avec le projet."""
    if os.environ.get("MIRADOR_CONFIG"):
        return Path(os.environ["MIRADOR_CONFIG"])
    local = Path.cwd() / "config" / "mirador.toml"
    return local if local.exists() else DEFAULT_CONFIG


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else find_config()
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root = path.resolve().parent.parent if path.exists() else Path.cwd()
    cfg = Config(
        root=root,
        paths=_fill(PathsConfig, data.get("paths", {})),
        model=_fill(ModelSettings, data.get("model", {})),
        training=_fill(TrainingConfig, data.get("training", {})),
        generation=_fill(GenerationConfig, data.get("generation", {})),
        web=_fill(WebConfig, data.get("web", {})),
        analysis=_fill(AnalysisConfig, data.get("analysis", {})),
    )
    if not cfg.web.cache_dir:
        cfg.web.cache_dir = str(cfg.path("cache"))
    return cfg
