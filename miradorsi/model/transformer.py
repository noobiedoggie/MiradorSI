"""MiradorLM : Transformer décodeur (style GPT) écrit de zéro sur le moteur maison.

Architecture :
    tokens → embedding → N × [RMSNorm → attention causale multi-têtes (RoPE) → +résiduel
                               RMSNorm → MLP (GELU)                        → +résiduel]
           → RMSNorm → projection sur le vocabulaire (poids liés à l'embedding)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from ..core import functional as F
from ..core.autograd import Tensor, get_default_dtype
from ..core.nn import Embedding, Linear, Module, RMSNorm
from . import streaming as S


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    context_length: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    seed: int = 1337

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model doit être divisible par n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("la dimension par tête doit être paire (RoPE)")


class CausalSelfAttention(Module):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator):
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.q = Linear(d, d, rng)
        self.k = Linear(d, d, rng)
        self.v = Linear(d, d, rng)
        self.out = Linear(d, d, rng, std=0.02 / np.sqrt(2 * cfg.n_layers))

    def _heads(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, cos: np.ndarray, sin: np.ndarray, mask: np.ndarray) -> Tensor:
        B, T, C = x.shape
        q = F.rope(self._heads(self.q(x), B, T), cos, sin)
        k = F.rope(self._heads(self.k(x), B, T), cos, sin)
        v = self._heads(self.v(x), B, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        weights = F.softmax(F.add_constant(scores, mask), axis=-1)
        y = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out(y)


class MLP(Module):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator):
        hidden = cfg.mlp_ratio * cfg.d_model
        self.up = Linear(cfg.d_model, hidden, rng)
        self.down = Linear(hidden, cfg.d_model, rng, std=0.02 / np.sqrt(2 * cfg.n_layers))

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(Module):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator):
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, rng)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg, rng)

    def forward(self, x: Tensor, cos, sin, mask) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.mlp(self.norm2(x))


class MiradorLM(Module):
    arch = "transformer"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.embed = Embedding(cfg.vocab_size, cfg.d_model, rng)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        dtype = get_default_dtype()
        head_dim = cfg.d_model // cfg.n_heads
        self._cos, self._sin = F.rope_tables(cfg.context_length, head_dim, cfg.rope_base, dtype)
        self._mask = np.triu(np.full((cfg.context_length, cfg.context_length), -1e9, dtype=dtype), k=1)

    # ------------------------------------------------------------ calcul
    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"Séquence trop longue ({T} > {self.cfg.context_length})")
        cos, sin, mask = self._cos[:T], self._sin[:T], self._mask[:T, :T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm(x)
        return x @ self.embed.weight.T  # poids liés : moins de paramètres, meilleure généralisation

    def loss(self, idx: np.ndarray, targets: np.ndarray, ignore_index: int = -1) -> Tensor:
        return F.cross_entropy(self.forward(idx), targets, ignore_index)

    # ------------------------------------------------------------ génération
    def init_state(self) -> dict:
        """Cache clé/valeur circulaire par couche (borné à context_length) + poids q/k/v fusionnés."""
        cfg, dtype = self.cfg, self.embed.weight.data.dtype
        hd = cfg.d_model // cfg.n_heads
        fused = [np.concatenate([b.attn.q.weight.data, b.attn.k.weight.data, b.attn.v.weight.data], 1)
                 for b in self.blocks]
        shape = (cfg.context_length, cfg.n_heads, hd)
        runtime = SimpleNamespace(  # poids pré-calculés : hors de l'état, ils ne comptent pas comme mémoire
            fused=fused, inv_freq=1.0 / (cfg.rope_base ** (np.arange(0, hd, 2, dtype=np.float64) / hd)))
        return {"position": 0, "runtime": runtime,
                "layers": [{"k": np.zeros(shape, dtype), "v": np.zeros(shape, dtype)} for _ in self.blocks]}

    def step(self, token: int, state: dict) -> np.ndarray:
        cfg = self.cfg
        d, H, hd = cfg.d_model, cfg.n_heads, cfg.d_model // cfg.n_heads
        position = state["position"]
        runtime = state["runtime"]
        angles = position * runtime.inv_freq
        angles = np.concatenate([angles, angles])
        dtype = self.embed.weight.data.dtype
        cos, sin = np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)
        slot, filled = position % cfg.context_length, min(position + 1, cfg.context_length)
        x = self.embed.weight.data[token]
        for block, w_qkv, cache in zip(self.blocks, runtime.fused, state["layers"]):
            z = S.rms_norm(x, block.norm1.weight.data, cfg.norm_eps) @ w_qkv
            q, k = z[:d].reshape(H, 1, hd), z[d:2 * d].reshape(H, hd)
            cache["k"][slot] = S.apply_rope(k, cos, sin)
            cache["v"][slot] = z[2 * d:].reshape(H, hd)
            keys, values = cache["k"][:filled], cache["v"][:filled]              # (n, H, hd)
            weights = S.softmax(S.apply_rope(q, cos, sin) @ keys.transpose(1, 2, 0) / np.sqrt(hd))
            x = x + (weights @ values.transpose(1, 0, 2)).reshape(-1) @ block.attn.out.weight.data
            h = S.rms_norm(x, block.norm2.weight.data, cfg.norm_eps)
            x = x + S.gelu(h @ block.mlp.up.weight.data) @ block.mlp.down.weight.data
        state["position"] = position + 1
        x = S.rms_norm(x, self.norm.weight.data, cfg.norm_eps)
        return x @ self.embed.weight.data.T

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 200, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.95, stop_ids: set[int] | None = None,
                 rng: np.random.Generator | None = None, **options) -> list[int]:
        return S.generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p, stop_ids, rng, **options)

    # ------------------------------------------------------------ sauvegarde
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        from .checkpoint import save_model
        save_model(self, path, extra)

    @classmethod
    def load(cls, path: str | Path) -> tuple["MiradorLM", dict]:
        from .checkpoint import load_model
        return load_model(path)


def sample(logits: np.ndarray, temperature: float = 0.8, top_k: int | None = 40,
           top_p: float | None = 0.95, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng()
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.size:
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p is not None and 0 < top_p < 1:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, top_p) + 1)]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return int(rng.choice(probs.size, p=probs))
