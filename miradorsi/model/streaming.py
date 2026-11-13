"""Inférence en flux (un token à la fois), en NumPy pur, sans graphe de calcul.

Chaque architecture fournit `init_state()` et `step(token, state) -> logits`.
La génération ne recalcule donc jamais le passé : c'est ce qui la rend rapide sur PC.
"""

from __future__ import annotations

import numpy as np

_GELU_K = np.sqrt(2.0 / np.pi)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_GELU_K * (x + 0.044715 * x ** 3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def rope_at(position: int, dim: int, base: float, dtype) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin de RoPE pour une position (identiques à functional.rope_tables)."""
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = position * inv_freq
    angles = np.concatenate([angles, angles])
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def apply_rope(v: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = v.shape[-1] // 2
    rotated = np.concatenate([-v[..., half:], v[..., :half]], axis=-1)
    return v * cos + rotated * sin


def penalize_repetition(logits: np.ndarray, recent: list[int], penalty: float) -> np.ndarray:
    """Rend moins probables les tokens déjà produits récemment (évite « en France, en France… »)."""
    if penalty == 1.0 or not recent:
        return logits
    logits = logits.copy()
    ids = np.fromiter(set(recent), dtype=np.int64)
    values = logits[ids]
    logits[ids] = np.where(values > 0, values / penalty, values * penalty)
    return logits


def generate(model, prompt_ids: list[int], max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int | None = 40, top_p: float | None = 0.95, stop_ids: set[int] | None = None,
             rng: np.random.Generator | None = None, repetition_penalty: float = 1.0,
             repetition_window: int = 64) -> list[int]:
    from .transformer import sample

    rng = rng or np.random.default_rng()
    stop_ids = stop_ids or set()
    state = model.init_state()
    logits = None
    for token in prompt_ids:  # lecture du prompt (« prefill »)
        logits = model.step(int(token), state)
    if logits is None:
        raise ValueError("prompt vide")
    out: list[int] = []
    for _ in range(max_new_tokens):
        scores = penalize_repetition(logits.astype(np.float64), out[-repetition_window:], repetition_penalty)
        next_id = sample(scores, temperature, top_k, top_p, rng)
        if next_id in stop_ids:
            break
        out.append(next_id)
        logits = model.step(next_id, state)
    return out


def state_bytes(state) -> int:
    """Mémoire occupée par l'état d'inférence (cache d'attention + états récurrents)."""
    if isinstance(state, np.ndarray):
        return state.nbytes
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(v) for v in state)
    return 0
