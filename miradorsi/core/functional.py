"""Opérations « fusionnées » pour le Transformer : forward et backward écrits à la main.

Les fusionner (au lieu de les composer à partir d'opérations élémentaires) rend
le calcul plus rapide, plus économe en mémoire et numériquement plus stable.
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, as_tensor

_GELU_K = np.sqrt(2.0 / np.pi)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    y = exp / exp.sum(axis=axis, keepdims=True)

    def backward(g):
        x.accumulate(y * (g - (g * y).sum(axis=axis, keepdims=True)))

    return Tensor.from_op(y, (x,), backward, "softmax")


def gelu(x: Tensor) -> Tensor:
    """GELU (approximation tanh, comme GPT-2)."""
    v = x.data
    inner = _GELU_K * (v + 0.044715 * v ** 3)
    t = np.tanh(inner)
    y = 0.5 * v * (1.0 + t)

    def backward(g):
        d_inner = _GELU_K * (1.0 + 3 * 0.044715 * v ** 2)
        dy = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t ** 2) * d_inner
        x.accumulate(g * dy)

    return Tensor.from_op(y, (x,), backward, "gelu")


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm : x / sqrt(moyenne(x²) + eps) * poids."""
    v = x.data
    r = 1.0 / np.sqrt((v * v).mean(axis=-1, keepdims=True) + eps)
    xn = v * r
    y = xn * weight.data

    def backward(g):
        if weight.requires_grad:
            weight.accumulate((g * xn).reshape(-1, v.shape[-1]).sum(axis=0))
        if x.requires_grad:
            gxn = g * weight.data
            x.accumulate(r * (gxn - xn * (gxn * xn).mean(axis=-1, keepdims=True)))

    return Tensor.from_op(y, (x, weight), backward, "rms_norm")


def embedding(weight: Tensor, indices: np.ndarray) -> Tensor:
    indices = np.asarray(indices, dtype=np.int64)

    def backward(g):
        full = np.zeros_like(weight.data)
        np.add.at(full, indices.reshape(-1), g.reshape(-1, weight.data.shape[-1]))
        weight.accumulate(full)

    return Tensor.from_op(weight.data[indices], (weight,), backward, "embedding")


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int = -1) -> Tensor:
    """Entropie croisée moyenne ; les cibles égales à `ignore_index` sont ignorées."""
    vocab = logits.data.shape[-1]
    flat = logits.data.reshape(-1, vocab)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    valid = targets != ignore_index
    count = max(int(valid.sum()), 1)
    safe_targets = np.where(valid, targets, 0)

    shifted = flat - flat.max(axis=-1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=-1))
    picked = shifted[np.arange(flat.shape[0]), safe_targets]
    loss = ((log_sum - picked) * valid).sum() / count

    def backward(g):
        probs = np.exp(shifted - log_sum[:, None])
        probs[np.arange(flat.shape[0]), safe_targets] -= 1.0
        probs *= (valid / count)[:, None]
        logits.accumulate((probs * g).reshape(logits.data.shape))

    return Tensor.from_op(np.asarray(loss), (logits,), backward, "cross_entropy")


def _rotate_half(v: np.ndarray) -> np.ndarray:
    half = v.shape[-1] // 2
    return np.concatenate([-v[..., half:], v[..., :half]], axis=-1)


def _rotate_half_transposed(v: np.ndarray) -> np.ndarray:
    half = v.shape[-1] // 2
    return np.concatenate([v[..., half:], -v[..., :half]], axis=-1)


def rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """Encodage positionnel rotatif (RoPE). x : (..., T, d), cos/sin : (T, d)."""
    y = x.data * cos + _rotate_half(x.data) * sin

    def backward(g):
        x.accumulate(g * cos + _rotate_half_transposed(g * sin))

    return Tensor.from_op(y, (x,), backward, "rope")


def rope_tables(length: int, dim: int, base: float = 10000.0, dtype=np.float32):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = np.outer(np.arange(length, dtype=np.float64), inv_freq)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def add_constant(x: Tensor, constant: np.ndarray) -> Tensor:
    """x + constante (masque d'attention) sans copier la constante dans le graphe."""
    return Tensor.from_op(x.data + constant, (x,), lambda g: x.accumulate(g), "add_const")


def concat(tensors: list[Tensor], axis: int = -1) -> Tensor:
    tensors = [as_tensor(t) for t in tensors]
    sizes = [t.data.shape[axis] for t in tensors]
    bounds = np.cumsum([0] + sizes)

    def backward(g):
        for t, start, end in zip(tensors, bounds[:-1], bounds[1:]):
            if t.requires_grad:
                index = [slice(None)] * g.ndim
                index[axis] = slice(start, end)
                t.accumulate(g[tuple(index)])

    data = np.concatenate([t.data for t in tensors], axis=axis)
    return Tensor.from_op(data, tensors, backward, "concat")
