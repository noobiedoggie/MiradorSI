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
    v2 = v * v                       # v*v*v est bien plus rapide que v ** 3
    t = np.tanh(_GELU_K * (v + 0.044715 * v2 * v))
    y = 0.5 * v * (1.0 + t)

    def backward(g):
        d_inner = _GELU_K * (1.0 + 3 * 0.044715 * v2)
        dy = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * d_inner
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
        # Somme des gradients par token : tri + réduction par segments (bien plus rapide que add.at)
        flat = indices.reshape(-1)
        rows = g.reshape(-1, weight.data.shape[-1])
        order = np.argsort(flat, kind="stable")
        tokens, starts = np.unique(flat[order], return_index=True)
        full = np.zeros_like(weight.data)
        full[tokens] = np.add.reduceat(rows[order], starts, axis=0)
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
    exp = np.exp(shifted)            # calculée une seule fois, réutilisée au backward
    total = exp.sum(axis=-1)
    log_sum = np.log(total)
    picked = shifted[np.arange(flat.shape[0]), safe_targets]
    loss = ((log_sum - picked) * valid).sum() / count
    del shifted

    def backward(g):
        probs = exp / total[:, None]
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


def sigmoid(x: Tensor) -> Tensor:
    y = 0.5 * (1.0 + np.tanh(0.5 * x.data))  # forme stable de 1 / (1 + e^-x)
    return Tensor.from_op(y, (x,), lambda g: x.accumulate(g * y * (1.0 - y)), "sigmoid")


def silu(x: Tensor) -> Tensor:
    s = 0.5 * (1.0 + np.tanh(0.5 * x.data))
    y = x.data * s

    def backward(g):
        x.accumulate(g * (s + x.data * s * (1.0 - s)))

    return Tensor.from_op(y, (x,), backward, "silu")


def causal_conv(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Convolution causale « en profondeur » le long du temps.

    x : (B, T, D), weight : (K, D), bias : (D,)  →  y[:, t] = Σ_k weight[k] · x[:, t-k] + bias
    """
    K = weight.data.shape[0]
    B, T, D = x.data.shape
    padded = np.concatenate([np.zeros((B, K - 1, D), dtype=x.data.dtype), x.data], axis=1)
    y = np.broadcast_to(bias.data, (B, T, D)).copy()
    for k in range(K):
        y += weight.data[k] * padded[:, K - 1 - k: K - 1 - k + T]

    def backward(g):
        if bias.requires_grad:
            bias.accumulate(g.sum(axis=(0, 1)))
        if weight.requires_grad:
            weight.accumulate(np.stack([(g * padded[:, K - 1 - k: K - 1 - k + T]).sum(axis=(0, 1))
                                        for k in range(K)]))
        if x.requires_grad:
            gp = np.zeros_like(padded)
            for k in range(K):
                gp[:, K - 1 - k: K - 1 - k + T] += g * weight.data[k]
            x.accumulate(gp[:, K - 1:])

    return Tensor.from_op(y, (x, weight, bias), backward, "causal_conv")


def linear_scan(a: Tensor, b: Tensor) -> Tensor:
    """Récurrence linéaire h_t = a_t · h_{t-1} + b_t (h_{-1} = 0), élément par élément.

    Les portes ne dépendant que de l'entrée, le coût est linéaire en T et l'état
    à mémoriser pendant la génération est un simple vecteur.
    """
    av, bv = a.data, b.data
    h = np.empty_like(bv)
    state = np.zeros_like(bv[:, 0])
    for t in range(bv.shape[1]):
        state = av[:, t] * state + bv[:, t]
        h[:, t] = state

    def backward(g):
        gb = np.empty_like(g)
        carry = np.zeros_like(g[:, 0])
        for t in range(g.shape[1] - 1, -1, -1):
            carry = g[:, t] + carry
            gb[:, t] = carry
            carry = carry * av[:, t]
        if b.requires_grad:
            b.accumulate(gb)
        if a.requires_grad:
            previous = np.concatenate([np.zeros_like(h[:, :1]), h[:, :-1]], axis=1)
            a.accumulate(gb * previous)

    return Tensor.from_op(h, (a, b), backward, "linear_scan")
