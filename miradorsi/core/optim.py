"""Optimiseur AdamW, découpage du gradient et planning du taux d'apprentissage."""

from __future__ import annotations

import math

import numpy as np

from .nn import Parameter


class AdamW:
    """Adam avec décroissance des poids découplée (Loshchilov & Hutter)."""

    def __init__(self, params: list[Parameter], lr: float = 3e-4, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        correction1 = 1.0 - b1 ** self.step_count
        correction2 = 1.0 - b2 ** self.step_count
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            if self.weight_decay and p.data.ndim >= 2:  # pas de décroissance sur normes/biais
                p.data -= lr * self.weight_decay * p.data
            p.data -= lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def clip_grad_norm(params: list[Parameter], max_norm: float) -> float:
    """Limite la norme globale du gradient ; renvoie la norme avant découpage."""
    total = math.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params if p.grad is not None))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Montée linéaire puis décroissance en cosinus jusqu'à base_lr * min_ratio."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
