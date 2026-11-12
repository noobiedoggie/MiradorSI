"""Briques de réseau de neurones : paramètres, modules et couches de base."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from . import functional as F
from .autograd import Tensor, get_default_dtype


class Parameter(Tensor):
    """Tenseur entraînable."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """Classe de base : découvre automatiquement les paramètres et sous-modules."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - à surcharger
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in vars(self).items():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            elif isinstance(value, Module):
                yield from value.named_parameters(full + ".")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{full}.{i}.")
                    elif isinstance(item, Parameter):
                        yield f"{full}.{i}", item

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"Poids manquants : {sorted(missing)[:5]}")
        for name, p in params.items():
            value = np.asarray(state[name], dtype=get_default_dtype())
            if value.shape != p.data.shape:
                raise ValueError(f"Forme incompatible pour {name}: {value.shape} != {p.data.shape}")
            p.data = value


def normal_init(rng: np.random.Generator, shape, std: float) -> np.ndarray:
    return (rng.standard_normal(shape) * std).astype(get_default_dtype())


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator,
                 bias: bool = False, std: float = 0.02):
        self.weight = Parameter(normal_init(rng, (in_features, out_features), std))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int, rng: np.random.Generator, std: float = 0.02):
        self.weight = Parameter(normal_init(rng, (num_embeddings, dim), std))

    def forward(self, indices: np.ndarray) -> Tensor:
        return F.embedding(self.weight, indices)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Parameter(np.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight, self.eps)
