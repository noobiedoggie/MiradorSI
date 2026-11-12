"""Moteur de différentiation automatique de Mirador SI, écrit de zéro avec NumPy.

Chaque opération construit un nœud dans un graphe de calcul. `Tensor.backward()`
parcourt ce graphe en ordre topologique inverse et propage les gradients
(rétropropagation en mode inverse).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable

import numpy as np

_DEFAULT_DTYPE = np.float32
_GRAD_ENABLED = True


def get_default_dtype():
    return _DEFAULT_DTYPE


def set_default_dtype(dtype) -> None:
    """Change le type flottant par défaut (float64 sert aux tests de gradient)."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


@contextlib.contextmanager
def no_grad():
    """Désactive la construction du graphe (inférence plus rapide et plus légère)."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Ramène un gradient diffusé (broadcast) à la forme d'origine en sommant."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def as_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


class Tensor:
    """Tableau NumPy qui mémorise comment il a été calculé."""

    __array_priority__ = 1000  # pour que `ndarray + Tensor` appelle Tensor.__radd__

    def __init__(self, data, requires_grad: bool = False):
        dtype = get_default_dtype()
        if isinstance(data, np.ndarray) and data.dtype == dtype:
            self.data = data
        else:
            self.data = np.asarray(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Callable[[np.ndarray], None] | None = None
        self._op = ""

    # ------------------------------------------------------------------ infos
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}{grad}, op={self._op or 'leaf'})"

    def __len__(self) -> int:
        return len(self.data)

    # -------------------------------------------------------- construction
    @staticmethod
    def from_op(data: np.ndarray, parents: Iterable["Tensor"], backward, op: str) -> "Tensor":
        """Crée le résultat d'une opération et le relie au graphe si besoin."""
        parents = tuple(parents)
        out = Tensor(data)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    def accumulate(self, grad: np.ndarray) -> None:
        """Ajoute une contribution au gradient de ce tenseur."""
        if not self.requires_grad:
            return
        if grad.shape != self.data.shape:
            grad = unbroadcast(grad, self.data.shape)
        if self.grad is None:
            self.grad = np.array(grad, dtype=self.data.dtype, copy=True)
        else:
            self.grad = self.grad + grad

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    # ----------------------------------------------------- rétropropagation
    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Ce tenseur ne dépend d'aucun paramètre entraînable.")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() sans argument exige un scalaire.")
            grad = np.ones_like(self.data)

        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

        self.grad = np.asarray(grad, dtype=self.data.dtype)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
            if node._parents:  # nœud intermédiaire : on libère le graphe
                node.grad = None
                node._backward = None
                node._parents = ()

    # ---------------------------------------------------- opérations de base
    def __add__(self, other) -> "Tensor":
        other = as_tensor(other)

        def backward(g):
            self.accumulate(g)
            other.accumulate(g)

        return Tensor.from_op(self.data + other.data, (self, other), backward, "add")

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return Tensor.from_op(-self.data, (self,), lambda g: self.accumulate(-g), "neg")

    def __sub__(self, other) -> "Tensor":
        return self + (-as_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return as_tensor(other) + (-self)

    def __mul__(self, other) -> "Tensor":
        other = as_tensor(other)

        def backward(g):
            if self.requires_grad:
                self.accumulate(g * other.data)
            if other.requires_grad:
                other.accumulate(g * self.data)

        return Tensor.from_op(self.data * other.data, (self, other), backward, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        other = as_tensor(other)

        def backward(g):
            if self.requires_grad:
                self.accumulate(g / other.data)
            if other.requires_grad:
                other.accumulate(-g * self.data / (other.data ** 2))

        return Tensor.from_op(self.data / other.data, (self, other), backward, "div")

    def __rtruediv__(self, other) -> "Tensor":
        return as_tensor(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        def backward(g):
            self.accumulate(g * exponent * self.data ** (exponent - 1))

        return Tensor.from_op(self.data ** exponent, (self,), backward, "pow")

    def __matmul__(self, other) -> "Tensor":
        other = as_tensor(other)
        a, b = self.data, other.data

        def backward(g):
            if self.requires_grad:
                self.accumulate(g @ np.swapaxes(b, -1, -2))
            if other.requires_grad:
                other.accumulate(np.swapaxes(a, -1, -2) @ g)

        return Tensor.from_op(a @ b, (self, other), backward, "matmul")

    # ------------------------------------------------------ formes et index
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.data.shape
        return Tensor.from_op(
            self.data.reshape(shape), (self,), lambda g: self.accumulate(g.reshape(original)), "reshape"
        )

    def transpose(self, *axes) -> "Tensor":
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inverse = tuple(np.argsort(axes))
        return Tensor.from_op(
            self.data.transpose(axes), (self,), lambda g: self.accumulate(g.transpose(inverse)), "transpose"
        )

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __getitem__(self, index) -> "Tensor":
        if isinstance(index, Tensor):
            index = index.data.astype(np.int64)

        def backward(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)
            self.accumulate(full)

        return Tensor.from_op(self.data[index], (self,), backward, "getitem")

    # ----------------------------------------------------------- réductions
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        shape = self.data.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.accumulate(np.broadcast_to(g, shape))

        return Tensor.from_op(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward, "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # ------------------------------------------------------ fonctions simples
    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        return Tensor.from_op(out_data, (self,), lambda g: self.accumulate(g * out_data), "exp")

    def log(self) -> "Tensor":
        return Tensor.from_op(np.log(self.data), (self,), lambda g: self.accumulate(g / self.data), "log")

    def tanh(self) -> "Tensor":
        out_data = np.tanh(self.data)
        return Tensor.from_op(
            out_data, (self,), lambda g: self.accumulate(g * (1.0 - out_data ** 2)), "tanh"
        )

    def relu(self) -> "Tensor":
        mask = self.data > 0
        return Tensor.from_op(self.data * mask, (self,), lambda g: self.accumulate(g * mask), "relu")
