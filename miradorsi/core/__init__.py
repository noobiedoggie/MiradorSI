"""Cœur numérique de Mirador SI : autograd, couches, optimiseur."""

from .autograd import Tensor, no_grad, set_default_dtype, get_default_dtype
from .nn import Module, Parameter, Linear, Embedding, RMSNorm
from .optim import AdamW, clip_grad_norm, cosine_lr

__all__ = [
    "Tensor", "no_grad", "set_default_dtype", "get_default_dtype",
    "Module", "Parameter", "Linear", "Embedding", "RMSNorm",
    "AdamW", "clip_grad_norm", "cosine_lr",
]
