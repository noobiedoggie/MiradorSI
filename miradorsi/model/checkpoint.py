"""Construction, sauvegarde et chargement des modèles (toutes architectures).

Quantification int8 : chaque matrice est stockée en entiers 8 bits avec un facteur
d'échelle par canal de sortie → fichier ~4× plus petit, écart de qualité minime.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..core.autograd import get_default_dtype
from .transformer import MiradorLM, ModelConfig
from .vigie import Vigie, VigieConfig

ARCHITECTURES = {
    "transformer": (MiradorLM, ModelConfig),
    "vigie": (Vigie, VigieConfig),
}


def build_model(arch: str, **config):
    if arch not in ARCHITECTURES:
        raise ValueError(f"Architecture inconnue « {arch} » (choix : {', '.join(ARCHITECTURES)})")
    model_cls, cfg_cls = ARCHITECTURES[arch]
    known = cfg_cls.__dataclass_fields__
    return model_cls(cfg_cls(**{k: v for k, v in config.items() if k in known}))


def model_arch(model) -> str:
    return getattr(model, "arch", "transformer")


# ------------------------------------------------------------------ int8
def _quant_axis(name: str) -> int:
    return 1 if name.startswith("embed") else 0  # embedding : une échelle par token


def quantize_array(weight: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(weight).max(axis=axis, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for name, value in state.items():
        if value.ndim == 2 and value.size >= 4096:
            q, scale = quantize_array(value, _quant_axis(name))
            out[f"q::{name}"], out[f"s::{name}"] = q, scale
        else:
            out[name] = value.astype(np.float32)
    return out


def dequantize_state(archive: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = {}
    dtype = get_default_dtype()
    for key, value in archive.items():
        if key.startswith("q::"):
            name = key[3:]
            state[name] = (value.astype(np.float32) * archive[f"s::{name}"]).astype(dtype)
        elif not key.startswith("s::"):
            state[key] = value
    return state


# ------------------------------------------------------------------ fichiers
def save_model(model, path: str | Path, extra: dict | None = None, quantize: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arch": model_arch(model), "config": asdict(model.cfg), "extra": extra or {},
            "quantized": quantize}
    state = model.state_dict()
    if quantize:
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **quantize_state(state))
    else:
        np.savez(path, __meta__=np.array(json.dumps(meta)), **state)
    return path


def load_model(path: str | Path):
    with np.load(Path(path), allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        arrays = {k: archive[k] for k in archive.files if k != "__meta__"}
    model = build_model(meta.get("arch", "transformer"), **meta["config"])
    model.load_state_dict(dequantize_state(arrays) if meta.get("quantized") else arrays)
    extra = dict(meta.get("extra", {}))
    extra["quantized"] = bool(meta.get("quantized"))
    return model, extra
