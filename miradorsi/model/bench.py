"""Banc d'essai reproductible : compare les architectures à budget égal.

Corpus : la documentation (docstrings) de la bibliothèque standard de Python,
disponible hors ligne sur tout PC. ~8 Mo de prose technique en anglais : assez pour
qu'un petit modèle ne puisse pas tout mémoriser. La validation se fait sur des
documents jamais vus pendant l'entraînement (découpage par document).

Mesures : perte de validation, bits par caractère (indépendant du tokenizer),
vitesse d'entraînement, vitesse de génération, mémoire d'inférence, taille du fichier.
"""

from __future__ import annotations

import ast
import copy
import math
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..core import AdamW, clip_grad_norm, cosine_lr, no_grad
from .checkpoint import build_model, dequantize_state, quantize_state, save_model
from .streaming import state_bytes
from .tokenizer import BPETokenizer
from .train import get_batch

# Configurations comparées à nombre de paramètres comparable (~2 M avec un vocabulaire de 1024).
DEFAULT_CANDIDATES: dict[str, dict] = {
    "Transformer (v0.1)": {"arch": "transformer", "d_model": 192, "n_layers": 4, "n_heads": 4},
    "Vigie": {"arch": "vigie", "d_model": 288, "d_embed": 128, "pattern": "RA", "n_loops": 2,
              "n_heads": 4, "n_kv_heads": 1, "window": 64},
    "Vigie compacte": {"arch": "vigie", "d_model": 192, "d_embed": 96, "pattern": "RA", "n_loops": 2,
                       "n_heads": 4, "n_kv_heads": 1, "window": 64},
}


def stdlib_documents(max_chars: int = 3_000_000) -> list[str]:
    root = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    if not root.exists():
        root = Path(ast.__file__).parent
    docs, total = [], 0
    for path in sorted(root.rglob("*.py")):
        if any(part in ("site-packages", "test", "tests", "idlelib") for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc and len(doc) >= 200:
                    docs.append(textwrap.dedent(doc).strip())
                    total += len(docs[-1])
                    if total >= max_chars:
                        return docs
    return docs


def _evaluate(model, data: np.ndarray, batch_size: int, context: int, batches: int) -> float:
    rng = np.random.default_rng(12345)  # mêmes fenêtres de validation pour tous les modèles
    with no_grad():
        return float(np.mean([model.loss(*get_batch(data, batch_size, context, rng)).item()
                              for _ in range(batches)]))


def _generation_speed(model, tokens: int = 200) -> tuple[float, int, int]:
    """(tokens/s, mémoire d'état après 256 tokens, après 2048 tokens)."""
    state = model.init_state()
    with no_grad():
        start = time.perf_counter()
        for i in range(tokens):
            model.step(i % model.cfg.vocab_size, state)
        speed = tokens / (time.perf_counter() - start)
        for i in range(tokens, 256):
            model.step(i % model.cfg.vocab_size, state)
        mem_256 = state_bytes(state)
        for i in range(256, 2048):
            model.step(i % model.cfg.vocab_size, state)
        mem_2048 = state_bytes(state)
    return speed, mem_256, mem_2048


def run_benchmark(candidates: dict[str, dict] | None = None, steps: int = 400, batch_size: int = 16,
                  context: int = 128, learning_rate: float = 2e-3, vocab_size: int = 1024,
                  max_chars: int = 3_000_000, eval_batches: int = 24,
                  log: Callable[[str], None] = print) -> list[dict]:
    candidates = candidates or DEFAULT_CANDIDATES
    log("Préparation du corpus (docstrings de la bibliothèque standard Python)…")
    docs = stdlib_documents(max_chars)
    val_docs, train_docs = docs[::20], [d for i, d in enumerate(docs) if i % 20]
    train_text, val_text = "\n\n".join(train_docs), "\n\n".join(val_docs)
    log(f"  {len(train_docs)} documents d'entraînement ({len(train_text) / 1e6:.2f} M caractères), "
        f"{len(val_docs)} de validation ({len(val_text) / 1e6:.2f} M)")
    tokenizer = BPETokenizer.train(train_text[:1_000_000], vocab_size)
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.int64)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.int64)
    chars_per_token = len(val_text) / len(val_ids)
    log(f"  tokenizer {tokenizer.vocab_size} tokens · {len(train_ids) / 1e6:.2f} M tokens d'entraînement · "
        f"{chars_per_token:.2f} caractères/token")

    results = []
    for name, spec in candidates.items():
        spec = dict(spec)
        arch = spec.pop("arch")
        model = build_model(arch, vocab_size=tokenizer.vocab_size, context_length=context, seed=0, **spec)
        params = model.num_parameters()
        log(f"\n▶ {name} — {params / 1e6:.2f} M paramètres")
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        rng = np.random.default_rng(0)  # mêmes lots d'entraînement pour tous
        start = time.perf_counter()
        for step in range(steps):
            lr = cosine_lr(step, steps, learning_rate, max(1, steps // 20))
            loss = model.loss(*get_batch(train_ids, batch_size, context, rng))
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            optimizer.step(lr)
            if (step + 1) % max(1, steps // 8) == 0:
                log(f"  étape {step + 1}/{steps} · perte {loss.item():.3f}")
        train_time = time.perf_counter() - start
        val = _evaluate(model, val_ids, batch_size, context, eval_batches)

        quantized = copy.deepcopy(model)
        quantized.load_state_dict(dequantize_state(quantize_state(model.state_dict())))
        val_int8 = _evaluate(quantized, val_ids, batch_size, context, eval_batches)
        with tempfile.TemporaryDirectory() as tmp:
            size_f32 = save_model(model, Path(tmp) / "m.npz").stat().st_size
            size_int8 = save_model(model, Path(tmp) / "q.npz", quantize=True).stat().st_size
        gen_speed, mem_256, mem_2048 = _generation_speed(model)
        result = {
            "name": name, "arch": arch, "params": params, "val_loss": val,
            "bits_per_char": val / math.log(2) / chars_per_token, "val_loss_int8": val_int8,
            "train_seconds": train_time, "train_tokens_per_s": steps * batch_size * context / train_time,
            "gen_tokens_per_s": gen_speed, "state_bytes_256": mem_256, "state_bytes_2048": mem_2048,
            "file_bytes_f32": size_f32, "file_bytes_int8": size_int8,
        }
        log(f"  validation {val:.3f} ({result['bits_per_char']:.3f} bits/caractère) · int8 {val_int8:.3f} · "
            f"{result['train_tokens_per_s']:.0f} tokens/s à l'entraînement · {gen_speed:.0f} tokens/s en génération")
        results.append(result)
    return results


def format_results(results: list[dict]) -> str:
    def kb(n):
        return f"{n / 1024:,.0f} Ko".replace(",", " ")

    lines = ["| Modèle | Paramètres | Perte val. ↓ | Bits/car. ↓ | Perte int8 | Entraînement | "
             "Génération | Mémoire @256 | Mémoire @2048 | Fichier f32 | Fichier int8 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['params'] / 1e6:.2f} M | {r['val_loss']:.3f} | {r['bits_per_char']:.3f} | "
            f"{r['val_loss_int8']:.3f} | {r['train_tokens_per_s']:,.0f} tok/s | {r['gen_tokens_per_s']:,.0f} tok/s | "
            f"{kb(r['state_bytes_256'])} | {kb(r['state_bytes_2048'])} | {kb(r['file_bytes_f32'])} | "
            f"{kb(r['file_bytes_int8'])} |".replace(",", " "))
    return "\n".join(lines)
