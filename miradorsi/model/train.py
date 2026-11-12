"""Entraînement de MiradorLM sur le corpus local + les dialogues d'identité."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..core import AdamW, clip_grad_norm, cosine_lr, no_grad
from ..identity import Identity
from .tokenizer import BPETokenizer
from .transformer import MiradorLM, ModelConfig

TEXT_SUFFIXES = {".txt", ".md"}


def _n(value: float) -> str:
    """12345 → « 12 345 » (séparateur de milliers à la française)."""
    return f"{value:,.0f}".replace(",", "\u202f")


def format_dialogue(user: str, assistant: str) -> str:
    return f"<|user|>{user}<|end|><|assistant|>{assistant}<|end|>\n"


def load_corpus_text(corpus_dir: Path) -> str:
    """Concatène tous les .txt / .md du dossier (récursivement), séparés par <|end|>."""
    if not corpus_dir.exists():
        return ""
    texts = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file() and path.name.lower() != "readme.md":
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                texts.append(content)
    return "\n<|end|>\n".join(texts) + ("\n<|end|>\n" if texts else "")


def identity_corpus(identity: Identity, repeats: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    pairs = identity.training_dialogues()
    chunks = []
    for _ in range(max(1, repeats)):
        for i in rng.permutation(len(pairs)):
            chunks.append(format_dialogue(*pairs[i]))
    return "".join(chunks)


def evaluate(model: MiradorLM, data: np.ndarray, batch_size: int, context: int, batches: int,
             rng: np.random.Generator) -> float:
    losses = []
    with no_grad():
        for _ in range(batches):
            x, y = get_batch(data, batch_size, context, rng)
            losses.append(model.loss(x, y).item())
    return float(np.mean(losses))


def get_batch(data: np.ndarray, batch_size: int, context: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - context - 1, size=batch_size)
    x = np.stack([data[s:s + context] for s in starts])
    y = np.stack([data[s + 1:s + context + 1] for s in starts])
    return x, y


def train(cfg: Config, steps: int | None = None, fresh: bool = False, retrain_tokenizer: bool = False,
          log: Callable[[str], None] = print) -> Path:
    identity = Identity.load(cfg.path("identity"))
    tcfg, mcfg = cfg.training, cfg.model
    steps = steps or tcfg.steps
    ckpt_dir = cfg.path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok_path, model_path = ckpt_dir / "tokenizer.json", ckpt_dir / "model.npz"

    corpus = load_corpus_text(cfg.path("corpus"))
    text = corpus + identity_corpus(identity, tcfg.identity_repeats, mcfg.seed)
    log(f"Corpus : {_n(len(corpus))} caractères de documents + dialogues d'identité "
        f"→ {_n(len(text))} caractères au total")

    if tok_path.exists() and not (retrain_tokenizer or fresh):
        tokenizer = BPETokenizer.load(tok_path)
        log(f"Tokenizer chargé ({tokenizer.vocab_size} tokens)")
    else:
        log(f"Entraînement du tokenizer BPE (cible : {mcfg.vocab_size} tokens)…")
        tokenizer = BPETokenizer.train(text, mcfg.vocab_size)
        tokenizer.save(tok_path)
        fresh = True  # nouveau vocabulaire → nouveau modèle
        log(f"Tokenizer prêt : {tokenizer.vocab_size} tokens")

    data = np.array(tokenizer.encode(text), dtype=np.int64)
    context = min(mcfg.context_length, max(16, len(data) // 4))
    split = int(len(data) * 0.92)
    train_data, val_data = data[:split], data[split:]
    if len(val_data) <= context + 1:
        val_data = data[-(context * 2 + 2):]
    log(f"{_n(len(data))} tokens (entraînement {_n(len(train_data))}, validation {_n(len(val_data))})")

    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=mcfg.context_length,
                            d_model=mcfg.d_model, n_layers=mcfg.n_layers, n_heads=mcfg.n_heads,
                            mlp_ratio=mcfg.mlp_ratio, seed=mcfg.seed)
    start_step = 0
    if model_path.exists() and not fresh:
        model, extra = MiradorLM.load(model_path)
        if model.cfg != model_cfg:
            log("Configuration modifiée : nouveau modèle initialisé.")
            model = MiradorLM(model_cfg)
        else:
            start_step = int(extra.get("step", 0))
            log(f"Reprise de l'entraînement à l'étape {start_step}")
    else:
        model = MiradorLM(model_cfg)
    log(f"MiradorLM : {_n(model.num_parameters())} paramètres, {model_cfg.n_layers} couches, "
        f"d={model_cfg.d_model}, contexte={model_cfg.context_length}")

    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    rng = np.random.default_rng(mcfg.seed + start_step)
    total = start_step + steps
    best_val = float("inf")
    t0 = time.time()
    tokens_seen = 0
    val_loss = float("nan")
    for step in range(start_step, total):
        lr = cosine_lr(step - start_step, steps, tcfg.learning_rate, tcfg.warmup_steps, tcfg.min_lr_ratio)
        x, y = get_batch(train_data, tcfg.batch_size, context, rng)
        loss = model.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), tcfg.grad_clip)
        optimizer.step(lr)
        tokens_seen += x.size

        done = step + 1
        if done % 10 == 0 or done == start_step + 1:
            speed = tokens_seen / max(time.time() - t0, 1e-9)
            log(f"étape {done:5d}/{total} | perte {loss.item():.3f} | lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | {_n(speed)} tokens/s")
        if done % tcfg.eval_every == 0 or done == total:
            val_loss = evaluate(model, val_data, tcfg.batch_size, context, tcfg.eval_batches, rng)
            log(f"  ↳ validation : perte {val_loss:.3f} (perplexité {np.exp(val_loss):.1f})")
            best_val = min(best_val, val_loss)
            model.save(model_path, extra={"step": done, "val_loss": val_loss, "identity": identity.name,
                                          "tokenizer": tok_path.name})

    log(f"Modèle sauvegardé : {model_path}")
    prompt = "<|user|>Qui es-tu ?<|end|><|assistant|>"
    sample_ids = model.generate(tokenizer.encode(prompt), max_new_tokens=60, temperature=0.5,
                                stop_ids={tokenizer.end_id}, rng=np.random.default_rng(0))
    log(f"Exemple — « Qui es-tu ? » → {tokenizer.decode(sample_ids, skip_special=True).strip()}")
    return model_path
