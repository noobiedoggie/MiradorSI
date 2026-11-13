"""Entraînement du modèle de Mirador SI (Vigie ou Transformer) sur le corpus local + l'identité."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..core import AdamW, clip_grad_norm, cosine_lr, no_grad
from ..identity import Identity
from .checkpoint import build_model, load_model, model_arch, save_model
from .tokenizer import BPETokenizer

TEXT_SUFFIXES = {".txt", ".md"}


def _n(value: float) -> str:
    """12345 → « 12 345 » (séparateur de milliers à la française)."""
    return f"{value:,.0f}".replace(",", "\u202f")


def format_dialogue(user: str, assistant: str) -> str:
    return f"<|user|>{user}<|end|><|assistant|>{assistant}<|end|>\n"


def load_corpus_documents(corpus_dir: Path) -> list[str]:
    """Tous les .txt / .md du dossier (récursivement), un document par fichier."""
    if not corpus_dir.exists():
        return []
    texts = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file() and path.name.lower() != "readme.md":
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                texts.append(content)
    return texts


def join_documents(texts: list[str]) -> str:
    return "".join(f"{t}\n<|end|>\n" for t in texts)


def load_corpus_text(corpus_dir: Path) -> str:
    return join_documents(load_corpus_documents(corpus_dir))


def split_documents(docs: list[str], fraction: float = 0.08) -> tuple[list[str], list[str]]:
    """Met de côté des documents ENTIERS pour la validation (jamais vus à l'entraînement).

    Avec un seul document, on garde sa fin : la validation reste du texte inédit.
    """
    if not docs:
        return [], []
    if len(docs) >= 5:
        val = docs[::max(2, round(1 / fraction))]
        return [d for d in docs if d not in val], val
    cut = int(sum(map(len, docs)) * (1 - fraction))
    train, val, seen = [], [], 0
    for d in docs:
        if seen + len(d) <= cut:
            train.append(d)
        elif seen >= cut:
            val.append(d)
        else:
            train.append(d[: cut - seen])
            val.append(d[cut - seen:])
        seen += len(d)
    return train, val


def identity_corpus(identity: Identity, repeats: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    pairs = identity.training_dialogues()
    chunks = []
    for _ in range(max(1, repeats)):
        for i in rng.permutation(len(pairs)):
            chunks.append(format_dialogue(*pairs[i]))
    return "".join(chunks)


def evaluate(model, data: np.ndarray, batch_size: int, context: int, batches: int,
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


def model_settings(cfg: Config, vocab_size: int) -> dict:
    settings = asdict(cfg.model)
    settings.pop("arch")
    settings["vocab_size"] = vocab_size
    return settings


def train(cfg: Config, steps: int | None = None, fresh: bool = False, retrain_tokenizer: bool = False,
          int8: bool = False, log: Callable[[str], None] = print) -> Path:
    identity = Identity.load(cfg.path("identity"))
    tcfg, mcfg = cfg.training, cfg.model
    steps = steps or tcfg.steps
    ckpt_dir = cfg.path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok_path, model_path = ckpt_dir / "tokenizer.json", ckpt_dir / "model.npz"

    docs = load_corpus_documents(cfg.path("corpus"))
    train_docs, val_docs = split_documents(docs)
    dialogues = identity_corpus(identity, tcfg.identity_repeats, mcfg.seed)
    train_text = join_documents(train_docs) + dialogues
    val_text = join_documents(val_docs)
    log(f"Corpus : {len(docs)} document(s), {_n(sum(map(len, docs)))} caractères "
        f"+ dialogues d'identité ({_n(len(dialogues))} caractères)")

    if tok_path.exists() and not (retrain_tokenizer or fresh):
        tokenizer = BPETokenizer.load(tok_path)
        log(f"Tokenizer chargé ({tokenizer.vocab_size} tokens)")
    else:
        log(f"Entraînement du tokenizer BPE (cible : {mcfg.vocab_size} tokens)…")
        tokenizer = BPETokenizer.train(train_text + val_text, mcfg.vocab_size)
        tokenizer.save(tok_path)
        fresh = True  # nouveau vocabulaire → nouveau modèle
        log(f"Tokenizer prêt : {tokenizer.vocab_size} tokens")

    train_data = np.array(tokenizer.encode(train_text), dtype=np.int64)
    val_data = np.array(tokenizer.encode(val_text), dtype=np.int64)
    context = min(mcfg.context_length, max(16, len(train_data) // 4))
    val_context = min(context, len(val_data) - 2)
    if val_context < 16:
        val_context = 0
        log("Corpus trop petit pour une validation séparée : ajoute des documents dans data/corpus/.")
    log(f"Tokens : entraînement {_n(len(train_data))}, validation {_n(len(val_data))} "
        f"(documents jamais vus à l'entraînement)")

    model = build_model(mcfg.arch, **model_settings(cfg, tokenizer.vocab_size))
    start_step = 0
    if model_path.exists() and not fresh:
        previous, extra = load_model(model_path)
        if model_arch(previous) == model_arch(model) and previous.cfg == model.cfg:
            model, start_step = previous, int(extra.get("step", 0))
            log(f"Reprise de l'entraînement à l'étape {start_step}")
        else:
            log("Architecture ou configuration modifiée : nouveau modèle initialisé.")
    describe = (f"{mcfg.n_layers} couches" if mcfg.arch == "transformer" else
                f"blocs {model.cfg.pattern} × {model.cfg.n_loops} passages, fenêtre {model.cfg.window}")
    log(f"Architecture {model_arch(model)} : {_n(model.num_parameters())} paramètres, {describe}, "
        f"d={model.cfg.d_model}")

    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    rng = np.random.default_rng(mcfg.seed + start_step)
    total = start_step + steps
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
            if val_context:
                val_loss = evaluate(model, val_data, tcfg.batch_size, val_context, tcfg.eval_batches, rng)
                log(f"  ↳ validation (texte inédit) : perte {val_loss:.3f} (perplexité {np.exp(val_loss):.1f})")
            save_model(model, model_path, extra={"step": done, "val_loss": val_loss, "identity": identity.name,
                                                 "tokenizer": tok_path.name})

    log(f"Modèle sauvegardé : {model_path} ({model_path.stat().st_size / 1e6:.1f} Mo)")
    if val_context and val_loss > np.log(tokenizer.vocab_size):
        log(f"⚠ Sur du texte inédit, le modèle fait moins bien que le hasard (perte {val_loss:.1f} > "
            f"{np.log(tokenizer.vocab_size):.1f}) : il a mémorisé son corpus au lieu de généraliser. "
            "Ajoute du texte (data/corpus/ ou `mirador corpus-web`) pour qu'il apprenne vraiment la langue.")
    if int8:
        q_path = save_model(model, ckpt_dir / "model-int8.npz",
                            extra={"step": total, "val_loss": val_loss, "identity": identity.name}, quantize=True)
        log(f"Version int8 : {q_path} ({q_path.stat().st_size / 1e6:.1f} Mo)")
    prompt = "<|user|>Qui es-tu ?<|end|><|assistant|>"
    sample_ids = model.generate(tokenizer.encode(prompt), max_new_tokens=60, temperature=0.5,
                                stop_ids={tokenizer.end_id}, rng=np.random.default_rng(0))
    log(f"Exemple — « Qui es-tu ? » → {tokenizer.decode(sample_ids, skip_special=True).strip()}")
    return model_path
