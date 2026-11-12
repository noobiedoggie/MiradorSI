"""Le « cerveau » conversationnel : MiradorLM + tokenizer + format de dialogue."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import GenerationConfig
from .tokenizer import BPETokenizer
from .transformer import MiradorLM


class Brain:
    def __init__(self, model: MiradorLM, tokenizer: BPETokenizer, gen: GenerationConfig | None = None,
                 meta: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.gen = gen or GenerationConfig()
        self.meta = meta or {}
        self.rng = np.random.default_rng()

    @classmethod
    def load(cls, checkpoint_dir: Path, gen: GenerationConfig | None = None) -> "Brain | None":
        model_path, tok_path = checkpoint_dir / "model.npz", checkpoint_dir / "tokenizer.json"
        if not (model_path.exists() and tok_path.exists()):
            return None
        model, meta = MiradorLM.load(model_path)
        return cls(model, BPETokenizer.load(tok_path), gen, meta)

    @property
    def stop_ids(self) -> set[int]:
        t = self.tokenizer
        return {t.special_id(s) for s in ("<|end|>", "<|user|>", "<|system|>") if s in t.special_to_id}

    def reply(self, message: str, history: list[tuple[str, str]] | None = None) -> str:
        budget = self.model.cfg.context_length - self.gen.max_new_tokens
        turns = list(history or [])[-4:]
        prompt_ids: list[int] = []
        while True:
            text = "".join(f"<|user|>{u}<|end|><|assistant|>{a}<|end|>" for u, a in turns)
            text += f"<|user|>{message}<|end|><|assistant|>"
            prompt_ids = self.tokenizer.encode(text)
            if len(prompt_ids) <= budget or not turns:
                break
            turns.pop(0)
        prompt_ids = prompt_ids[-max(budget, 8):]
        out = self.model.generate(prompt_ids, self.gen.max_new_tokens, self.gen.temperature,
                                  self.gen.top_k, self.gen.top_p, self.stop_ids, self.rng)
        return self.tokenizer.decode(out, skip_special=True).strip()

    def complete(self, text: str, max_new_tokens: int | None = None) -> str:
        ids = self.tokenizer.encode(text)[-self.model.cfg.context_length:]
        out = self.model.generate(ids, max_new_tokens or self.gen.max_new_tokens, self.gen.temperature,
                                  self.gen.top_k, self.gen.top_p, {self.tokenizer.end_id}, self.rng)
        return self.tokenizer.decode(out, skip_special=True)
