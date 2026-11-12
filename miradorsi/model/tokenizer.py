"""Tokenizer BPE au niveau des octets, écrit de zéro.

Le texte est d'abord découpé en « pré-tokens » (mots, nombres, ponctuation,
espaces), puis chaque pré-token est converti en octets UTF-8. L'entraînement
fusionne itérativement la paire de symboles adjacents la plus fréquente.
Comme on part des 256 octets, n'importe quel texte (accents, emojis…) est
encodable sans token inconnu.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]

# [^\W\d_] = une lettre Unicode (y compris é, ç, ß…)
PRETOKEN_PATTERN = (
    r"'(?:[sdmtlnj]|ll|ve|re)|"   # contractions (it's, l', j'…)
    r" ?[^\W\d_]+|"                # mots
    r" ?\d{1,3}|"                  # nombres par paquets de 3 chiffres
    r" ?[^\s\w]+|"                 # ponctuation et symboles
    r"\s+(?!\S)|\s+"               # espaces
)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 special_tokens: list[str] | None = None):
        self.pattern = re.compile(PRETOKEN_PATTERN)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self.special_tokens = list(special_tokens if special_tokens is not None else SPECIAL_TOKENS)
        self._build()

    # --------------------------------------------------------------- interne
    def _build(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_to_id = {tok: base + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        if self.special_tokens:
            self._special_re = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
        else:
            self._special_re = None
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def special_id(self, token: str) -> int:
        return self.special_to_id[token]

    @property
    def end_id(self) -> int:
        return self.special_to_id["<|end|>"]

    # ----------------------------------------------------------- entraînement
    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens: list[str] | None = None,
              verbose: bool = False) -> "BPETokenizer":
        special_tokens = list(special_tokens if special_tokens is not None else SPECIAL_TOKENS)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size doit être ≥ 256 + nombre de tokens spéciaux")

        tok = cls([], special_tokens)
        counts: Counter[str] = Counter()
        for segment in tok._split_special(text):
            if segment not in tok.special_to_id:
                counts.update(tok.pattern.findall(segment))

        words = [list(piece.encode("utf-8")) for piece in counts]
        freqs = list(counts.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = {}
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                pair_where.setdefault(pair, set()).add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_count < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in list(pair_where.get(best, ())):
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged = _merge(w, best, new_id)
                words[wi] = merged
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += f
                    pair_where.setdefault(pair, set()).add(wi)
            pair_where.pop(best, None)
            if verbose and (step + 1) % 200 == 0:
                print(f"  fusion {step + 1}/{n_merges}")

        tok.merges = merges
        tok._build()
        return tok

    # --------------------------------------------------------------- encodage
    def _split_special(self, text: str) -> list[str]:
        if self._special_re is None:
            return [text]
        return [s for s in self._special_re.split(text) if s]

    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            ids = _merge(ids, best, 256 + self.ranks[best])
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        segments = self._split_special(text) if allow_special else [text]
        for segment in segments:
            if allow_special and segment in self.special_to_id:
                out.append(self.special_to_id[segment])
                continue
            for piece in self.pattern.findall(segment):
                out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids, skip_special: bool = False) -> str:
        chunks: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if not skip_special:
                    chunks.append(self.id_to_special[i].encode("utf-8"))
            elif i in self.vocab:
                chunks.append(self.vocab[i])
        return b"".join(chunks).decode("utf-8", errors="replace")

    # -------------------------------------------------------- sauvegarde
    def save(self, path: str | Path) -> None:
        data = {"type": "mirador-bpe", "version": 1, "merges": self.merges,
                "special_tokens": self.special_tokens}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
