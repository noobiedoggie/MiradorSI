"""Vigie : l'architecture compacte de Mirador SI, pensée pour tourner sur un PC.

Idées combinées (chacune connue séparément, l'assemblage est propre à Mirador SI) :

1. **Profondeur recyclée** — quelques blocs *uniques* sont réappliqués `n_loops` fois
   (comme on relit un texte). On obtient la profondeur d'un grand réseau avec les
   paramètres d'un petit. Chaque passage garde ses propres normalisations : le même
   bloc peut ainsi jouer un rôle différent à chaque tour.
2. **Mélangeur hybride** — un bloc *récurrent* (mémoire continue, coût constant par
   token) alterne avec un bloc d'*attention locale* (fenêtre glissante, précise pour
   recopier un nom ou un chiffre proche).
3. **Mémoire multi-échelles** — chaque canal de la récurrence démarre avec une durée
   de mémoire différente (de 2 à 512 tokens), du mot au paragraphe.
4. **Attention multi-requêtes** — toutes les têtes partagent une seule paire clé/valeur :
   cache d'inférence 4× plus petit.
5. **MLP SwiGLU** et **embeddings factorisés** (vocabulaire projeté dans un petit espace).

Conséquence pour un PC : pendant la génération, la mémoire et le temps par token sont
**constants** (état récurrent + fenêtre d'attention bornée), quelle que soit la longueur
de la conversation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import functional as F
from ..core.autograd import Tensor, get_default_dtype
from ..core.nn import Embedding, Linear, Module, Parameter, RMSNorm
from . import streaming as S


@dataclass
class VigieConfig:
    vocab_size: int = 2048
    context_length: int = 256      # longueur des séquences d'entraînement
    d_model: int = 256
    d_embed: int = 128             # 0 ou d_model = pas de factorisation
    pattern: str = "RA"            # blocs uniques : R = récurrent, A = attention locale
    n_loops: int = 2               # nombre de passages sur les blocs uniques
    n_heads: int = 4
    n_kv_heads: int = 1
    window: int = 64               # portée de l'attention locale
    conv_kernel: int = 4
    mlp_hidden: int = 0            # 0 = automatique (≈ 8/3 · d_model)
    memory_min: float = 2.0        # durées de mémoire initiales de la récurrence (tokens)
    memory_max: float = 512.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    seed: int = 1337

    def __post_init__(self):
        if not self.pattern or set(self.pattern) - {"R", "A"}:
            raise ValueError("pattern doit contenir uniquement R et/ou A")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model % n_heads et n_heads % n_kv_heads doivent valoir 0")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("la dimension par tête doit être paire (RoPE)")
        if self.mlp_hidden <= 0:
            self.mlp_hidden = int(math.ceil(8 * self.d_model / 3 / 32) * 32)
        if self.d_embed <= 0:
            self.d_embed = self.d_model

    @property
    def factorized(self) -> bool:
        return self.d_embed != self.d_model

    @property
    def applications(self) -> int:
        return self.n_loops * len(self.pattern)


class RecurrentMixer(Module):
    """Récurrence à portes : h_t = a_t ⊙ h_{t-1} + (1 - a_t) ⊙ conv(x)_t ; sortie = h ⊙ GELU(porte)."""

    def __init__(self, cfg: VigieConfig, rng: np.random.Generator):
        d = cfg.d_model
        out_std = 0.02 / math.sqrt(2 * cfg.applications)
        self.value = Linear(d, d, rng)
        self.retention = Linear(d, d, rng, bias=True, std=0.01)
        self.gate = Linear(d, d, rng)
        self.out = Linear(d, d, rng, std=out_std)
        conv = rng.standard_normal((cfg.conv_kernel, d)) * (0.1 / cfg.conv_kernel)
        conv[0] += 1.0  # démarre proche de l'identité
        self.conv_w = Parameter(conv)
        self.conv_b = Parameter(np.zeros(d))
        timescales = np.exp(np.linspace(math.log(cfg.memory_min), math.log(cfg.memory_max), d))
        self.retention.bias.data[:] = np.log(timescales - 1.0 + 1e-6)  # σ(b) = 1 - 1/τ

    def forward(self, x: Tensor, **_) -> Tensor:
        v = F.causal_conv(self.value(x), self.conv_w, self.conv_b)
        a = F.sigmoid(self.retention(x))
        h = F.linear_scan(a, (1.0 - a) * v)
        return self.out(h * F.gelu(self.gate(x)))

class LocalAttention(Module):
    """Attention causale à fenêtre glissante, multi-requêtes, avec RoPE."""

    def __init__(self, cfg: VigieConfig, rng: np.random.Generator):
        d = cfg.d_model
        self.cfg = cfg
        self.head_dim = d // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.q = Linear(d, d, rng)
        self.k = Linear(d, cfg.n_kv_heads * self.head_dim, rng)
        self.v = Linear(d, cfg.n_kv_heads * self.head_dim, rng)
        self.out = Linear(d, d, rng, std=0.02 / math.sqrt(2 * cfg.applications))

    def forward(self, x: Tensor, cos=None, sin=None, mask=None) -> Tensor:
        B, T, C = x.shape
        nkv, g, hd = self.cfg.n_kv_heads, self.groups, self.head_dim
        q = F.rope(self.q(x).reshape(B, T, nkv, g, hd).transpose(0, 2, 3, 1, 4), cos, sin)
        k = F.rope(self.k(x).reshape(B, T, nkv, 1, hd).transpose(0, 2, 3, 1, 4), cos, sin)
        v = self.v(x).reshape(B, T, nkv, 1, hd).transpose(0, 2, 3, 1, 4)
        scores = (q @ k.transpose(0, 1, 2, 4, 3)) * (1.0 / math.sqrt(hd))
        weights = F.softmax(F.add_constant(scores, mask), axis=-1)
        y = (weights @ v).transpose(0, 3, 1, 2, 4).reshape(B, T, C)
        return self.out(y)

class SwiGLU(Module):
    def __init__(self, cfg: VigieConfig, rng: np.random.Generator):
        d, h = cfg.d_model, cfg.mlp_hidden
        self.w_gate = Linear(d, h, rng)
        self.w_up = Linear(d, h, rng)
        self.w_down = Linear(h, d, rng, std=0.02 / math.sqrt(2 * cfg.applications))

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class SharedBlock(Module):
    """Les poids d'un bloc, partagés entre tous ses passages."""

    def __init__(self, kind: str, cfg: VigieConfig, rng: np.random.Generator):
        self.kind = kind
        self.mixer = RecurrentMixer(cfg, rng) if kind == "R" else LocalAttention(cfg, rng)
        self.mlp = SwiGLU(cfg, rng)


class Vigie(Module):
    arch = "vigie"

    def __init__(self, cfg: VigieConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.embed = Embedding(cfg.vocab_size, cfg.d_embed, rng)
        if cfg.factorized:
            self.proj_in = Linear(cfg.d_embed, cfg.d_model, rng, std=1.0 / math.sqrt(cfg.d_embed))
            self.proj_out = Linear(cfg.d_model, cfg.d_embed, rng, std=1.0 / math.sqrt(cfg.d_model))
        self.blocks = [SharedBlock(kind, cfg, rng) for kind in cfg.pattern]
        # normalisations propres à chaque passage : le bloc partagé « sait » à quel tour il est
        self.mix_norms = [RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.applications)]
        self.mlp_norms = [RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.applications)]
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        dtype = get_default_dtype()
        head_dim = cfg.d_model // cfg.n_heads
        self._cos, self._sin = F.rope_tables(cfg.context_length, head_dim, cfg.rope_base, dtype)
        i = np.arange(cfg.context_length)
        allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < cfg.window)
        self._mask = np.where(allowed, 0.0, -1e9).astype(dtype)

    def _schedule(self):
        """Ordre d'application : (index du passage, bloc partagé)."""
        app = 0
        for _ in range(self.cfg.n_loops):
            for block in self.blocks:
                yield app, block
                app += 1

    # ------------------------------------------------------------ entraînement
    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        T = idx.shape[1]
        if T > self.cfg.context_length:
            raise ValueError(f"Séquence trop longue ({T} > {self.cfg.context_length})")
        cos, sin, mask = self._cos[:T], self._sin[:T], self._mask[:T, :T]
        x = self.embed(idx)
        if self.cfg.factorized:
            x = self.proj_in(x)
        for app, block in self._schedule():
            x = x + block.mixer(self.mix_norms[app](x), cos=cos, sin=sin, mask=mask)
            x = x + block.mlp(self.mlp_norms[app](x))
        x = self.norm(x)
        if self.cfg.factorized:
            x = self.proj_out(x)
        return x @ self.embed.weight.T

    def loss(self, idx: np.ndarray, targets: np.ndarray, ignore_index: int = -1) -> Tensor:
        return F.cross_entropy(self.forward(idx), targets, ignore_index)

    # ------------------------------------------------------------ inférence en flux
    def init_state(self) -> dict:
        """Prépare un moteur d'inférence (poids fusionnés) et son état vide."""
        runtime = VigieRuntime(self)
        return runtime.init_state()

    def step(self, token: int, state: dict) -> np.ndarray:
        return state["runtime"].step(token, state)

    def prefill(self, tokens, state: dict) -> np.ndarray:
        """Lit tout un texte d'un coup (même état que `step` jeton par jeton, bien plus vite)."""
        return state["runtime"].prefill(tokens, state)

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 200, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.95, stop_ids: set[int] | None = None,
                 rng: np.random.Generator | None = None, **options) -> list[int]:
        return S.generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p, stop_ids, rng, **options)


class VigieRuntime:
    """Moteur d'inférence pour PC : un token à la fois, mémoire constante.

    Optimisations : matrices lisant la même entrée fusionnées en un seul produit,
    tampons circulaires pré-alloués (aucune réallocation par token), RoPE calculé une
    fois par token et partagé par toutes les couches d'attention.
    """

    def __init__(self, model: Vigie):
        cfg = self.cfg = model.cfg
        self.dtype = model.embed.weight.data.dtype
        self.embed = model.embed.weight.data
        self.embed_t = np.ascontiguousarray(self.embed.T)
        self.proj_in = model.proj_in.weight.data if cfg.factorized else None
        self.proj_out = model.proj_out.weight.data if cfg.factorized else None
        self.final_norm = model.norm.weight.data
        self.head_dim = cfg.d_model // cfg.n_heads
        self.inv_freq = 1.0 / (cfg.rope_base ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim))
        fused: dict[int, dict] = {}
        for block in model.blocks:
            mlp = block.mlp
            weights = {"kind": block.kind, "w_mlp": np.concatenate([mlp.w_gate.weight.data, mlp.w_up.weight.data], 1),
                       "w_down": mlp.w_down.weight.data, "hidden": cfg.mlp_hidden}
            m = block.mixer
            if block.kind == "R":
                d = cfg.d_model
                weights.update(
                    w_in=np.concatenate([m.value.weight.data, m.retention.weight.data, m.gate.weight.data], 1),
                    b_in=np.concatenate([np.zeros(d, self.dtype), m.retention.bias.data, np.zeros(d, self.dtype)]),
                    conv_now=m.conv_w.data[0], conv_past=m.conv_w.data[1:], conv_b=m.conv_b.data, w_out=m.out.weight.data)
            else:
                weights.update(w_in=np.concatenate([m.q.weight.data, m.k.weight.data, m.v.weight.data], 1),
                               w_out=m.out.weight.data)
            fused[id(block)] = weights
        self.layers = [(fused[id(block)], model.mix_norms[app].weight.data, model.mlp_norms[app].weight.data)
                       for app, block in model._schedule()]
        self.has_attention = "A" in cfg.pattern

    def init_state(self) -> dict:
        cfg, dt = self.cfg, self.dtype
        layers = []
        for weights, _, _ in self.layers:
            if weights["kind"] == "R":
                layers.append({"conv": np.zeros((cfg.conv_kernel - 1, cfg.d_model), dt),
                               "h": np.zeros(cfg.d_model, dt)})
            else:
                shape = (cfg.window, cfg.n_kv_heads, self.head_dim)
                layers.append({"k": np.zeros(shape, dt), "v": np.zeros(shape, dt)})
        return {"runtime": self, "position": 0, "layers": layers}

    def _norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return x * (weight / np.sqrt(np.dot(x, x) / x.size + self.cfg.norm_eps))

    def _norm_rows(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return x * (weight / np.sqrt((x * x).mean(axis=1, keepdims=True) + self.cfg.norm_eps))

    def prefill(self, tokens, state: dict) -> np.ndarray:
        """Lit `tokens` en une passe matricielle et met l'état à jour exactement comme autant
        d'appels à `step` ; renvoie les logits du dernier jeton. La récurrence reste une boucle
        (un produit par jeton sur un vecteur), tout le reste est calculé en bloc."""
        cfg, d, hd = self.cfg, self.cfg.d_model, self.head_dim
        tokens = np.asarray(tokens, dtype=np.int64)
        T = len(tokens)
        if T == 0:
            raise ValueError("prefill : texte vide")
        x = self.embed[tokens]
        if self.proj_in is not None:
            x = x @ self.proj_in
        position = state["position"]
        if self.has_attention:
            positions = position + np.arange(T)
            angles = positions[:, None] * self.inv_freq[None, :]
            angles = np.concatenate([angles, angles], 1)
            cos, sin = np.cos(angles).astype(self.dtype), np.sin(angles).astype(self.dtype)
            half, kv, window = hd // 2, cfg.n_kv_heads * hd, cfg.window
            past = min(position, window)
            past_pos = np.arange(position - past, position)
            all_pos = np.concatenate([past_pos, positions])
            visible = (all_pos[None, :] <= positions[:, None]) & (all_pos[None, :] > positions[:, None] - window)

            def rotate(t, c, s_):  # t : (T, …, hd)
                return t * c + np.concatenate([-t[..., half:], t[..., :half]], -1) * s_
        for (w, mix_norm, mlp_norm), st in zip(self.layers, state["layers"]):
            z = self._norm_rows(x, mix_norm) @ w["w_in"]
            if w["kind"] == "R":
                z += w["b_in"]
                u, a, g = z[:, :d], S.sigmoid(z[:, d:2 * d]), z[:, 2 * d:]
                buf = st["conv"]                                   # buf[0] = entrée précédente
                history = np.concatenate([buf[::-1], u], 0)        # du plus ancien au plus récent
                k = len(buf)
                v = w["conv_now"] * u + w["conv_b"]
                for j in range(k):                                 # conv_past[j] × u(t-1-j)
                    v = v + w["conv_past"][j] * history[k - 1 - j: k - 1 - j + T]
                if k:
                    st["conv"] = history[::-1][:k].copy()
                h = st["h"]
                hs = np.empty_like(v)
                for t in range(T):
                    h = a[t] * h + (1.0 - a[t]) * v[t]
                    hs[t] = h
                st["h"] = h
                x = x + (hs * S.gelu(g)) @ w["w_out"]
            else:
                q = rotate(z[:, :d].reshape(T, cfg.n_kv_heads, -1, hd), cos[:, None, None, :], sin[:, None, None, :])
                kn = rotate(z[:, d:d + kv].reshape(T, cfg.n_kv_heads, hd), cos[:, None, :], sin[:, None, :])
                vn = z[:, d + kv:].reshape(T, cfg.n_kv_heads, hd)
                slots = past_pos % window
                keys = np.concatenate([st["k"][slots], kn], 0)      # (N, kv, hd)
                values = np.concatenate([st["v"][slots], vn], 0)
                scores = np.einsum("tkgh,nkh->kgtn", q, keys) * (1.0 / np.sqrt(hd))
                scores = np.where(visible[None, None], scores, -np.inf)
                y = np.einsum("kgtn,nkh->tkgh", S.softmax(scores), values)
                keep = np.arange(max(0, T - window), T)            # le tampon garde les derniers
                st["k"][positions[keep] % window] = kn[keep]
                st["v"][positions[keep] % window] = vn[keep]
                x = x + y.reshape(T, -1) @ w["w_out"]
            gu = self._norm_rows(x, mlp_norm) @ w["w_mlp"]
            hidden = w["hidden"]
            x = x + (S.silu(gu[:, :hidden]) * gu[:, hidden:]) @ w["w_down"]
        state["position"] = position + T
        last = self._norm(x[-1], self.final_norm)
        if self.proj_out is not None:
            last = last @ self.proj_out
        return last @ self.embed_t

    def step(self, token: int, state: dict) -> np.ndarray:
        cfg, d, hd = self.cfg, self.cfg.d_model, self.head_dim
        x = self.embed[token]
        if self.proj_in is not None:
            x = x @ self.proj_in
        position = state["position"]
        if self.has_attention:
            angles = position * self.inv_freq
            angles = np.concatenate([angles, angles])
            cos, sin = np.cos(angles).astype(self.dtype), np.sin(angles).astype(self.dtype)
            half = hd // 2
            slot, filled = position % cfg.window, min(position + 1, cfg.window)
            kv = cfg.n_kv_heads * hd
        for (w, mix_norm, mlp_norm), st in zip(self.layers, state["layers"]):
            z = self._norm(x, mix_norm) @ w["w_in"]
            if w["kind"] == "R":
                z += w["b_in"]
                u, a, g = z[:d], S.sigmoid(z[d:2 * d]), z[2 * d:]
                buf = st["conv"]
                v = w["conv_now"] * u + (w["conv_past"] * buf).sum(axis=0) + w["conv_b"]
                if len(buf):
                    buf[1:] = buf[:-1].copy()
                    buf[0] = u
                st["h"] = a * st["h"] + (1.0 - a) * v
                x = x + (st["h"] * S.gelu(g)) @ w["w_out"]
            else:
                q = z[:d].reshape(cfg.n_kv_heads, -1, hd)
                k = z[d:d + kv].reshape(cfg.n_kv_heads, hd)
                q = q * cos + np.concatenate([-q[..., half:], q[..., :half]], -1) * sin
                st["k"][slot] = k * cos + np.concatenate([-k[..., half:], k[..., :half]], -1) * sin
                st["v"][slot] = z[d + kv:].reshape(cfg.n_kv_heads, hd)
                keys, values = st["k"][:filled], st["v"][:filled]             # (n, kv, hd)
                scores = q @ keys.transpose(1, 2, 0) * (1.0 / np.sqrt(hd))     # (kv, g, n)
                y = S.softmax(scores) @ values.transpose(1, 0, 2)               # (kv, g, hd)
                x = x + y.reshape(-1) @ w["w_out"]
            gu = self._norm(x, mlp_norm) @ w["w_mlp"]
            hidden = w["hidden"]
            x = x + (S.silu(gu[:hidden]) * gu[hidden:]) @ w["w_down"]
        state["position"] = position + 1
        x = self._norm(x, self.final_norm)
        if self.proj_out is not None:
            x = x @ self.proj_out
        return x @ self.embed_t
