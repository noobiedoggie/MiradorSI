import json

import numpy as np
import pytest

from miradorsi.core import AdamW
from miradorsi.core.autograd import set_default_dtype
from miradorsi.core.optim import clip_grad_norm
from miradorsi.model.checkpoint import build_model, load_model, save_model
from miradorsi.model.streaming import state_bytes
from miradorsi.model.transformer import MiradorLM, ModelConfig
from miradorsi.model.vigie import Vigie, VigieConfig

TINY = VigieConfig(vocab_size=64, context_length=24, d_model=32, d_embed=16, pattern="RA", n_loops=2,
                   n_heads=4, n_kv_heads=2, window=6, seed=0)


def test_forward_shape_and_causality():
    model = Vigie(TINY)
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    b = np.array([[1, 2, 3, 4, 9, 9, 9, 9]])
    la, lb = model.forward(a).data, model.forward(b).data
    assert la.shape == (1, 8, 64)
    np.testing.assert_allclose(la[0, :4], lb[0, :4], rtol=1e-5, atol=1e-6)
    assert not np.allclose(la[0, 4:], lb[0, 4:])


@pytest.mark.parametrize("model", [Vigie(TINY), MiradorLM(ModelConfig(vocab_size=64, context_length=24,
                                                                      d_model=32, n_layers=2, n_heads=2))])
def test_streaming_matches_full_forward(model):
    """La génération token par token doit donner exactement les mêmes logits que l'entraînement."""
    ids = np.random.default_rng(3).integers(0, 64, 20)  # 20 > fenêtre (6) : teste le glissement
    full = model.forward(ids[None]).data[0]
    state = model.init_state()
    steps = np.stack([model.step(int(t), state) for t in ids])
    np.testing.assert_allclose(steps, full, rtol=1e-4, atol=1e-5)


def test_inference_memory_is_constant():
    model = Vigie(TINY)
    state = model.init_state()
    sizes = []
    for i in range(40):
        model.step(i % 64, state)
        sizes.append(state_bytes(state))
    assert sizes[10] == sizes[39]  # au-delà de la fenêtre, la mémoire ne grandit plus


def test_weight_sharing_keeps_model_compact():
    one = Vigie(VigieConfig(**{**TINY.__dict__, "n_loops": 1}))
    three = Vigie(VigieConfig(**{**TINY.__dict__, "n_loops": 3}))
    extra = three.num_parameters() - one.num_parameters()
    assert extra == 2 * 2 * 2 * TINY.d_model  # seules les normalisations par passage s'ajoutent


def test_full_model_gradient_check():
    set_default_dtype(np.float64)
    try:
        cfg = VigieConfig(vocab_size=11, context_length=8, d_model=8, d_embed=4, pattern="RA", n_loops=2,
                          n_heads=2, n_kv_heads=1, window=3, mlp_hidden=16, seed=1)
        model = Vigie(cfg)
        idx = np.array([[1, 3, 5, 7, 2, 4]])
        tgt = np.array([[3, 5, 7, 2, 4, 6]])
        model.loss(idx, tgt).backward()
        rng = np.random.default_rng(0)
        for name, p in model.named_parameters():
            flat = p.data.reshape(-1)
            for i in rng.choice(flat.size, size=min(3, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + 1e-6
                plus = model.loss(idx, tgt).item()
                flat[i] = old - 1e-6
                minus = model.loss(idx, tgt).item()
                flat[i] = old
                numeric = (plus - minus) / 2e-6
                assert abs(p.grad.reshape(-1)[i] - numeric) < 1e-6 + 1e-4 * abs(numeric), name
    finally:
        set_default_dtype(np.float32)


def test_learns_a_sequence():
    model = Vigie(TINY)
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    seq = np.random.default_rng(0).integers(0, 64, (2, 21))
    first = None
    for _ in range(60):
        loss = model.loss(seq[:, :-1], seq[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    assert loss.item() < first * 0.25


def test_checkpoint_roundtrip_and_int8(tmp_path):
    model = Vigie(VigieConfig(**{**TINY.__dict__, "d_model": 64, "d_embed": 64}))
    x = np.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    save_model(model, tmp_path / "f.npz", extra={"step": 3})
    save_model(model, tmp_path / "q.npz", quantize=True)
    loaded, extra = load_model(tmp_path / "f.npz")
    assert isinstance(loaded, Vigie) and extra["step"] == 3 and not extra["quantized"]
    np.testing.assert_allclose(loaded.forward(x).data, model.forward(x).data)
    quantized, extra = load_model(tmp_path / "q.npz")
    assert extra["quantized"]
    assert (tmp_path / "q.npz").stat().st_size < (tmp_path / "f.npz").stat().st_size / 2.5
    diff = np.abs(quantized.forward(x).data - model.forward(x).data).max()
    assert diff < 0.05


def test_loads_v01_checkpoints_without_arch(tmp_path):
    model = MiradorLM(ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=1, n_heads=2))
    path = tmp_path / "old.npz"
    meta = {"config": model.cfg.__dict__, "extra": {"step": 1}}  # format de la v0.1
    np.savez(path, __meta__=np.array(json.dumps(meta)), **model.state_dict())
    loaded, _ = load_model(path)
    assert isinstance(loaded, MiradorLM)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build_model("inconnue")
    with pytest.raises(ValueError):
        VigieConfig(pattern="RX")


def test_split_documents_holds_out_whole_texts():
    from miradorsi.model.train import split_documents

    docs = [f"document {i} " * 20 for i in range(30)]
    train, val = split_documents(docs)
    assert val and not set(train) & set(val) and len(train) + len(val) == 30
    train, val = split_documents(["un seul long document. " * 50])
    assert len(train) == 1 and len(val) == 1 and len(val[0]) < len(train[0])


@pytest.mark.parametrize("arch", ["vigie", "transformer"])
def test_training_pipeline_end_to_end(tmp_path, arch):
    from miradorsi.config import load_config
    from miradorsi.model.brain import Brain
    from miradorsi.model.train import train

    cfg = load_config()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("Mirador observe et vérifie les sources. " * 40, encoding="utf-8")
    cfg.paths.corpus, cfg.paths.checkpoints = str(tmp_path / "corpus"), str(tmp_path / "ckpt")
    cfg.model.arch, cfg.model.vocab_size, cfg.model.context_length = arch, 300, 32
    cfg.model.d_model, cfg.model.d_embed, cfg.model.n_layers = 32, 16, 1
    cfg.training.batch_size, cfg.training.eval_every, cfg.training.identity_repeats = 2, 2, 1
    logs = []
    train(cfg, steps=3, fresh=True, int8=True, log=logs.append)
    assert (tmp_path / "ckpt" / "model.npz").exists() and (tmp_path / "ckpt" / "model-int8.npz").exists()
    assert any("validation (texte inédit)" in line for line in logs)
    brain = Brain.load(tmp_path / "ckpt")
    assert getattr(brain.model, "arch") == arch
    assert isinstance(brain.reply("Qui es-tu ?"), str)


def test_training_mixes_identity_dialogues_into_large_corpus(tmp_path):
    from miradorsi.config import load_config
    from miradorsi.model.train import train

    cfg = load_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(6):  # gros corpus : les dialogues y seraient noyés sans mélange
        (corpus / f"doc{i}.txt").write_text(f"Texte encyclopédique numéro {i} sur la science. " * 600,
                                            encoding="utf-8")
    cfg.paths.corpus, cfg.paths.checkpoints = str(corpus), str(tmp_path / "ckpt")
    cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model, cfg.model.d_embed = 300, 32, 32, 16
    cfg.training.batch_size, cfg.training.eval_every, cfg.training.identity_repeats = 5, 2, 1
    logs = []
    train(cfg, steps=4, fresh=True, log=logs.append)
    assert any(line.startswith("Mélange : 1/5") for line in logs)
    assert any("★ meilleur" in line for line in logs)


def test_repetition_penalty_discourages_recent_tokens():
    from miradorsi.model.streaming import penalize_repetition

    logits = np.array([2.0, 1.0, -1.0, 0.5])
    out = penalize_repetition(logits, [0, 2, 0], 2.0)
    assert out[0] == 1.0 and out[2] == -2.0 and out[1] == 1.0     # positifs divisés, négatifs multipliés
    assert np.array_equal(penalize_repetition(logits, [0], 1.0), logits)


def test_interrupted_training_resumes_where_it_stopped(tmp_path, monkeypatch):
    import miradorsi.model.train as T
    from miradorsi.config import load_config
    from miradorsi.core.optim import cosine_lr

    cfg = load_config()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("Mirador observe et vérifie les sources. " * 60, encoding="utf-8")
    cfg.paths.corpus, cfg.paths.checkpoints = str(tmp_path / "corpus"), str(tmp_path / "ckpt")
    cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model, cfg.model.d_embed = 300, 32, 32, 16
    cfg.training.batch_size, cfg.training.identity_repeats = 2, 1
    cfg.training.eval_every, cfg.training.checkpoint_every, cfg.training.warmup_steps = 100, 2, 2

    class Coupure(Exception):
        pass

    real_get_batch, calls = T.get_batch, {"n": 0}

    def get_batch_then_crash(*args):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Coupure  # panne simulée pendant l'étape 3 (dernier point de reprise : étape 2)
        return real_get_batch(*args)

    monkeypatch.setattr(T, "get_batch", get_batch_then_crash)
    with pytest.raises(Coupure):
        T.train(cfg, steps=6, fresh=True, log=lambda line: None)
    monkeypatch.setattr(T, "get_batch", real_get_batch)

    logs: list[str] = []
    T.train(cfg, steps=999, log=logs.append)              # `steps` ignoré : on reprend le plan initial
    assert any("Reprise d'un entraînement interrompu : étape 2/6" in l for l in logs)
    step3 = next(l for l in logs if l.startswith("étape     3/6"))
    expected = cosine_lr(2, 6, cfg.training.learning_rate, 2, cfg.training.min_lr_ratio)
    assert f"lr {expected:.2e}" in step3                  # le planning continue, il ne redémarre pas
    assert (tmp_path / "ckpt" / "reprise-optim.npz").exists()
