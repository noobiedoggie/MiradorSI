import numpy as np

from miradorsi.core import AdamW
from miradorsi.core.optim import clip_grad_norm, cosine_lr
from miradorsi.model.tokenizer import BPETokenizer
from miradorsi.model.transformer import MiradorLM, ModelConfig, sample

TINY = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=2, seed=0)


def test_forward_shape():
    model = MiradorLM(TINY)
    logits = model.forward(np.zeros((3, 10), dtype=np.int64))
    assert logits.shape == (3, 10, 64)


def test_causality_future_tokens_do_not_leak():
    model = MiradorLM(TINY)
    a = np.array([[1, 2, 3, 4, 5, 6]])
    b = np.array([[1, 2, 3, 9, 9, 9]])
    la, lb = model.forward(a).data, model.forward(b).data
    np.testing.assert_allclose(la[0, :3], lb[0, :3], rtol=1e-5, atol=1e-6)
    assert not np.allclose(la[0, 3:], lb[0, 3:])


def test_overfits_a_small_sequence():
    model = MiradorLM(TINY)
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    seq = np.random.default_rng(0).integers(0, 64, (2, 17))
    first = None
    for _ in range(60):
        loss = model.loss(seq[:, :-1], seq[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    assert loss.item() < first * 0.25


def test_save_load_roundtrip(tmp_path):
    model = MiradorLM(TINY)
    path = tmp_path / "m.npz"
    model.save(path, extra={"step": 7})
    loaded, extra = MiradorLM.load(path)
    x = np.array([[3, 1, 4, 1, 5]])
    np.testing.assert_allclose(model.forward(x).data, loaded.forward(x).data)
    assert extra["step"] == 7 and loaded.cfg == TINY


def test_generate_and_sampling():
    model = MiradorLM(TINY)
    out = model.generate([1, 2, 3], max_new_tokens=30, rng=np.random.default_rng(0))
    assert len(out) == 30 and all(0 <= t < 64 for t in out)  # dépasse le contexte sans erreur
    assert sample(np.array([0.0, 5.0, 1.0]), temperature=0) == 1


def test_cosine_schedule():
    assert cosine_lr(0, 100, 1.0, 10) == 0.1
    assert abs(cosine_lr(10, 100, 1.0, 10) - 1.0) < 1e-9
    assert abs(cosine_lr(100, 100, 1.0, 10) - 0.1) < 1e-9


def test_tokenizer_roundtrip_and_specials(tmp_path):
    text = ("Mirador SI analyse les sources. Les sources fiables sont recoupées. " * 20
            + "Émoji 🔭, accents éèàç, chiffres 12 345,6 %.")
    tok = BPETokenizer.train(text, vocab_size=300)
    assert tok.vocab_size <= 300
    sample_text = "<|user|>Qui es-tu ?<|end|><|assistant|>Mirador SI 🔭 — déjà vérifié.<|end|>"
    ids = tok.encode(sample_text)
    assert tok.decode(ids) == sample_text
    assert tok.special_id("<|user|>") in ids
    assert len(tok.encode("les sources")) < len("les sources".encode("utf-8"))  # compression
    tok.save(tmp_path / "tok.json")
    again = BPETokenizer.load(tmp_path / "tok.json")
    assert again.encode(sample_text) == ids


def test_brain_reply_uses_dialogue_format():
    from miradorsi.config import GenerationConfig
    from miradorsi.model.brain import Brain

    tok = BPETokenizer.train("Bonjour, je suis Mirador SI. " * 30, vocab_size=300)
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_length=64, d_model=32, n_layers=1, n_heads=2)
    brain = Brain(MiradorLM(cfg), tok, GenerationConfig(max_new_tokens=12))
    answer = brain.reply("Bonjour", history=[("Salut", "Bonjour !")] * 20)  # historique tronqué
    assert isinstance(answer, str) and "<|" not in answer
