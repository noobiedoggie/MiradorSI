from miradorsi.agent import MiradorAgent, _parse_depth
from miradorsi.analysis.researcher import DeepAnalyzer
from miradorsi.config import load_config
from miradorsi.identity import Identity
from miradorsi.model.train import identity_corpus, load_corpus_text

from test_analysis import FakeFetcher, FakeSearch


def make_agent(tmp_path):
    cfg = load_config()
    cfg.paths.reports = str(tmp_path / "reports")
    identity = Identity.load(cfg.path("identity"))
    search, fetcher = FakeSearch(), FakeFetcher()
    analyzer = DeepAnalyzer(search, fetcher, cfg.analysis, analyst=identity.name)
    return MiradorAgent(cfg, identity, analyzer, search, fetcher, brain=None)


def test_identity_loaded_from_config():
    identity = Identity.load(load_config().path("identity"))
    assert identity.name == "Mirador SI"
    assert "Mirador SI" in identity.system_prompt()
    assert len(identity.training_dialogues()) >= 15
    assert "<|user|>Qui es-tu ?<|end|><|assistant|>Je suis Mirador SI." in identity_corpus(identity, 1)


def test_corpus_loader_skips_readme():
    text = load_corpus_text(load_config().path("corpus"))
    assert "Observer, vérifier, comprendre" in text
    assert "Corpus d'entraînement" not in text


def test_parse_depth():
    assert _parse_depth("le climat -p 3") == ("le climat", 3)
    assert _parse_depth("le climat") == ("le climat", None)


def test_agent_routing(tmp_path):
    agent = make_agent(tmp_path)
    assert "/analyse" in agent.handle("/aide")
    assert "Mirador SI" in agent.handle("/identite")
    assert "Mirador SI" in agent.handle("Comment t'appelles-tu ?")   # mémoire d'identité sans modèle
    assert "mirador train" in agent.handle("Raconte une blague sur les chats")
    answer = agent.handle("/analyse Quelle est la puissance solaire installée en France ? -p 1")
    assert "Rapport complet" in answer and "[1]" in answer
    assert agent.last_report.depth == 1
    assert list((tmp_path / "reports").glob("*.md"))
    answer = agent.handle("analyse : puissance solaire en France")
    assert "Analyse (" in answer
