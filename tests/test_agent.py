from miradorsi.agent import MiradorAgent, _parse_depth, is_small_talk
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


def test_small_talk_is_answered_without_web_search(tmp_path):
    for message in ["yo", "Salut ça va ?", "Merci !", "bonjour Mirador SI", "T'es qui ?", "ok", "Yo, ça dit quoi"]:
        assert is_small_talk(message), message
    for message in ["Napoléon", "yo-yo", "Salut, explique-moi la fusion nucléaire", "Hey Jude chanson des Beatles",
                    "Bonjour, quelle est la population de Lyon ?"]:
        assert not is_small_talk(message), message
    agent = make_agent(tmp_path)
    assert agent.handle("yo").startswith("Salut ! Je suis Mirador SI.")
    assert agent.handle("Bonsoir").startswith("Bonsoir !")
    assert "Ça va bien" in agent.handle("salut ça va ?")
    assert agent.handle("merci").startswith("Avec plaisir")
    assert "Mirador SI" in agent.handle("Comment tu t'appelles ?")
    assert agent.search.queries == []  # aucune recherche lancée pour de la conversation


def test_offline_answers_are_reliable_or_honest(tmp_path):
    from miradorsi.agent import OFFLINE_LIMIT, PERSONAL_REPLY, calculate, is_insult

    assert calculate("1+1") == "1 + 1 = 2"
    assert calculate("combien font 12 x 7 ?") == "12 × 7 = 84"
    assert calculate("2^10") == "2^10 = 1 024"
    assert calculate("10/4") == "10 ÷ 4 = 2,5"
    assert calculate("-5+2") == "-5 + 2 = -3"
    assert "zéro" in calculate("1/0")
    for text in ["2025", "Covid-19", "2-3 jours", "Napoléon", "9**999999"]:
        assert calculate(text) is None, text
    assert is_insult("connard") and is_insult("t'es nul") and not is_insult("le film est nul ? je me demande")

    agent = make_agent(tmp_path)
    agent.brain = object()  # un modèle est chargé : il ne doit plus inventer de réponse au hasard
    assert agent.handle("1+1") == "1 + 1 = 2"
    assert agent.handle("connard").startswith("Je reste poli.")
    assert agent.handle("j'aime le caca et toi?") == PERSONAL_REPLY
    assert agent.handle("C'est quoi la photosynthèse ?") == OFFLINE_LIMIT  # sans bibliothèque ni internet
    assert "Mirador SI" in agent.handle("Qui es-tu ?")
