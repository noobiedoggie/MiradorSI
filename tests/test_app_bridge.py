"""Pont de l'application Android : messages d'erreur compréhensibles."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "android" / "app" / "src" / "main" / "python"))  # après le dépôt : son miradorsi prime

from mirador_app.bridge import explain_errors  # noqa: E402


def test_explain_errors_hides_urls_and_merges_duplicates():
    errors = [
        "lecture impossible : Échec de https://fr.wikipedia.org/w/api.php?action=query : HTTP 429 sur https://fr.wikipedia.org/w/api.php",
        "lecture impossible : Échec de https://en.wikipedia.org/w/api.php?a=1 : HTTP 429 sur https://en.wikipedia.org/w/api.php",
        "duckduckgo : DuckDuckGo a demandé une vérification anti-robot",
        "lecture impossible : Échec de https://www.exemple.fr/page : <urlopen error timed out>",
    ]
    lines = explain_errors(errors, limit=5)
    assert lines == [
        "Wikipédia reçoit trop de requêtes depuis ton réseau : réessaie dans une minute",
        "DuckDuckGo bloque les recherches automatiques pour le moment",
        "exemple.fr est injoignable : vérifie la connexion internet",
    ]
    assert not any("http" in line for line in lines)


def test_bridge_prepares_requests_for_open_models(tmp_path, monkeypatch):
    import json

    from mirador_app import bridge
    from test_agent import make_agent

    monkeypatch.setattr(bridge, "_agent", make_agent(tmp_path))
    exact = json.loads(bridge.ask("1+1", "local", 1, "[]", False, True))
    assert exact["text"] == "1 + 1 = 2" and "llm" not in exact            # calcul : réponse exacte
    chat = json.loads(bridge.ask("yo", "local", 1, "[]", False, True))
    assert chat["llm_kind"] == "chat" and chat["llm"][-1] == {"role": "user", "content": "yo"}
    assert chat["llm"][0]["role"] == "system" and "Mirador SI" in chat["llm"][0]["content"]
    sourced = json.loads(bridge.ask("Quelle est la puissance solaire installée en France ?", "local", 1,
                                    json.dumps([["Bonjour", "Salut !"]]), True, True))
    assert sourced["llm_kind"] == "sources" and sourced["llm_sources"]
    user = sourced["llm"][-1]["content"]
    assert user.startswith("Sources :\n[") and "20 GW" in user and user.rstrip().endswith("?")
    assert [m["role"] for m in sourced["llm"]] == ["system", "user", "assistant", "user"]
    assert {s["id"] for s in sourced["llm_sources"]} <= {int(n) for n in __import__("re").findall(r"^\[(\d+)\]", user, 8)}
    free = json.loads(bridge.ask("Quelle est la puissance solaire installée en France ?", "local", 1, "[]", False, True))
    assert free["llm_kind"] == "free"                                        # hors ligne, rien trouvé
    plain = json.loads(bridge.ask("yo", "local", 1, "[]", False, False))
    assert "llm" not in plain                                                # sans modèle ouvert : inchangé
