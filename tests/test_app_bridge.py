"""Pont de l'application Android : messages d'erreur compréhensibles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "android" / "app" / "src" / "main" / "python"))

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
