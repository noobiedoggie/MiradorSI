"""Pont entre l'application Android (Kotlin) et Mirador SI (Python).

L'application appelle `start()` une fois, puis `ask()` pour chaque message.
Les réponses sont renvoyées en JSON pour rester simples à lire côté Kotlin.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_agent = None
_lock = threading.Lock()
LIBRARY_FILE = os.path.join("bibliotheque", "bibliotheque-fr.db")  # dans le stockage de l'application


def _configure_certificates() -> None:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def start(files_dir: str) -> str:
    """Charge l'identité, le modèle local et les outils internet. Renvoie la fiche d'identité."""
    global _agent
    with _lock:
        if _agent is None:
            _configure_certificates()
            from miradorsi.agent import MiradorAgent
            from miradorsi.config import load_config

            cfg = load_config(_HERE / "config" / "mirador.toml")
            cfg.web.cache_dir = os.path.join(files_dir, "cache")
            cfg.paths.reports = os.path.join(files_dir, "rapports")
            cfg.paths.library = os.path.join(files_dir, LIBRARY_FILE)
            _agent = MiradorAgent.create(cfg, load_brain=True)
        return json.dumps(_info(_agent), ensure_ascii=False)


def library_info() -> str:
    """État de la bibliothèque hors ligne (articles de Wikipédia sur le téléphone)."""
    library = _agent.library if _agent is not None else None
    return json.dumps({"installed": library is not None,
                       "description": library.describe() if library is not None else ""}, ensure_ascii=False)


def library_load(path: str) -> str:
    """Branche la bibliothèque téléchargée ; une erreur si le fichier n'est pas une bibliothèque."""
    from miradorsi.library import Library

    library = Library(path)  # lève une exception si le fichier est invalide
    with _lock:
        _agent.use_library(library)
    return library_info()


def _info(agent) -> dict:
    identity = agent.identity
    info = {
        "name": identity.name, "version": identity.version, "tagline": identity.tagline,
        "description": identity.description, "principles": identity.rules,
        "has_model": agent.brain is not None, "model": "",
    }
    if agent.brain is not None:
        model = agent.brain.model
        params = model.num_parameters()
        arch = getattr(model, "arch", "transformer")
        label = f"{arch.capitalize()} · {params / 1e6:.1f} M de paramètres".replace(".", ",")
        if agent.brain.meta.get("quantized"):
            label += " · compressé int8"
        info["model"] = label
    return info


def ask(message: str, mode: str, depth: int = 2, history_json: str = "[]") -> str:
    """mode = "analyse" (recherche sur internet) ou "local" (modèle embarqué, hors ligne)."""
    if _agent is None:
        return json.dumps({"text": "Mirador SI démarre encore, réessaie dans un instant.", "sources": []})
    try:
        # calculs, salutations, politesse, questions sur Mirador : réponse directe, sans internet
        quick = _agent.quick_reply(message)
        if quick:
            return json.dumps({"text": _app_wording(quick), "sources": []}, ensure_ascii=False)
        _agent.history = [tuple(pair) for pair in json.loads(history_json)][-4:]
        if mode == "analyse":
            from miradorsi.reasoning import resolve_follow_up

            question = resolve_follow_up(message, _agent.history)  # « Et sa capitale ? »
            reasoned = _agent.reason(question, online=True)        # âges, comparaisons, chaînes
            if reasoned is not None:
                text, report = reasoned
                return json.dumps({"text": text, "sources": _sources(report, text)}, ensure_ascii=False)
            return json.dumps(_analyse(question, int(depth)), ensure_ascii=False)
        # hors ligne : outils, mémoire d'identité, raisonnement, puis bibliothèque (Wikipédia)
        text, report = _agent.offline_reply(message)
        return json.dumps({"text": _app_wording(text), "sources": _sources(report, text)}, ensure_ascii=False)
    except Exception as exc:  # l'application ne doit jamais planter sur une erreur Python
        return json.dumps({"text": f"Désolé, une erreur est survenue : {exc}", "sources": []},
                          ensure_ascii=False)


def _app_wording(text: str) -> str:
    """Les réponses parlent des commandes de la console (/analyse) : dans l'appli, c'est la loupe."""
    return (text.replace("Utilise la commande /analyse suivie de ta question.",
                         "Passe en mode Analyse (la loupe) et pose ta question.")
                .replace("Pour une réponse sourcée : /analyse …", "Pour une réponse sourcée : mode Analyse (la loupe)")
                .replace("télécharge la bibliothèque hors ligne (sur l'accueil)",
                         "télécharge la bibliothèque hors ligne (bandeau de l'accueil)"))


def _analyse(question: str, depth: int) -> dict:
    from miradorsi.analysis.answer import compose_answer

    report = _agent.analyzer.analyze(question, depth)
    if not report.findings:
        offline = _agent.library_answer(question)
        if offline is not None:  # pas d'internet (ou rien trouvé) : la bibliothèque prend le relais
            text = ("Je n'ai pas pu chercher sur internet ; voici ce que dit ma bibliothèque hors ligne.\n\n"
                    + compose_answer(offline, offline=True))
            return {"text": text, "sources": _sources(offline, text)}
        incidents = "\n".join(f"• {e}" for e in explain_errors(report.errors))
        text = ("Je n'ai pas pu collecter d'information exploitable. Vérifie la connexion internet "
                "ou reformule la question.")
        return {"text": text + (f"\n\n{incidents}" if incidents else ""), "sources": []}

    text = compose_answer(report)
    return {"text": text, "sources": _sources(report, text)}


def _sources(report, text: str) -> list[dict]:
    """Les sources citées dans la réponse ([1], [3]…), dans l'ordre de leurs numéros."""
    if report is None:
        return []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return [{"id": s.id, "title": s.title, "url": s.url, "domain": s.domain,
             "reliability": s.credibility.label} for s in report.sources if s.id in cited]


def explain_errors(errors: list[str], limit: int = 3) -> list[str]:
    """Traduit les erreurs techniques (URL, codes HTTP) en phrases courtes, sans doublons."""
    out: list[str] = []
    for error in errors:
        host = re.search(r"https?://(?:www\.)?([^/\s:]+)", error)
        site = host.group(1) if host else error.split(" :", 1)[0].strip()
        site = "Wikipédia" if site.endswith("wikipedia.org") else site
        if "anti-robot" in error:
            text = "DuckDuckGo bloque les recherches automatiques pour le moment"
        elif "HTTP 429" in error:
            text = f"{site} reçoit trop de requêtes depuis ton réseau : réessaie dans une minute"
        elif re.search(r"HTTP 40[13]", error):
            text = f"{site} refuse l'accès"
        elif re.search(r"HTTP 5\d\d", error):
            text = f"{site} est en panne pour le moment"
        elif re.search(r"timed out|timeout|Network is unreachable|Name or service|"
                       r"No address|getaddrinfo|Temporary failure|Connection (refused|reset)", error, re.I):
            text = f"{site} est injoignable : vérifie la connexion internet"
        elif "robots.txt" in error:
            text = f"{site} interdit la lecture automatique"
        else:
            text = re.sub(r"https?://\S+", site, error)[:160]
        if text not in out:
            out.append(text)
    return out[:limit]
