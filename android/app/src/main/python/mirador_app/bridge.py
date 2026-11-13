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
            _agent = MiradorAgent.create(cfg, load_brain=True)
        return json.dumps(_info(_agent), ensure_ascii=False)


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
        if mode == "analyse":
            return json.dumps(_analyse(message, int(depth)), ensure_ascii=False)
        _agent.history = [tuple(pair) for pair in json.loads(history_json)][-4:]
        return json.dumps({"text": _app_wording(_agent.handle(message)), "sources": []}, ensure_ascii=False)
    except Exception as exc:  # l'application ne doit jamais planter sur une erreur Python
        return json.dumps({"text": f"Désolé, une erreur est survenue : {exc}", "sources": []},
                          ensure_ascii=False)


def _app_wording(text: str) -> str:
    """Les réponses parlent des commandes de la console (/analyse) : dans l'appli, c'est la loupe."""
    return (text.replace("Utilise la commande /analyse suivie de ta question.",
                         "Passe en mode Analyse (la loupe) et pose ta question.")
                .replace("Pour une réponse sourcée : /analyse …", "Pour une réponse sourcée : mode Analyse (la loupe)"))


def _analyse(question: str, depth: int) -> dict:
    from miradorsi.analysis.answer import compose_answer

    report = _agent.analyzer.analyze(question, depth)
    if not report.findings:
        incidents = "\n".join(f"• {e}" for e in explain_errors(report.errors))
        text = ("Je n'ai pas pu collecter d'information exploitable. Vérifie la connexion internet "
                "ou reformule la question.")
        return {"text": text + (f"\n\n{incidents}" if incidents else ""), "sources": []}

    text = compose_answer(report)
    sources = [{"id": s.id, "title": s.title, "url": s.url, "domain": s.domain,
                "reliability": s.credibility.label} for s in report.sources]
    return {"text": text, "sources": sources}


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
