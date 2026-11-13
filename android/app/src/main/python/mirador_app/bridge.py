"""Pont entre l'application Android (Kotlin) et Mirador SI (Python).

L'application appelle `start()` une fois, puis `ask()` pour chaque message.
Les réponses sont renvoyées en JSON pour rester simples à lire côté Kotlin.
"""

from __future__ import annotations

import json
import os
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
        if mode == "analyse":
            return json.dumps(_analyse(message, int(depth)), ensure_ascii=False)
        _agent.history = [tuple(pair) for pair in json.loads(history_json)][-4:]
        return json.dumps({"text": _agent.handle(message), "sources": []}, ensure_ascii=False)
    except Exception as exc:  # l'application ne doit jamais planter sur une erreur Python
        return json.dumps({"text": f"Désolé, une erreur est survenue : {exc}", "sources": []},
                          ensure_ascii=False)


def _analyse(question: str, depth: int) -> dict:
    report = _agent.analyzer.analyze(question, depth)
    if not report.findings:
        incidents = "\n".join(f"• {e}" for e in report.errors[:3])
        text = ("Je n'ai pas pu collecter d'information exploitable. Vérifie la connexion internet "
                "ou reformule la question.")
        return {"text": text + (f"\n\n{incidents}" if incidents else ""), "sources": []}

    def cite(ids):
        return "".join(f"[{i}]" for i in ids)

    best = sorted(report.findings, key=lambda f: -f.confidence)[:3]
    lines = ["Synthèse", " ".join(f"{f.text} {cite(f.citations)}" for f in best), "", "Constats"]
    for i, f in enumerate(report.findings, 1):
        lines.append(f"{i}. {f.text} {cite(f.citations)} — confiance {f.confidence_label}")
    if report.corroborated:
        lines += ["", "Chiffres confirmés par plusieurs sites"]
        for g in report.corroborated[:4]:
            values = " / ".join(sorted({f.display() for f in g.facts}))
            lines.append(f"• {values} ({', '.join(g.context[:4])}) {cite(g.source_ids)}")
    if report.divergent:
        lines += ["", f"⚠ {len(report.divergent)} divergence(s) de chiffres entre sources : à vérifier."]
    lines += ["", f"{len(report.sources)} sources lues · {len(report.queries)} recherches · "
                  f"{report.duration_seconds:.0f} s"]
    sources = [{"id": s.id, "title": s.title, "url": s.url, "domain": s.domain,
                "reliability": s.credibility.label} for s in report.sources]
    return {"text": "\n".join(lines), "sources": sources}
