"""Ligne de commande : `mirador <commande>` (ou `python -m miradorsi <commande>`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .identity import Identity


def _progress(message: str) -> None:
    print(f"  {message}", file=sys.stderr, flush=True)


def cmd_info(cfg, args) -> int:
    from .model.brain import Brain

    identity = Identity.load(cfg.path("identity"))
    print(identity.banner())
    print(identity.presentation())
    print()
    brain = Brain.load(cfg.path("checkpoints"), cfg.generation)
    if brain:
        m = brain.model
        params = f"{m.num_parameters():,}".replace(",", "\u202f")
        print(f"Modèle      : {params} paramètres")
        if getattr(m, "arch", "transformer") == "vigie":
            print(f"Architecture: Vigie · blocs {m.cfg.pattern} × {m.cfg.n_loops} passages · d={m.cfg.d_model} · "
                  f"attention locale {m.cfg.window} tokens · vocabulaire {m.cfg.vocab_size}")
        else:
            print(f"Architecture: Transformer · {m.cfg.n_layers} couches · d={m.cfg.d_model} · "
                  f"{m.cfg.n_heads} têtes · contexte {m.cfg.context_length} · vocabulaire {m.cfg.vocab_size}")
        if brain.meta.get("quantized"):
            print("Format      : compressé int8")
        if "val_loss" in brain.meta:
            print(f"Entraînement: étape {brain.meta.get('step')} · perte de validation "
                  f"{brain.meta['val_loss']:.3f}")
    else:
        print("Modèle      : pas encore entraîné → `mirador train`")
    print(f"Recherche   : {', '.join(cfg.web.search_providers)}")
    print(f"Analyse     : profondeur par défaut {cfg.analysis.default_depth}")
    print(f"Config      : {cfg.root / 'config'}")
    return 0


def cmd_train(cfg, args) -> int:
    from .model.train import train

    if args.steps:
        cfg.training.steps = args.steps
    if args.arch:
        cfg.model.arch = args.arch
    train(cfg, steps=args.steps, fresh=args.fresh, retrain_tokenizer=args.retrain_tokenizer, int8=args.int8)
    return 0


def cmd_quantize(cfg, args) -> int:
    from .model.checkpoint import load_model, save_model

    source = cfg.path("checkpoints") / "model.npz"
    if not source.exists():
        print("Aucun modèle à compresser : lance d'abord `mirador train`.")
        return 2
    model, extra = load_model(source)
    target = save_model(model, cfg.path("checkpoints") / "model-int8.npz", extra=extra, quantize=True)
    before, after = source.stat().st_size, target.stat().st_size
    print(f"{source.name} ({before / 1e6:.1f} Mo) → {target.name} ({after / 1e6:.1f} Mo), "
          f"{before / after:.1f}× plus petit")
    return 0


def cmd_bench(cfg, args) -> int:
    import json

    from .model.bench import format_results, run_benchmark

    results = run_benchmark(steps=args.steps, max_chars=args.chars)
    table = format_results(results)
    print("\n" + table)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def cmd_analyze(cfg, args) -> int:
    from .agent import MiradorAgent

    agent = MiradorAgent.create(cfg, on_progress=None if args.quiet else _progress, load_brain=False)
    report = agent.analyzer.analyze(args.question, args.depth)
    output = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Rapport écrit : {args.output}", file=sys.stderr)
    else:
        saved = agent.save_report(report)
        print(output)
        print(f"\n(Rapport enregistré : {saved})", file=sys.stderr)
    return 0 if report.findings else 2


def cmd_search(cfg, args) -> int:
    from .agent import MiradorAgent

    agent = MiradorAgent.create(cfg, load_brain=False)
    print(agent.search_web(args.query, args.n))
    return 0


def cmd_read(cfg, args) -> int:
    from .agent import MiradorAgent

    agent = MiradorAgent.create(cfg, load_brain=False)
    print(agent.read(args.url, args.chars))
    return 0


def cmd_ask(cfg, args) -> int:
    from .agent import MiradorAgent

    agent = MiradorAgent.create(cfg, on_progress=_progress)
    print(agent.handle(args.message))
    return 0


def cmd_chat(cfg, args) -> int:
    from .agent import MiradorAgent

    agent = MiradorAgent.create(cfg, on_progress=_progress)
    print(agent.identity.banner())
    if agent.brain is None:
        print("(Réseau de neurones non entraîné : `mirador train` pour lui apprendre à parler.)")
    print("Tape /aide pour la liste des commandes.\n")
    name = agent.identity.short_name
    while True:
        try:
            message = input("toi › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("/quitter", "/quit", "/exit", "quitter", "exit"):
            break
        if message:
            print(f"{name} › {agent.handle(message)}\n")
    print("À bientôt.")
    return 0


def cmd_corpus_web(cfg, args) -> int:
    from .model.corpus import build_web_corpus
    from .web.http import WebClient
    from .web.search import Fetcher, WikipediaSearch

    client = WebClient(cfg.web)
    out_dir = cfg.path("corpus") / "web"
    print(f"Collecte de Wikipédia ({args.lang}) → {out_dir}")
    topics = list(args.topics)
    if args.topics_file:
        topics += [l.strip() for l in Path(args.topics_file).read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.lstrip().startswith("#")]
    if args.exact:
        from .model.corpus import build_wikipedia_bulk
        files = build_wikipedia_bulk(topics, client, out_dir, args.lang, args.max_chars)
    else:
        files = build_web_corpus(topics, WikipediaSearch(client, [args.lang]), Fetcher(client),
                                 out_dir, args.lang, args.per_topic, args.max_chars)
    print(f"{len(files)} document(s) ajouté(s). Relance `mirador train --retrain-tokenizer` pour apprendre.")
    cmd_lexicon(cfg, args)
    return 0 if files else 2


def cmd_lexicon(cfg, args) -> int:
    from .analysis.lexicon import build_idf, save_lexicon
    from .model.train import load_corpus_documents

    docs = load_corpus_documents(cfg.path("corpus"))
    if not docs:
        print("Corpus vide : rien à analyser.")
        return 2
    idf = build_idf(docs)
    path = save_lexicon(idf, cfg.path("lexicon"), source=f"{len(docs)} documents de {cfg.paths.corpus}")
    print(f"Lexique : {len(idf)} mots, rareté calculée sur {len(docs)} documents → {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirador", description="Mirador SI — IA d'analyse créée de zéro")
    parser.add_argument("--config", help="chemin vers mirador.toml (défaut : config/mirador.toml)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="identité et état du modèle").set_defaults(func=cmd_info)

    p = sub.add_parser("train", help="entraîner le réseau de neurones")
    p.add_argument("--steps", type=int, help="nombre d'étapes (défaut : config)")
    p.add_argument("--fresh", action="store_true", help="repartir de zéro")
    p.add_argument("--retrain-tokenizer", action="store_true", help="réapprendre le vocabulaire")
    p.add_argument("--arch", choices=["vigie", "transformer"], help="architecture (défaut : config)")
    p.add_argument("--int8", action="store_true", help="exporter aussi une version compressée int8")
    p.set_defaults(func=cmd_train)

    sub.add_parser("quantize", help="compresser le modèle entraîné en int8 (~4× plus petit)"
                   ).set_defaults(func=cmd_quantize)

    p = sub.add_parser("bench", help="comparer les architectures sur ce PC")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--chars", type=int, default=3_000_000, help="taille du corpus de test")
    p.add_argument("-o", "--output", help="résultats JSON")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("analyze", aliases=["analyse"], help="analyse approfondie sur internet")
    p.add_argument("question")
    p.add_argument("-d", "--depth", type=int, choices=[1, 2, 3], help="1 rapide · 2 standard · 3 approfondie")
    p.add_argument("-o", "--output", help="fichier de sortie")
    p.add_argument("--json", action="store_true", help="sortie JSON")
    p.add_argument("-q", "--quiet", action="store_true", help="sans suivi de progression")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("search", aliases=["cherche"], help="recherche web")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("read", aliases=["lire"], help="lire une page web")
    p.add_argument("url")
    p.add_argument("--chars", type=int, default=3000)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("ask", help="poser une question (une seule fois)")
    p.add_argument("message")
    p.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="conversation interactive").set_defaults(func=cmd_chat)

    p = sub.add_parser("corpus-web", help="enrichir le corpus d'entraînement depuis Wikipédia")
    p.add_argument("topics", nargs="*")
    p.add_argument("--topics-file", help="fichier texte : un sujet par ligne")
    p.add_argument("--lang", default="fr")
    p.add_argument("--per-topic", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=60_000, help="taille max par article")
    p.add_argument("--exact", action="store_true",
                   help="les sujets sont des titres exacts : collecte groupée (10 articles par requête)")
    p.set_defaults(func=cmd_corpus_web)

    sub.add_parser("lexique", help="recalculer la rareté des mots à partir du corpus"
                   ).set_defaults(func=cmd_lexicon)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if not getattr(args, "func", None):
        args.func = cmd_chat
    return args.func(cfg, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
