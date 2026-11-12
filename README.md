# Mirador SI

> **Observer. Vérifier. Comprendre.**

Mirador SI est une IA d'analyse **construite de zéro**. Elle cherche sur internet, lit les
pages, recoupe les sources et produit des rapports d'analyse sourcés, avec un niveau de
confiance pour chaque constat.

Il n'y a ni modèle pré-entraîné, ni PyTorch, ni API d'IA externe. Tout est écrit ici, en Python
avec NumPy : le calcul des gradients, le Transformer, le tokenizer, l'optimiseur, le client web
et le moteur d'analyse.

---

## Démarrage rapide

```bash
# Python ≥ 3.11 ; seule dépendance : numpy
pip install -e .            # installe la commande `mirador`
# (ou, sans installation : python -m miradorsi <commande>)

mirador info                # identité et état du modèle
mirador train               # entraîne le réseau de neurones (~15–20 min sur un CPU 4 cœurs)
mirador chat                # conversation interactive

mirador analyze "Quel est l'impact de l'IA sur l'emploi en France ?" -d 3
mirador search "énergie solaire France 2025"
mirador read https://fr.wikipedia.org/wiki/Mirador
mirador corpus-web "intelligence artificielle" "esprit critique"   # enrichit le corpus
```

Dans `mirador chat` :

| Commande | Effet |
|---|---|
| `/analyse <question> [-p 1\|2\|3]` | analyse approfondie sur internet (rapport enregistré dans `reports/`) |
| `/cherche <requête>` | résultats de recherche bruts |
| `/lire <url>` | lire une page |
| `/identite` | présentation de Mirador SI |
| `analyse : …` | même chose que `/analyse`, en langage naturel |

---

## Architecture

```
miradorsi/
├── core/            Moteur numérique, écrit de zéro
│   ├── autograd.py    tenseurs + rétropropagation automatique (graphe de calcul)
│   ├── functional.py  softmax, GELU, RMSNorm, RoPE, entropie croisée (forward + backward)
│   ├── nn.py          Module, Linear, Embedding, RMSNorm
│   └── optim.py       AdamW, découpage du gradient, planning cosinus
├── model/           Le modèle de langage
│   ├── tokenizer.py   BPE au niveau des octets (entraînable sur ton corpus)
│   ├── transformer.py MiradorLM : Transformer décodeur, attention causale multi-têtes
│   ├── train.py       boucle d'entraînement, validation, sauvegarde
│   ├── brain.py       format de dialogue et génération
│   └── corpus.py      collecte de corpus depuis Wikipédia
├── web/             Accès à internet (bibliothèque standard uniquement)
│   ├── http.py        client : cache disque, robots.txt, politesse, gzip, reprises
│   ├── extract.py     HTML → texte propre (retire menus, pubs, scripts)
│   └── search.py      DuckDuckGo, Wikipédia, SearXNG + agrégateur + lecteur de pages
├── analysis/        Analyse approfondie
│   ├── researcher.py  boucle de recherche itérative et analyse multi-sources
│   ├── rank.py        BM25, TF-IDF, TextRank, MMR
│   ├── facts.py       extraction des chiffres, accords et divergences entre sources
│   ├── credibility.py fiabilité des sources (règles lisibles et modifiables)
│   └── report.py      rapport Markdown
├── identity.py      identité (chargée depuis config/identity.toml)
├── agent.py         l'agent : relie identité, modèle, web et analyse
└── cli.py           ligne de commande
config/
├── identity.toml    ← qui est Mirador SI (nom, ton, principes…)
└── mirador.toml     ← l'environnement (taille du modèle, web, analyse)
data/corpus/         ← textes d'entraînement
```

### Le modèle : MiradorLM

Un Transformer décodeur de type GPT, écrit sur le moteur maison :

- embeddings de tokens, **attention causale multi-têtes avec RoPE**, MLP GELU, RMSNorm, connexions résiduelles
- poids partagés entre l'embedding et la couche de sortie
- configuration par défaut : 4 couches, d=192, 4 têtes, contexte de 256 tokens, **~2 M de paramètres**
- tokenizer **BPE au niveau des octets** : il encode tout texte (accents, emojis) sans token inconnu
- chaque gradient est vérifié par différences finies dans `tests/test_autograd.py`

### L'analyse approfondie

```
question
  │ 1. planifier : la question est déclinée en angles (définition, chiffres, enjeux, actualité…)
  │ 2. chercher  : plusieurs moteurs ; résultats fusionnés, dédoublonnés, priorisés
  │ 3. lire      : pages téléchargées en parallèle, texte extrait, copies écartées
  │ 4. relancer  : notions récurrentes repérées → requêtes de suivi (profondeur 2–3)
  │ 5. analyser  : passages classés (BM25), phrases recoupées entre sources (TF-IDF),
  │                centralité (TextRank), sélection sans redondance (MMR)
  │ 6. vérifier  : chiffres extraits ; accords et divergences entre sources
  ▼ 7. rapporter : synthèse, constats [sources], chiffres clés, divergences, pistes, sources
rapport Markdown (ou JSON avec --json)
```

Garde-fous :

- les constats sont des **extraits fidèles** des sources : rien n'est inventé ;
- un chiffre différent ne compte jamais comme confirmation (20 GW ≠ 18 GW, c'est une divergence) ;
- chaque source reçoit une note de fiabilité **expliquée** ;
- les erreurs réseau ne bloquent pas l'analyse : elles sont listées dans le rapport.

| Profondeur | Requêtes | Sources max | Relances |
|---|---|---|---|
| 1 rapide | 2 | 6 | 0 |
| 2 standard | 5 + suivi | 12 | 1 |
| 3 approfondie | 8 + suivi | 20 | 2 |

---

## Personnaliser

| Je veux… | Où |
|---|---|
| changer le nom, le ton, les principes | `config/identity.toml`, puis `mirador train` |
| un modèle plus gros | `[model]` dans `config/mirador.toml` (`d_model`, `n_layers`, `context_length`) |
| lui apprendre un domaine | ajoute des `.txt` dans `data/corpus/` ou lance `mirador corpus-web …`, puis `mirador train --retrain-tokenizer` |
| mes propres moteurs de recherche | `search_providers` + `searxng_url` dans `[web]` (SearXNG est auto-hébergeable) |
| changer les sources jugées fiables | `REFERENCE_DOMAINS` / `LOW_TRUST_DOMAINS` dans `miradorsi/analysis/credibility.py` |
| des analyses plus larges | `[analysis]` dans `config/mirador.toml` |

---

## Limites actuelles, en toute franchise

- **Le réseau de neurones est petit et n'a vu que peu de texte.** Il connaît son identité et
  le style de ses dialogues, mais il ne « raisonne » pas comme un grand modèle. Pour cela il
  faudrait des milliards de tokens et des GPU. L'architecture est prête : il suffit
  d'augmenter la configuration et le corpus.
- **Les analyses ne dépendent pas du réseau de neurones.** Elles utilisent des algorithmes de
  recherche d'information (BM25, TF-IDF, TextRank) : c'est fiable, traçable et sourcé, mais
  c'est de l'extraction, pas de la rédaction libre.
- L'entraînement tourne sur CPU (NumPy). Comme NumPy et CuPy partagent la même API, un portage
  GPU est envisageable.
- Les PDF ne sont pas encore lus. DuckDuckGo peut parfois bloquer les robots ; Wikipédia et
  SearXNG prennent alors le relais.

## Feuille de route

1. Grossir le corpus (Wikipédia, documents métier) et le modèle.
2. Portage GPU (CuPy) et cache clé/valeur pour une génération plus rapide.
3. Utiliser MiradorLM pour **rédiger** la synthèse à partir des extraits sourcés.
4. Lecture des PDF, interface web, mémoire des analyses passées.

## Tests

```bash
pip install pytest
python -m pytest -q
```
