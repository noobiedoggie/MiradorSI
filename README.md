# Mirador SI

> **Observer. Vérifier. Comprendre.**

Mirador SI est une IA d'analyse **construite de zéro**. Elle cherche sur internet, lit les
pages, recoupe les sources et produit des rapports d'analyse sourcés, avec un niveau de
confiance pour chaque constat.

Il n'y a ni modèle pré-entraîné, ni PyTorch, ni API d'IA externe. Tout est écrit ici, en Python
avec NumPy : le calcul des gradients, le réseau de neurones, le tokenizer, l'optimiseur, le
client web et le moteur d'analyse.

Depuis la v0.2, le réseau de neurones utilise **Vigie**, une architecture compacte conçue pour
Mirador SI et pensée pour tourner sur un PC ordinaire, sans carte graphique. À taille égale, elle
fait **mieux que le Transformer de la v0.1** (voir le [banc d'essai](#banc-dessai)).

---

## Démarrage rapide

```bash
# Python ≥ 3.11 ; seule dépendance : numpy
pip install -e .            # installe la commande `mirador`
# (ou, sans installation : python -m miradorsi <commande>)

mirador info                # identité et état du modèle
mirador train --int8        # entraîne le réseau (~12 min sur un CPU 4 cœurs) + version compressée
mirador chat                # conversation interactive
mirador bench               # compare les architectures sur TON PC (~15 min)
mirador quantize            # compresse un modèle entraîné en int8 (~4× plus petit)

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
│   ├── functional.py  softmax, GELU, SiLU, RMSNorm, RoPE, convolution causale,
│   │                  récurrence linéaire, entropie croisée (forward + backward)
│   ├── nn.py          Module, Linear, Embedding, RMSNorm
│   └── optim.py       AdamW, découpage du gradient, planning cosinus
├── model/           Le modèle de langage
│   ├── tokenizer.py   BPE au niveau des octets (entraînable sur ton corpus)
│   ├── vigie.py       Vigie : architecture compacte (v0.2, par défaut) + moteur d'inférence PC
│   ├── transformer.py MiradorLM : Transformer décodeur classique (v0.1)
│   ├── streaming.py   génération token par token (cache, mémoire bornée)
│   ├── checkpoint.py  sauvegarde, chargement, compression int8
│   ├── bench.py       banc d'essai reproductible entre architectures
│   ├── train.py       boucle d'entraînement, validation sur textes inédits
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

### Le modèle : Vigie

```
token ─► embedding factorisé (vocabulaire → 128 → 288)
           │
           │   ┌──────────── 2 blocs UNIQUES, réutilisés 2 fois (profondeur recyclée) ───────────┐
           ├──►│ R  récurrence à portes + convolution courte   │  mémoire continue, coût constant │
           │   │    └► MLP SwiGLU                               │                                  │
           ├──►│ A  attention locale (64 tokens), 1 clé/valeur  │  précision sur le contexte proche│
           │   │    └► MLP SwiGLU                               │                                  │
           │   └─────────── normalisations propres à chaque passage ─────────────────────────────┘
           ▼
         RMSNorm ─► projection ─► vocabulaire (poids liés à l'embedding)
```

Cinq idées, chacune connue séparément. Leur **assemblage** est propre à Mirador SI :

1. **Profondeur recyclée.** Deux blocs uniques sont appliqués deux fois de suite, comme une
   relecture : la profondeur de 4 couches avec les paramètres de 2. Chaque passage garde ses
   propres normalisations pour que le même bloc puisse jouer un rôle différent à chaque tour.
2. **Mélangeur hybride.** Un bloc *récurrent* (une mémoire qui se met à jour à chaque token) alterne
   avec un bloc d'*attention locale* (précise sur les 64 derniers tokens). La récurrence porte
   l'information lointaine, l'attention retrouve un nom ou un chiffre proche.
3. **Mémoire multi-échelles.** Chaque canal de la récurrence démarre avec une durée de mémoire
   différente, de 2 à 512 tokens : du mot au paragraphe.
4. **Attention multi-requêtes.** Toutes les têtes partagent une seule paire clé/valeur, ce qui
   rend le cache 4× plus petit.
5. **MLP SwiGLU et embeddings factorisés.** Plus d'expressivité par paramètre.

**Pourquoi c'est fait pour un PC :** pendant la génération, la mémoire et le temps par token sont
**constants**, quelle que soit la longueur du texte (état récurrent + fenêtre bornée). Le moteur
d'inférence fusionne les matrices et n'alloue rien par token. Le modèle compressé en int8 pèse
**~2 Mo**.

Garanties vérifiées par les tests :
- chaque gradient est vérifié numériquement, opération par opération **et sur le modèle entier** ;
- la génération token par token donne **exactement** les mêmes résultats que le calcul
  d'entraînement ;
- la mémoire d'inférence ne grandit pas avec la longueur du texte.

Le Transformer de la v0.1 reste disponible : `arch = "transformer"` dans `config/mirador.toml`
ou `mirador train --arch transformer`.

### Banc d'essai

Les architectures sont comparées sur les mêmes données, avec les mêmes réglages :
- **Données :** 2,9 M de caractères de documentation Python, disponibles hors ligne sur tout PC.
- **Entraînement :** 400 étapes × 2 048 tokens, avec les mêmes lots pour tous les modèles.
- **Validation :** sur **des documents jamais vus**.
- **Matériel :** CPU 4 cœurs, NumPy. Une seule graine aléatoire par modèle.

Reproductible avec `mirador bench`. La génération est mesurée un token à la fois, sur 2 048 tokens.
Les vitesses varient de ±15 % selon la charge de la machine, mais leur ordre reste stable.

| Modèle | Paramètres | Perte val. ↓ | Bits/caractère ↓ | Entraînement | Génération | Mémoire d'inférence | Fichier int8 |
|---|---|---|---|---|---|---|---|
| Transformer v0.1 (4 couches, d=192) | 1,97 M | 3,380 | 2,127 | 2 312 tok/s | ~600 tok/s | 1 536 Ko, **256 tokens max** | 1,8 Mo |
| **Vigie** (RA×2, d=288) — *défaut* | 2,08 M | **3,055** | **1,923** | 3 434 tok/s | ~355 tok/s | **81 Ko, illimité** | 1,9 Mo |
| Vigie sans recyclage (RARA×1, d=192) | 1,84 M | 3,175 | 1,998 | 5 227 tok/s | ~800 tok/s | 55 Ko, illimité | 1,7 Mo |
| **Vigie compacte** (RA×2, d=192) | **0,97 M** | 3,259 | 2,051 | **5 923 tok/s** | **~780 tok/s** | **55 Ko, illimité** | **0,9 Mo** |

Ce qu'on en retient :
- À taille égale, **Vigie** obtient −9,6 % de bits par caractère et s'entraîne 1,5× plus vite.
- La **Vigie compacte**, avec **moitié moins de paramètres**, fait encore mieux que le Transformer,
  génère ~1,3× plus vite et utilise 28× moins de mémoire.
- Le **recyclage de profondeur** compte : sans lui, à taille comparable, la perte passe de 3,055 à 3,175.
- La **compression int8** ne dégrade pas la qualité de façon mesurable (écart < 0,001), pour un
  fichier 4× plus petit.
- **Le prix à payer :** la Vigie par défaut relit ses blocs deux fois, donc calcule deux fois plus
  par token et génère moins vite que le Transformer (~355 contre ~600 tokens/s). Cela reste
  instantané pour discuter (une réponse de 160 tokens en moins d'une demi-seconde). Sur un PC
  modeste, choisis la Vigie compacte : `d_model = 192`, `d_embed = 96`.

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
| un modèle plus gros ou plus léger | `[model]` dans `config/mirador.toml` (`d_model`, `n_loops`, `pattern`, `window`) |
| la Vigie compacte (PC modeste) | `d_model = 192` et `d_embed = 96` dans `[model]` |
| comparer des variantes sur ma machine | `mirador bench` (ou `run_benchmark()` dans `miradorsi/model/bench.py`) |
| lui apprendre un domaine | ajoute des `.txt` dans `data/corpus/` ou lance `mirador corpus-web …`, puis `mirador train --retrain-tokenizer` |
| mes propres moteurs de recherche | `search_providers` + `searxng_url` dans `[web]` (SearXNG est auto-hébergeable) |
| changer les sources jugées fiables | `REFERENCE_DOMAINS` / `LOW_TRUST_DOMAINS` dans `miradorsi/analysis/credibility.py` |
| des analyses plus larges | `[analysis]` dans `config/mirador.toml` |

---

## Limites actuelles, en toute franchise

- **Le réseau de neurones est petit et n'a vu que peu de texte (~7 000 tokens).** Sur un texte
  qu'il n'a jamais vu, il fait aujourd'hui moins bien que le hasard : il a *mémorisé* son corpus.
  `mirador train` l'affiche désormais clairement. Il connaît son identité et
  le style de ses dialogues, mais il ne « raisonne » pas comme un grand modèle. Une meilleure
  architecture améliore ce qu'il apprend de chaque token, mais **c'est la quantité de texte qui
  le rendra savant** : ajoute des documents avec `mirador corpus-web` ou dans `data/corpus/`.
- Le banc d'essai porte sur de petits modèles (~1–2 M de paramètres) et de la prose technique en
  anglais. Les écarts sont nets, mais ils restent à confirmer à plus grande échelle.
- **Les analyses ne dépendent pas du réseau de neurones.** Elles utilisent des algorithmes de
  recherche d'information (BM25, TF-IDF, TextRank) : c'est fiable, traçable et sourcé, mais
  c'est de l'extraction, pas de la rédaction libre.
- L'entraînement tourne sur CPU (NumPy). Comme NumPy et CuPy partagent la même API, un portage
  GPU est envisageable.
- Les PDF ne sont pas encore lus. DuckDuckGo peut parfois bloquer les robots ; Wikipédia et
  SearXNG prennent alors le relais.

## Feuille de route

1. Grossir le corpus (Wikipédia, documents métier) et le modèle.
2. Portage GPU (CuPy) pour entraîner plus gros, plus vite.
3. Utiliser le réseau de neurones pour **rédiger** la synthèse à partir des extraits sourcés.
4. Lecture des PDF, interface web, mémoire des analyses passées.

## Tests

```bash
pip install pytest
python -m pytest -q
```
