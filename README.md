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
mirador corpus-web --topics-file data/sujets_fr.txt            # ~150 articles (recherche, ~25 min)
mirador corpus-web --exact --topics-file data/sujets_fr_2.txt  # titres exacts : 10 articles par requête
mirador corpus-web --exact --topics-file data/sujets_fr_3.txt  # (~1 min chacun)
mirador train --int8        # entraîne le réseau (~1 h 30 sur un CPU 4 cœurs) + version compressée
mirador chat                # conversation interactive
mirador bench               # compare les architectures sur TON PC (~15 min)
mirador quantize            # compresse un modèle entraîné en int8 (~4× plus petit)

mirador analyze "Quel est l'impact de l'IA sur l'emploi en France ?" -d 3
mirador search "énergie solaire France 2025"
mirador read https://fr.wikipedia.org/wiki/Mirador
mirador corpus-web "intelligence artificielle" "esprit critique"   # ajoute tes propres sujets
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

## Application Android

L'APK est compilé automatiquement par GitHub Actions (`.github/workflows/android.yml`) à chaque
modification de `android/`, `miradorsi/` ou du modèle embarqué.

**Pour l'installer :**
1. Sur GitHub, ouvre l'onglet **Actions**, puis le dernier passage réussi de « APK Android ».
2. Télécharge l'artefact **MiradorSI-apk**, qui contient `MiradorSI.apk`.
3. Sur le téléphone, ouvre le fichier et autorise l'installation d'applications de cette source.

**Ce que fait l'application :**
- **Analyse sur internet** (icône loupe) : la même analyse approfondie que sur PC, qui recherche,
  lit, recoupe puis cite ses sources. Les sources sont cliquables.
- **Modèle local** (icône ordinateur) : discussion avec le réseau de neurones Vigie embarqué,
  **hors ligne**.
- **Dictée vocale**, historique des conversations sur le téléphone, choix de la profondeur
  d'analyse (bouton +).

**Comment c'est fait :**
- Interface en Kotlin et Jetpack Compose.
- Le code Python de Mirador SI et numpy sont embarqués tels quels grâce à
  [Chaquopy](https://chaquo.com/chaquopy/) : c'est le même moteur que sur PC.
- Le modèle compressé int8 est dans `android/app/src/main/python/mirador_app/modele/`. Pour
  changer de modèle, remplace `model-int8.npz` et `tokenizer.json` par ceux de `checkpoints/`.
- L'APK est un APK de débogage, installable directement. Pour le Play Store, il faudrait une clé
  de signature (secret GitHub) et une compilation `assembleRelease`.

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

### v0.4 : 2,5 fois plus grand, corpus quadruplé

| | v0.3 | **v0.4** |
|---|---|---|
| Paramètres | 4,3 M (d=384) | **10,7 M** (d=640, attention locale de 128 tokens) |
| Corpus d'entraînement | 474 articles, 5,2 M tokens | **2 825 articles, 21,4 M tokens** (75 M caractères) |
| Entraînement | 4 000 étapes (16 M tokens vus) | **9 000 étapes** (37 M tokens vus), ~11 h sur CPU |
| **Bits par caractère** sur 196 articles inédits pour les deux ↓ | 1,662 | **1,448 (−13 %)** |
| Fichier int8 (même score que la version complète) | 4,0 Mo | 9,9 Mo |

L'entraînement a survécu à plusieurs redémarrages de la machine grâce aux points de reprise
(`checkpoints/reprise.npz`, toutes les 50 étapes). C'est ce modèle qui est embarqué dans
l'application Android. Il écrit un français plus juste, mais à cette taille **les faits qu'il
invente restent faux** : les réponses de Mirador SI viennent des sources (internet ou bibliothèque
hors ligne), pas de sa mémoire.

### v0.3 : un modèle plus puissant, toujours compact

| | v0.2 | **v0.3** |
|---|---|---|
| Paramètres | 2,5 M (d=288) | **4,3 M** (d=384, attention locale de 128 tokens) |
| Corpus d'entraînement | 153 articles, 1,6 M tokens | **474 articles, 5,2 M tokens** (19 M caractères) |
| Entraînement | 1 500 étapes (6 M tokens vus) | **4 000 étapes** (16 M tokens vus), ~1 h 30 sur CPU |
| **Bits par caractère** sur 24 articles inédits pour les deux ↓ | 1,703 | **1,483 (−13 %)** |
| Fichier int8 (même score que la version complète) | 2,4 Mo | 4,0 Mo |
| Génération sur CPU, un token à la fois | ~250 tokens/s | ~150 tokens/s |

Ce qui a rendu cela possible :
- **Un moteur 37 % plus rapide**, à résultats identiques (gradients vérifiés) :
  - un seul grand produit matriciel par couche au lieu d'un par lot ;
  - plus de copies inutiles de gradients ;
  - une GELU sans puissance générique ;
  - une exponentielle de la softmax réutilisée.
- **La collecte groupée** (`corpus-web --exact`) : 10 articles Wikipédia par requête, convertis du
  wikitexte en texte propre. 290 articles en une minute au lieu de deux heures, avec bien moins de
  requêtes vers Wikipédia.
- **Une pénalité de répétition** à la génération (`repetition_penalty`), qui supprime les
  boucles du type « en France, en France, en France… ».

En pratique, la v0.3 reste dans le sujet de son amorce (la Lune → planètes, Soleil), là où la v0.2
partait n'importe où. Mais **les faits qu'elle écrit restent inventés** : à 4 M de paramètres,
elle a appris la forme du français encyclopédique, pas encore le savoir.

### Vigie-500M : la configuration existe, l'entraînement demande des GPU

`config/mirador-500m.toml` décrit une Vigie de **496,7 M de paramètres** : d=1280, 28 blocs,
attention à requêtes groupées, contexte de 512 tokens. Elle se construit et apprend : sa perte
passe de 8,41 à 7,71 en trois étapes. Voici ce qu'elle coûte, mesuré sur un CPU 4 cœurs :

| | v0.3 (4,3 M) | Vigie-500M |
|---|---|---|
| Entraînement | ~2 600 tokens/s | **~50 tokens/s** |
| Un passage sur le corpus actuel (5,2 M tokens) | ~35 min | ~29 h |
| Entraînement complet (~20 tokens par paramètre ≈ 10 milliards de tokens) | — | ~6 ans sur ce CPU |
| Mémoire pour l'entraîner | < 1 Go | 10,6 Go |
| Génération | ~150 tokens/s | 1,7 token/s |
| Fichier (int8) | 4,0 Mo | ~0,5 Go |

À temps de calcul égal sur CPU, la 500M lit environ 60 fois moins de texte que la v0.3, et
finit **moins bonne**. La taille ne paie que si le calcul et les données suivent : pour cette
configuration, il faut un moteur sur carte graphique et une dizaine de milliards de tokens.

Pour l'essayer quand même (les points de reprise vont dans `checkpoints-500m/`) :
`mirador --config config/mirador-500m.toml train`

### Historique : premier entraînement sur Wikipédia (v0.2)

`mirador corpus-web --topics-file data/sujets_fr.txt` collecte des articles de Wikipédia en
français sur 81 sujets de culture générale. Résultats obtenus avec Vigie (2,5 M de paramètres,
1 500 étapes, ~35 min sur CPU) :

| | Avant (corpus local) | Après (corpus Wikipédia) |
|---|---|---|
| Texte d'entraînement | 7 000 tokens | **1,6 M tokens** (153 articles, 6,1 M caractères) |
| Perte sur 13 articles **jamais vus** | 10,9 (pire que le hasard : 6,9) | **3,88** (hasard : 8,32) |
| Perplexité · bits par caractère | — | 48 · 1,66 |
| Réponses sur son identité | correctes | correctes (les dialogues gardent 3 séquences sur 16 par lot) |
| Fichier int8 | 2,0 Mo | 2,4 Mo, **même score** que la version complète |

Le modèle a donc vraiment appris le français au lieu de réciter. Son texte libre a des mots, des
tournures et une mise en page de type encyclopédie, mais **pas encore de sens suivi** : à cette
taille, c'est normal. La perte baissait encore en fin d'entraînement, donc plus d'étapes et plus
d'articles l'amélioreront.

Notes :
- Les articles collectés restent sur ta machine (`data/corpus/web/`, ignoré par git). Le texte de
  Wikipédia est sous licence CC BY-SA.
- `data/lexique_fr.json` ne contient que la rareté de chaque mot (aucune phrase). Il sert à
  l'analyse et se recalcule avec `mirador lexique`.

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
- deux pages d'**un même site** ne se confirment pas l'une l'autre : seule une source *indépendante*
  corrobore (Wikipédia recopie souvent des sections d'un article à l'autre) ;
- les **noms propres** de la question sont des contraintes : à « … en France ? », une phrase sur
  l'Allemagne est écartée, même si elle est très proche par ailleurs ;
- les mots **rares** de la question pèsent plus lourd (« solaire » plus que « production ») grâce
  au lexique de fond (`data/lexique_fr.json`), calculé sur le corpus ;
- chaque source reçoit une note de fiabilité **expliquée** ;
- les erreurs réseau ne bloquent pas l'analyse : elles sont listées dans le rapport ;
- quand un site répond « trop de requêtes » (HTTP 429), Mirador attend le délai demandé
  (`Retry-After`) puis ralentit durablement pour ce site.

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

- **Le réseau de neurones reste petit (10,7 M de paramètres) et a lu 21 M de tokens.** Il connaît
  son identité et écrit un français plausible mot à mot, mais il ne « raisonne » pas et ne sait
  pas répondre à une question de culture générale. Un grand modèle a des milliards de paramètres
  et lit des milliers de milliards de tokens. Pour progresser : plus d'articles
  (`mirador corpus-web`), plus d'étapes, un modèle plus large. Si le corpus est trop petit et que
  le modèle se met à mémoriser, `mirador train` le signale.
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
