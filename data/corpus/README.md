# Corpus d'entraînement

Tous les fichiers `.txt` et `.md` de ce dossier (sauf ce README) servent à entraîner
le réseau de neurones de Mirador SI.

- Ajoute tes propres textes ici (UTF-8), un document par fichier.
- `mirador corpus-web "sujet 1" "sujet 2"` télécharge des articles Wikipédia dans `web/`.
- Les dialogues d'identité (`config/identity.toml`) sont ajoutés automatiquement.

Après avoir ajouté du texte : `mirador train --retrain-tokenizer`.
