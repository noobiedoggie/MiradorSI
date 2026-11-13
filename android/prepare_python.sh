#!/usr/bin/env bash
# Copie le code Python de Mirador SI et ses données dans l'application avant la compilation.
set -euo pipefail
cd "$(dirname "$0")/.."
P=android/app/src/main/python
rm -rf "$P/miradorsi"
cp -r miradorsi "$P/"
find "$P/miradorsi" -name "__pycache__" -type d -prune -exec rm -rf {} +
mkdir -p "$P/mirador_app/config" "$P/mirador_app/data"
cp config/identity.toml "$P/mirador_app/config/identity.toml"
cp data/lexique_fr.json "$P/mirador_app/data/lexique_fr.json"
ls -la "$P/mirador_app/modele"
