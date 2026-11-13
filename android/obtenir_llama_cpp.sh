#!/usr/bin/env bash
# Place les sources de llama.cpp (moteur des modèles ouverts) dans app/src/main/cpp/llama.cpp.
# Version fixée : celle livrée avec llama-cpp-python 0.3.36 sur PyPI, vérifiée par son empreinte SHA-256.
set -euo pipefail
cd "$(dirname "$0")"
CIBLE=app/src/main/cpp/llama.cpp
VERSION=0.3.36
URL=https://files.pythonhosted.org/packages/ec/e9/e7de2b0463ea3ffbf0ede6cb21b58c1258a8f6521aae45ca773a59fe7cf3/llama_cpp_python-$VERSION.tar.gz
EMPREINTE=832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e
if [ -f "$CIBLE/CMakeLists.txt" ]; then
  echo "llama.cpp déjà présent : $CIBLE"
  exit 0
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -sSfL --retry 3 -o "$TMP/source.tar.gz" "$URL"
echo "$EMPREINTE  $TMP/source.tar.gz" | sha256sum -c -
tar xzf "$TMP/source.tar.gz" -C "$TMP" "llama_cpp_python-$VERSION/vendor/llama.cpp"
mv "$TMP/llama_cpp_python-$VERSION/vendor/llama.cpp" "$CIBLE"
echo "llama.cpp installé dans $CIBLE"
