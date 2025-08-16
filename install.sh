#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

echo "🔄 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing wheel-enabled tokenizers..."
pip install --only-binary=:all: tokenizers==0.20.1

echo "📦 Installing remaining requirements..."
pip install -r requirements.txt
