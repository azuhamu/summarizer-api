#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

echo "🔄 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing wheel-enabled tokenizers..."
pip install --only-binary=:all: tokenizers==0.21.2

echo "📦 Installing remaining requirements..."
pip install -r requirements.txt

echo "🔍 Checking dependency conflicts..."
# エラー終了させずに衝突を表示
pip check || echo "⚠️ pip check found potential dependency conflicts"
