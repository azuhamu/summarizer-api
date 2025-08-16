#!/bin/bash
set -e

echo "📦 Installing Python dependencies with --prefer-binary..."

# 最新の pip, setuptools, wheel にアップグレード
pip install --upgrade pip setuptools wheel

# requirements.txt を wheel 優先でインストール
pip install --prefer-binary -r requirements.txt

echo "✅ Installation complete."
