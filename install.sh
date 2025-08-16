#!/bin/bash
set -e

echo "📦 Installing Python dependencies with --prefer-binary..."

pip install --upgrade pip setuptools wheel
pip install --prefer-binary --no-build-isolation -r requirements.txt

echo "✅ Installation complete."
