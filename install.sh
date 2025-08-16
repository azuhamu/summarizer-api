#!/bin/bash
set -e

echo "📦 Installing Python dependencies with --prefer-binary..."

pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt

echo "✅ Installation complete."
