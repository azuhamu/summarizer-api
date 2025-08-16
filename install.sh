#!/usr/bin/env bash
set -e

echo "🚀 Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing tokenizers wheel first..."
pip install --prefer-binary tokenizers==0.19.1

echo "📦 Installing remaining dependencies..."
pip install --prefer-binary -r requirements.txt
