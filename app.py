#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import os
from transformers import pipeline

# 環境変数から APIキー取得
API_KEY = os.environ.get("API_KEY", "")

# モデル読み込み
generator = pipeline(
    "text-generation",
    model="rinna/japanese-gpt2-medium",
    tokenizer="rinna/japanese-gpt2-medium"
)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    # APIキー認証
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth.split(" ")[1] != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        result = generator(
            prompt,
            max_length=200,
            temperature=0.8,
            do_sample=True,
            top_p=0.95
        )
        text = result[0]["generated_text"].strip()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
