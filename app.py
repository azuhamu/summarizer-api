#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API - /tweet エンドポイント
Render側でX投稿処理(post_to_x.py)を呼び出すシンプルな実装例
"""

from flask import Flask, request, jsonify
from post_to_x import post_to_x  # 同ディレクトリ内の投稿処理関数をインポート

app = Flask(__name__)

@app.route("/")
def health():
    """ヘルスチェック用エンドポイント"""
    return {"status": "ok"}

@app.route("/tweet", methods=["POST"])
def tweet():
    """JSONで渡された'text'をXに投稿"""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        result = post_to_x(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Renderのデフォルトポートは環境変数PORTを利用する
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
