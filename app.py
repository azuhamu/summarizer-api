import os
import gc
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# === 環境変数からAPIキー取得 ===
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("環境変数 API_KEY が設定されていません")

# 最大トークン数制限
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", 512))

# === モデル初期化（起動時に1回だけロード） ===
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small",
    device=-1  # CPUモード
)

@app.route("/summarize", methods=["POST"])
def summarize():
    # === APIキー認証 ===
    if request.headers.get("x-api-key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # === JSON取得 ===
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # === トークン長制限 ===
        tokenized = summarizer.tokenizer.tokenize(text)
        if len(tokenized) > MAX_INPUT_TOKENS:
            tokenized = tokenized[:MAX_INPUT_TOKENS]
            text = summarizer.tokenizer.convert_tokens_to_string(tokenized)

        # === 要約実行 ===
        result = summarizer(
            text,
            max_length=140,
            min_length=10,
            truncation=True,
            do_sample=False
        )

        # メモリ解放
        del tokenized
        gc.collect()

        return jsonify({"summary": result[0]["summary_text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Render のヘルスチェック用エンドポイント"""
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
