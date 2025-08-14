import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# === 環境変数からAPIキー取得 ===
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("環境変数 API_KEY が設定されていません")

# === モデル初期化 ===
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

    # === リクエストデータ取得 ===
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # === 要約処理 ===
    try:
        result = summarizer(
            text,
            max_length=140,
            min_length=10,
            do_sample=False
        )
        return jsonify({"summary": result[0]["summary_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    """動作確認用の簡易エンドポイント"""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
