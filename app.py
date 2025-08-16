from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# モデル名は環境変数から取得（render.yamlで指定）
model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")

# 要約パイプラインを初期化（低メモリ構成）
summarizer = pipeline(
    "summarization",
    model=model_name,
    device=-1  # CPU推奨（Render free plan）
)

# 長文をチャンク分割
def chunk_text(text, max_tokens=150):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    chunks = list(chunk_text(text))
    results = summarizer(
        chunks,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    summary = " ".join([r["summary_text"] for r in results])
    return jsonify({"summary": summary})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
