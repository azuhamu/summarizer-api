import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")

# summarizer を初期化（デバイス自動選択：CPU）
summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    device=-1
)

MAX_CHARS = 2000  # 1チャンク文字数
CHUNK_OVERLAP = 200  # コンテキスト重複

def chunk_text(text, max_chars=MAX_CHARS, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    input_text = data.get("text", "")
    if not input_text.strip():
        return jsonify({"error": "No text provided"}), 400

    chunks = chunk_text(input_text)
    summaries = []
    for ch in chunks:
        summary = summarizer(ch, max_length=200, min_length=30, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    final_summary = " ".join(summaries)
    return jsonify({"summary": final_summary})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
