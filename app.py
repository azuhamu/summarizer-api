from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
model_name = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
summarizer = pipeline("summarization", model=model_name, device=-1)

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
    results = summarizer(chunks, max_length=130, min_length=30, do_sample=False)
    summary = " ".join([r["summary_text"] for r in results])
    return jsonify({"summary": summary})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
