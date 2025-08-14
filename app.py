from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

summarizer = None  # モデルは初回アクセス時にのみロード

@app.route("/summarize", methods=["POST"])
def summarize():
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU利用
        )

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        summary = summarizer(
            data["text"],
            max_length=130,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
