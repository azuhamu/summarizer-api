import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-cnn")
MAX_TOKENS = 512
CHUNK_SIZE = 400  # マージンあり（MAX_TOKENS未満）

app = Flask(__name__)

# モデル & トークナイザーをロード（CPU向け）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def split_into_chunks(text):
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    chunks = split_into_chunks(text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            max_length=MAX_TOKENS,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=MAX_TOKENS
            )
        summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))

    final_summary = " ".join(summaries)
    return jsonify({"summary": final_summary})

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
