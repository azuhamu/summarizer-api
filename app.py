from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from loguru import logger

app = Flask(__name__)

# モデルとトークナイザーのロード（Rust不要）
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# GPU or CPU 自動選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # トークナイズ（Python実装）
        inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推論
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
