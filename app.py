import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
MAX_TOKENS = 512
CHUNK_SIZE = 400

app = Flask(__name__)

device_map = "auto"
kwargs = {
    "device_map": device_map,
    "low_cpu_mem_usage": True
}
if torch.cuda.is_available():
    kwargs["load_in_8bit"] = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **kwargs)

def split_into_chunks(text):
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+CHUNK_SIZE])
            for i in range(0, len(tokens), CHUNK_SIZE)]

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    summaries = []
    for chunk in split_into_chunks(text):
        inputs = tokenizer(chunk, max_length=MAX_TOKENS, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_length=MAX_TOKENS)
        summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return jsonify({"summary": " ".join(summaries)})

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
