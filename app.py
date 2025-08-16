from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

model_name = "facebook/bart-large-cnn"
model = None

def get_model():
    global model
    if model is None:
        device = 0 if torch.cuda.is_available() else -1
        model = pipeline("summarization", model=model_name, device=device)
    return model

def chunk_text(text, max_tokens=1024):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    summarizer = get_model()
    
    summaries = []
    for chunk in chunk_text(text):
        summaries.append(summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"])
    return jsonify({"summary": " ".join(summaries)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
