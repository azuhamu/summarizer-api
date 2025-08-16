import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 環境変数から最大トークン数を取得（デフォルト512）
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 512))

# サマライザー初期化（例：t5-small）
summarizer = pipeline("summarization", model="t5-small")

@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]

    # トークン制限（事前カット）
    tokens = summarizer.tokenizer.tokenize(text)
    if len(tokens) > MAX_INPUT_TOKENS:
        tokens = tokens[:MAX_INPUT_TOKENS]
        text = summarizer.tokenizer.convert_tokens_to_string(tokens)

    # サマリ生成（保険のtruncation=True）
    try:
        summary = summarizer(
            text,
            max_length=140,
            min_length=10,
            truncation=True,
            do_sample=False
        )
        return jsonify({"summary": summary[0]['summary_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
