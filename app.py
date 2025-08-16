from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 軽量モデルでメモリ使用量削減
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPUのみ
)

MAX_TOKENS = 512
CHUNK_SIZE = 500

def chunk_text(text, max_tokens=CHUNK_SIZE):
    tokens = summarizer.tokenizer.tokenize(text)
    for i in range(0, len(tokens), max_tokens):
        yield summarizer.tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Summarizer API running"})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        chunks = list(chunk_text(text))
        partial_summaries = []

        for chunk in chunks:
            summary = summarizer(
                chunk,
                max_length=128,
                min_length=30,
                truncation=True
            )[0]["summary_text"]
            partial_summaries.append(summary)

        if len(partial_summaries) > 1:
            final_input = " ".join(partial_summaries)
            final_summary = summarizer(
                final_input,
                max_length=128,
                min_length=30,
                truncation=True
            )[0]["summary_text"]
        else:
            final_summary = partial_summaries[0]

        return jsonify({
            "summary": final_summary,
            "chunks_processed": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
