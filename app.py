from flask import Flask, request, jsonify
from transformers import pipeline
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# ✅ モデルを事前ロード（初回遅延対策）
summarizer = pipeline("summarization", model="t5-small")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            raise BadRequest("Missing or empty 'text' field.")

        # ✅ 文字数制限（t5-smallは長文に弱いため）
        if len(text) > 1000:
            return jsonify({"error": "Text too long. Please limit to 1000 characters."}), 400

        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return jsonify({"summary": summary[0]["summary_text"]})

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

# ✅ Render用ポート対応（gunicornが使うので明示不要だが、ローカル用に残してもOK）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
