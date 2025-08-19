from flask import Flask, request, jsonify
from post_to_x import post_to_x

app = Flask(__name__)

@app.route("/")
def health():
    return {"status": "ok"}

@app.route("/tweet", methods=["POST"])
def tweet():
    # JSON を安全に取得
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        result = post_to_x(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
