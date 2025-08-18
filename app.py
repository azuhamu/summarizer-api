from flask import Flask, request, jsonify
from post_to_x import post_to_x

app = Flask(__name__)

@app.route("/")
def health():
    return {"status": "ok"}

@app.route("/tweet", methods=["POST"])
def tweet():
    data = request.json
    text = data.get("text", "")
    result = post_to_x(text)
    return jsonify(result)
