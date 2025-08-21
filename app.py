from flask import Flask, request, jsonify
from transformers import pipeline, set_seed
import torch

app = Flask(__name__)

# --- AIモデルの準備 ---
# モデルをより軽量な 'small' バージョンに変更してメモリ使用量を削減
try:
    print("🔄 Loading AI Model: rinna/japanese-gpt2-small...")
    generator = pipeline(
        'text-generation',
        # ▼▼▼ ここが最重要の変更点 ▼▼▼
        model='rinna/japanese-gpt2-small', 
        # ▲▲▲ ここが最重要の変更点 ▲▲▲
        device=-1, # CPUを使用
        torch_dtype=torch.float16
    )
    print("✅ AI Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")
    generator = None

@app.route("/")
def health():
    status = "ok" if generator else "error: model not loaded"
    return {"status": status}

# 文章生成のエンドポイント
@app.route("/generate", methods=["GET"])
def generate_text():
    if not generator:
        return jsonify({"error": "AI model is not available"}), 503

    title = request.args.get("title", "").strip()
    excerpt = request.args.get("excerpt", "").strip()

    if not title:
        return jsonify({"error": "title is a required parameter"}), 400

    prompt = f"""
以下のブログ記事のタイトルと抜粋を元に、読者の興味を引くような魅力的なツイートを作成してください。

# 記事タイトル:
{title}

# 記事の抜粋:
{excerpt}

# 生成ツイート:
"""

    try:
        set_seed(torch.randint(0, 10000, (1,)).item())
        generated_outputs = generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            no_repeat_ngram_size=2
        )
        
        generated_text = generated_outputs[0]['generated_text']
        tweet_text = generated_text.split("# 生成ツイート:")[1].strip()

        return jsonify({"generated_text": tweet_text})

    except Exception as e:
        return jsonify({"error": f"AI generation failed: {str(e)}"}), 500

# サーバー起動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
