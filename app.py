from flask import Flask, request, jsonify
from transformers import pipeline, set_seed
import torch

app = Flask(__name__)

# --- AIモデルの準備 ---
# AIモデルをロードします。サービスの起動時に一度だけ実行されるので少し時間がかかります。
# device=0 はGPUを使う設定ですが、Renderの無料プランではCPU(-1)になります。
# torch_dtypeで半精度浮動小数点数を使うと、メモリ使用量と速度が改善します。
try:
    generator = pipeline(
        'text-generation',
        model='rinna/japanese-gpt2-medium',
        device=-1, # CPUを使用
        torch_dtype=torch.float16
    )
    print("✅ AI Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")
    generator = None

@app.route("/")
def health():
    # APIが生きているか、モデルがロードできたかを確認できる
    status = "ok" if generator else "error: model not loaded"
    return {"status": status}

# 文章生成のエンドポイント（PHPから呼び出される）
@app.route("/generate", methods=["GET"])
def generate_text():
    if not generator:
        return jsonify({"error": "AI model is not available"}), 503

    # PHPから送られてくるパラメータを取得
    title = request.args.get("title", "").strip()
    excerpt = request.args.get("excerpt", "").strip()

    if not title:
        return jsonify({"error": "title is a required parameter"}), 400

    # --- AIへの指示（プロンプト）を作成 ---
    # このプロンプトの書き方で、生成される文章の質が大きく変わります。
    prompt = f"""
以下のブログ記事のタイトルと抜粋を元に、読者の興味を引くような魅力的なツイートを作成してください。

# 記事タイトル:
{title}

# 記事の抜粋:
{excerpt}

# 生成ツイート:
"""

    try:
        # AIで文章を生成
        set_seed(torch.randint(0, 10000, (1,)).item()) # 毎回違う結果にするための乱数シード
        generated_outputs = generator(
            prompt,
            max_length=150, # 生成する最大長 (プロンプト含む)
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            no_repeat_ngram_size=2 # 同じフレーズの繰り返しを防ぐ
        )
        
        # 生成されたテキストだけを抽出
        generated_text = generated_outputs[0]['generated_text']
        # プロンプト部分を削除して、生成されたツイート部分だけを取り出す
        tweet_text = generated_text.split("# 生成ツイート:")[1].strip()

        # PHPに生成した文章を返す
        return jsonify({"generated_text": tweet_text})

    except Exception as e:
        return jsonify({"error": f"AI generation failed: {str(e)}"}), 500

# サーバー起動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
