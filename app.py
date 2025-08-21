from flask import Flask, request, jsonify
from transformers import pipeline, set_seed
import torch

app = Flask(__name__)

# --- AIモデルの準備 ---
# メモリ使用量を削減するため、より軽量な 'xsmall' バージョンに変更しました。
try:
    print("🔄 Loading AI Model: rinna/japanese-gpt2-xsmall...")
    generator = pipeline(
        'text-generation',
        # ▼▼▼ メモリ不足対策のための最重要変更点 ▼▼▼
        model='rinna/japanese-gpt2-xsmall', 
        # ▲▲▲ rinna/japanese-gpt2-small から変更しました ▲▲▲
        device=-1, # CPUを使用
        # CPU実行の場合、float16は速度低下の可能性があるためコメントアウトを推奨
        # torch_dtype=torch.float16 
    )
    print("✅ AI Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")
    generator = None

@app.route("/")
def health():
    """サーバーのヘルスチェック用エンドポイント"""
    status = "ok" if generator else "error: model not loaded"
    return {"status": status}

@app.route("/generate", methods=["GET"])
def generate_text():
    """ブログのタイトルと抜粋からツイートを生成するエンドポイント"""
    if not generator:
        return jsonify({"error": "AI model is not available"}), 503

    # URLパラメータから記事のタイトルと抜粋を取得
    title = request.args.get("title", "").strip()
    excerpt = request.args.get("excerpt", "").strip()

    # titleパラメータは必須
    if not title:
        return jsonify({"error": "title is a required parameter"}), 400

    # AIに入力するプロンプト（指示文）を作成
    prompt = f"""
以下のブログ記事のタイトルと抜粋を元に、読者の興味を引くような魅力的なツイートを作成してください。

# 記事タイトル:
{title}

# 記事の抜粋:
{excerpt}

# 生成ツイート:
"""

    try:
        # 毎回違う結果を出すために乱数シードを設定
        set_seed(torch.randint(0, 10000, (1,)).item())
        
        # テキスト生成の実行
        generated_outputs = generator(
            prompt,
            max_length=150,              # 生成する文章の最大長
            num_return_sequences=1,      # 生成する文章の数
            do_sample=True,              # サンプリングを有効化
            top_k=50,                    # 上位k個の単語からサンプリング
            top_p=0.95,                  # 上位p%の単語からサンプリング
            temperature=0.9,             # 生成の多様性を調整 (高いほど多様)
            no_repeat_ngram_size=2       # 同じ2単語の繰り返しを防ぐ
        )
        
        # 生成されたテキスト全体から、必要な部分だけを抽出
        generated_text = generated_outputs[0]['generated_text']
        # "# 生成ツイート:" の後ろの部分を取り出す
        tweet_text = generated_text.split("# 生成ツイート:")[1].strip()

        # 結果をJSON形式で返す
        return jsonify({"generated_text": tweet_text})

    except Exception as e:
        # エラーが発生した場合
        return jsonify({"error": f"AI generation failed: {str(e)}"}), 500

# サーバーを起動
if __name__ == "__main__":
    # host="0.0.0.0" で外部からのアクセスを受け付ける
    app.run(host="0.0.0.0", port=8000)
