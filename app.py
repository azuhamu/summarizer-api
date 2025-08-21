from flask import Flask, request, jsonify
from post_to_x import post_to_x

app = Flask(__name__)

# サーバーが正常に起動しているかを確認するためのエンドポイント
@app.route("/")
def health():
    return {"status": "ok"}

# PHPから呼び出され、投稿文を生成してTwitterに投稿するエンドポイント
@app.route("/tweet", methods=["GET"]) # PHPからのGETリクエストを受け付ける
def tweet():
    # --- STEP 1: PHPから送られてくるURLパラメータを取得 ---
    # request.args.get() を使って、URLの ? 以降のパラメータを取得します
    title = request.args.get("title", "").strip()
    link = request.args.get("link", "").strip()
    
    # excerpt は現在使用していませんが、将来的に要約機能などで利用可能です
    # excerpt = request.args.get("excerpt", "").strip()

    # --- STEP 2: 必要なパラメータが揃っているかチェック ---
    if not title or not link:
        # title と link は必須なので、どちらかが欠けていればエラーを返す
        return jsonify({"error": "title and link are required parameters"}), 400

    # --- STEP 3: Twitterに投稿する本文を生成 ---
    # ここで投稿文のフォーマットを自由に定義できます。
    # 例: 「【記事のタイトル】\n\n【記事のURL】」
    post_text = f"{title}\n\n{link}"

    try:
        # --- STEP 4: 生成した投稿文をTwitterに投稿 ---
        # post_to_x.py の関数を呼び出し、実際にツイート処理を行います
        result = post_to_x(post_text)

        # --- STEP 5: 処理結果をPHPに返す ---
        # Twitterへの投稿が成功したかどうかをチェック
        if result.get("status") == "success":
            # 成功した場合、PHP側がログ出力などで使えるように'text'キーを含んだJSONを返す
            return jsonify({
                "status": "success",
                "text": post_text,  # PHPが受け取るための生成済み投稿文
                "tweet_response": result.get("tweet") # Twitterからの詳細な応答
            })
        else:
            # post_to_x内でエラーが発生した場合（認証情報エラーなど）
            return jsonify(result), 500

    except Exception as e:
        # 予期せぬエラーが発生した場合
        return jsonify({"error": str(e)}), 500

# サーバーを起動するための記述
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
