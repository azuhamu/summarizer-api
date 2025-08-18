import os
import requests
from requests_oauthlib import OAuth1
from bs4 import BeautifulSoup

# 認証情報（環境変数）
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# エンドポイント
POST_TWEET_URL = "https://api.twitter.com/1.1/statuses/update.json"
UPLOAD_MEDIA_URL = "https://upload.twitter.com/1.1/media/upload.json"

# HTML除去関数
def strip_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

# 画像アップロード関数
def upload_media(image_path, auth):
    with open(image_path, "rb") as img:
        files = {"media": img}
        response = requests.post(UPLOAD_MEDIA_URL, auth=auth, files=files)
        if response.status_code == 200:
            media_id = response.json().get("media_id_string")
            return media_id
        else:
            raise Exception(f"Media upload failed: {response.text}")

# 投稿関数（HTML除去 + 画像付き）
def post_to_x(text, image_path=None):
    if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET]):
        return {"error": "Missing credentials"}

    auth = OAuth1(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)

    clean_text = strip_html(text)
    payload = {"status": clean_text}

    try:
        if image_path:
            media_id = upload_media(image_path, auth)
            payload["media_ids"] = media_id

        response = requests.post(POST_TWEET_URL, auth=auth, data=payload)
        if response.status_code == 200:
            return {"status": "success", "tweet": response.json()}
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "body": response.text
            }
    except Exception as e:
        return {"status": "exception", "message": str(e)}
