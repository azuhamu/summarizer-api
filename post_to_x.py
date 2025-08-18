import os
import requests
from requests_oauthlib import OAuth1

# Twitter API credentials（環境変数から取得）
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Twitter API v1.1 endpoint
POST_URL = "https://api.twitter.com/1.1/statuses/update.json"

def post_to_x(text):
    if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET]):
        return {"error": "Missing credentials"}

    auth = OAuth1(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)

    payload = {"status": text}

    try:
        response = requests.post(POST_URL, auth=auth, data=payload)
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
