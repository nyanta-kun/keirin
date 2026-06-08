"""X (Twitter) 投稿モジュール"""
import os
from pathlib import Path


def _load_credentials() -> dict:
    env_file = Path(__file__).parent.parent.parent / ".env"
    env = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return {
        "api_key":            env.get("X_API_KEY")            or os.environ.get("X_API_KEY", ""),
        "api_key_secret":     env.get("X_API_KEY_SECRET")     or os.environ.get("X_API_KEY_SECRET", ""),
        "access_token":       env.get("X_ACCESS_TOKEN")       or os.environ.get("X_ACCESS_TOKEN", ""),
        "access_token_secret":env.get("X_ACCESS_TOKEN_SECRET")or os.environ.get("X_ACCESS_TOKEN_SECRET", ""),
    }


def post_tweet(text: str, reply_to_id: str = None) -> str | None:
    """ツイートを投稿。成功時はツイートIDを返す。"""
    try:
        import tweepy
    except ImportError:
        print("[X] tweepy がインストールされていません: pip install tweepy")
        return None

    creds = _load_credentials()
    if not all(creds.values()):
        print("[X] API認証情報が未設定です（.env に X_API_KEY 等を追記してください）")
        return None

    client = tweepy.Client(
        consumer_key=creds["api_key"],
        consumer_secret=creds["api_key_secret"],
        access_token=creds["access_token"],
        access_token_secret=creds["access_token_secret"],
    )

    kwargs: dict = {"text": text}
    if reply_to_id:
        kwargs["in_reply_to_tweet_id"] = reply_to_id

    try:
        resp = client.create_tweet(**kwargs)
        return str(resp.data["id"])
    except tweepy.TweepyException as e:
        print(f"[X] 投稿失敗: {e}")
        return None


def post_thread(texts: list[str]) -> bool:
    """複数ツイートをスレッドとして投稿。1件でも失敗したら False を返す。"""
    reply_id = None
    for text in texts:
        tweet_id = post_tweet(text, reply_to_id=reply_id)
        if tweet_id is None:
            return False
        reply_id = tweet_id
    return True
