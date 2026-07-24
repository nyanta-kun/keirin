"""Discord Webhook 通知

チャンネルごとに個別のWebhook URLを使う（2026-07-24〜「二軸探偵」サーバー
5チャンネル構成）。呼び出し側は必ず channel を明示指定すること
（黙って別チャンネルに届く事故を防ぐため、デフォルト値は用意しない）。
"""
import os
import mimetypes
import urllib.request
import urllib.error
import json
from pathlib import Path

# channel キー → .env の環境変数名
_WEBHOOK_ENV_KEYS: dict[str, str] = {
    "picks": "DISCORD_WEBHOOK_URL_PICKS",        # 朝夕の推奨
    "prerace": "DISCORD_WEBHOOK_URL_PRERACE",    # 発走前個別通知
    "results": "DISCORD_WEBHOOK_URL_RESULTS",    # 成績報告
    "netkeirin": "DISCORD_WEBHOOK_URL_NETKEIRIN",  # netkeirin入稿完了
    "system": "DISCORD_WEBHOOK_URL_SYSTEM",      # システム障害
}


def _load_webhook_url(channel: str) -> str:
    if channel not in _WEBHOOK_ENV_KEYS:
        raise ValueError(
            f"未知のDiscord通知チャンネルキー: {channel!r}（有効な値: "
            f"{sorted(_WEBHOOK_ENV_KEYS)}）"
        )
    env_key = _WEBHOOK_ENV_KEYS[channel]
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{env_key}="):
                return line.split("=", 1)[1].strip()
    return os.environ.get(env_key, "")


def send(content: str, channel: str) -> bool:
    """Discord にメッセージを送信。成功で True を返す。

    channel: "picks" / "prerace" / "results" / "netkeirin" / "system" のいずれか。
    """
    url = _load_webhook_url(channel)
    if not url:
        print(f"[Discord] {_WEBHOOK_ENV_KEYS[channel]} が未設定です")
        return False

    payload = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (keirin-ai, 1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 204
    except urllib.error.URLError as e:
        print(f"[Discord] 送信失敗: {e}")
        return False


def send_file(filepath: str, channel: str, caption: str = "") -> bool:
    """Discord にファイルを添付送信する（multipart/form-data）。

    channel: "picks" / "prerace" / "results" / "netkeirin" / "system" のいずれか。
    """
    url = _load_webhook_url(channel)
    if not url:
        print(f"[Discord] {_WEBHOOK_ENV_KEYS[channel]} が未設定です")
        return False

    path = Path(filepath)
    if not path.exists():
        print(f"[Discord] ファイルが見つかりません: {filepath}")
        return False

    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    boundary = "keirin_discord_boundary_20260101"

    parts: list[bytes] = []
    if caption:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="content"\r\n\r\n'
            f"{caption}\r\n".encode()
        )
    file_data = path.read_bytes()
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
        + file_data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "DiscordBot (keirin-ai, 1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 204)
    except urllib.error.URLError as e:
        print(f"[Discord] ファイル送信失敗: {e}")
        return False
