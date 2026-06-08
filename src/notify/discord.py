"""Discord Webhook 通知"""
import os
import mimetypes
import urllib.request
import urllib.error
import json
from pathlib import Path


def _load_webhook_url() -> str:
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DISCORD_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("DISCORD_WEBHOOK_URL", "")


def send(content: str) -> bool:
    """Discord にメッセージを送信。成功で True を返す。"""
    url = _load_webhook_url()
    if not url:
        print("[Discord] DISCORD_WEBHOOK_URL が未設定です")
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


def send_file(filepath: str, caption: str = "") -> bool:
    """Discord にファイルを添付送信する（multipart/form-data）。"""
    url = _load_webhook_url()
    if not url:
        print("[Discord] DISCORD_WEBHOOK_URL が未設定です")
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
