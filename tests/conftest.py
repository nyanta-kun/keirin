"""pytest 共通設定: リポジトリルートを import パスに追加（src.* / scripts/* を解決）。

加えて **テストから実際の Discord 送信が飛ばないよう全テストで遮断**する（下記
`_block_discord`）。2026-08-04、`tests/test_three_head_rebuild_guard.py` の
「行が空の窓ではガードが発火しない」ケースが `rebuild_pg_atomic` の0件警告経路を
通り、`notify_discord_warning()` → 実webhookで **本番の #システム障害 チャンネルへ
警告が5通投稿された**（ローカルは .env に本番webhookが入っているため）。
テストが本番の通知先を汚すのは検査として明確に誤りなので、個別テストの
monkeypatch 漏れに依存しない形で塞ぐ。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_configure(config):
    """CI 環境（SQLite DB 不在）でもテーブルが存在するようスキーマを初期化。

    本番は VPS PostgreSQL へ一本化済み（2026-07-22〜）で get_connection() は
    KEIRIN_DB_URL 未設定時に例外を送出する。テストだけは明示的に
    KEIRIN_ALLOW_SQLITE_FALLBACK=1 を立ててローカル SQLite を使う。
    """
    import os
    if not os.environ.get("KEIRIN_DB_URL"):
        os.environ["KEIRIN_ALLOW_SQLITE_FALLBACK"] = "1"
        from src.database import init_db
        init_db()


@pytest.fixture(autouse=True)
def _block_discord(monkeypatch):
    """全テストで Discord webhook URL の解決を潰し、実送信を不可能にする。

    `src.notify.discord.send()` / `send_file()` は呼び出しのたびに
    `_load_webhook_url()` を引くため、ここを空文字にすれば送信前に False で返る
    （モジュール側で `from ... import send` 済みでも効く）。

    通知が飛んだかを検証したいテストは、従来どおり呼び出し側モジュールの
    `notify_discord_warning` 等を個別に monkeypatch すること。
    """
    monkeypatch.setattr("src.notify.discord._load_webhook_url", lambda channel: "")
