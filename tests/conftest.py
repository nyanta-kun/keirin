"""pytest 共通設定: リポジトリルートを import パスに追加（src.* / scripts/* を解決）。"""
import sys
from pathlib import Path

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
