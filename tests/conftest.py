"""pytest 共通設定: リポジトリルートを import パスに追加（src.* / scripts/* を解決）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_configure(config):
    """CI 環境（SQLite DB 不在）でもテーブルが存在するようスキーマを初期化。"""
    import os
    if not os.environ.get("KEIRIN_DB_URL"):
        from src.database import init_db
        init_db()
