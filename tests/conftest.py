"""pytest 共通設定: リポジトリルートを import パスに追加（src.* / scripts/* を解決）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
