"""凍結vintageモデルのハッシュ付きマニフェスト管理。

## 背景

2026-07-28、四半期vintageモデル18本がアドホック実験で無断上書きされ、
honest walk-forward検証の再現性が失われる事故が発生した
（memory `keirin_s7_foundational_rethink_2026_07_29` 参照）。これを受けて
`src/models/trainer.py::save_model()` に「vintage命名規則に一致するファイルは
既に存在する場合、force=True を明示しない限り上書き禁止」というガードと
chmod 444（読み取り専用化）による第二の防御線が実装された。

しかし、このガードは **同名ファイルが存在するかどうか** でしか判定しない。
そのため `rm data/models/lgbm_wt_eval_m2401.pkl` のように **先にファイルを
削除してから** 再度 `save_model()` を呼べば `path.exists()` が False になり、
`force=True` を指定せずとも無警告で新規保存として成功してしまう
（chmod 444 もファイル自体の削除には無力。Unixのファイル削除権限は
ファイル自身のモードではなく親ディレクトリの書き込み権限で決まるため）。

## 設計

「一度でも凍結vintageとして保存された」という事実を、ファイル実体とは
独立にJSONマニフェスト（`data/models/vintage_manifest.json`）へ記録する。
`save_model()` は vintage 名で保存しようとした際、ファイル実体の有無に
加えてこのマニフェストの登録有無も確認し、**マニフェストに登録済みなら
ファイルが存在しなくても保存を拒否する**（`force=True` 明示時のみ許可）。
これにより「rmしてから再作成する」という事故と同型の操作経路を塞ぐ。

マニフェストの1エントリは以下を保持する:
    {
      "sha256": "...",          # 保存時点のファイルのSHA256
      "size_bytes": 12345,
      "registered_at": "2026-07-31T12:00:00"
    }

## 設計上の限界（重要）

- **マニフェスト自体（このJSONファイル）は保護されない。** マニフェストを
  直接編集・削除する操作には無力であり、それを検知する仕組みも本実装には
  含まれない。これは「気づかずに壊れる」事故を「意図的に改ざんする」行為へ
  格上げする効果に留まる（＝過失は防げるが、悪意や重大な誤操作には
  最終的には無力）。
- マニフェストと実ファイルの整合性は `verify_manifest()` を明示的に
  実行しない限りチェックされない（`save_model()` は保存時にしか触らない）。
- さらに強固にするには `chattr +i`（immutable属性）等のOSレベルの保護が
  有効だが、root権限が必要なため本実装では採用していない。
- ファイルシステムを跨いだ移動・バックアップからの手動復元を行うと、
  ハッシュが変わらなくても `registered_at` 等のメタ情報は復元されない
  （ハッシュ自体は内容が同一なら一致するため実害は小さい）。
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .model_io import atomic_write_json

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MANIFEST_PATH = MODEL_DIR / "vintage_manifest.json"

# 凍結vintageモデルの命名規則（四半期q2401等・旧非標準w2/w3・月次m2401=YYMM等）。
# src/models/trainer.py::save_model() の書き込み保護と同一の正規表現。
# 単一の情報源とするため、trainer.py 側はこのモジュールから import する。
VINTAGE_NAME_RE = re.compile(r"_(q\d{4}|w\d+|m\d{4})$")


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """ファイルのSHA256ハッシュ値（16進文字列）を計算する。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """マニフェストJSONを読み込む。存在しない場合は空の構造を返す。"""
    mp = manifest_path or MANIFEST_PATH
    if not mp.exists():
        return {"models": {}}
    with open(mp, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("models", {})
    return data


def save_manifest(manifest: dict[str, Any], manifest_path: Path | None = None) -> None:
    """マニフェストJSONをアトミックに保存する。"""
    mp = manifest_path or MANIFEST_PATH
    atomic_write_json(manifest, mp)


def is_registered(name: str, manifest_path: Path | None = None) -> bool:
    """`name`（拡張子なしのモデル名）がマニフェストに登録済みかどうか。"""
    manifest = load_manifest(manifest_path)
    return name in manifest["models"]


def register(name: str, path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """凍結モデルをマニフェストに登録（または更新）する。

    Returns:
        登録したエントリの内容。
    """
    entry = {
        "sha256": sha256_of_file(path),
        "size_bytes": Path(path).stat().st_size,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest = load_manifest(manifest_path)
    manifest["models"][name] = entry
    save_manifest(manifest, manifest_path)
    return entry


def verify_manifest(
    model_dir: Path | None = None, manifest_path: Path | None = None
) -> dict[str, list[str]]:
    """マニフェストと実ファイルの整合性を検証する。

    Returns:
        {"missing": [...], "hash_mismatch": [...], "ok": [...]}
        - missing: マニフェストには登録されているがファイル実体が無い
                   （= rm等で削除された可能性）
        - hash_mismatch: ファイルは存在するが内容のSHA256が一致しない
                         （= 内容が書き換えられた可能性）
        - ok: 整合性の取れているエントリ
    """
    md = model_dir or MODEL_DIR
    manifest = load_manifest(manifest_path)
    missing: list[str] = []
    hash_mismatch: list[str] = []
    ok: list[str] = []
    for name, entry in manifest["models"].items():
        path = md / f"{name}.pkl"
        if not path.exists():
            missing.append(name)
            continue
        actual = sha256_of_file(path)
        if actual != entry.get("sha256"):
            hash_mismatch.append(name)
        else:
            ok.append(name)
    return {"missing": missing, "hash_mismatch": hash_mismatch, "ok": ok}
