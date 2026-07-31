"""モデル・メタデータ永続化の共通ヘルパー（アトミック書き込み）。

`open(path, "wb")` はオープンした時点で既存ファイルを即座に0バイトへ
切り詰める。そのため `pickle.dump()` の完了前にプロセスが異常終了
（OOM kill・電源断・スリープ中の強制終了・シグナル等）すると、直前まで
正常だったファイルが「存在するが壊れている」状態のまま残ってしまう。

本モジュールは「同一ディレクトリ内の一時ファイルへ書き切ってから
`os.replace()` でアトミックに差し替える」方式を提供し、この種の
部分書き込みによる汚染を防ぐ。`os.replace()` は同一ファイルシステム上で
あればPOSIX上atomicであり、かつ置換先ファイルが読み取り専用（chmod 444等）
であっても親ディレクトリに書き込み権限があれば成功する（rename可否は
対象ファイル自身のモードではなく親ディレクトリの書き込み権限で決まるため）。
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, write_fn) -> None:
    """`write_fn(file_obj)` の結果を `path` へアトミックに書き込む共通ロジック。

    一時ファイルは `path` と同一ディレクトリに作成する（`os.replace()` の
    アトミック性はファイルシステムを跨ぐと保証されないため）。
    途中で例外が発生した場合は一時ファイルを削除して後始末する。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """`obj` を `path` へpickleでアトミックに保存する。

    書き込み完了前にプロセスが異常終了しても、`path` には旧内容
    （まだ存在しない場合は何も）が残るだけで、破損した中間状態には
    ならない。
    """
    _atomic_write(Path(path), lambda f: pickle.dump(obj, f))


def atomic_write_text(text: str, path: Path, encoding: str = "utf-8") -> None:
    """テキストを `path` へアトミックに書き込む（`.meta.json` 等のサイドカー用）。"""
    _atomic_write(Path(path), lambda f: f.write(text.encode(encoding)))


def atomic_write_json(obj: Any, path: Path) -> None:
    """JSONシリアライズ可能な `obj` を `path` へアトミックに保存する。"""
    atomic_write_text(json.dumps(obj, ensure_ascii=False, indent=2), path)
