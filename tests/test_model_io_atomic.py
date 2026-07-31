"""D-3: モデル保存のアトミック化（`src/models/model_io.py`）のテスト。

`tmp_path` のみを操作対象とし、本物の `data/models/` には一切触れない。
"""
import pickle
from pathlib import Path

import pytest

from src.models.model_io import atomic_pickle_dump, atomic_write_json, atomic_write_text


def test_atomic_pickle_dump_creates_loadable_file(tmp_path: Path):
    """通常経路: 保存したファイルはpickle.loadで元のオブジェクトへ復元できる。"""
    target = tmp_path / "model.pkl"
    obj = {"a": 1, "b": [1, 2, 3]}
    atomic_pickle_dump(obj, target)

    assert target.exists()
    with open(target, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == obj


def test_atomic_pickle_dump_no_leftover_tmp_files(tmp_path: Path):
    """保存後、同ディレクトリに一時ファイル(.tmp)が残らないこと。"""
    target = tmp_path / "model.pkl"
    atomic_pickle_dump({"x": 1}, target)

    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == [], f"一時ファイルが残存: {leftovers}"


def test_atomic_pickle_dump_overwrite_replaces_content(tmp_path: Path):
    """既存ファイルへの上書きが正しく新内容に置き換わること。"""
    target = tmp_path / "model.pkl"
    atomic_pickle_dump({"version": 1}, target)
    atomic_pickle_dump({"version": 2}, target)

    with open(target, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == {"version": 2}

    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_atomic_pickle_dump_cleans_up_tmp_on_failure(tmp_path: Path):
    """pickle.dump が失敗しても一時ファイルが残らず、元のファイルも破壊されないこと。"""
    target = tmp_path / "model.pkl"
    atomic_pickle_dump({"safe": True}, target)  # 事前に正常なファイルを置いておく

    # ローカル関数（lambda等）はモジュールの qualified name で解決できず pickle 不可
    # → AttributeError/PicklingError 系の例外が送出される
    unpicklable = lambda: None  # noqa: E731
    with pytest.raises(Exception):
        atomic_pickle_dump(unpicklable, target)

    # 元のファイルが無事であること（破損した中間状態で上書きされていない）
    with open(target, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == {"safe": True}

    # 一時ファイルが残っていないこと
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == [], f"失敗時に一時ファイルが残存: {leftovers}"


def test_atomic_pickle_dump_over_readonly_target_succeeds(tmp_path: Path):
    """chmod 444（読み取り専用）の既存ファイルに対してもアトミック上書きが成功すること。

    D-3で扱う「chmod 444とos.replaceの順序関係」の直接的な検証。
    os.replace()の可否は対象ファイル自身のモードではなく親ディレクトリの
    書き込み権限で決まるため、読み取り専用ファイルへの置換も成功するはず。
    """
    import stat

    target = tmp_path / "model.pkl"
    atomic_pickle_dump({"version": 1}, target)
    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444

    atomic_pickle_dump({"version": 2}, target)  # 例外を出さずに成功するはず

    with open(target, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == {"version": 2}


def test_atomic_write_text_and_json(tmp_path: Path):
    """`.meta.json` サイドカー相当のテキスト/JSON書き込みもアトミックに行えること。"""
    text_path = tmp_path / "note.txt"
    atomic_write_text("hello", text_path)
    assert text_path.read_text(encoding="utf-8") == "hello"

    json_path = tmp_path / "meta.json"
    atomic_write_json({"k": "v", "n": 1}, json_path)
    import json
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"k": "v", "n": 1}

    leftovers = [p for p in tmp_path.iterdir() if p not in (text_path, json_path)]
    assert leftovers == []
