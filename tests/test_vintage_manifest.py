"""D-4: `rm` 耐性のある vintage 凍結保護（`src/models/vintage_manifest.py`）のテスト。

すべて `tmp_path` に隔離した model_dir / manifest_path を使い、
本物の `data/models/` には一切触れない。
"""
import pickle
from pathlib import Path

from src.models import vintage_manifest as vm


def _make_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_register_and_verify_ok(tmp_path: Path):
    model_dir = tmp_path / "models"
    manifest_path = tmp_path / "vintage_manifest.json"
    path = model_dir / "dummy_eval_m9901.pkl"
    _make_pkl(path, {"v": 1})

    entry = vm.register("dummy_eval_m9901", path, manifest_path=manifest_path)
    assert "sha256" in entry and "registered_at" in entry
    assert vm.is_registered("dummy_eval_m9901", manifest_path=manifest_path)

    result = vm.verify_manifest(model_dir=model_dir, manifest_path=manifest_path)
    assert result == {"missing": [], "hash_mismatch": [], "ok": ["dummy_eval_m9901"]}


def test_verify_detects_missing_file(tmp_path: Path):
    """マニフェストに登録済みだがファイル実体が無い（rm相当）ケースを検知できること。"""
    model_dir = tmp_path / "models"
    manifest_path = tmp_path / "vintage_manifest.json"
    path = model_dir / "dummy_eval_m9902.pkl"
    _make_pkl(path, {"v": 1})
    vm.register("dummy_eval_m9902", path, manifest_path=manifest_path)

    path.unlink()  # rm 相当（ファイル実体だけ削除、マニフェストはそのまま）

    result = vm.verify_manifest(model_dir=model_dir, manifest_path=manifest_path)
    assert result["missing"] == ["dummy_eval_m9902"]
    assert result["hash_mismatch"] == []
    assert result["ok"] == []


def test_verify_detects_hash_mismatch(tmp_path: Path):
    """ファイルは存在するが内容が変わっている場合を検知できること。"""
    model_dir = tmp_path / "models"
    manifest_path = tmp_path / "vintage_manifest.json"
    path = model_dir / "dummy_eval_m9903.pkl"
    _make_pkl(path, {"v": 1})
    vm.register("dummy_eval_m9903", path, manifest_path=manifest_path)

    path.write_bytes(b"tampered-bytes")

    result = vm.verify_manifest(model_dir=model_dir, manifest_path=manifest_path)
    assert result["hash_mismatch"] == ["dummy_eval_m9903"]
    assert result["missing"] == []
    assert result["ok"] == []


def test_load_manifest_missing_file_returns_empty_structure(tmp_path: Path):
    manifest_path = tmp_path / "does_not_exist.json"
    manifest = vm.load_manifest(manifest_path)
    assert manifest == {"models": {}}


def test_vintage_name_re_matches_expected_patterns():
    """命名規則の正規表現が想定パターンに一致すること（trainer.pyと共有の正本）。"""
    for name in ["lgbm_wt_eval_m2401", "lgbm_wt_win_m2607", "lgbm_wt_eval_q2401",
                 "lgbm_wt_eval_w2", "lgbm_wt_eval_w3"]:
        assert vm.VINTAGE_NAME_RE.search(name), f"{name} はvintageとして検出されるべき"
    for name in ["lgbm_wt", "lgbm_wt_eval", "lgbm_pair", "lgbm_upset"]:
        assert not vm.VINTAGE_NAME_RE.search(name), f"{name} はvintageではないはず"
