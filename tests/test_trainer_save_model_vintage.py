"""D-3/D-4: `src/models/trainer.py::save_model()` の統合テスト。

- アトミック保存（一時ファイルが残らない）
- vintage命名規則に一致するモデルの凍結保護:
    - 同名ファイルが既に存在する場合に拒否
    - ファイルは存在しないがマニフェストに登録済みの場合に拒否（rm耐性・D-4）
    - 非vintage名は従来通り上書きできる
    - force=True で明示的に上書きできる（マニフェストも更新される）

`trainer.MODEL_DIR` と `vintage_manifest.MANIFEST_PATH` を monkeypatch で
`tmp_path` 配下に差し替え、本物の `data/models/` には一切触れない。
"""
import pickle
import stat
from pathlib import Path

import pytest

from src.models import trainer
from src.models import vintage_manifest as vm


@pytest.fixture()
def isolated_model_dir(tmp_path, monkeypatch):
    """save_model()が参照するMODEL_DIR/MANIFEST_PATHをtmp_path配下に差し替える。"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    manifest_path = model_dir / "vintage_manifest.json"
    monkeypatch.setattr(trainer, "MODEL_DIR", model_dir)
    monkeypatch.setattr(vm, "MANIFEST_PATH", manifest_path)
    return model_dir


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_save_model_non_vintage_overwrite_ok(isolated_model_dir):
    """非vintage名（例: lgbm_wt）は何度でも上書き保存できる。"""
    trainer.save_model({"v": 1}, "lgbm_wt")
    trainer.save_model({"v": 2}, "lgbm_wt")  # 例外なく上書きできる

    path = isolated_model_dir / "lgbm_wt.pkl"
    assert _load(path) == {"v": 2}
    # 非vintageなのでマニフェストに登録されない
    assert not vm.is_registered("lgbm_wt")


def test_save_model_atomic_no_leftover_tmp(isolated_model_dir):
    trainer.save_model({"v": 1}, "lgbm_wt")
    leftovers = [p for p in isolated_model_dir.iterdir()
                 if p.name not in ("lgbm_wt.pkl",)]
    assert leftovers == [], f"一時ファイルが残存: {leftovers}"


def test_save_model_vintage_rejects_when_file_exists(isolated_model_dir):
    """vintage名でファイルが既に存在する場合、force無指定なら拒否される（従来挙動）。"""
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9901")

    with pytest.raises(FileExistsError):
        trainer.save_model({"v": 2}, "lgbm_wt_eval_m9901")

    # 拒否されたので内容は変わっていない
    path = isolated_model_dir / "lgbm_wt_eval_m9901.pkl"
    assert _load(path) == {"v": 1}


def test_save_model_vintage_rejects_when_manifest_registered_but_file_missing(
    isolated_model_dir,
):
    """今回塞いだ穴: `rm` でファイル実体を消してもマニフェスト登録があれば拒否される。"""
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9902")
    path = isolated_model_dir / "lgbm_wt_eval_m9902.pkl"
    assert path.exists()
    assert vm.is_registered("lgbm_wt_eval_m9902")

    # rm 相当（ファイル実体だけ削除。マニフェストはそのまま残る）
    path.chmod(stat.S_IWUSR | stat.S_IRUSR)  # 444のままだとunlinkできない環境を考慮
    path.unlink()
    assert not path.exists()

    with pytest.raises(FileExistsError):
        trainer.save_model({"v": 2}, "lgbm_wt_eval_m9902")

    # 拒否されたのでファイルは再作成されていない
    assert not path.exists()


def test_save_model_vintage_force_overwrite_works(isolated_model_dir):
    """force=True なら既存vintageモデルを明示的に上書きでき、マニフェストも更新される。"""
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9903")
    first_entry = vm.load_manifest()["models"]["lgbm_wt_eval_m9903"]

    trainer.save_model({"v": 2}, "lgbm_wt_eval_m9903", force=True)

    path = isolated_model_dir / "lgbm_wt_eval_m9903.pkl"
    assert _load(path) == {"v": 2}
    second_entry = vm.load_manifest()["models"]["lgbm_wt_eval_m9903"]
    assert second_entry["sha256"] != first_entry["sha256"]


def test_save_model_vintage_force_overwrite_after_rm_works(isolated_model_dir):
    """force=True であれば rm 後・マニフェスト登録済みの状態からも再作成できる
    （D-4は「無警告での」上書きだけを防ぐ設計であり、明示的なforceは常に許可する）。
    """
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9904")
    path = isolated_model_dir / "lgbm_wt_eval_m9904.pkl"
    path.chmod(stat.S_IWUSR | stat.S_IRUSR)
    path.unlink()

    trainer.save_model({"v": 2}, "lgbm_wt_eval_m9904", force=True)
    assert _load(path) == {"v": 2}


def test_save_model_vintage_becomes_readonly(isolated_model_dir):
    """vintageモデルは保存後 chmod 444（読み取り専用）になる。"""
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9905")
    path = isolated_model_dir / "lgbm_wt_eval_m9905.pkl"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def test_save_model_vintage_readonly_file_can_be_force_overwritten(isolated_model_dir):
    """chmod 444の既存ファイルに対してforce=Trueでの上書き（アトミックreplace）が
    例外なく成功すること（D-3で確認したchmod 444とos.replaceの順序関係の統合テスト）。
    """
    trainer.save_model({"v": 1}, "lgbm_wt_eval_m9906")
    path = isolated_model_dir / "lgbm_wt_eval_m9906.pkl"
    assert stat.S_IMODE(path.stat().st_mode) == (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    trainer.save_model({"v": 2}, "lgbm_wt_eval_m9906", force=True)
    assert _load(path) == {"v": 2}
