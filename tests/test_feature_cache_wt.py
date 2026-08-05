"""特徴量キャッシュ（load_features_wt）の回帰テスト（2026-08-05 新設）。

本テストが守る不変条件:
  1. **保存失敗を握り潰さない**——書けないまま有効化されていると「毎回フル計算
     ＋指紋クエリ」で素の実行より遅くなる。2026-08-05 に実際に踏んだ
     （pyarrow 未導入で parquet が書けず、例外を握り潰して 54% 遅くなっていた）。
     黙って効かないキャッシュは、遅いだけでなく「効いているつもり」を作るので有害。
  2. **指紋が変わったら古いキャッシュを使わない**——黙って古い特徴量で学習する経路は
     作らない。廃止されたローカル SQLite が起こした事故（picks_history 消失・
     2026-07-20）と同じ形になるため。
  3. 既定は無効（環境変数 KEIRIN_FEATURE_CACHE=1 で opt-in）。

DBアクセスなし（load/build を差し替えて純関数として検証する）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.preprocessing.feature_wt as fw


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """DBを触らずに load/build/指紋 を差し替える。"""
    calls = {"build": 0}

    def fake_load(min_date=None, max_date=None):
        return pd.DataFrame({"race_key": ["a", "b"], "frame_no": [1, 2]})

    def fake_build(df):
        calls["build"] += 1
        out = df.copy()
        out["feat"] = [1.0, 2.0]
        return out

    monkeypatch.setattr(fw, "load_raw_data_wt", fake_load)
    monkeypatch.setattr(fw, "build_features_wt", fake_build)
    monkeypatch.setattr(fw, "FEATURE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fw, "_wt_data_fingerprint", lambda a, b: "100_20260805")
    monkeypatch.delenv("KEIRIN_FEATURE_CACHE", raising=False)
    return calls, tmp_path


def test_disabled_by_default(stub, monkeypatch):
    """環境変数が無ければキャッシュは作られない（既定は無効）。"""
    calls, tmp = stub
    fw.load_features_wt("2026-01-01", "2026-01-31")
    fw.load_features_wt("2026-01-01", "2026-01-31")
    assert calls["build"] == 2
    assert list(tmp.glob("*.pkl")) == []


def test_enabled_by_env(stub, monkeypatch):
    calls, tmp = stub
    monkeypatch.setenv("KEIRIN_FEATURE_CACHE", "1")
    fw.load_features_wt("2026-01-01", "2026-01-31")
    assert calls["build"] == 1
    assert len(list(tmp.glob("*.pkl"))) == 1


def test_second_call_hits_cache_and_returns_same_data(stub):
    calls, _tmp = stub
    a = fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
    b = fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
    assert calls["build"] == 1, "2回目は再計算しないこと"
    pd.testing.assert_frame_equal(a, b)


def test_fingerprint_change_invalidates(stub, monkeypatch):
    """データが増えたら（指紋が変われば）古いキャッシュを使わず作り直す。"""
    calls, tmp = stub
    fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
    assert calls["build"] == 1
    monkeypatch.setattr(fw, "_wt_data_fingerprint", lambda a, b: "200_20260806")
    fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
    assert calls["build"] == 2, "指紋が変わったら再計算すること"
    names = sorted(p.name for p in tmp.glob("*.pkl"))
    assert len(names) == 1 and "200_20260806" in names[0], "古い指紋は削除されること"


def test_different_period_does_not_share_cache(stub):
    """期間ごとにキャッシュを持つこと（切り出し流用は安全でないため）。

    広い期間から切り出すと race_point の得点補完（読み込み範囲全体の中央値）に
    依存する7列がズレる。実測済み: race_point / score_rank / score_z /
    line_rp_sum / line_rp_max / line_rp_mean / line_rp_gap_top。
    """
    calls, tmp = stub
    fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
    fw.load_features_wt("2026-01-01", "2026-02-28", use_cache=True)
    assert calls["build"] == 2
    assert len(list(tmp.glob("*.pkl"))) == 2


def test_save_failure_is_not_swallowed(stub, monkeypatch):
    """保存できないときは例外を上げる（握り潰すと素の実行より遅くなるだけ）。"""
    _calls, tmp = stub

    def boom(self, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_pickle", boom)
    with pytest.raises(OSError):
        fw.load_features_wt("2026-01-01", "2026-01-31", use_cache=True)
