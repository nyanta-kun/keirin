"""隊列推定位置（feature_wt.add_formation_features_wt）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT,
    FORMATION_COLS_WT,
    add_formation_features_wt,
)


def _race(race_key: str, rows: list[dict]) -> pd.DataFrame:
    base = dict(race_key=race_key, line_group=1, line_size=1, line_pos=1, b_rate_90=0.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_lines_are_ordered_by_b_rate_not_line_group():
    """隊列は line_group の並び順ではなく b_rate_90 で決まる。

    winticket の line_group は予想並びの配列インデックスに過ぎず隊列の前後を表さない
    （実測: 第1ラインの先頭が実際にBを取る率50.5%に対し第2/第3も約30%）。
    """
    df = _race("R1", [
        dict(frame_no=1, line_group=1, line_size=2, line_pos=1, b_rate_90=0.10),
        dict(frame_no=2, line_group=1, line_size=2, line_pos=2, b_rate_90=0.02),
        dict(frame_no=3, line_group=2, line_size=2, line_pos=1, b_rate_90=0.80),
        dict(frame_no=4, line_group=2, line_size=2, line_pos=2, b_rate_90=0.03),
    ])
    out = add_formation_features_wt(df).set_index("frame_no")
    assert out.loc[3, "formation_pos_frac"] == pytest.approx(0.0)
    assert out.loc[4, "formation_pos_frac"] == pytest.approx(1 / 3)
    assert out.loc[1, "formation_pos_frac"] == pytest.approx(2 / 3)
    assert out.loc[2, "formation_pos_frac"] == pytest.approx(1.0)
    # ライン単位の順位も先行ライン=0
    assert out.loc[3, "formation_line_rank"] == pytest.approx(0.0)
    assert out.loc[1, "formation_line_rank"] == pytest.approx(1.0)


def test_line_members_stay_together_in_line_pos_order():
    """ラインの構成員は分断されず、ライン内は line_pos 順に並ぶ。"""
    df = _race("R1", [
        dict(frame_no=1, line_group=1, line_size=3, line_pos=3, b_rate_90=0.01),
        dict(frame_no=2, line_group=1, line_size=3, line_pos=1, b_rate_90=0.70),
        dict(frame_no=3, line_group=1, line_size=3, line_pos=2, b_rate_90=0.05),
        dict(frame_no=4, line_group=None, line_size=1, line_pos=1, b_rate_90=0.30),
    ])
    out = add_formation_features_wt(df).set_index("frame_no")
    assert list(out.loc[[2, 3, 1, 4], "formation_pos_frac"]) == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0])


def test_normalized_across_car_counts():
    """7車と9車で範囲が揃う（車数依存の閾値を持たない）。

    9車は「後方」が4車以上になるなど境界が車数で動くため、生の順位ではなく
    正規化位置を使う。
    """
    for n in (7, 9):
        rows = [dict(frame_no=i, b_rate_90=1.0 - i / 100) for i in range(1, n + 1)]
        out = add_formation_features_wt(_race("R", rows))
        assert out["formation_pos_frac"].min() == pytest.approx(0.0)
        assert out["formation_pos_frac"].max() == pytest.approx(1.0)


def test_row_order_is_preserved_across_races():
    """複数レースを渡しても行順が変わらない（下流の列代入がずれないため）。"""
    df = pd.concat([
        _race("R1", [dict(frame_no=i, b_rate_90=i / 10) for i in (1, 2, 3)]),
        _race("R2", [dict(frame_no=i, b_rate_90=i / 10) for i in (4, 5)]),
    ], ignore_index=True)
    out = add_formation_features_wt(df)
    assert list(out["race_key"]) == ["R1", "R1", "R1", "R2", "R2"]
    assert list(out["frame_no"]) == [1, 2, 3, 4, 5]


def test_degenerates_gracefully_without_b_rate():
    """b_rate_90 が無い場合は 0 埋めで返す（列欠損で落ちない）。"""
    df = _race("R1", [dict(frame_no=i) for i in (1, 2, 3)]).drop(columns="b_rate_90")
    out = add_formation_features_wt(df)
    for c in FORMATION_COLS_WT:
        assert c in out.columns
        assert out[c].eq(0.0).all()


def test_columns_are_in_feature_list_and_finite():
    df = _race("R1", [dict(frame_no=i, b_rate_90=i / 10) for i in range(1, 8)])
    out = add_formation_features_wt(df)
    for c in FORMATION_COLS_WT:
        assert c in FEATURE_COLS_WT, c
        assert np.isfinite(out[c].astype(float)).all(), c


def test_load_model_rejects_mismatched_feature_set(tmp_path, monkeypatch):
    """特徴量『数』が同じでも『中身』が違うモデルは拒否する。

    LightGBM は列数さえ合えば素通しするため、旧特徴セットのモデルが無言で
    誤った予測を返す。実例: 2026-07-31に ex_spurt_pct/ex_thrust_pct を除去して
    48→46、2026-08-03に formation_* を追加して46→48。旧48は列数が一致するため
    LightGBM の Fatal では止まらない。
    """
    import pickle
    from types import SimpleNamespace

    import src.models.trainer as trainer

    def _Fake(cols):
        return SimpleNamespace(feature_name_=list(cols))

    monkeypatch.setattr(trainer, "MODEL_DIR", tmp_path)

    # 中身が現行と一致 → 通る
    (tmp_path / "lgbm_wt_ok.pkl").write_bytes(pickle.dumps(_Fake(FEATURE_COLS_WT)))
    assert trainer.load_model("lgbm_wt_ok") is not None

    # 列数は同じだが formation_* の代わりに旧特徴 → 拒否
    old48 = [c for c in FEATURE_COLS_WT if c not in FORMATION_COLS_WT]
    old48 += ["ex_spurt_pct", "ex_thrust_pct"]
    assert len(old48) == len(FEATURE_COLS_WT)
    (tmp_path / "lgbm_wt_old48.pkl").write_bytes(pickle.dumps(_Fake(old48)))
    with pytest.raises(ValueError, match="特徴量セット"):
        trainer.load_model("lgbm_wt_old48")

    # ks ルート（lgbm_wt 以外）は検証対象外
    (tmp_path / "lgbm_v6.pkl").write_bytes(pickle.dumps(_Fake(["a", "b"])))
    assert trainer.load_model("lgbm_v6") is not None
