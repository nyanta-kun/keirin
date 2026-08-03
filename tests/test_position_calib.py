"""隊列位置バイアス補正（src/preprocessing/position_calib.py）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.position_calib import (
    POSCAL_FEATURE_COLS,
    add_position_features,
    apply_position_calibration,
)


def _race(race_key: str, rows: list[dict]) -> pd.DataFrame:
    base = dict(race_key=race_key, race_point=100.0, line_group=1, line_size=1,
                line_pos=1, pred_prob=0.5, pc_p_b=0.1)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_pos_frac_follows_b_probability_not_line_group():
    """隊列は line_group の並び順ではなく B取得確率で決まる。

    winticket の line_group は予想並びの配列インデックスに過ぎず、隊列の前後を
    表さない（実測: 第1ラインの先頭がBを取る率50.5%に対し第2/第3も約30%）。
    line_group=2 のラインの方がBを取りそうなら、そちらが前になるべき。
    """
    df = _race("R1", [
        dict(frame_no=1, line_group=1, line_size=2, line_pos=1, pc_p_b=0.10),
        dict(frame_no=2, line_group=1, line_size=2, line_pos=2, pc_p_b=0.02),
        dict(frame_no=3, line_group=2, line_size=2, line_pos=1, pc_p_b=0.80),
        dict(frame_no=4, line_group=2, line_size=2, line_pos=2, pc_p_b=0.03),
    ])
    out = add_position_features(df).set_index("frame_no")
    # line_group=2 が先頭ラインになる
    assert out.loc[3, "pc_pos_frac"] == pytest.approx(0.0)
    assert out.loc[4, "pc_pos_frac"] == pytest.approx(1 / 3)
    assert out.loc[1, "pc_pos_frac"] == pytest.approx(2 / 3)
    assert out.loc[2, "pc_pos_frac"] == pytest.approx(1.0)


def test_line_members_stay_together_in_line_pos_order():
    """ラインの構成員は分断されず、ライン内は line_pos 順に並ぶ。"""
    df = _race("R1", [
        dict(frame_no=1, line_group=1, line_size=3, line_pos=3, pc_p_b=0.01),
        dict(frame_no=2, line_group=1, line_size=3, line_pos=1, pc_p_b=0.70),
        dict(frame_no=3, line_group=1, line_size=3, line_pos=2, pc_p_b=0.05),
        dict(frame_no=4, line_group=None, line_size=1, line_pos=1, pc_p_b=0.30),
    ])
    out = add_position_features(df).set_index("frame_no")
    assert out.loc[2, "pc_pos_frac"] == pytest.approx(0.0)
    assert out.loc[3, "pc_pos_frac"] == pytest.approx(1 / 3)
    assert out.loc[1, "pc_pos_frac"] == pytest.approx(2 / 3)
    assert out.loc[4, "pc_pos_frac"] == pytest.approx(1.0)   # 単騎は最後方


def test_pos_frac_is_normalized_across_car_counts():
    """7車と9車で pos_frac の範囲が揃う（車数依存の閾値を持たない）。

    9車は後方が4車以上になるなど「後方」の境界が車数で動くため、生の順位ではなく
    正規化位置を入力にする。
    """
    for n in (7, 9):
        rows = [dict(frame_no=i, line_group=None, line_size=1, line_pos=1,
                     pc_p_b=1.0 - i / 100) for i in range(1, n + 1)]
        out = add_position_features(_race("R", rows))
        assert out["pc_pos_frac"].min() == pytest.approx(0.0)
        assert out["pc_pos_frac"].max() == pytest.approx(1.0)
        assert out["pc_n_entries"].eq(n).all()


def test_rp_adv_is_diff_from_others_mean():
    """競走得点差は「自分以外の平均」との差。"""
    df = _race("R1", [
        dict(frame_no=1, race_point=120.0, pc_p_b=0.9),
        dict(frame_no=2, race_point=90.0, pc_p_b=0.1),
        dict(frame_no=3, race_point=90.0, pc_p_b=0.1),
    ])
    out = add_position_features(df).set_index("frame_no")
    assert out.loc[1, "pc_rp_adv"] == pytest.approx(30.0)
    assert out.loc[2, "pc_rp_adv"] == pytest.approx(90.0 - 105.0)


def test_rank_frac_uses_base_probability():
    """予測順位はベースモデルの確率で決まり、0〜1に正規化される。"""
    df = _race("R1", [
        dict(frame_no=1, pred_prob=0.20, pc_p_b=0.5),
        dict(frame_no=2, pred_prob=0.80, pc_p_b=0.1),
        dict(frame_no=3, pred_prob=0.50, pc_p_b=0.1),
    ])
    out = add_position_features(df).set_index("frame_no")
    assert out.loc[2, "pc_rank_frac"] == pytest.approx(0.0)
    assert out.loc[3, "pc_rank_frac"] == pytest.approx(0.5)
    assert out.loc[1, "pc_rank_frac"] == pytest.approx(1.0)


def test_multiple_races_keep_input_row_order():
    """複数レースを渡しても行順が入れ替わらない（下流の列代入がずれないため）。"""
    df = pd.concat([
        _race("R1", [dict(frame_no=i, pc_p_b=i / 10) for i in (1, 2, 3)]),
        _race("R2", [dict(frame_no=i, pc_p_b=i / 10) for i in (4, 5)]),
    ], ignore_index=True)
    out = add_position_features(df)
    assert list(out["race_key"]) == ["R1", "R1", "R1", "R2", "R2"]
    assert list(out["frame_no"]) == [1, 2, 3, 4, 5]


def test_all_feature_columns_present_and_finite():
    df = _race("R1", [dict(frame_no=i, pc_p_b=i / 10) for i in range(1, 8)])
    out = add_position_features(df)
    for c in POSCAL_FEATURE_COLS:
        assert c in out.columns, c
        assert np.isfinite(out[c].astype(float)).all(), c


def test_seven_car_races_are_not_calibrated(monkeypatch):
    """7車には適用しない。7車の改善は◎◯への寄りで説明され、質は上がらないため。

    7S/7A は overlap∈{0,1}（市場と不一致）だけを対象とするランクなので、
    軸が◎◯側へ寄ると改善ではなく7Bへの再分類になり、母集団だけが減る。
    """
    import src.models.trainer as trainer

    class _Calib:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.99)])

    class _B:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])

    monkeypatch.setattr(trainer, "load_model",
                        lambda name: _B() if name.endswith("_b") else _Calib())
    monkeypatch.setattr("src.preprocessing.feature_wt.prepare_X", lambda d: d[["frame_no"]])

    seven = _race("R7", [dict(frame_no=i, pred_prob=i / 10) for i in range(1, 8)])
    out, applied = apply_position_calibration(seven)
    assert applied is False
    assert list(out["pred_prob"]) == list(seven["pred_prob"])

    nine = _race("R9", [dict(frame_no=i, pred_prob=i / 10) for i in range(1, 10)])
    out, applied = apply_position_calibration(nine)
    assert applied is True
    assert out["pred_prob"].eq(0.99).all()
    assert list(out["pred_prob_raw"]) == list(nine["pred_prob"])


def test_apply_is_noop_without_models(monkeypatch):
    """モデルが無いときは無適用で返し、補正前の値を必ず残す。

    無言でフォールバックさせないため、適用有無は戻り値で受け取れること。
    """
    import src.models.trainer as trainer

    def _missing(name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(trainer, "load_model", _missing)
    df = _race("R1", [dict(frame_no=i, pred_prob=i / 10, pc_p_b=0.1) for i in (1, 2, 3)])
    out, applied = apply_position_calibration(df)
    assert applied is False
    assert list(out["pred_prob"]) == list(df["pred_prob"])
    assert list(out["pred_prob_raw"]) == list(df["pred_prob"])
