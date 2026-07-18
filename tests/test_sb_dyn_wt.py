"""add_sb_dyn_features_wt（レース単位S/B・上がりローリング4特徴）の純粋テスト。

history 引数注入で DB アクセスなしに検証する（point-in-time 保証・
closed="left" の当日除外・レース内相対化・新人/ラベル欠損の 0.0 補完）。
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT,
    SB_DYN_COLS_WT,
    add_sb_dyn_features_wt,
)


def _hist(rows: list[tuple]) -> pd.DataFrame:
    """(race_key, player_id, res_standing, res_back, final_half, race_date) から履歴を作る。"""
    return pd.DataFrame(rows, columns=[
        "race_key", "player_id", "res_standing", "res_back",
        "final_half", "race_date"])


def test_rates_and_fh_computed_from_past_races_only():
    """過去2走から4特徴が正しく計算され、当日のレースは窓に入らない。"""
    history = _hist([
        # 過去レース1（2026-04-01）: A が B取り・上がり最速（11.5 vs 中央値12.0）
        ("r1", "A", 1, 1, 11.5, "2026-04-01"),
        ("r1", "X", 0, 0, 12.0, "2026-04-01"),
        ("r1", "Y", 0, 0, 12.5, "2026-04-01"),
        # 過去レース2（2026-05-01）: A は B取りなし・上がり中央値+0.5
        ("r2", "A", 0, 0, 12.5, "2026-05-01"),
        ("r2", "X", 1, 1, 12.0, "2026-05-01"),
        ("r2", "Y", 0, 0, 11.5, "2026-05-01"),
        # 当日レース（2026-06-01）: 未確定（ラベル NaN）— 窓に入ってはいけない
        ("r3", "A", None, None, None, "2026-06-01"),
        ("r3", "X", None, None, None, "2026-06-01"),
    ])
    df = pd.DataFrame({
        "race_key": ["r3"], "player_id": ["A"], "race_date": ["2026-06-01"],
    })
    out = add_sb_dyn_features_wt(df, history=history)
    row = out.iloc[0]
    assert row["b_rate_90"] == pytest.approx(0.5)        # 2走中1回
    assert row["s_rate_90"] == pytest.approx(0.5)
    # fh_rel: r1 = 11.5-12.0 = -0.5, r2 = 12.5-12.0 = +0.5 → 平均 0.0
    assert row["fh_rel_90"] == pytest.approx(0.0)
    assert row["fh_best_rate_90"] == pytest.approx(0.5)  # r1 のみ最速


def test_current_day_excluded_closed_left():
    """当日にラベルが入っていても closed="left" で窓から除外される。"""
    history = _hist([
        ("r1", "A", 0, 0, 12.0, "2026-05-01"),
        ("r1", "X", 1, 1, 11.5, "2026-05-01"),
        # 当日（確定済みラベルあり＝再計算シナリオ）: B取り・最速でも窓に入らない
        ("r2", "A", 1, 1, 11.0, "2026-06-01"),
        ("r2", "X", 0, 0, 12.0, "2026-06-01"),
    ])
    df = pd.DataFrame({
        "race_key": ["r2"], "player_id": ["A"], "race_date": ["2026-06-01"],
    })
    out = add_sb_dyn_features_wt(df, history=history)
    row = out.iloc[0]
    assert row["b_rate_90"] == pytest.approx(0.0)   # 過去走 r1 のみ（B取りなし）
    assert row["fh_best_rate_90"] == pytest.approx(0.0)  # r1 は X が最速


def test_rookie_and_prelabel_default_zero():
    """履歴なし（新人）・ラベル欠損のみ（2024-01以前相当）は 0.0。"""
    history = _hist([
        # ラベル欠損の過去走のみ（バックフィル対象外期間の想定）
        ("r0", "B", None, None, None, "2026-04-01"),
    ])
    df = pd.DataFrame({
        "race_key": ["r9", "r9"], "player_id": ["B", "NEW"],
        "race_date": ["2026-06-01", "2026-06-01"],
    })
    out = add_sb_dyn_features_wt(df, history=history)
    for c in SB_DYN_COLS_WT:
        assert (out[c] == 0.0).all()


def test_backward_compat_without_required_columns():
    """player_id / race_date が無い df は既定値 0.0 で埋めて返す。"""
    df = pd.DataFrame({"race_key": ["r1"]})
    out = add_sb_dyn_features_wt(df)
    for c in SB_DYN_COLS_WT:
        assert (out[c] == 0.0).all()


def test_sb_dyn_cols_in_feature_cols():
    """4特徴が FEATURE_COLS_WT（48特徴化）に含まれる。"""
    for c in SB_DYN_COLS_WT:
        assert c in FEATURE_COLS_WT
    assert len(FEATURE_COLS_WT) == 48
