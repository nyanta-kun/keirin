"""add_rp_trend_features_wt（競走得点トレンド4特徴）の純粋テスト。

history 引数注入で DB アクセスなしに検証する（point-in-time 保証・
closed="left" の当日除外・新人 0.0 補完・後方互換ガード）。
"""
import pandas as pd
import pytest

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT,
    RP_TREND_COLS_WT,
    add_rp_trend_features_wt,
)


def _hist(rows: list[tuple]) -> pd.DataFrame:
    """(player_id, race_point, race_date[, finish_order]) タプル列から履歴を作る。

    finish_order 省略時は 1（確定済み＝値は実得点）。None は未確定
    （wave-picks の AIスコア上書きが残存し得る行）を表す。
    """
    norm = [(r + (1,))[:4] for r in rows]
    return pd.DataFrame(
        norm, columns=["player_id", "race_point", "race_date", "finish_order"])


def test_rp_trend_positive_for_improving_negative_for_declining():
    """得点が上昇中の選手は rp_trend > 0、下降中は < 0。"""
    history = _hist([
        # 上昇中の選手 up: 40 → 45 → 55 → 60 → 65
        ("up", 40.0, "2025-11-01"),
        ("up", 45.0, "2025-12-01"),
        ("up", 55.0, "2026-02-01"),
        ("up", 60.0, "2026-03-01"),
        ("up", 65.0, "2026-04-01"),
        # 下降中の選手 down: 70 → 65 → 55 → 50 → 45
        ("down", 70.0, "2025-11-01"),
        ("down", 65.0, "2025-12-01"),
        ("down", 55.0, "2026-02-01"),
        ("down", 50.0, "2026-03-01"),
        ("down", 45.0, "2026-04-01"),
    ])
    df = pd.DataFrame({
        "player_id": ["up", "down"],
        "race_date": ["2026-04-01", "2026-04-01"],
        "race_point": [65.0, 45.0],
    })
    out = add_rp_trend_features_wt(df, history=history)

    # up: ma90(=[01-01,04-01) の 02-01, 03-01) = 57.5 / ma180 = 50.0
    row = out[out["player_id"] == "up"].iloc[0]
    assert row["rp_trend"] == pytest.approx(7.5)
    assert row["rp_prev_delta"] == pytest.approx(65.0 - 60.0)
    assert row["rp_delta_90"] == pytest.approx(65.0 - 57.5)
    assert row["rp_delta_180"] == pytest.approx(65.0 - 50.0)

    # down: ma90 = 52.5 / ma180 = 60.0（up の鏡像）
    row = out[out["player_id"] == "down"].iloc[0]
    assert row["rp_trend"] == pytest.approx(-7.5)
    assert row["rp_prev_delta"] == pytest.approx(45.0 - 50.0)
    assert row["rp_delta_90"] == pytest.approx(45.0 - 52.5)
    assert row["rp_delta_180"] == pytest.approx(45.0 - 60.0)


def test_rolling_excludes_current_day_closed_left():
    """当日の得点は rolling 窓に混入しない（closed="left"・同一日複数走含む）。

    当日 100 点が窓に混入すると平均が 50 を超えるため検出できる。
    """
    history = _hist([
        ("p1", 50.0, "2026-03-01"),
        # 当日は同一節で2走（得点は同値でなくても median 集約→窓からは除外）
        ("p1", 100.0, "2026-04-01"),
        ("p1", 100.0, "2026-04-01"),
    ])
    df = pd.DataFrame({
        "player_id": ["p1", "p1"],   # 同一日に複数出走の行が df 側にもあるケース
        "race_date": ["2026-04-01", "2026-04-01"],
        "race_point": [100.0, 100.0],
    })
    out = add_rp_trend_features_wt(df, history=history)

    # 窓は前日以前のみ＝ma90 = ma180 = 50.0（100 が混入すると 75 になる）
    assert out["rp_delta_90"].to_numpy() == pytest.approx(100.0 - 50.0)
    assert out["rp_delta_180"].to_numpy() == pytest.approx(100.0 - 50.0)
    # 前回値も前日以前（同一日は shift の対象外＝日次集約済）
    assert out["rp_prev_delta"].to_numpy() == pytest.approx(100.0 - 50.0)
    assert out["rp_trend"].to_numpy() == pytest.approx(0.0)


def test_no_history_player_gets_zero():
    """履歴が無い選手（新人・履歴初日）は全特徴 0.0。"""
    history = _hist([
        # rookie の履歴なし。veteran は当日行のみ（初出走＝過去窓が空）
        ("veteran", 55.0, "2026-04-01"),
    ])
    df = pd.DataFrame({
        "player_id": ["rookie", "veteran"],
        "race_date": ["2026-04-01", "2026-04-01"],
        "race_point": [48.0, 55.0],
    })
    out = add_rp_trend_features_wt(df, history=history)
    for c in RP_TREND_COLS_WT:
        assert (out[c] == 0.0).all(), c


def test_unconfirmed_history_rows_are_excluded_from_aggregation():
    """finish_order=NULL の履歴行（AIスコア汚染値）は rolling 平均・rp_prev に影響しない。

    汚染値 35.0 が集計に混入すると ma90 は (50+60+35)/3 = 48.33、
    rp_prev は NaN（→0.0 fill）になるため、期待値との差で混入を検出できる。
    """
    history = _hist([
        ("p1", 50.0, "2026-03-01", 1),
        ("p1", 60.0, "2026-03-20", 1),
        # 結果未再収集の汚染行（最新の実値 60 より後ろ＝rp_prev のスキップも検証）
        ("p1", 35.0, "2026-03-25", None),
        # 当日行（未確定）＝merge キーとしては残る
        ("p1", 70.0, "2026-04-01", None),
    ])
    df = pd.DataFrame({
        "player_id": ["p1"],
        "race_date": ["2026-04-01"],
        "race_point": [70.0],
    })
    out = add_rp_trend_features_wt(df, history=history)
    row = out.iloc[0]

    # 実値のみで ma90 = ma180 = (50+60)/2 = 55.0（汚染混入なら 48.33）
    assert row["rp_delta_90"] == pytest.approx(70.0 - 55.0)
    assert row["rp_delta_180"] == pytest.approx(70.0 - 55.0)
    # rp_prev は NaN 行を飛ばして直前の実値 60.0（汚染混入なら 35.0 → delta 35.0）
    assert row["rp_prev_delta"] == pytest.approx(70.0 - 60.0)
    assert row["rp_trend"] == pytest.approx(0.0)


def test_missing_columns_backward_compat():
    """player_id / race_date が無い df は既定値 0.0 で埋める（後方互換）。"""
    df = pd.DataFrame({"race_point": [50.0, 60.0]})
    out = add_rp_trend_features_wt(df, history=_hist([]))
    assert len(out) == 2
    for c in RP_TREND_COLS_WT:
        assert (out[c] == 0.0).all(), c


def test_feature_cols_contains_rp_trend():
    """FEATURE_COLS_WT に rp_trend 4特徴が含まれる（2026-07-18 の sb_dyn 4特徴追加で
    末尾は sb_dyn になったため位置ではなく包含で検証・48特徴）。"""
    for c in RP_TREND_COLS_WT:
        assert c in FEATURE_COLS_WT
    assert len(FEATURE_COLS_WT) == 48
