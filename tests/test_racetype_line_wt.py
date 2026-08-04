"""race_type フラグ・ライン実力特徴（2026-08-04 追加・48→60特徴）のテスト。

採用根拠は scripts/exp_racetype_field_ab.py の A/B（2窓×5seed）:
    +rt_line: ΔAUC +0.00302/+0.00333・Δ1位3着内 +0.39pt/+0.19pt
    （参考: 2026-08-03 に採用した隊列推定位置は ΔAUC +0.00303 で同等）
"""
import pandas as pd
import pytest

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, LINE_STRENGTH_COLS_WT, RACE_TYPE_COLS_WT,
    add_line_strength_features_wt, add_race_type_features_wt,
)


def test_cols_in_feature_cols():
    for c in RACE_TYPE_COLS_WT + LINE_STRENGTH_COLS_WT:
        assert c in FEATURE_COLS_WT


@pytest.mark.parametrize("race_type,expected", [
    ("決勝", {"rt_is_final"}),
    ("準決勝", {"rt_is_semifinal"}),
    ("チャレンジ決勝", {"rt_is_final"}),
    ("ガールズ予選(第１走)", {"rt_is_heat"}),
    ("初特選", {"rt_is_tokusen", "rt_is_hatsu"}),
    ("特選", {"rt_is_tokusen"}),
    ("選抜", {"rt_is_senbatsu"}),
    ("一般", {"rt_is_ippan"}),
    ("特一般", {"rt_is_ippan"}),
    # 学習データに無い種別でもフラグが分解して立つ（序数エンコードにしない理由）
    ("ヤンググランプリ", set()),
])
def test_race_type_flags(race_type, expected):
    df = pd.DataFrame({"race_type": [race_type]})
    out = add_race_type_features_wt(df)
    got = {c for c in RACE_TYPE_COLS_WT if out.iloc[0][c] == 1}
    assert got == expected, f"{race_type}: {got} != {expected}"


def test_race_type_semifinal_is_not_final():
    """「準決勝」は「決勝」を含む文字列だが rt_is_final を立ててはいけない。"""
    out = add_race_type_features_wt(pd.DataFrame({"race_type": ["準決勝"]}))
    assert out.iloc[0]["rt_is_final"] == 0
    assert out.iloc[0]["rt_is_semifinal"] == 1


def test_race_type_missing_column_is_safe():
    """race_type 列が無い/NaN でも例外を出さず全フラグ0で通す（旧データ互換）。"""
    out = add_race_type_features_wt(pd.DataFrame({"dummy": [1]}))
    for c in RACE_TYPE_COLS_WT:
        assert out.iloc[0][c] == 0
    out2 = add_race_type_features_wt(pd.DataFrame({"race_type": [None]}))
    for c in RACE_TYPE_COLS_WT:
        assert out2.iloc[0][c] == 0


def _sample_race() -> pd.DataFrame:
    """3車ライン(合計270) + 2車ライン(130) + 単騎50 + 単騎40 の7車レース。"""
    return pd.DataFrame({
        "race_key": ["r1"] * 7,
        "frame_no": [1, 2, 3, 4, 5, 6, 7],
        "race_point": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
        "line_group": [1, 1, 1, 2, 2, None, None],
    })


def test_line_strength_values():
    out = add_line_strength_features_wt(_sample_race()).set_index("frame_no")
    # 3車ライン
    assert out.loc[1, "line_rp_sum"] == pytest.approx(270.0)
    assert out.loc[1, "line_rp_max"] == pytest.approx(100.0)
    assert out.loc[1, "line_rp_mean"] == pytest.approx(90.0)
    assert out.loc[1, "line_rank_by_rp"] == 0
    assert out.loc[1, "line_rp_gap_top"] == pytest.approx(0.0)
    # 2車ライン
    assert out.loc[4, "line_rp_sum"] == pytest.approx(130.0)
    assert out.loc[4, "line_rank_by_rp"] == 1
    assert out.loc[4, "line_rp_gap_top"] == pytest.approx(140.0)
    # 単騎は1車ラインとして個別に扱う（line_group 欠損でも合算されない）
    assert out.loc[6, "line_rp_sum"] == pytest.approx(50.0)
    assert out.loc[7, "line_rp_sum"] == pytest.approx(40.0)
    assert out.loc[6, "line_rank_by_rp"] == 2
    assert out.loc[7, "line_rank_by_rp"] == 3


def test_line_strength_varies_within_race():
    """レース内で値が変わること（＝軸選定に効きうる）を保証する回帰テスト。

    同時に検証した rp_mean/rp_std 等の「レース単位で全車同値」の特徴は
    1位3着内率を改善せず不採用になった。ここが両者の本質的な違い。
    """
    out = add_line_strength_features_wt(_sample_race())
    assert out["line_rp_gap_top"].nunique() > 1
    assert out["line_rp_sum"].nunique() > 1


def test_line_strength_no_nan():
    out = add_line_strength_features_wt(_sample_race())
    assert out[LINE_STRENGTH_COLS_WT].isna().sum().sum() == 0
