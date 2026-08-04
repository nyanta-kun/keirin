"""3ヘッド軸選定（2026-08-04 導入）のテスト。

  軸1 = pred_win 最上位
  軸2 = z(pred_prob) − RANK_AXIS2_BAD_WEIGHT × z(bad_prob) の最上位（軸1を除く）

bad_probs が未供給なら旧ロジック（win上位3 ∩ top3上位3 の重なり）へフォールバックする。
"""
import pytest

from src.strategy_wt import (
    RANK_AXIS2_BAD_WEIGHT, _race_zscore, rank_7s_select_axis,
)


def test_zscore_basic():
    z = _race_zscore({1: 1.0, 2: 2.0, 3: 3.0})
    assert z[2] == pytest.approx(0.0)
    assert z[1] < 0 < z[3]
    assert z[1] == pytest.approx(-z[3])


def test_zscore_all_same_is_zero():
    """全車同値のとき 0 を返す（ゼロ除算しない）。"""
    z = _race_zscore({1: 0.5, 2: 0.5, 3: 0.5})
    assert set(z.values()) == {0.0}


def test_zscore_empty():
    assert _race_zscore({}) == {}


def _probs(pw, p3, bad=None):
    return ({i + 1: v for i, v in enumerate(pw)},
            {i + 1: v for i, v in enumerate(p3)},
            {i + 1: v for i, v in enumerate(bad)} if bad else None)


def test_axis1_is_win_top_not_top3_top():
    """軸1は pred_win 最上位。pred_prob 最上位とズレていても pred_win を採る。"""
    pw, p3, bad = _probs(
        pw=[0.10, 0.40, 0.20, 0.10, 0.10, 0.05, 0.05],   # 2番が最上位
        p3=[0.80, 0.50, 0.60, 0.40, 0.30, 0.20, 0.10],   # 1番が最上位
        bad=[0.10] * 7)
    a1, a2, _ = rank_7s_select_axis(pw, p3, bad)
    assert a1 == 2


def test_axis2_penalised_by_bad_prob():
    """bad が高い車は軸2から外れる。同じ p3 でも bad の低い方が選ばれる。"""
    pw, p3, bad = _probs(
        pw=[0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        p3=[0.80, 0.60, 0.60, 0.10, 0.10, 0.10, 0.10],   # 2番と3番が同値
        bad=[0.10, 0.90, 0.10, 0.50, 0.50, 0.50, 0.50])  # 2番だけ大敗確率が高い
    a1, a2, _ = rank_7s_select_axis(pw, p3, bad)
    assert a1 == 1
    assert a2 == 3, "bad が高い2番ではなく3番が軸2になるべき"


def test_axis2_excludes_axis1():
    pw, p3, bad = _probs(
        pw=[0.90, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        p3=[0.90, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05],
        bad=[0.10] * 7)
    a1, a2, _ = rank_7s_select_axis(pw, p3, bad)
    assert a1 == 1 and a2 != 1


def test_axis_sum_is_top3_prob_sum():
    """axis_sum は従来どおり軸2車の pred_prob 合計（ゲート閾値の意味を変えない）。"""
    pw, p3, bad = _probs(
        pw=[0.40, 0.30, 0.10, 0.10, 0.10, 0.05, 0.05],
        p3=[0.70, 0.60, 0.30, 0.20, 0.10, 0.10, 0.05],
        bad=[0.10] * 7)
    a1, a2, asum = rank_7s_select_axis(pw, p3, bad)
    assert asum == pytest.approx(p3[a1] + p3[a2])


def test_bad_weight_zero_equals_p3_top():
    """bad_weight=0 なら軸2は単純に pred_prob 最上位（軸1を除く）。"""
    pw, p3, bad = _probs(
        pw=[0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        p3=[0.80, 0.60, 0.65, 0.10, 0.10, 0.10, 0.10],
        bad=[0.10, 0.90, 0.90, 0.10, 0.10, 0.10, 0.10])
    _, a2, _ = rank_7s_select_axis(pw, p3, bad, bad_weight=0.0)
    assert a2 == 3, "bad を無視すれば p3 最上位の3番"


def test_fallback_to_legacy_without_bad():
    """bad_probs 未供給なら旧ロジック（重なり方式）。"""
    pw = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.05, 5: 0.03, 6: 0.01, 7: 0.01}
    p3 = {1: 0.80, 2: 0.70, 3: 0.60, 4: 0.30, 5: 0.20, 6: 0.10, 7: 0.05}
    got = rank_7s_select_axis(pw, p3)
    assert got is not None
    a1, a2, _ = got
    # 重なり{1,2,3} が3車 → top3_probs 上位2 = 1,2
    assert (a1, a2) == (1, 2)


def test_legacy_returns_none_when_no_overlap():
    """旧ロジックは重なり0なら None（この挙動は維持する）。"""
    pw = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    p3 = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.9, 5: 0.8, 6: 0.7, 7: 0.1}
    assert rank_7s_select_axis(pw, p3) is None


def test_three_head_never_returns_none_on_valid_input():
    """3ヘッド版は重なりを要求しないため、有効な入力なら必ず軸が立つ。

    旧ロジックが None を返すケース（重なり0）でも選定できることが、
    件数が減らない理由のひとつ。
    """
    pw = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    p3 = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.9, 5: 0.8, 6: 0.7, 7: 0.1}
    bad = {i: 0.2 for i in range(1, 8)}
    got = rank_7s_select_axis(pw, p3, bad)
    assert got is not None
    a1, a2, _ = got
    assert a1 == 1 and a2 == 4


def test_default_weight_value():
    """既定の重みは掃引で選んだ 0.3（軸1には掛けない＝w1=0）。

    honest walk-forward 4窓（2025-07〜2026-07・約4,300推奨）で、的中・ROIとも
    4窓すべて改善し符号反転が無かった唯一の構成が w1=0.0 × w2=0.3。
    詳細は strategy_wt.RANK_AXIS2_BAD_WEIGHT のコメント参照。
    """
    assert RANK_AXIS2_BAD_WEIGHT == 0.3


def test_insufficient_data_returns_none():
    assert rank_7s_select_axis({}, {}) is None
    assert rank_7s_select_axis({1: 0.5}, {1: 0.5}, {1: 0.1}) is None
