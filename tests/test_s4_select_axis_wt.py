"""strategy_wt.s4_select_axis（S4・単勝×複勝指数トップ3重なり軸選定）の純関数テスト。

2026-07-21 導入: 単勝指数(pred_win)上位3 ∩ 複勝指数(pred_top3)上位3 の重なり車から
軸2車を選ぶ。重なり>=2ならpred_top3上位2、重なり==1ならその1車+残りのpred_top3最上位。
"""
from src.strategy_wt import s4_select_axis


def test_overlap_of_three_picks_top2_by_top3():
    # win上位3 = {1,2,3}, top3上位3 = {1,2,3}（完全一致・重なり3車）
    win_probs = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10, 6: 0.0, 7: 0.0}
    top3_probs = {1: 0.90, 2: 0.85, 3: 0.60, 4: 0.50, 5: 0.40, 6: 0.30, 7: 0.20}
    result = s4_select_axis(win_probs, top3_probs)
    assert result is not None
    axis1, axis2, axis_sum = result
    assert axis1 == 1  # top3_probs最上位
    assert axis2 == 2  # top3_probs2位
    assert abs(axis_sum - (0.90 + 0.85)) < 1e-9


def test_overlap_of_two_picks_both_as_axis():
    # win上位3 = {1,2,4}, top3上位3 = {1,2,3} → 重なりは{1,2}
    win_probs = {1: 0.30, 2: 0.25, 3: 0.05, 4: 0.20, 5: 0.10, 6: 0.05, 7: 0.05}
    top3_probs = {1: 0.90, 2: 0.60, 3: 0.55, 4: 0.10, 5: 0.10, 6: 0.05, 7: 0.05}
    result = s4_select_axis(win_probs, top3_probs)
    assert result is not None
    axis1, axis2, axis_sum = result
    assert {axis1, axis2} == {1, 2}
    assert abs(axis_sum - (0.90 + 0.60)) < 1e-9


def test_overlap_of_one_uses_top3_runner_up_as_second_axis():
    # win上位3 = {1,4,5}, top3上位3 = {1,2,3} → 重なりは{1}のみ
    win_probs = {1: 0.30, 2: 0.05, 3: 0.05, 4: 0.25, 5: 0.20, 6: 0.10, 7: 0.05}
    top3_probs = {1: 0.90, 2: 0.60, 3: 0.55, 4: 0.10, 5: 0.05, 6: 0.05, 7: 0.05}
    result = s4_select_axis(win_probs, top3_probs)
    assert result is not None
    axis1, axis2, axis_sum = result
    assert axis1 == 1  # 重なりの唯一車
    assert axis2 == 2  # 残りでpred_top3最上位
    assert abs(axis_sum - (0.90 + 0.60)) < 1e-9


def test_returns_none_when_no_overlap():
    # win上位3 = {4,5,6}, top3上位3 = {1,2,3} → 重なりなし
    win_probs = {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.30, 5: 0.25, 6: 0.20, 7: 0.10}
    top3_probs = {1: 0.90, 2: 0.60, 3: 0.55, 4: 0.10, 5: 0.05, 6: 0.05, 7: 0.05}
    assert s4_select_axis(win_probs, top3_probs) is None


def test_returns_none_when_insufficient_entries():
    assert s4_select_axis({1: 0.5, 2: 0.3}, {1: 0.5, 2: 0.3}) is None


def test_returns_none_when_empty():
    assert s4_select_axis({}, {1: 0.5, 2: 0.3, 3: 0.2}) is None
    assert s4_select_axis({1: 0.5, 2: 0.3, 3: 0.2}, {}) is None
