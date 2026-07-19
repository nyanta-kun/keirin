"""strategy_wt.s1w_select / s1w_gate（S1新設計・win軸1着固定×相手2車選定）の純関数テスト。

2026-07-19 導入: win model(1着専用モデル)のレース内1位を軸に固定し、
3着内モデルで軸を除いた上位2頭を相手に選ぶ。
"""
from src.strategy_wt import S1W_TOP3_GAP_MIN, s1w_gate, s1w_select


def test_select_axis_is_win_model_top1():
    win_probs = {1: 0.5, 2: 0.3, 3: 0.2}
    top3_probs = {1: 0.4, 2: 0.35, 3: 0.25}
    result = s1w_select(win_probs, top3_probs)
    assert result is not None
    axis, p1, p2, top3_gap = result
    assert axis == 1  # win_probsの1位


def test_select_partners_exclude_axis_top3_ranked():
    win_probs = {1: 0.5, 2: 0.3, 3: 0.2}
    top3_probs = {1: 0.2, 2: 0.5, 3: 0.3}
    axis, p1, p2, top3_gap = s1w_select(win_probs, top3_probs)
    assert axis == 1
    assert p1 == 2  # 軸(1)を除いたtop3_probs上位
    assert p2 == 3
    assert abs(top3_gap - (0.5 - 0.3)) < 1e-9


def test_select_returns_none_when_insufficient_remainder():
    win_probs = {1: 0.6, 2: 0.4}
    top3_probs = {1: 0.6, 2: 0.4}
    assert s1w_select(win_probs, top3_probs) is None


def test_select_returns_none_when_empty():
    assert s1w_select({}, {1: 0.5}) is None
    assert s1w_select({1: 0.5}, {}) is None


def test_gate_passes_at_threshold():
    assert s1w_gate(S1W_TOP3_GAP_MIN) is True


def test_gate_fails_below_threshold():
    assert s1w_gate(S1W_TOP3_GAP_MIN - 0.01) is False


def test_gate_passes_above_threshold():
    assert s1w_gate(0.30) is True
