"""strategy_wt.s7a_daily_select/s9a_daily_select（S7/S9の境界ランク7A/9A・2026-07-27導入）の純関数テスト。"""
from src.strategy_wt import (
    S7_AXIS_SUM_MAX, S7_ENTROPY_MAX, S7_MARK3_OVERLAP_MAX, S9_ENTROPY_MAX,
    s7a_daily_select, s9a_daily_select,
)


def _cand(axis_sum=1.0, entropy=1.0, wt_overlap_n=0, mark3=0):
    return {"axis_sum": axis_sum, "entropy": entropy,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": mark3}


# ── s7a_daily_select ──

def test_7a_all_gates_pass_is_excluded():
    """3ゲート全合格はS7本体の対象であり7Aには含まれない。"""
    c = _cand(axis_sum=S7_AXIS_SUM_MAX, entropy=S7_ENTROPY_MAX, mark3=S7_MARK3_OVERLAP_MAX)
    assert s7a_daily_select([c]) == []


def test_7a_axis_sum_only_fail_is_included():
    c = _cand(axis_sum=S7_AXIS_SUM_MAX + 0.1, entropy=S7_ENTROPY_MAX, mark3=S7_MARK3_OVERLAP_MAX)
    assert s7a_daily_select([c]) == [c]


def test_7a_entropy_only_fail_is_included():
    c = _cand(axis_sum=S7_AXIS_SUM_MAX, entropy=S7_ENTROPY_MAX + 0.1, mark3=S7_MARK3_OVERLAP_MAX)
    assert s7a_daily_select([c]) == [c]


def test_7a_mark3_only_fail_is_included():
    c = _cand(axis_sum=S7_AXIS_SUM_MAX, entropy=S7_ENTROPY_MAX, mark3=S7_MARK3_OVERLAP_MAX + 1)
    assert s7a_daily_select([c]) == [c]


def test_7a_two_gates_fail_is_excluded():
    c = _cand(axis_sum=S7_AXIS_SUM_MAX + 0.1, entropy=S7_ENTROPY_MAX + 0.1, mark3=S7_MARK3_OVERLAP_MAX)
    assert s7a_daily_select([c]) == []


def test_7a_wt_overlap_two_or_none_excluded_even_if_one_gate_fails():
    c2 = _cand(axis_sum=S7_AXIS_SUM_MAX + 0.1, wt_overlap_n=2)
    cn = _cand(axis_sum=S7_AXIS_SUM_MAX + 0.1, wt_overlap_n=None)
    assert s7a_daily_select([c2, cn]) == []


def test_7a_mark3_missing_excluded():
    c = _cand(axis_sum=S7_AXIS_SUM_MAX + 0.1, mark3=None)
    assert s7a_daily_select([c]) == []


def test_7a_sorted_by_axis_sum_ascending():
    low = _cand(axis_sum=0.5, entropy=S7_ENTROPY_MAX + 0.1)
    high = _cand(axis_sum=1.2, entropy=S7_ENTROPY_MAX + 0.1)
    assert s7a_daily_select([high, low]) == [low, high]


# ── s9a_daily_select（axis_sumゲートなし・entropy/mark3の2ゲートのみ） ──

def test_9a_all_gates_pass_is_excluded():
    c = _cand(entropy=S9_ENTROPY_MAX, mark3=S7_MARK3_OVERLAP_MAX)
    assert s9a_daily_select([c]) == []


def test_9a_entropy_only_fail_is_included():
    c = _cand(entropy=S9_ENTROPY_MAX + 0.1, mark3=S7_MARK3_OVERLAP_MAX)
    assert s9a_daily_select([c]) == [c]


def test_9a_mark3_only_fail_is_included():
    c = _cand(entropy=S9_ENTROPY_MAX, mark3=S7_MARK3_OVERLAP_MAX + 1)
    assert s9a_daily_select([c]) == [c]


def test_9a_both_gates_fail_is_excluded():
    c = _cand(entropy=S9_ENTROPY_MAX + 0.1, mark3=S7_MARK3_OVERLAP_MAX + 1)
    assert s9a_daily_select([c]) == []


def test_9a_wt_overlap_two_or_none_excluded():
    c2 = _cand(entropy=S9_ENTROPY_MAX + 0.1, wt_overlap_n=2)
    cn = _cand(entropy=S9_ENTROPY_MAX + 0.1, wt_overlap_n=None)
    assert s9a_daily_select([c2, cn]) == []
