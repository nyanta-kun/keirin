"""strategy_wt.s4_gate_label（S4表示ランク分岐・2026-07-23 SS+新設）の純関数テスト。"""
from src.strategy_wt import s4_gate_label


def test_overlap_zero_with_no_top_class_is_ssplus():
    assert s4_gate_label(0, "A2", "A3") == "SS+"


def test_overlap_zero_with_axis1_top_class_is_ss():
    assert s4_gate_label(0, "S1", "A3") == "SS"


def test_overlap_zero_with_axis2_top_class_is_ss():
    assert s4_gate_label(0, "A3", "A1") == "SS"


def test_overlap_zero_with_both_top_class_is_ss():
    assert s4_gate_label(0, "S1", "A1") == "SS"


def test_overlap_zero_without_class_info_falls_back_to_ss():
    assert s4_gate_label(0, None, None) == "SS"
    assert s4_gate_label(0) == "SS"


def test_overlap_one_is_s_regardless_of_class():
    assert s4_gate_label(1, "S1", "A1") == "S"
    assert s4_gate_label(1, "A2", "A3") == "S"


def test_overlap_two_or_none_is_none():
    assert s4_gate_label(2, "A2", "A3") is None
    assert s4_gate_label(None, "A2", "A3") is None
