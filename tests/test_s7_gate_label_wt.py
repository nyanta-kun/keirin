"""strategy_wt.s7_gate_label（S7表示ランク分岐）の純関数テスト。

2026-07-23に導入した観察用サブランク"SS+"（軸2車の級班に各グレード最上位を
含まないSS内訳）は、サンプル数不足のため2026-07-27にユーザー判断で廃止し
SSへ統合した。axis1_class/axis2_classは廃止後もコール側互換のため引数として
残しているが、結果には影響しない。
"""
from src.strategy_wt import s7_gate_label


def test_overlap_zero_is_ss_regardless_of_class():
    assert s7_gate_label(0, "A2", "A3") == "SS"
    assert s7_gate_label(0, "S1", "A3") == "SS"
    assert s7_gate_label(0, "A3", "A1") == "SS"
    assert s7_gate_label(0, "S1", "A1") == "SS"


def test_overlap_zero_without_class_info_is_ss():
    assert s7_gate_label(0, None, None) == "SS"
    assert s7_gate_label(0) == "SS"


def test_overlap_one_is_s_regardless_of_class():
    assert s7_gate_label(1, "S1", "A1") == "S"
    assert s7_gate_label(1, "A2", "A3") == "S"


def test_overlap_two_or_none_is_none():
    assert s7_gate_label(2, "A2", "A3") is None
    assert s7_gate_label(None, "A2", "A3") is None
