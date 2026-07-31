"""strategy_wt.rank_7s_gate_label（S7表示ランク分岐）の純関数テスト。

サブランクは2段階で廃止された:
- 2026-07-23導入の"SS+"（軸2車の級班に各グレード最上位を含まないSS内訳）は
  サンプル数不足のため2026-07-27にユーザー判断で廃止しSSへ統合。
- SS自体（7SS/9SS・重なり0）も2024-09以降の発生頻度が月0〜4件まで激減した
  ため2026-07-31に廃止しSへ統合（commit e994758）。

結果として現在は重なり0/1がともに"S"を返す。axis1_class/axis2_classは
廃止後もコール側互換のため引数として残しているが、結果には影響しない。
"""
from src.strategy_wt import rank_7s_gate_label


def test_overlap_zero_is_s_regardless_of_class():
    """重なり0は2026-07-31にSSからSへ統合された（級班によらず一律）。"""
    assert rank_7s_gate_label(0, "A2", "A3") == "S"
    assert rank_7s_gate_label(0, "S1", "A3") == "S"
    assert rank_7s_gate_label(0, "A3", "A1") == "S"
    assert rank_7s_gate_label(0, "S1", "A1") == "S"


def test_overlap_zero_without_class_info_is_s():
    assert rank_7s_gate_label(0, None, None) == "S"
    assert rank_7s_gate_label(0) == "S"


def test_ss_and_ss_plus_are_never_returned():
    """廃止済みサブランクが復活していないことを保証する回帰テスト。"""
    for overlap in (0, 1, 2, None):
        for classes in ((None, None), ("S1", "A1"), ("A2", "A3")):
            assert rank_7s_gate_label(overlap, *classes) not in ("SS", "SS+")


def test_overlap_one_is_s_regardless_of_class():
    assert rank_7s_gate_label(1, "S1", "A1") == "S"
    assert rank_7s_gate_label(1, "A2", "A3") == "S"


def test_overlap_two_or_none_is_none():
    assert rank_7s_gate_label(2, "A2", "A3") is None
    assert rank_7s_gate_label(None, "A2", "A3") is None
