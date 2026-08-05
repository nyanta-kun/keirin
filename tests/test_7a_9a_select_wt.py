"""strategy_wt.rank_7a_daily_select/rank_9a_daily_select（S7/S9の境界ランク7A/9A・2026-07-27導入）の純関数テスト。

2026-07-31: S7がmark3ゲートを撤廃したことに伴い、7Aも2ゲート化(axis_sum/entropyの
みで判定)した。9Aは変更なし（entropy/mark3の2ゲートのまま）。
"""
from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_7S_MARK3_OVERLAP_MAX, RANK_9S_ENTROPY_MAX,
    rank_7a_daily_select, rank_9a_daily_select,
)


def _cand(axis_sum=1.0, entropy=1.0, wt_overlap_n=0, mark3=0):
    return {"axis_sum": axis_sum, "entropy": entropy,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": mark3}


# ── rank_7a_daily_select（2026-07-31: axis_sum/entropyの2ゲートに変更） ──

def test_7a_all_gates_pass_is_excluded():
    """2ゲート全合格はS7本体の対象であり7Aには含まれない。"""
    c = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX, entropy=RANK_7S_ENTROPY_MAX)
    assert rank_7a_daily_select([c]) == []


def test_7a_axis_sum_only_fail_is_included():
    c = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, entropy=RANK_7S_ENTROPY_MAX)
    assert rank_7a_daily_select([c]) == [c]


def test_7a_entropy_only_fail_is_now_excluded():
    """【2026-08-05改定】entropy だけ不合格は 7SS へ分離したため 7A には含まれない。

    旧仕様（不合格ちょうど1つ）では 7A に入っていた。7A は
    「axis_sum だけ不合格」＝軸2車が堅い群のみを指す。
    詳細は strategy_wt.rank_7a_daily_select / RANK_7SS_STAKE のコメント参照。
    """
    c = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX, entropy=RANK_7S_ENTROPY_MAX + 0.1)
    assert rank_7a_daily_select([c]) == []


def test_7a_both_gates_fail_is_excluded():
    c = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, entropy=RANK_7S_ENTROPY_MAX + 0.1)
    assert rank_7a_daily_select([c]) == []


def test_7a_wt_overlap_two_or_none_excluded_even_if_one_gate_fails():
    c2 = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, wt_overlap_n=2)
    cn = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, wt_overlap_n=None)
    assert rank_7a_daily_select([c2, cn]) == []


def test_7a_mark3_no_longer_affects_selection():
    """2026-07-31撤廃: mark3の値・欠損はもはや7Aの判定に一切影響しない。"""
    ok = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, entropy=RANK_7S_ENTROPY_MAX, mark3=None)
    assert rank_7a_daily_select([ok]) == [ok]


def test_7a_mark3_only_fail_no_longer_qualifies_for_7a():
    """axis_sum/entropyが両方合格（旧仕様ならmark3のみ不合格で7A対象）の場合、
    2ゲート化後はS7本体の対象となり7Aには含まれない（新S7との重複防止）。"""
    c = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX, entropy=RANK_7S_ENTROPY_MAX, mark3=RANK_7S_MARK3_OVERLAP_MAX + 1)
    assert rank_7a_daily_select([c]) == []


def test_7a_sorted_by_axis_sum_ascending():
    # axis_sum だけ不合格（entropy は合格）が 7A の対象。
    low = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.1, entropy=RANK_7S_ENTROPY_MAX)
    high = _cand(axis_sum=RANK_7S_AXIS_SUM_MAX + 0.5, entropy=RANK_7S_ENTROPY_MAX)
    assert rank_7a_daily_select([high, low]) == [low, high]


# ── rank_9a_daily_select（axis_sumゲートなし・entropy/mark3の2ゲートのみ） ──

def test_9a_all_gates_pass_is_excluded():
    c = _cand(entropy=RANK_9S_ENTROPY_MAX, mark3=RANK_7S_MARK3_OVERLAP_MAX)
    assert rank_9a_daily_select([c]) == []


def test_9a_entropy_only_fail_is_included():
    c = _cand(entropy=RANK_9S_ENTROPY_MAX + 0.1, mark3=RANK_7S_MARK3_OVERLAP_MAX)
    assert rank_9a_daily_select([c]) == [c]


def test_9a_mark3_only_fail_is_included():
    c = _cand(entropy=RANK_9S_ENTROPY_MAX, mark3=RANK_7S_MARK3_OVERLAP_MAX + 1)
    assert rank_9a_daily_select([c]) == [c]


def test_9a_both_gates_fail_is_excluded():
    c = _cand(entropy=RANK_9S_ENTROPY_MAX + 0.1, mark3=RANK_7S_MARK3_OVERLAP_MAX + 1)
    assert rank_9a_daily_select([c]) == []


def test_9a_wt_overlap_two_or_none_excluded():
    c2 = _cand(entropy=RANK_9S_ENTROPY_MAX + 0.1, wt_overlap_n=2)
    cn = _cand(entropy=RANK_9S_ENTROPY_MAX + 0.1, wt_overlap_n=None)
    assert rank_9a_daily_select([c2, cn]) == []
