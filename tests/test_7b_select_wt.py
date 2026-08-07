"""7B（◎◯一致 × 順序も一致 × 準決勝・三連複3点）の選出・相手絞りの回帰テスト。

⚠️ **2026-08-05 に 7B は中身を全面的に入れ替えた。**
  旧7B（2026-08-03〜08-05）: overlap==2 ∧ order**不一致**。全窓で控除率75%を越えず廃止
  新7B（本テストの対象）  : overlap==2 ∧ order**一致** ∧ race_type=="準決勝"
**order_disagree の向きが真逆**なので、picks_history の旧7B行と成績を合算してはいけない。

新7Bは7車レースの被覆マップで見つけた「現行ランクがどこも触っていない空白」のうち、
3窓（掃引・確認・未使用期間）を生き残った唯一の定義。ROI 82.8 / 83.0 / 81.7% と
**水準が±0.7ptに収まる**ことが採用根拠。

本テストが守る不変条件:
  1. 7SS/7S/7A（wt_overlap_n∈{0,1}）と 7B（==2）が母集団として完全に排他であること
     ＝同一レースが両方に選出されない（重複計上・二重入稿の防止／7Bは純増）
  2. `race_type` が**完全一致**であること。「チャレンジ準決勝」「ガールズ準決勝」は
     別値として実在し、部分一致にすると未検証の母集団が約30%混入する
  3. order_disagree が True/None のときフェイルセーフで除外されること
  4. race_type 欠損時も推奨を増やさない側に倒すこと
  5. 相手絞りが「△を除外してから上位K車」の順序で行われること
     （先に上位K車を取ってから△を除くと点数が減り、設計と実績が乖離する）
  6. judge_rank_7b が朝の候補JSONではなく発走前の盤面から相手を再計算すること

DBアクセスなし（純関数のみ）。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import src.strategy_wt as sw
from notify_prerace_wt import judge_rank_7b


def _cand(rk, overlap, disagree, entropy=1.80, axis_sum=1.4, race_type="準決勝"):
    return {"race_key": rk, "wt_overlap_n": overlap, "order_disagree": disagree,
            "entropy": entropy, "axis_sum": axis_sum, "race_type": race_type}


# ── 1. 選出ゲート ────────────────────────────────────────────────


def test_selects_only_overlap2_with_order_agreement_in_semifinal():
    """新7B = overlap==2 ∧ order**一致** ∧ 準決勝（2026-08-05 定義入替）。

    ⚠️ 旧7B（〜2026-08-05）は order**不一致**を取っていた＝向きが真逆。
    """
    cands = [
        _cand("ok", 2, False),                        # ◎◯一致 ∧ 順序一致 ∧ 準決勝
        _cand("order_disagree", 2, True),             # 旧7Bの母集団（今は対象外）
        _cand("overlap1", 1, False),                  # 7SS/7S/7A の母集団
        _cand("overlap0", 0, False),
        _cand("mark_missing", 2, None),               # ◎欠損＝判定不能
        _cand("heats", 2, False, race_type="予選"),    # 準決勝でない
    ]
    got = [c["race_key"] for c in sw.rank_7b_select_pool(cands)]
    assert got == ["ok"]


def test_race_type_must_match_exactly_not_substring():
    """「チャレンジ準決勝」「ガールズ準決勝」を拾わないこと。

    ⚠️ これらは `race_type` の**別値として実在する**（掃引窓で
    チャレンジ準決勝 915件・ガールズ準決勝 16件）。検証は `== "準決勝"` の完全一致
    でしか行っておらず、`in` 判定にすると未検証の母集団が約30%混入して
    掃引窓ですら ROI 82.8→81.7% に薄まる。
    """
    for rt in ("チャレンジ準決勝", "ガールズ準決勝", "準決勝A", "準決"):
        assert sw.rank_7b_select_pool([_cand("x", 2, False, race_type=rt)]) == [], rt
    assert len(sw.rank_7b_select_pool([_cand("x", 2, False, race_type="準決勝")])) == 1


def test_missing_race_type_is_failsafe_excluded():
    """race_type 欠損（wt_races 未取込等）は推奨を増やさない側に倒す。"""
    assert sw.rank_7b_select_pool([_cand("x", 2, False, race_type=None)]) == []
    c = _cand("y", 2, False)
    del c["race_type"]
    assert sw.rank_7b_select_pool([c]) == []


def test_order_disagree_true_and_none_are_both_excluded():
    """`is False` 判定であること（`is not True` だと None が通ってしまう）。

    overlap==2 は ◎◯ が両方存在しないと成立しないため None とは構造的に
    両立しない（掃引窓 18,440件すべて False を実測確認）。だが overlap の定義が
    将来変わったときに黙って通さないためのフェイルセーフとして明示的に弾く。
    """
    assert sw.rank_7b_select_pool([_cand("x", 2, True)]) == []
    assert sw.rank_7b_select_pool([_cand("x", 2, None)]) == []


def test_mutually_exclusive_with_7ss_7s_and_7a():
    """同一候補集合に対し 7SS / 7S / 7A / 7B の選出が互いに素であること。

    7B は overlap==2、他は overlap∈{0,1} なので構造的に排他＝**純増**。
    """
    cands = []
    for i, (ov, dis) in enumerate([(0, False), (1, False), (2, False), (2, True), (1, True)]):
        cands.append(_cand(f"r{i}", ov, dis, entropy=1.70, axis_sum=1.2))
    s7 = {c["race_key"] for c in sw.rank_7s_daily_select(cands)}
    s7a = {c["race_key"] for c in sw.rank_7a_daily_select(cands)}
    s7ss = {c["race_key"] for c in sw.rank_7ss_daily_select(cands)}
    s7b = {c["race_key"] for c in sw.rank_7b_select_pool(cands)}
    assert s7b, "選出されること（素通りテスト化の防止）"
    for other in (s7, s7a, s7ss):
        assert other.isdisjoint(s7b)


def test_selection_sorted_by_entropy_ascending():
    cands = [_cand("hi", 2, False, entropy=1.90), _cand("lo", 2, False, entropy=1.60)]
    assert [c["race_key"] for c in sw.rank_7b_select_pool(cands)] == ["lo", "hi"]


# ── 2. 相手絞り ──────────────────────────────────────────────────

PROBS = {2: 0.50, 3: 0.40, 4: 0.30, 5: 0.20, 6: 0.10}


def test_select_legs_drops_ana_before_taking_topk():
    """△を除外してから上位K車を取る（順序が逆だと2点に痩せる）。"""
    # △=3（確率2位）。除外してから上位3 → 2,4,5
    assert sw.rank_7b_select_legs([2, 3, 4, 5, 6], PROBS, wt_ana=3) == [2, 4, 5]
    assert len(sw.rank_7b_select_legs([2, 3, 4, 5, 6], PROBS, wt_ana=3)) == sw.RANK_7B_LEGS


def test_select_legs_without_ana_takes_plain_topk():
    assert sw.rank_7b_select_legs([2, 3, 4, 5, 6], PROBS, wt_ana=None) == [2, 3, 4]


def test_select_legs_when_ana_outside_top_k():
    """△が元々上位K外なら結果は△なしの場合と同じ（余計に削らない）。"""
    assert sw.rank_7b_select_legs([2, 3, 4, 5, 6], PROBS, wt_ana=6) == [2, 3, 4]


def test_select_legs_shrinks_when_others_are_few():
    """欠車等で相手が少ない場合は取れるだけ返す（例外にしない）。"""
    assert sw.rank_7b_select_legs([2, 3], PROBS, wt_ana=3) == [2]


# ── 3. 順序不一致判定 ────────────────────────────────────────────


def test_order_disagree_basic():
    win = {1: 0.40, 2: 0.35, 3: 0.10}
    assert sw.rank_7b_order_disagree(win, wt_honmei=2) is True   # モデル1位=1 ≠ ◎2
    assert sw.rank_7b_order_disagree(win, wt_honmei=1) is False  # 一致
    assert sw.rank_7b_order_disagree(win, wt_honmei=None) is None
    assert sw.rank_7b_order_disagree({}, wt_honmei=1) is None


# ── 4. 発走前ライブ判定 ──────────────────────────────────────────


def _full_trio(cars, odds=5.0):
    return {frozenset(c): odds for c in combinations(cars, 3)}


def _live_cand(**kw):
    base = {"axis1": 1, "axis2": 7, "wt_ana": 5,
            "top3_probs": {"1": .9, "7": .8, "2": .7, "3": .6, "5": .55, "4": .4, "6": .3}}
    base.update(kw)
    return base


def test_judge_buys_three_legs_excluding_ana():
    d, det = judge_rank_7b(_live_cand(), _full_trio([1, 2, 3, 4, 5, 6, 7]))
    assert d == "buy"
    # △=5 を除いた相手上位3車 = 2,3,4
    assert det["combos"] == ["1-2-7", "1-3-7", "1-4-7"]
    assert det["dropped_ana"] == 5


def test_judge_recomputes_legs_from_board_not_from_json():
    """朝の legs_7b をそのまま使わず盤面から再計算すること。

    legs_7b に故意に誤った値を入れても、top3_probs があれば盤面基準の
    正しい3点が選ばれる。
    """
    cand = _live_cand(legs_7b=[6, 6, 6])
    _, det = judge_rank_7b(cand, _full_trio([1, 2, 3, 4, 5, 6, 7]))
    assert det["combos"] == ["1-2-7", "1-3-7", "1-4-7"]


def test_judge_falls_back_to_legs_7b_when_probs_missing():
    """旧形式（top3_probs なし）の候補は legs_7b へフォールバックする。"""
    cand = {"axis1": 1, "axis2": 7, "wt_ana": 5, "legs_7b": [2, 4, 6]}
    d, det = judge_rank_7b(cand, _full_trio([1, 2, 3, 4, 5, 6, 7]))
    assert d == "buy"
    assert det["combos"] == ["1-2-7", "1-4-7", "1-6-7"]


def test_judge_skips_on_scratched_board():
    d, det = judge_rank_7b(_live_cand(), _full_trio([1, 2, 3, 4, 5, 6]))
    assert d == "skip"
    assert "欠車" in det["skip_reason"]


def test_judge_skips_when_axis_absent_from_board():
    cand = _live_cand(axis1=9)
    d, det = judge_rank_7b(cand, _full_trio([1, 2, 3, 4, 5, 6, 7]))
    assert d == "skip"
    assert det["skip_reason"] == "軸が盤面に不在"


def test_judge_returns_unknown_without_odds():
    assert judge_rank_7b(_live_cand(), {})[0] == "不明"


def test_judge_ignores_placeholder_odds():
    """未確定プレースホルダ(9999.9等)は盤面構築から除外される。"""
    trio = {k: 9999.9 for k in _full_trio([1, 2, 3, 4, 5, 6, 7])}
    assert judge_rank_7b(_live_cand(), trio)[0] == "不明"


# ── 廃止（2026-08-07・ユーザー判断）────────────────────────────────

def test_7bは稼働しており7Cより後ろで入稿される():
    """2026-08-07: 一度は廃止したが、同日中に**「7C の下に置き、重複は 7C・
    独自レースだけ 7B」**へユーザー判断が変わった。

    7C との重複では 7C が実質的中率で上回る（39.0% vs 31.6%）一方、
    7B は 7C が拾わないレースを 3.14件/日 持つ。優先順位だけでこれを実現する。
    """
    from scripts.netkeirin_submit_wt import RANK_ORDER

    assert sw.RANK_7B_STOPPED is False
    ok = _cand("ok", 2, False)
    assert sw.rank_7b_daily_select([ok]) != []
    # netkeirin は1レース1商品。**7C より後ろ**に置くことで重複は 7C が取る。
    assert RANK_ORDER.index("7B") > RANK_ORDER.index("7C")
