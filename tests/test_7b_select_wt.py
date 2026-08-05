"""7B（◎◯一致だが順序・相手で不一致・三連複3点）の選出・相手絞りの回帰テスト。

7B は 2026-08-03 新設。7S/7A が構造的に枯渇（軸2車がWT公式印◎◯と完全一致する
レース＝wt_overlap_n==2 が母集団の8割近くを占めるようになった）したことへの増枠で、
「軸は市場と一致するが、順序（モデル1位≠◎）と相手（△を買い目から外す）で不一致」
という部分的不一致だけを拾う。

本テストが守る不変条件:
  1. 7S/7A（wt_overlap_n∈{0,1}）と 7B（==2）が母集団として完全に排他であること
     ＝同一レースが両方に選出されない（重複計上・二重入稿の防止）
  2. order_disagree が None（WT◎欠損で判定不能）のときフェイルセーフで除外されること
  3. 相手絞りが「△を除外してから上位K車」の順序で行われること
     （先に上位K車を取ってから△を除くと点数が減り、設計と実績が乖離する）
  4. judge_rank_7b が朝の候補JSONではなく発走前の盤面から相手を再計算すること

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

import pytest

import src.strategy_wt as sw
from notify_prerace_wt import judge_rank_7b


def _cand(rk, overlap, disagree, entropy=1.80, axis_sum=1.4):
    return {"race_key": rk, "wt_overlap_n": overlap, "order_disagree": disagree,
            "entropy": entropy, "axis_sum": axis_sum}


@pytest.fixture
def gen_enabled(monkeypatch):
    """候補生成の停止フラグを一時的に解除する（2026-08-05〜）。

    7B は 2026-08-05 に候補生成を停止した（`RANK_7B_GENERATION_STOPPED`）が、
    **判定ロジック自体は残す**方針のため、ロジックの回帰テストはフラグを外して行う。
    フラグを外さずに書くと「常に空」を検証するだけの**素通りテスト**になり、
    将来 False に戻したときにロジックの壊れを検出できない。
    """
    monkeypatch.setattr(sw, "RANK_7B_GENERATION_STOPPED", False)


# ── 0. 候補生成の停止（2026-08-05 ユーザー判断）────────────────────


def test_generation_is_stopped_by_default():
    """既定では候補が1件も出ないこと（停止フラグが効いていること）。

    ⚠️ このテストが落ちたら 7B が本番で再稼働している。
    停止理由は strategy_wt.RANK_7B_GENERATION_STOPPED のコメント参照。
    """
    assert sw.RANK_7B_GENERATION_STOPPED is True
    assert sw.rank_7b_daily_select([_cand("ok", 2, True)]) == []


def test_reusable_parts_survive_the_stop():
    """停止しても後継ランクが使う部品は残っていること。

    `rank_7b_select_legs`（△除外3点）は空白3×準決勝など後継候補でも使うため、
    7B の停止と一緒に消してはいけない。
    """
    assert sw.rank_7b_select_legs([2, 3, 4, 5, 6], PROBS, wt_ana=3) == [2, 4, 5]
    assert sw.RANK_7B_STAKE > 0


# ── 1. 選出ゲート（フラグを外してロジック本体を検証）──────────────


def test_selects_only_overlap2_with_order_disagreement(gen_enabled):
    cands = [
        _cand("ok", 2, True),
        _cand("consensus", 2, False),      # 順序も一致＝完全コンセンサス
        _cand("overlap1", 1, True),        # 7S/7A の母集団
        _cand("overlap0", 0, True),
        _cand("mark_missing", 2, None),    # ◎欠損＝判定不能
    ]
    got = [c["race_key"] for c in sw.rank_7b_daily_select(cands)]
    assert got == ["ok"]


def test_none_order_disagree_is_failsafe_excluded(gen_enabled):
    """order_disagree=None（◎欠損）は「不一致でない」として扱い除外する。

    True 以外を全て弾く実装であること（`is True` 判定）。truthy 判定だと
    None が False 扱いになるのは同じだが、将来 0/1 等が入った場合に
    意図せず通るのを防ぐ。
    """
    assert sw.rank_7b_daily_select([_cand("x", 2, None)]) == []
    assert sw.rank_7b_daily_select([_cand("x", 2, False)]) == []


def test_mutually_exclusive_with_7s_and_7a(gen_enabled):
    """同一候補集合に対し 7S / 7A / 7B の選出結果が互いに素であること。"""
    cands = []
    for i, (ov, dis) in enumerate([(0, True), (1, True), (2, True), (2, False), (1, False)]):
        c = _cand(f"r{i}", ov, dis, entropy=1.70, axis_sum=1.2)
        cands.append(c)
    s7 = {c["race_key"] for c in sw.rank_7s_daily_select(cands)}
    s7a = {c["race_key"] for c in sw.rank_7a_daily_select(cands)}
    s7b = {c["race_key"] for c in sw.rank_7b_daily_select(cands)}
    assert s7b, "フラグ解除時は選出されること（素通りテスト化の防止）"
    assert s7.isdisjoint(s7b)
    assert s7a.isdisjoint(s7b)
    assert s7.isdisjoint(s7a)


def test_selection_sorted_by_entropy_ascending(gen_enabled):
    cands = [_cand("hi", 2, True, entropy=1.90), _cand("lo", 2, True, entropy=1.60)]
    assert [c["race_key"] for c in sw.rank_7b_daily_select(cands)] == ["lo", "hi"]


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
