"""notify_prerace_wt.judge_m（M・◎不一致×システム◎/ペーパートレード）の純関数テスト。

judge_m はライブ三連複盤面（{frozenset: odds}）とM候補ペア（朝確定済み1件）、
同一レースの U 判定記録（あれば）のみで判定する:
  ① 盤面7車（有効オッズ 0<ov<9000 の掲載車）— 欠車なら見送り
  ② 盤面min三連複オッズ(mto) >= U_MTO_MIN(4.3)（Uと同じ凍結値）
  ③ U優先の重複排除: {rk}#U が buy ∧ ペアが同一集合 → 見送り（skip_reason="U重複"）
  ④ 買い目 = 軸2車 + 残り5車 のうちオッズ >= U_LEG_MIN_ODDS(15.0) のみ。0点なら見送り

judge_u との相違点: 市場順位条件なし・ペア選定なし（朝に1件確定済み）・U重複排除あり。
"""
from itertools import combinations

import notify_prerace_wt as np_wt  # scripts/ は conftest で path 追加済


def _lookup(odds=20.0, overrides=None, cars=range(1, 8)):
    """全三連複組合せに一律オッズを張った盤面（overrides で個別上書き）。"""
    ov = overrides or {}
    return {frozenset(c): float(ov.get(frozenset(c), odds))
            for c in combinations(cars, 3)}


PAIR = {"m1": 4, "mate": 5}


def test_buy_basic():
    """盤面7車・mto充足・全目15倍以上 → buy 成立（市場順位条件はなし）。"""
    dec, det = np_wt.judge_m(PAIR, _lookup(20.0))
    assert dec == "buy"
    assert det["m1"] == 4
    assert det["mate"] == 5
    assert det["mto"] == 20.0
    assert det["combos"] == ["1-4-5", "2-4-5", "3-4-5", "4-5-6", "4-5-7"]
    assert len(det["leg_odds"]) == 5          # 全5目のオッズを記録
    assert det["skip_reason"] is None


def test_skip_board_not_7cars():
    """盤面6車（欠車発生）→ 見送り。"""
    dec, det = np_wt.judge_m(PAIR, _lookup(20.0, cars=range(1, 7)))
    assert dec == "skip"
    assert "欠車" in det["skip_reason"]


def test_skip_mto_below_threshold():
    """盤面min三連複オッズ < U_MTO_MIN(4.3) → 見送り。"""
    dec, det = np_wt.judge_m(PAIR, _lookup(4.0))
    assert dec == "skip"
    assert det["mto"] == 4.0
    assert "mto" in det["skip_reason"]


def test_skip_no_leg_above_15():
    """15倍以上の買い目が0点 → 見送り（mto=14.0≥4.3 は通過するが全目<15倍）。"""
    dec, det = np_wt.judge_m(PAIR, _lookup(14.0))
    assert dec == "skip"
    assert det["mto"] == 14.0
    assert det["combos"] == []
    assert "15倍以上の目なし" in det["skip_reason"]


def test_skip_u_duplicate_same_pair():
    """同一レースの U が buy かつ同一ペア集合（順序不問）→ U優先で M は見送り。"""
    u_dec = {"decision": "buy", "dark": 5, "mate": 4}  # {5,4} == {4,5}
    dec, det = np_wt.judge_m(PAIR, _lookup(20.0), u_decision=u_dec)
    assert dec == "skip"
    assert "U重複" in det["skip_reason"]


def test_buy_when_u_exists_with_different_pair():
    """U が buy でもペアが異なる集合なら M は独立に buy 成立。"""
    u_dec = {"decision": "buy", "dark": 1, "mate": 2}
    dec, det = np_wt.judge_m(PAIR, _lookup(20.0), u_decision=u_dec)
    assert dec == "buy"
    assert det["combos"] == ["1-4-5", "2-4-5", "3-4-5", "4-5-6", "4-5-7"]
    assert det["skip_reason"] is None


def test_buy_when_u_skip_even_same_pair():
    """U が skip 記録（buy でない）なら同一ペアでも重複排除しない → buy。"""
    u_dec = {"decision": "skip", "dark": 4, "mate": 5}
    dec, det = np_wt.judge_m(PAIR, _lookup(20.0), u_decision=u_dec)
    assert dec == "buy"


def test_unknown_when_no_board():
    """盤面（trio_lookup）なし → 不明（次分再試行・skip記録しない）。"""
    dec, det = np_wt.judge_m(PAIR, {})
    assert dec == "不明"
    assert det["skip_reason"] is None
