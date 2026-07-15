"""notify_prerace_wt.judge_u（U・波乱ライン連れ込み/ペーパートレード）の純関数テスト。

judge_u はライブ三連複盤面（{frozenset: odds}）とU候補ペアのみで判定する:
  ① 盤面7車（有効オッズ 0<ov<9000 の掲載車）— 欠車なら見送り
  ② 盤面min三連複オッズ(mto) >= U_MTO_MIN(4.3)
  ③ 穴の市場評価順位（Σ1/オッズ の降順）が 4〜7位
  ④ 複数ペアは「穴のモデル順位最小 → 車番最小」で1ペアに決定
  ⑤ 買い目 = 軸2車 + 残り5車 のうちオッズ >= U_LEG_MIN_ODDS(15.0) のみ。0点なら見送り
"""
from itertools import combinations

import notify_prerace_wt as np_wt  # scripts/ は conftest で path 追加済


def _lookup(odds=20.0, overrides=None, cars=range(1, 8)):
    """全三連複組合せに一律オッズを張った盤面（overrides で個別上書き）。

    一律オッズでは市場評価 q_i が全車同点になり、順位は車番タイブレークで
    1..7 位になる（車番 = 市場順位）。
    """
    ov = overrides or {}
    return {frozenset(c): float(ov.get(frozenset(c), odds))
            for c in combinations(cars, 3)}


PAIRS = [{"dark": 4, "dark_model_rank": 3, "mate": 5}]


def test_buy_basic():
    """盤面7車・mto充足・穴=市場4位・全目15倍以上 → buy 成立。"""
    dec, det = np_wt.judge_u(PAIRS, _lookup(20.0))
    assert dec == "buy"
    assert det["dark"] == 4
    assert det["mate"] == 5
    assert det["mkt_rank"] == 4
    assert det["mto"] == 20.0
    assert det["combos"] == ["1-4-5", "2-4-5", "3-4-5", "4-5-6", "4-5-7"]
    assert len(det["leg_odds"]) == 5          # 全5目のオッズを記録
    assert det["skip_reason"] is None


def test_skip_board_not_7cars():
    """盤面6車（欠車発生）→ 見送り。"""
    dec, det = np_wt.judge_u(PAIRS, _lookup(20.0, cars=range(1, 7)))
    assert dec == "skip"
    assert "欠車" in det["skip_reason"]


def test_skip_mto_below_threshold():
    """盤面min三連複オッズ < U_MTO_MIN(4.3) → 見送り。"""
    dec, det = np_wt.judge_u(PAIRS, _lookup(4.0))
    assert dec == "skip"
    assert det["mto"] == 4.0
    assert "mto" in det["skip_reason"]


def test_skip_dark_market_rank_top3():
    """穴の市場評価順位が3位以内 → 「穴」ではない → 見送り。"""
    pairs = [{"dark": 2, "dark_model_rank": 1, "mate": 3}]
    dec, det = np_wt.judge_u(pairs, _lookup(20.0))
    assert dec == "skip"
    assert "市場順位" in det["skip_reason"]


def test_skip_no_leg_above_15():
    """成立ペアはあるが 15倍以上の買い目が0点 → 見送り。

    1,2,3絡みの組は5倍（市場上位を1,2,3に固定）、4-7内の組は14倍。
    mto=5.0≥4.3・穴4は市場4位で通過するが、全買い目 {4,5,t} が15倍未満。
    """
    board = {}
    for c in combinations(range(1, 8), 3):
        k = frozenset(c)
        board[k] = 5.0 if (k & {1, 2, 3}) else 14.0
    dec, det = np_wt.judge_u(PAIRS, board)
    assert dec == "skip"
    assert det["mkt_rank"] == 4
    assert det["combos"] == []
    assert "15倍以上の目なし" in det["skip_reason"]


def test_multi_pair_priority_model_rank_then_frame():
    """複数ペア成立時: 穴のモデル順位最小 → 同点なら車番最小 の1ペアに決定。"""
    pairs = [
        {"dark": 6, "dark_model_rank": 2, "mate": 7},
        {"dark": 5, "dark_model_rank": 3, "mate": 6},
        {"dark": 4, "dark_model_rank": 2, "mate": 5},
    ]
    dec, det = np_wt.judge_u(pairs, _lookup(20.0))
    assert dec == "buy"
    # model_rank=2 が2件（dark 6 と 4）→ 車番最小の dark=4 を採用
    assert det["dark"] == 4
    assert det["mate"] == 5


def test_pair_filtered_by_market_rank_falls_back_to_valid_pair():
    """市場3位以内の穴ペアは除外され、4位以下の別ペアで成立する。"""
    pairs = [
        {"dark": 1, "dark_model_rank": 1, "mate": 2},  # 市場1位 → 対象外
        {"dark": 4, "dark_model_rank": 3, "mate": 5},
    ]
    dec, det = np_wt.judge_u(pairs, _lookup(20.0))
    assert dec == "buy"
    assert det["dark"] == 4


def test_unknown_when_no_board():
    """盤面（trio_lookup）なし → 不明（次分再試行・skip記録しない）。"""
    dec, det = np_wt.judge_u(PAIRS, {})
    assert dec == "不明"
    assert det["skip_reason"] is None
