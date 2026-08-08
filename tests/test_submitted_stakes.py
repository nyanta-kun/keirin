"""採点の賭け金が「実際に入稿した額」になっていることの検査（2026-08-08）。

## 背景（実際に起きた実害）

入稿は点ごとの傾斜配分で金額を決めるのに、採点は**全点に同じ単価**を掛けていた。

立川3R(2026-08-08): 的中 `3=5=7` の入稿額 1,300円・三連複配当 1,250円/100円
→ 正しくは **16,250円**。記録は 10,000÷5点=2,000円で計算した **25,000円**。

⚠️ **ずれ方に方向性がある。** 傾斜配分は高オッズの点ほど薄く張るので、
**高配当が当たったときほど過大に記録される**。`bet_detail` を持つ全13的中の
実測で 記録 219,330 / 正しい 186,810 ＝ +17.4% 過大、
2日間サマリー ROI は 0.750 → 0.698（約5pt甘い）だった。
"""

from __future__ import annotations

import json

from src.submitted_stakes import resolve_payout, submitted_stakes

# 立川3R(2026-08-08) の実際の入稿記録。
_TACHIKAWA_3R = {
    "total": 10000,
    "source": "model",
    "lines": [
        {"bet_type": "3連複", "combo": "1=3=7", "stake": 3000, "odds": 5.6},
        {"bet_type": "3連複", "combo": "3=4=7", "stake": 3000, "odds": 2.5},
        {"bet_type": "3連複", "combo": "2=3=7", "stake": 1400, "odds": 45.0},
        {"bet_type": "3連複", "combo": "3=5=7", "stake": 1300, "odds": 11.2},
        {"bet_type": "3連複", "combo": "3=6=7", "stake": 1300, "odds": None},
    ],
}


class _Conn:
    """conn.execute(...).fetchone() だけを満たす最小のスタブ。"""

    def __init__(self, bet_detail):
        self._bd = bet_detail

    def execute(self, _sql, _params=None):
        conn = self

        class _Cur:
            def fetchone(self):
                return (json.dumps(conn._bd),) if conn._bd is not None else None

        return _Cur()


def test_uses_the_actual_submitted_stake_not_the_average():
    """立川3R の再現。均等割り(2,000)ではなく実際の 1,300 を使う。"""
    pay, bet = resolve_payout(
        _Conn(_TACHIKAWA_3R), "20260808_28_03", "7C",
        hit=True, winning_key=frozenset({3, 5, 7}),
        odds_payout=1250, fallback_stake=2000, n_combos=5,
    )

    assert pay == 16250, "1,250円/100円 × 1,300円 = 16,250円"
    assert pay != 25000, "均等割り 2,000円で計算してはいけない（実害のあった値）"
    assert bet == 10000


def test_overstatement_direction_is_reproduced_by_the_old_rule():
    """旧ルールは高配当の的中でこそ過大になる（回帰の向きを固定する）。"""
    old = 1250 * 2000 // 100   # 均等割り
    new = 1250 * 1300 // 100   # 実際の入稿額
    assert old > new
    assert old - new == 8750


def test_falls_back_when_no_submission_record():
    """入稿記録が無い（2026-08-07 以前・未入稿）ときは従来計算のまま。"""
    pay, bet = resolve_payout(
        _Conn(None), "20260701_28_03", "7C",
        hit=True, winning_key=frozenset({3, 5, 7}),
        odds_payout=1250, fallback_stake=2000, n_combos=5,
    )
    assert (pay, bet) == (25000, 10000)


def test_miss_still_reports_the_real_total_investment():
    """不的中でも投資額は入稿記録の合計を使う（端数の寄せで単価×点数と一致しない）。"""
    pay, bet = resolve_payout(
        _Conn(_TACHIKAWA_3R), "20260808_28_03", "7C",
        hit=False, winning_key=frozenset({1, 2, 3}),
        odds_payout=0, fallback_stake=2000, n_combos=5,
    )
    assert pay == 0
    assert bet == 10000


def test_falls_back_when_winning_combo_absent_from_submission():
    """入稿記録に的中点が無い（欠車で組み替わった等）ときは黙って0円にしない。"""
    pay, bet = resolve_payout(
        _Conn(_TACHIKAWA_3R), "20260808_28_03", "7C",
        hit=True, winning_key=frozenset({1, 2, 5}),
        odds_payout=900, fallback_stake=2000, n_combos=5,
    )
    assert pay == 900 * 2000 // 100


def test_trifecta_keeps_order_but_trio_does_not():
    """三連単は順序つき、三連複は順序なしで照合する。"""
    bd = {
        "total": 9600,
        "lines": [
            {"bet_type": "3連単", "combo": "5-3-8", "stake": 1600},
            {"bet_type": "3連複", "combo": "3=5=8", "stake": 2000},
        ],
    }
    stakes, total = submitted_stakes(_Conn(bd), "rk", "9H1")

    assert stakes[(5, 3, 8)] == 1600
    assert (3, 5, 8) not in stakes, "三連単は順序が違えば別の点"
    assert stakes[frozenset({3, 5, 8})] == 2000
    assert total == 9600
