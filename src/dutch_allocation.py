"""高配当ランク(7H1 / 9H1)向けのダッチ配分（2026-08-09 新設・STEP3 §2B）。

## なぜ高配当ランクにだけ入れるのか

的中率ランク(7C/7A/7S)では**目を切ってはいけない**。的中の 94〜97% は
三連複払戻 2,000円未満の「人気の目」が作っており、それを外すと的中率の源泉ごと
失う（仕様書 §2.2）。一方 7H1/9H1 は元々人気の目に依存しておらず、
的中の約4割が払戻 2,000円超・5% が1万円超。ここでは低オッズ目を切って
「当たれば必ず予算を上回る」形に寄せる方が素直に効く。

⚠️ `src/stake_allocation.py` の docstring にある「レース単位のゲートは入れない」は
**的中率ランクの傾斜配分についての結論**であって、本モジュールとは母集団が違う。
混同しないこと（あちらは人気の目を残す前提、こちらは切る前提）。

## 手順（仕様書 §2B）

1. 朝オッズ(`wt_odds_snapshot` morning)で **2.0倍未満の目を除外**
2. 残りから **Σ(1/オッズ) ≤ 1/1.3** を満たすまで**低オッズ目から切る**
3. `s_i = B * (1/o_i) / Σ(1/o)` で配分し、**1点上限 5,000円**・100円単位へ丸める
4. 購入可否は **①保証 1.3倍 ②最低 2.5倍 ③EV 1.3** の3条件。少点数(〜2点)でも
   成立すれば購入し、不成立なら**そのレースを見送る**

「保証」= 最も払戻が小さい目が当たったときの払戻 ÷ 総賭け金。Σ(1/o)≤1/1.3 は
全額を賭けきった理想配分での保証であり、**丸めと1点上限で目減りする**ので
実際の保証は配分後の実額から計算し直す（`dutch_min_return`）。
"""

from __future__ import annotations

BUDGET_DEFAULT = 10_000
UNIT_DEFAULT = 100
PER_POINT_CAP_DEFAULT = 5_000

ODDS_FLOOR = 2.0
"""この倍率未満の目は最初に捨てる（仕様書 §2B）。"""

MIN_RETURN = 1.3
"""目標保証倍率。Σ(1/o) ≤ 1/MIN_RETURN を満たすまで低オッズ目から切る。"""

MIN_ODDS = 2.5
"""購入条件②: 採用した目の最低オッズがこれ以上。"""

EV_MIN = 1.3
"""購入条件③: Σ(p_i * o_i) がこれ以上。"""


class DutchResult:
    """ダッチ配分の結果。

    buy が False のとき stakes は空で、reason に見送り理由が入る。
    """

    __slots__ = ("stakes", "buy", "reason", "min_return", "min_odds", "ev", "dropped")

    def __init__(
        self,
        stakes: dict,
        buy: bool,
        reason: str,
        min_return: float,
        min_odds: float,
        ev: float,
        dropped: list,
    ) -> None:
        self.stakes = stakes
        self.buy = buy
        self.reason = reason
        self.min_return = min_return
        self.min_odds = min_odds
        self.ev = ev
        self.dropped = dropped

    def as_dict(self) -> dict:
        """ログ・記録用の dict 表現を返す。"""
        return {
            "buy": self.buy,
            "reason": self.reason,
            "dutch_min_return": round(self.min_return, 4),
            "min_odds": round(self.min_odds, 2),
            "ev": round(self.ev, 4),
            "n_legs": len(self.stakes),
            "n_dropped": len(self.dropped),
            "total": sum(self.stakes.values()),
        }


def _select_legs(odds: dict, min_return: float) -> tuple[list, list]:
    """Σ(1/o) ≤ 1/min_return を満たすまで低オッズ目から切る。

    returns (採用する目, 切った目)。オッズ降順に足していき、条件を満たす最大集合を返す。
    """
    ranked = sorted(odds, key=lambda k: -odds[k])
    kept: list = []
    total_inv = 0.0
    limit = 1.0 / min_return
    for key in ranked:
        inv = 1.0 / odds[key]
        if total_inv + inv > limit:
            break  # これ以上足すと保証が崩れる。以降(より低オッズ)も同様
        kept.append(key)
        total_inv += inv
    dropped = [k for k in ranked if k not in set(kept)]
    return kept, dropped


def _round_stakes(
    odds: dict, legs: list, budget: int, unit: int, cap: int
) -> dict:
    """s_i = B*(1/o_i)/Σ を 1点上限と単位で丸める。

    上限で余った分は、上限に達していない目へ**保証が最も低い目から**戻す
    （保証を上げる方向にしか使わない）。
    """
    inv = {k: 1.0 / odds[k] for k in legs}
    total_inv = sum(inv.values())
    if total_inv <= 0:
        return {}
    raw = {k: budget * inv[k] / total_inv for k in legs}
    stakes = {k: min(int(raw[k] // unit) * unit, cap) for k in legs}

    # 端数・上限で余った予算を、払戻が小さい目から順に上限まで積む
    while True:
        rest = budget - sum(stakes.values())
        if rest < unit:
            break
        room = [k for k in legs if stakes[k] + unit <= cap]
        if not room:
            break
        target = min(room, key=lambda k: stakes[k] * odds[k])
        stakes[target] += unit
    return {k: v for k, v in stakes.items() if v > 0}


def dutch_allocate(
    odds: dict,
    probs: dict | None = None,
    budget: int = BUDGET_DEFAULT,
    unit: int = UNIT_DEFAULT,
    cap: int = PER_POINT_CAP_DEFAULT,
    odds_floor: float = ODDS_FLOOR,
    min_return: float = MIN_RETURN,
    min_odds: float = MIN_ODDS,
    ev_min: float = EV_MIN,
) -> DutchResult:
    """ダッチ配分を計算し、購入可否まで判定して返す。

    odds:  {買い目キー: 朝オッズ}。キーは呼び出し側の表現（frozenset / tuple）のまま。
    probs: {買い目キー: 的中確率}。EV 判定に使う。None なら EV 判定を課さない
           （朝オッズはあるがモデル確率が無い経路のため。その旨 reason に残す）。

    朝オッズが1点も無いレースは buy=False（保証を計算できないため見送り）。
    仕様書 §7 のフォールバックは呼び出し側の責務とする。
    """
    usable = {k: float(v) for k, v in (odds or {}).items() if v and float(v) >= odds_floor}
    if not usable:
        return DutchResult({}, False, "no_odds_above_floor", 0.0, 0.0, 0.0, list(odds or {}))

    legs, dropped = _select_legs(usable, min_return)
    dropped += [k for k in (odds or {}) if k not in usable]
    if not legs:
        return DutchResult({}, False, "no_legs_after_dutch", 0.0, 0.0, 0.0, dropped)

    stakes = _round_stakes(usable, legs, budget, unit, cap)
    if not stakes:
        return DutchResult({}, False, "no_stake_after_rounding", 0.0, 0.0, 0.0, dropped)

    total = sum(stakes.values())
    # 保証 = 最も払戻の小さい目が来たときの払戻 ÷ 総賭け金（丸め・上限の後で測り直す）
    realized = min(stakes[k] * usable[k] for k in stakes) / total
    lowest = min(usable[k] for k in stakes)
    ev = sum(probs.get(k, 0.0) * usable[k] for k in stakes) if probs else 0.0

    if realized < min_return:
        return DutchResult(stakes, False, "min_return", realized, lowest, ev, dropped)
    if lowest < min_odds:
        return DutchResult(stakes, False, "min_odds", realized, lowest, ev, dropped)
    if probs is not None and ev < ev_min:
        return DutchResult(stakes, False, "ev", realized, lowest, ev, dropped)

    reason = "ok" if probs is not None else "ok_no_ev_check"
    return DutchResult(stakes, True, reason, realized, lowest, ev, dropped)
