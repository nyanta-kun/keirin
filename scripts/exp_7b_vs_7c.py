"""7B と 7C はどれだけ重なるか / どちらが良いか（2026-08-07・ユーザー依頼）。

ユーザー: 「7Bと7Cはほぼ重なるように思われます。どちらの方が成績良いですか？
            7Cの方が良い場合、7Bを廃止とします」

- 7B: ◎○一致 × **順序一致** × **準決勝** / 相手3点（`legs_7b`）
- 7C: モデル3着内率の**上位2車の合計 >= 1.44** ∧ 相手4点以上 ∧ 低配当パターンでない

母集団の定義がまったく違うので、まず**本当に重なっているのか**を数える。
そのうえで、重なる部分・重ならない部分それぞれで成績を比べる。

⚠️ 7C は picks_history へ未 backfill なので、確定仕様を vintage 予測
   （`axis_detail_7car.pkl`）の上で再現する（`exp_netkeirin_gami_allocation.py`
   と同じ `build_7c` 相当）。7B は picks_history（walk-forward 再構築済み）を使う。
⚠️ 読み取り専用。
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.stake_allocation import tilted_stakes  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN, RACE_BUDGET,
    rank_7c_is_lowpay_pattern, rank_7c_select_legs, unit_stake,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
_SEP_RE = re.compile(r"[-=]")


def build_7c(races):
    """確定仕様の 7C を再現する。returns {race_key: (axis1, axis2, legs)}"""
    out = {}
    for r in races:
        p3 = r["p3"]
        if len(p3) != 7:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2 = ranked[0], ranked[1]
        if p3[a1] + p3[a2] < RANK_7C_P3_SUM_MIN:
            continue
        legs = rank_7c_select_legs(ranked[2:], p3)
        if len(legs) < RANK_7C_LEGS_MIN:
            continue
        if rank_7c_is_lowpay_pattern(p3, r.get("line") or {}):
            continue
        out[r["rk"]] = (a1, a2, legs)
    return out


def settle(axis1, axis2, legs, board, top3, p3=None):
    """(投資, 払戻, 的中, 実質的中)。

    p3 を渡すと**本番と同じ傾斜配分**（モデル単独＝朝オッズが無い期間の規則）。
    渡さなければ均等配分。7C は傾斜配分の対象・7B は対象外なので、
    本番構成での比較には 7C だけ p3 を渡す。
    """
    thirds = [t for t in legs if frozenset({axis1, axis2, t}) in board]
    if not thirds:
        return None
    if p3:
        st, _ = tilted_stakes(thirds, None, p3, budget=RACE_BUDGET)
    else:
        u = unit_stake(len(thirds), RACE_BUDGET)
        st = {t: u for t in thirds}
    bet = sum(st.values())
    win = next((t for t in thirds if frozenset({axis1, axis2, t}) == top3), None)
    ret = int(st[win] * board[top3]) // 10 * 10 if win is not None else 0
    return bet, ret, int(win is not None), int(ret >= bet and ret > 0)


def main():
    races = pickle.load(open(DETAIL, "rb"))
    detail = {r["rk"]: r for r in races}
    c7 = build_7c(races)

    with get_connection() as conn:
        rows = list(conn.execute("""
            SELECT race_date, race_key, pred_combo, hit, payout, bet_amount
            FROM keirin.picks_history WHERE rank = 'RANK_7B'
        """))
    b7 = {}
    for r in rows:
        base = r["race_key"].split("#")[0]
        head = str(r["pred_combo"] or "").split(" ")[0]
        if "-" not in head:
            continue
        ax, lg = head.split("-", 1)
        axes = [int(x) for x in ax.replace("=", ",").split(",") if x.strip().isdigit()]
        legs = [int(x) for x in lg.split(",") if x.strip().isdigit()]
        if len(axes) == 2 and legs:
            b7[base] = (axes[0], axes[1], legs)

    common = sorted(set(b7) & set(c7))
    print(f"7B {len(b7):,} レース / 7C {len(c7):,} レース / **重なり {len(common):,} レース**")
    print(f"  7B のうち 7C にも入る: {100*len(common)/max(len(b7),1):.1f}%")
    print(f"  7C のうち 7B にも入る: {100*len(common)/max(len(c7),1):.1f}%")

    same_axis = sum(1 for k in common if set(b7[k][:2]) == set(c7[k][:2]))
    print(f"  重なったレースで**軸2車が一致**: {100*same_axis/max(len(common),1):.1f}%")

    def agg(keys, src, label, tilt=False):
        n = bet = ret = hit = rhit = 0
        for k in keys:
            d = detail.get(k)
            if d is None or frozenset(d["top3"]) not in d["board"]:
                continue
            s = settle(*src[k], d["board"], frozenset(d["top3"]),
                       d["p3"] if tilt else None)
            if s is None:
                continue
            n += 1
            bet += s[0]; ret += s[1]; hit += s[2]; rhit += s[3]
        if not n:
            return
        print(f"  {label:<22s} n={n:>6,d}  的中{100*hit/n:>5.1f}%  "
              f"実質的中{100*rhit/n:>5.1f}%  ROI{100*ret/bet:>6.1f}%")

    print("\n=== 重なったレース（同一母集団での直接比較）===")
    agg(common, b7, "7B 3点・均等")
    agg(common, b7, "7B 3点・傾斜配分", tilt=True)
    agg(common, c7, "7C 4-5点・均等")
    agg(common, c7, "7C 4-5点・傾斜配分", tilt=True)

    print("\n=== それぞれの独自部分 ===")
    agg(sorted(set(b7) - set(c7)), b7, "7B のみ・均等")
    agg(sorted(set(b7) - set(c7)), b7, "7B のみ・傾斜配分", tilt=True)
    agg(sorted(set(c7) - set(b7)), c7, "7C のみ・傾斜配分", tilt=True)

    print("\n=== 全体 ===")
    agg(sorted(b7), b7, "7B 全体・均等（現行）")
    agg(sorted(b7), b7, "7B 全体・傾斜配分", tilt=True)
    agg(sorted(c7), c7, "7C 全体・均等")
    agg(sorted(c7), c7, "7C 全体・傾斜配分（本番）", tilt=True)

    print("\n=== 7B を廃止したら何が失われるか ===")
    only_b = sorted(set(b7) - set(c7))
    days = len({detail[k]["date"] for k in b7 if k in detail}) or 1
    print(f"  7B のうち 7C が拾わないレース: {len(only_b):,} 件"
          f"（{len(only_b)/days:.2f} 件/日）")


if __name__ == "__main__":
    main()
