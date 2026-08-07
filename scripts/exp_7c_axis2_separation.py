"""7C: 軸2車が「本当に2車だけ抜けている」ことを要求すべきか（2026-08-07）。

ユーザー指摘:
  「合計144%以上は 1車に90%集中して2・3車目が55%くらいで並ぶ場合にも成立する。
    その場合、軸の2車目が外れて3車目が入ることが考えられる（4車目以下が僅差の
    混戦も同様）。**2車のみが抜けている条件**として見る必要はないか」

⚠️ これは「合計の**代わりに**差を使う」（測定済み・劣ると確定）ではなく、
   **合計に差を追加する2次元条件**なので別物として測る。

確定仕様（軸=p3上位2車 / 相手=p3>=15% / 相手4点以上 / 予算10,000円）の上で:

  【A】軸2の不安定さは実在するか — gap23（2位と3位の差）別に
       「軸2が3着内に入った率」「軸1だけ入った率」を見る
  【B】gap23 の下限を足す（件数が減るので下限 sum2 を緩めて件数を揃えた版も）
  【C】min2（2位の複勝率そのもの）の下限を足す
       ＝「合計が大きい」のではなく「**両方とも強い**」ことを要求する
  【D】3車目の弱さを直接見る（sum2 と gap23 の2次元セル）

⚠️ オッズは精算にのみ使う。読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN, rank_7c_select_legs, rank_7c_unit_stake,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
ALL_PER_DAY = 76.3


def build(races) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, board = r["p3"], r["board"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2, a3, a4 = ranked[0], ranked[1], ranked[2], ranked[3]
        legs = rank_7c_select_legs(ranked[2:], p3)
        combos = [frozenset({a1, a2, t}) for t in legs
                  if frozenset({a1, a2, t}) in board]
        if not combos:
            continue
        k = len(combos)
        s = rank_7c_unit_stake(k)
        od = board[top3]
        hit = top3 in combos
        rows.append(dict(
            rk=r["rk"], date=r["date"],
            win="確認" if r["date"] <= CONFIRM_END else "掃引",
            sum2=p3[a1] + p3[a2], min2=p3[a2],
            gap23=p3[a2] - p3[a3], gap34=p3[a3] - p3[a4],
            k=k, legs_ok=int(k >= RANK_7C_LEGS_MIN),
            bet=k * s, ret=(round(od * s) // 10 * 10) if hit else 0,
            hit=int(hit), low=int(hit and od <= 2.0), odds=od,
            # 軸の当否を分解する
            a1_in=int(a1 in top3), a2_in=int(a2 in top3),
            a3_in=int(a3 in top3),
        ))
    return pd.DataFrame(rows)


def line(tag, g, days, base_low=None):
    if len(g) < 100:
        print(f"{tag:26s} （n={len(g)} で不足）")
        return
    low = g.low.sum() / days
    red = f"{100*(1-g.low.sum()/base_low):5.1f}" if base_low else "    —"
    print(f"{tag:26s} {len(g)/days:6.2f} {100*len(g)/days/ALL_PER_DAY:5.1f}% "
          f"{100*g.hit.mean():6.2f} {g.hit.sum()/days:6.2f} "
          f"{low:6.2f} {red:>6s} {100*g.ret.sum()/g.bet.sum():6.1f}")


HEAD = (f"{'条件':26s} {'件/日':>6s} {'全体比':>6s} {'的中%':>6s} {'的中/日':>6s} "
        f"{'低配当/日':>6s} {'削減%':>6s} {'ROI%':>6s}")


def main() -> None:
    d = build(pickle.load(open(DETAIL, "rb")))
    ev, cf = d[d.win == "掃引"], d[d.win == "確認"]
    de, dc = ev.date.nunique(), cf.date.nunique()

    base_e = ev[(ev.sum2 >= RANK_7C_P3_SUM_MIN) & (ev.legs_ok == 1)]
    base_c = cf[(cf.sum2 >= RANK_7C_P3_SUM_MIN) & (cf.legs_ok == 1)]
    bl = base_e.low.sum()

    print("=== 【A】gap23（2位と3位の差）別に、軸1・軸2の当否を分解（評価窓・7C母集団）===")
    print("   ※軸2が外れているだけなのか、そもそも軸1も来ていないのかを見る")
    g = base_e.assign(band=pd.cut(
        base_e.gap23, [-1, 0.02, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["~2pt", "2-5pt", "5-10pt", "10-15pt", "15-25pt", "25pt~"]))
    print(f"{'gap23':>10s} {'n':>7s} {'軸1が3着内%':>11s} {'軸2が3着内%':>11s} "
          f"{'二軸的中%':>9s} {'3位が3着内%':>11s} {'ROI%':>6s}")
    for b, x in g.groupby("band", observed=True):
        print(f"{str(b):>10s} {len(x):7,d} {100*x.a1_in.mean():11.2f} "
              f"{100*x.a2_in.mean():11.2f} {100*x.hit.mean():9.2f} "
              f"{100*x.a3_in.mean():11.2f} {100*x.ret.sum()/x.bet.sum():6.1f}")

    print("\n=== 【B】gap23 の下限を足す（下限 sum2 は 1.44 のまま＝件数が減る）===")
    print(HEAD)
    line("現行（gap23 条件なし）", base_e, de, bl)
    for th in (0.02, 0.05, 0.08, 0.10, 0.15):
        line(f"  gap23 >= {th:.0%}", base_e[base_e.gap23 >= th], de, bl)

    print("\n=== 【C】min2（2位の複勝率そのもの）の下限を足す ===")
    print("   ＝『合計が大きい』ではなく『両方とも強い』ことを要求する")
    print(HEAD)
    line("現行（min2 条件なし）", base_e, de, bl)
    for th in (0.55, 0.60, 0.65, 0.70, 0.75):
        line(f"  min2 >= {th:.0%}", base_e[base_e.min2 >= th], de, bl)

    print("\n=== 【D】件数を 23.5件/日 に揃えて公平比較（下限 sum2 を再調整）===")
    target = 23.48 * de
    print(HEAD)
    line("現行（合計のみ）", base_e, de, bl)
    for name, col, th in (("gap23>=2%", "gap23", 0.02), ("gap23>=5%", "gap23", 0.05),
                          ("gap23>=10%", "gap23", 0.10),
                          ("min2>=55%", "min2", 0.55), ("min2>=60%", "min2", 0.60),
                          ("min2>=65%", "min2", 0.65)):
        pool = ev[(ev.legs_ok == 1) & (ev[col] >= th)]
        if len(pool) < target:
            print(f"{'  ' + name + ' + 下限調整':26s} （母数不足 {len(pool)/de:.2f}件/日）")
            continue
        lo = pool.sort_values("sum2", ascending=False).sum2.iloc[int(target)]
        line(f"  {name} + 合計>={lo:.3f}", pool[pool.sum2 >= lo], de, bl)

    print("\n=== 【E】確認窓での再現（【D】で良かった条件を固定して一度きり検証）===")
    print(HEAD)
    line("現行（合計のみ）", base_c, dc, base_c.low.sum())
    for name, col, th in (("gap23>=5%", "gap23", 0.05), ("min2>=60%", "min2", 0.60),
                          ("min2>=65%", "min2", 0.65)):
        pool_e = ev[(ev.legs_ok == 1) & (ev[col] >= th)]
        if len(pool_e) < target:
            continue
        lo = pool_e.sort_values("sum2", ascending=False).sum2.iloc[int(target)]
        pool_c = cf[(cf.legs_ok == 1) & (cf[col] >= th) & (cf.sum2 >= lo)]
        line(f"  {name} + 合計>={lo:.3f}", pool_c, dc, base_c.low.sum())

    print("\n=== 【F】軸2が外れたレースは何が起きているか（評価窓・7C母集団）===")
    miss = base_e[base_e.a2_in == 0]
    print(f"  軸2が3着内に入らなかった: {len(miss):,}R ({100*len(miss)/len(base_e):.1f}%)")
    print(f"    うち3位（軸から外した車）が3着内: {100*miss.a3_in.mean():.1f}%")
    print(f"    その場合の gap23 平均: {100*miss[miss.a3_in==1].gap23.mean():.2f}pt "
          f"（軸2が入ったレースは {100*base_e[base_e.a2_in==1].gap23.mean():.2f}pt）")
    print("  ※ gap23 が小さいほど『3位を軸にすべきだった』が起きやすいなら、")
    print("    gap23 下限に意味がある。差が無ければ入れ替えは避けられない。")


if __name__ == "__main__":
    main()
