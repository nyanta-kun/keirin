"""軸1は「単勝率1位」と「3着内率1位」のどちらが上か（2026-08-06・探索）。

ユーザーの問い:
  「レースのレンジを三連複10〜50倍に絞るとして、三連複の軸を選ぶのに
    単勝率・三着内率のどちらの1位が上か」

測り方:
  - 母集団: 7車立て。①全体 ②7S+7A（overlap∈{0,1} ∧ entropy 合格）
  - レンジ: **結果の三連複配当** が [10,50) のレース（＝欲しい配当帯）
    ⚠️ これは事後条件（オラクル）。**規則Aと規則Bを同じ土俵で比べる**用途にのみ使う。
       この条件下の ROI を絶対値として読んではいけない
  - 比較1: 軸1単独の3着内率（a1 が実際に3着内に入った率）
  - 比較2: 二軸総流し全体（軸2は現行の3ヘッド規則で、軸1を除いて選び直す）
  - 比較3: **2規則が食い違ったレースだけ**の直接対決（最も検出力が高い）

⚠️ オッズは wt_odds（最終）。DB 書き込みなし。掃引窓/確認窓を分けて出す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100
RNG = np.random.default_rng(42)

BANDS = [(0, 5), (5, 10), (10, 50), (50, 1e9), (0, 1e9)]
BAND_LBL = {(0, 5): "〜5倍", (5, 10): "5〜10倍", (10, 50): "**10〜50倍**",
            (50, 1e9): "50倍〜", (0, 1e9): "全帯"}


def a1_win(r):
    return max(r["pw"], key=lambda k: r["pw"][k])


def a1_top3(r):
    return max(r["p3"], key=lambda k: r["p3"][k])


def a2_of(r, a1):
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["pb"])
    sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in r["p3"]}
    return max((k for k in sc if k != a1), key=lambda k: sc[k])


def rows_for(races, a1fn, pool_7s7a: bool):
    out = []
    for r in races:
        a1 = a1fn(r)
        a2 = a2_of(r, a1)
        ent = rank_7s_field_entropy(r["p3"])
        if pool_7s7a:
            ov = rank_7s_wt_overlap_n(
                a1, a2, next((k for k, m in r["mk"].items() if m == 1), None),
                next((k for k, m in r["mk"].items() if m == 2), None))
            if ov not in (0, 1) or ent > RANK_7S_ENTROPY_MAX:
                continue
        legs = [x for x in r["p3"] if x not in (a1, a2)
                and frozenset({a1, a2, x}) in r["board"]]
        if not legs:
            continue
        odds = r["board"][frozenset(r["top3"])]
        rest = r["top3"] - {a1, a2}
        hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
        out.append(dict(rk=r["rk"], date=r["date"], odds=odds,
                        a1=a1, a2=a2, a1_in=int(a1 in r["top3"]),
                        a2_in=int(a2 in r["top3"]), hit=int(hit),
                        bet=len(legs) * STAKE,
                        ret=(round(odds * 100) // 10 * 10) if hit else 0))
    return pd.DataFrame(out)


def band_slice(df, lo, hi, win):
    t = df[(df.odds >= lo) & (df.odds < hi)]
    return t[(t.date <= CONFIRM_END) == (win == "確認")]


def cmp_table(dw, dt, title, pool_lbl):
    print(f"\n  ══ {title}（母集団: {pool_lbl}）")
    print(f"     {'配当帯':<12}{'窓':<5}{'n':>6}"
          f"{'軸1の3着内(単勝1位)':>20}{'(3着内1位)':>13}{'差':>8}"
          f"{'  的中(単勝/3着内)':>20}{'  ROI(単勝/3着内)':>20}")
    for lo, hi in BANDS:
        for win in ("掃引", "確認"):
            a = band_slice(dw, lo, hi, win)
            b = band_slice(dt, lo, hi, win)
            if len(a) < 30:
                continue
            print(f"     {BAND_LBL[(lo,hi)]:<12}{win:<5}{len(a):>6}"
                  f"{100*a.a1_in.mean():>19.1f}%{100*b.a1_in.mean():>12.1f}%"
                  f"{100*(a.a1_in.mean()-b.a1_in.mean()):>+8.1f}"
                  f"{100*a.hit.mean():>11.1f}% /{100*b.hit.mean():>6.1f}%"
                  f"{100*a.ret.sum()/a.bet.sum():>11.1f}% /{100*b.ret.sum()/b.bet.sum():>6.1f}%")


def duel(dw, dt, pool_lbl):
    """2規則が食い違ったレースだけの直接対決。"""
    m = dw.merge(dt[["rk", "a1", "a1_in", "hit", "bet", "ret"]], on="rk",
                 suffixes=("_w", "_t"))
    d = m[m.a1_w != m.a1_t]
    print(f"\n  ══ 直接対決: 単勝1位 ≠ 3着内1位 のレースだけ（母集団: {pool_lbl}）")
    print(f"     一致率 {100*(1-len(d)/len(m)):.1f}%（食い違い {len(d):,}R / 全{len(m):,}R）")
    print(f"     {'配当帯':<12}{'窓':<5}{'n':>6}{'単勝1位の3着内':>16}{'3着内1位の3着内':>16}"
          f"{'差':>8}{'  的中':>16}{'  ROI':>16}")
    for lo, hi in BANDS:
        for win in ("掃引", "確認"):
            t = d[(d.odds >= lo) & (d.odds < hi)]
            t = t[(t.date <= CONFIRM_END) == (win == "確認")]
            if len(t) < 30:
                continue
            aw, at = t.a1_in_w.mean(), t.a1_in_t.mean()
            se = np.sqrt((t.a1_in_w - t.a1_in_t).var(ddof=1) / len(t))
            print(f"     {BAND_LBL[(lo,hi)]:<12}{win:<5}{len(t):>6}{100*aw:>15.1f}%"
                  f"{100*at:>15.1f}%{100*(aw-at):>+8.1f}"
                  f"{100*t.hit_w.mean():>9.1f}% /{100*t.hit_t.mean():>4.1f}%"
                  f"{100*t.ret_w.sum()/t.bet_w.sum():>9.1f}% /"
                  f"{100*t.ret_t.sum()/t.bet_t.sum():>4.1f}%"
                  + (f"   t={(aw-at)/se:+.2f}" if se > 0 else ""))


def main():
    races = pd.read_pickle(DETAIL)
    print(f"[cache] {DETAIL.name} {len(races):,} レース（7車立て・2024-07〜2026-08）")
    print("⚠️ 配当帯は『結果の三連複配当』での事後分割。規則の優劣比較にのみ使う")

    for pool, lbl in ((True, "7S+7A"), (False, "7車立て全体")):
        dw = rows_for(races, a1_win, pool)
        dt = rows_for(races, a1_top3, pool)
        cmp_table(dw, dt, "軸1規則の比較", lbl)
        duel(dw, dt, lbl)


if __name__ == "__main__":
    main()
