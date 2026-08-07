"""レース選別基準の比較: 説明可能な単純規則 vs ペアモデル（2026-08-07）。

ユーザー提案（岐阜4Rの例）:
  「2・4 の複勝率が抜けているので、この2車を二軸とすることが考えられる」
  → 選別基準を「**複勝率(p3)の2位と3位の差が大きいレース**」に置く案。
    説明しやすく、ブランド（二軸探偵）の文脈にそのまま乗る。

同じ網羅率（全レースの約30% = 22.9件/日）に揃えて、下記を横並びで比較する:

  R1 p3 の 2位−3位 差（＝「2車が抜けている」の直接的な表現・ユーザー案）
  R2 p3 上位2車の合計（現行 7S の axis_sum と同型）
  R3 p3 上位2車のうち低い方（両方が高いことを要求）
  R4 レース内エントロピー（現行 7S/7A のゲート）
  R5 ペアモデルのスコア（＝そのペアが同時に3着内に入る確率）
  R6 R1 と R5 の合成

軸2車は常に「その選別基準に対応する2車」を使う（R1〜R4 は p3 上位2車、
R5/R6 はモデルが選んだペア）。⚠️ オッズ不使用・読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import rank_7s_field_entropy  # noqa: E402

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
SCORED = REPO / "data" / "exp_cache" / "pair_axis_scored_v2.pkl"
TEST_START = "2025-07-01"
STAKE = 100
ALL_PER_DAY = 76.3
TARGET_PER_DAY = 22.9   # 全レースの約30%


def main() -> None:
    races = {r["rk"]: r for r in pickle.load(open(DETAIL, "rb"))}
    te = pd.read_pickle(SCORED)
    pick = te.loc[te.groupby("rk").s_bin.idxmax()][["rk", "date", "hi", "lo", "s_bin"]]
    pick = pick.set_index("rk")

    rows = []
    for rk, p in pick.iterrows():
        r = races[rk]
        p3, board = r["p3"], r["board"]
        top3 = frozenset(r["top3"])
        if top3 not in board or len(p3) != 7:
            continue
        cars = sorted(p3)
        srt = sorted(p3.values(), reverse=True)
        t2 = sorted(p3, key=lambda k: -p3[k])[:2]
        mhi, mlo = int(p.hi), int(p.lo)

        def payoff(x, y):
            legs = [frozenset({x, y, z}) for z in cars if z not in (x, y)
                    and frozenset({x, y, z}) in board]
            if not legs:
                return None
            hit = int(top3 in legs)
            return hit, len(legs) * STAKE, (round(board[top3] * 100) // 10 * 10) if hit else 0

        a = payoff(*t2)
        b = payoff(mhi, mlo)
        if a is None or b is None:
            continue
        rows.append(dict(
            rk=rk, date=p.date,
            gap23=srt[1] - srt[2], sum2=srt[0] + srt[1], min2=srt[1],
            ent=rank_7s_field_entropy(p3), score=p.s_bin,
            hit_p3=a[0], bet_p3=a[1], ret_p3=a[2],
            hit_md=b[0], bet_md=b[1], ret_md=b[2],
            same_pick=int({mhi, mlo} == set(t2)),
            odds=board[top3],
        ))
    d = pd.DataFrame(rows)
    days = d.date.nunique()
    d["score_r"] = d.score.rank(pct=True)
    d["gap_r"] = d.gap23.rank(pct=True)
    d["mix"] = d.score_r + d.gap_r

    rules = [
        ("R1 p3 2位−3位差（ユーザー案）", "gap23", False, "p3"),
        ("R2 p3 上位2車の合計", "sum2", False, "p3"),
        ("R3 p3 2位の高さ", "min2", False, "p3"),
        ("R4 エントロピー（低い順）", "ent", True, "p3"),
        ("R5 ペアモデル スコア", "score", False, "md"),
        ("R6 R1×R5 合成", "mix", False, "md"),
        ("R5' ペアモデル（軸は p3 上位2車）", "score", False, "p3"),
    ]
    k = int(TARGET_PER_DAY * days)
    print(f"評価窓 {TEST_START}〜  {len(d):,}R / {days}日 "
          f"(7車 {len(d)/days:.1f}件/日・全体の {100*len(d)/days/ALL_PER_DAY:.0f}%)")
    print(f"選別数を {TARGET_PER_DAY}件/日（全体の30%）= {k:,}R に統一して比較\n")
    print(f"{'選別基準':34s} {'二軸的中%':>9s} {'的中/日':>7s} {'ROI%':>7s} "
          f"{'平均配当':>7s} {'軸一致%':>7s}")
    print(f"{'（参考）無選別・全7車 軸=p3上位2車':34s} "
          f"{100*d.hit_p3.mean():9.2f} {d.hit_p3.sum()/days:7.2f} "
          f"{100*d.ret_p3.sum()/d.bet_p3.sum():7.1f} "
          f"{d[d.hit_p3==1].odds.mean():7.2f} {'-':>7s}")
    print(f"{'（参考）無選別・全7車 軸=モデル':34s} "
          f"{100*d.hit_md.mean():9.2f} {d.hit_md.sum()/days:7.2f} "
          f"{100*d.ret_md.sum()/d.bet_md.sum():7.1f} "
          f"{d[d.hit_md==1].odds.mean():7.2f} {100*d.same_pick.mean():7.1f}")
    print()
    for name, col, asc, ax in rules:
        g = d.sort_values(col, ascending=asc).iloc[:k]
        h, bet, ret = f"hit_{ax}", f"bet_{ax}", f"ret_{ax}"
        print(f"{name:34s} {100*g[h].mean():9.2f} {g[h].sum()/days:7.2f} "
              f"{100*g[ret].sum()/g[bet].sum():7.1f} "
              f"{g[g[h]==1].odds.mean():7.2f} {100*g.same_pick.mean():7.1f}")

    # 網羅率を振ってユーザー案とモデルの差の形を見る
    print("\n=== 網羅率を振った比較（二軸的中%）===")
    print(f"{'件/日':>7s} {'全体比':>7s} | {'R1 gap23':>9s} {'R5 モデル':>9s} {'差pt':>6s}")
    for per_day in (10, 15, 19, 22.9, 30, 40, 63.8):
        kk = int(per_day * days)
        if kk > len(d):
            kk = len(d)
        g1 = d.sort_values("gap23", ascending=False).iloc[:kk]
        g5 = d.sort_values("score", ascending=False).iloc[:kk]
        print(f"{per_day:7.1f} {100*per_day/ALL_PER_DAY:6.1f}% | "
              f"{100*g1.hit_p3.mean():9.2f} {100*g5.hit_md.mean():9.2f} "
              f"{100*(g5.hit_md.mean()-g1.hit_p3.mean()):6.2f}")

    # ユーザー案の閾値を実運用向けに提示
    print("\n=== R1（p3 2位−3位差）を絶対閾値にした場合 ===")
    thr = d.gap23.quantile(1 - k / len(d))
    s = d[d.gap23 >= thr]
    n_day = s.groupby("date").size()
    print(f"閾値 gap23 >= {thr:.4f}  → {len(s)/days:.1f}件/日 "
          f"({100*len(s)/days/ALL_PER_DAY:.0f}%) 二軸的中 {100*s.hit_p3.mean():.2f}%  "
          f"ROI {100*s.ret_p3.sum()/s.bet_p3.sum():.1f}%")
    print(f"日次件数: 中央値 {n_day.median():.0f} / 最小 {n_day.min()} / "
          f"最大 {n_day.max()} / 0件の日 {days-len(n_day)}日")

    # 四半期安定性（両案）
    print("\n=== 四半期別（22.9件/日に統一）===")
    d["q"] = pd.PeriodIndex(pd.to_datetime(d.date), freq="Q").astype(str)
    g1 = d.sort_values("gap23", ascending=False).iloc[:k]
    g5 = d.sort_values("score", ascending=False).iloc[:k]
    for q in sorted(d.q.unique()):
        a, b = g1[g1.q == q], g5[g5.q == q]
        if len(a) < 100 or len(b) < 100:
            continue
        print(f"  {q}  R1 n={len(a):5,d} {100*a.hit_p3.mean():5.2f}%  |  "
              f"R5 n={len(b):5,d} {100*b.hit_md.mean():5.2f}%")


if __name__ == "__main__":
    main()
