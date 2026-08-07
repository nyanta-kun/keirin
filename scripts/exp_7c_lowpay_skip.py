"""7C: 「当たっても配当が低すぎるレース」を朝に見送れるか（2026-08-07）。

ユーザー要望（岐阜2R 2026-08-07・的中して三連複 ¥110）:
  「このオッズ（三連複が200円以下）はレースの見送りとしたいが、
    **正確なオッズは直前まで確認できない**」

つまり target は「勝ち三連複の配当 <= 2.0倍」。これを**オッズ非依存の量**で
朝8:00に予測できるかを測る。使える予測子（すべて pred_top3 から作れる）:

  P1 上位2車の合計          （選別に使っている量そのもの）
  P2 上位3車の合計          （3着まで含めて堅いか）
  P3 最有力3車組の素朴確率  （p3 の積を35組で正規化した最大値）
  P4 買う目の素朴確率合計   （実際に買う組だけの確率＝当たりやすさ）

評価: 見送り率 vs 「低配当的中」の削減率 vs 的中率・ROI の劣化。
比較として **朝オッズ**（wt_odds_snapshot・2026-06-08〜のみ）と
**発走15分前の最終オッズ**（＝理論上限。netkeirin入稿後なので商品には使えないが
picks_history/Web の見送り記録には使える）も並べる。

⚠️ 買い目は確定仕様（軸=p3上位2車 / 相手=p3>=15% / 1レース10,000円の予算枠）。
⚠️ 読み取りのみ。
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_BUDGET, RANK_7C_LEG_P3_MIN, RANK_7C_P3_SUM_MIN,
    rank_7c_select_legs, rank_7c_unit_stake,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
LOWPAY = 2.0          # 「低配当」の定義（三連複 200円以下）
ALL_PER_DAY = 76.3


def naive_combo_probs(p3: dict[int, float]) -> dict[frozenset, float]:
    out = {frozenset(c): p3[c[0]] * p3[c[1]] * p3[c[2]]
           for c in itertools.combinations(sorted(p3), 3)}
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else out


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
        a1, a2 = ranked[0], ranked[1]
        s2 = p3[a1] + p3[a2]
        if s2 < RANK_7C_P3_SUM_MIN:
            continue
        others = ranked[2:]
        legs = rank_7c_select_legs(others, p3)
        combos = [frozenset({a1, a2, t}) for t in legs
                  if frozenset({a1, a2, t}) in board]
        if not combos:
            continue
        k = len(combos)
        s = rank_7c_unit_stake(k)
        hit = top3 in combos
        od = board[top3]
        nc = naive_combo_probs(p3)
        rows.append(dict(
            rk=r["rk"], date=r["date"],
            win="確認" if r["date"] <= CONFIRM_END else "掃引",
            k=k, bet=k * s, ret=(round(od * s) // 10 * 10) if hit else 0,
            hit=int(hit), odds=od,
            lowpay_hit=int(hit and od <= LOWPAY),
            # 予測子（すべてオッズ非依存）
            P1_sum2=s2,
            P2_sum3=s2 + p3[ranked[2]],
            P3_pmax=max(nc.values()),
            P4_pbuy=sum(nc[c] for c in combos),
            # 参考: 買った目の最低オッズ（最終・理論上限の判定用）
            min_buy_odds=min(board[c] for c in combos),
        ))
    return pd.DataFrame(rows)


def sweep(d: pd.DataFrame, col: str, label: str) -> None:
    """予測子 col の上位から見送った場合の効果。"""
    ev = d[d.win == "掃引"]
    cf = d[d.win == "確認"]
    days_e, days_c = ev.date.nunique(), cf.date.nunique()
    base_low = ev.lowpay_hit.sum()
    print(f"\n-- {label} --")
    print(f"{'見送り率':>8s} {'閾値':>9s} {'件/日':>7s} {'全体比':>6s} | "
          f"{'的中%':>6s} {'低配当的中/日':>11s} {'削減%':>6s} {'ROI%':>6s} | "
          f"{'確認 的中%':>9s} {'ROI%':>6s}")
    for frac in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        thr = ev[col].quantile(1 - frac) if frac > 0 else float("inf")
        ge = ev[ev[col] < thr] if frac > 0 else ev
        gc = cf[cf[col] < thr] if frac > 0 else cf
        red = 100 * (1 - ge.lowpay_hit.sum() / max(base_low, 1))
        print(f"{frac:8.0%} {thr:9.4f} {len(ge)/days_e:7.2f} "
              f"{100*len(ge)/days_e/ALL_PER_DAY:5.1f}% | "
              f"{100*ge.hit.mean():6.2f} {ge.lowpay_hit.sum()/days_e:11.2f} "
              f"{red:6.1f} {100*ge.ret.sum()/ge.bet.sum():6.1f} | "
              f"{100*gc.hit.mean():9.2f} {100*gc.ret.sum()/gc.bet.sum():6.1f}")


def main() -> None:
    d = build(pickle.load(open(DETAIL, "rb")))
    ev = d[d.win == "掃引"]
    days = ev.date.nunique()
    print(f"7C 確定仕様（軸=p3上位2車 / 相手=p3>={RANK_7C_LEG_P3_MIN:.0%} / "
          f"予算{RANK_7C_BUDGET:,}円）")
    print(f"評価窓 {len(ev):,}R / {days}日 = {len(ev)/days:.2f}件/日  "
          f"平均{ev.k.mean():.2f}点")
    print(f"  的中 {100*ev.hit.mean():.2f}%  ROI {100*ev.ret.sum()/ev.bet.sum():.1f}%")
    print(f"  **低配当的中（{LOWPAY}倍以下）: {ev.lowpay_hit.sum():,}件 = "
          f"{ev.lowpay_hit.sum()/days:.2f}件/日 "
          f"（的中の {100*ev.lowpay_hit.sum()/ev.hit.sum():.1f}%）**")

    print("\n=== 的中時オッズの分布（評価窓・的中レースのみ）===")
    h = ev[ev.hit == 1]
    for lo, hi in ((0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0),
                   (10.0, 1e9)):
        n = ((h.odds >= lo) & (h.odds < hi)).sum()
        print(f"  {lo:5.1f}〜{hi if hi < 1e9 else 999:5.1f}倍: {n:6,d}件 "
              f"({100*n/len(h):5.1f}%)  {n/days:5.2f}件/日")

    print("\n=== 朝に見送れるか（オッズ非依存の予測子）===")
    for col, lab in (("P1_sum2", "P1 上位2車の合計"),
                     ("P2_sum3", "P2 上位3車の合計"),
                     ("P3_pmax", "P3 最有力3車組の素朴確率"),
                     ("P4_pbuy", "P4 買う目の素朴確率合計")):
        sweep(d, col, lab)

    # 参考: 最終オッズが使えた場合の理論上限（＝発走15分前判定でならできる）
    print("\n=== 参考: 発走15分前の最終オッズで切った場合（理論上限）===")
    print("   ※netkeirin 入稿後なので商品からは外せない。picks_history/Web の")
    print("     見送り記録・ペーパー成績にのみ適用できる。")
    print(f"{'条件':>22s} {'件/日':>7s} {'的中%':>6s} {'低配当的中/日':>11s} {'ROI%':>6s}")
    for thr in (None, 1.5, 2.0, 2.5, 3.0):
        g = ev if thr is None else ev[ev.min_buy_odds >= thr]
        lab = "切らない" if thr is None else f"買い目最低オッズ>={thr}"
        print(f"{lab:>22s} {len(g)/days:7.2f} {100*g.hit.mean():6.2f} "
              f"{g.lowpay_hit.sum()/days:11.2f} "
              f"{100*g.ret.sum()/g.bet.sum():6.1f}")


if __name__ == "__main__":
    main()
