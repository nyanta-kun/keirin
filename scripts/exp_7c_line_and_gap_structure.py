"""7C: 混戦時のライン関係と、1-2位差／2-3位差の連動（2026-08-07）。

ユーザー質問:
  Q1「混戦の場合、1位との同ライン・別ラインは考慮必要ですか？」
  Q2「1位と2位の差が少ない場合、3位も近いことはないか？」

Q2 は事実確認（gap12 と gap23 が連動しているなら、
「1-2位が僅差＝3位も近い＝軸2が不安定」という一連の懸念が1つの量に集約される）。
Q1 は軸2車の同時発生に効くはずのライン関係（memory: 同ライン lift 1.122 /
別ライン 0.672）が **7C の母集団の中でも**残っているかを見る。

確定仕様（軸=p3上位2車 / 相手=p3>=15% / 相手4点以上 / 予算10,000円）の上で:
  【1】gap12 × gap23 の連動（Q2）
  【2】同ライン/別ライン × 混戦度（gap12帯）別の二軸的中率・ROI（Q1）
  【3】ライン条件を選別に足す価値（件数を揃えた公平比較）
  【4】ライン構成の詳細（分戦数・単騎・番手位置）で更に分けられるか

⚠️ オッズは精算にのみ使う。読み取りのみ。
"""
from __future__ import annotations

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
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN, rank_7c_select_legs, rank_7c_unit_stake,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
ALL_PER_DAY = 76.3


def load_line_meta(keys):
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute("""SELECT race_key, frame_no, line_group, line_pos,
                              line_size, n_lines
                       FROM keirin.wt_entries WHERE race_key = ANY(%s)""", (keys,))
        out: dict[str, dict[int, tuple]] = {}
        for rk, f, g, p, s, n in cur.fetchall():
            out.setdefault(rk, {})[int(f)] = (g, p, s, n)
        return out


def build(races, meta) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, board = r["p3"], r["board"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        e = meta.get(r["rk"])
        if not e or len(e) < 7:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2, a3, a4 = ranked[0], ranked[1], ranked[2], ranked[3]
        # ⚠️ 下限は**ここで掛けない**。ライン条件を足したときに下限を緩めて
        #    件数を揃える比較ができなくなるため（最初の実装で母数不足になった）。
        if p3[a1] + p3[a2] < 1.15:
            continue
        legs = rank_7c_select_legs(ranked[2:], p3)
        combos = [frozenset({a1, a2, t}) for t in legs
                  if frozenset({a1, a2, t}) in board]
        if len(combos) < RANK_7C_LEGS_MIN:
            continue
        k = len(combos)
        s = rank_7c_unit_stake(k)
        od = board[top3]
        hit = top3 in combos

        g1, pos1, sz1, nl = e[a1]
        g2, pos2, sz2, _ = e[a2]
        g3 = e[a3][0]
        same = int(g1 is not None and g1 == g2)
        rows.append(dict(
            rk=r["rk"], date=r["date"],
            win="確認" if r["date"] <= CONFIRM_END else "掃引",
            sum2=p3[a1] + p3[a2],
            gap12=p3[a1] - p3[a2], gap23=p3[a2] - p3[a3], gap34=p3[a3] - p3[a4],
            same_line=same,
            a3_same_as_a1=int(g1 is not None and g1 == g3),
            adjacent=int(same and pos1 is not None and pos2 is not None
                         and abs(pos1 - pos2) == 1),
            a1_leads=int(same and pos1 is not None and pos2 is not None
                         and pos1 < pos2),
            n_lines=nl if nl is not None else np.nan,
            a1_solo=int(sz1 == 1), a2_solo=int(sz2 == 1),
            k=k, bet=k * s, ret=(round(od * s) // 10 * 10) if hit else 0,
            hit=int(hit), low=int(hit and od <= 2.0), odds=od,
            a1_in=int(a1 in top3), a2_in=int(a2 in top3),
        ))
    return pd.DataFrame(rows)


def cell(g, days):
    if len(g) < 150:
        return None
    return (len(g) / days, 100 * g.hit.mean(),
            100 * g.ret.sum() / g.bet.sum(), g.low.sum() / days)


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    meta = load_line_meta([r["rk"] for r in races])
    d = build(races, meta)
    ev, cf = d[d.win == "掃引"], d[d.win == "確認"]
    de, dc = ev.date.nunique(), cf.date.nunique()
    print(f"7C母集団 評価窓 {len(ev):,}R/{de}日  確認窓 {len(cf):,}R/{dc}日\n")

    # ── Q2 ───────────────────────────────────────────────────────────
    print("=== 【1】Q2: 1-2位差が小さいと3位も近いか（評価窓）===")
    ev2 = ev.assign(b12=pd.cut(ev.gap12, [-1, 0.02, 0.05, 0.10, 0.20, 1.0],
                               labels=["~2pt", "2-5pt", "5-10pt", "10-20pt", "20pt~"]))
    print(f"{'1-2位差':>10s} {'n':>7s} {'2-3位差 平均':>12s} {'中央値':>8s} "
          f"{'2-3位差<5ptの率':>14s} {'二軸的中%':>9s}")
    for b, g in ev2.groupby("b12", observed=True):
        print(f"{str(b):>10s} {len(g):7,d} {100*g.gap23.mean():12.2f}pt "
              f"{100*g.gap23.median():7.2f}pt {100*(g.gap23 < 0.05).mean():13.1f}% "
              f"{100*g.hit.mean():9.2f}")
    print(f"\n  相関 corr(1-2位差, 2-3位差) = {ev.gap12.corr(ev.gap23):+.3f}")
    print("  ※ 正なら『1-2位が離れているほど2-3位も離れる』、"
          "負なら『1-2位が僅差なら2-3位は離れる』")

    # ── Q1 ───────────────────────────────────────────────────────────
    print("\n=== 【2】Q1: 軸2車の同ライン/別ライン × 混戦度（評価窓）===")
    print(f"{'1-2位差':>10s} | {'別ライン n':>9s} {'的中%':>6s} {'ROI%':>6s} | "
          f"{'同ライン n':>9s} {'的中%':>6s} {'ROI%':>6s} | {'差pt':>6s}")
    for b, g in ev2.groupby("b12", observed=True):
        a, s = g[g.same_line == 0], g[g.same_line == 1]
        if len(a) < 100 or len(s) < 100:
            continue
        print(f"{str(b):>10s} | {len(a):9,d} {100*a.hit.mean():6.2f} "
              f"{100*a.ret.sum()/a.bet.sum():6.1f} | {len(s):9,d} "
              f"{100*s.hit.mean():6.2f} {100*s.ret.sum()/s.bet.sum():6.1f} | "
              f"{100*(s.hit.mean()-a.hit.mean()):6.2f}")
    print("\n  全体（混戦度によらず）:")
    for w, g0, days in (("評価窓", ev, de), ("確認窓", cf, dc)):
        a, s = g0[g0.same_line == 0], g0[g0.same_line == 1]
        print(f"    {w}  別ライン {len(a)/days:5.2f}件/日 的中 {100*a.hit.mean():5.2f}% "
              f"ROI {100*a.ret.sum()/a.bet.sum():5.1f}%  |  "
              f"同ライン {len(s)/days:5.2f}件/日 的中 {100*s.hit.mean():5.2f}% "
              f"ROI {100*s.ret.sum()/s.bet.sum():5.1f}%")

    # ── 【3】選別に足す価値 ───────────────────────────────────────────
    print("\n=== 【3】ライン条件を選別に足す（件数を23.5件/日に揃える）===")
    tgt = 23.48 * de
    print(f"{'条件':26s} {'sum2下限':>8s} | {'評価 的中%':>9s} {'ROI%':>6s} "
          f"{'低配当/日':>7s} | {'確認 的中%':>9s} {'ROI%':>6s}")
    base = ev[ev.sum2 >= RANK_7C_P3_SUM_MIN]
    basec = cf[cf.sum2 >= RANK_7C_P3_SUM_MIN]
    print(f"{'現行（ライン条件なし）':26s} {RANK_7C_P3_SUM_MIN:8.3f} | "
          f"{100*base.hit.mean():9.2f} {100*base.ret.sum()/base.bet.sum():6.1f} "
          f"{base.low.sum()/de:7.2f} | {100*basec.hit.mean():9.2f} "
          f"{100*basec.ret.sum()/basec.bet.sum():6.1f}")
    for lab, mask_e, mask_c in (
        ("軸2車が同一ライン", ev.same_line == 1, cf.same_line == 1),
        ("軸2車が別ライン", ev.same_line == 0, cf.same_line == 0),
        ("同一ライン ∧ 隣接", ev.adjacent == 1, cf.adjacent == 1),
        ("同一ライン ∧ 軸1が前", ev.a1_leads == 1, cf.a1_leads == 1),
    ):
        pool = ev[mask_e]
        if len(pool) < tgt:
            print(f"{lab:26s} （母数不足 {len(pool)/de:.2f}件/日）")
            continue
        lo = pool.sort_values("sum2", ascending=False).sum2.iloc[int(tgt)]
        g = pool[pool.sum2 >= lo]
        gc = cf[mask_c & (cf.sum2 >= lo)]
        print(f"{lab:26s} {lo:8.3f} | {100*g.hit.mean():9.2f} "
              f"{100*g.ret.sum()/g.bet.sum():6.1f} {g.low.sum()/de:7.2f} | "
              f"{100*gc.hit.mean():9.2f} {100*gc.ret.sum()/gc.bet.sum():6.1f}")

    # ── 【4】ライン構成の詳細 ─────────────────────────────────────────
    print("\n=== 【4】ライン構成の詳細（評価窓・7C母集団）===")
    for col, lab in (("n_lines", "分戦数"), ("a3_same_as_a1", "3位が軸1と同ライン"),
                     ("a1_solo", "軸1が単騎"), ("a2_solo", "軸2が単騎")):
        print(f"-- {lab} --")
        for v, g in ev.groupby(col):
            c = cell(g, de)
            if c:
                print(f"    {str(v):>6s}: {c[0]:5.2f}件/日 的中 {c[1]:5.2f}% "
                      f"ROI {c[2]:5.1f}% 低配当 {c[3]:4.2f}件/日")


if __name__ == "__main__":
    main()
