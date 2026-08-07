"""ベースモデル設計 第2段: ブレンド重み・点数の掃引 + 朝オッズ実装可能性（2026-08-07）。

第1段（exp_base_model_full_coverage.py）で「三連複の市場確率上位N組を買う」が
的中率・ROI ともに最良と出た。ここで確かめるのは3点:

  【1】モデルを混ぜる余地はどこかにあるか（ブレンド重み w の掃引）
  【2】点数 N と的中率・ROI のトレードオフの形
  【3】🔴 本番は朝8:00入稿。**朝オッズで選んでも同じ的中率が出るか**
       （board は最終オッズ。ここが崩れると設計ごと成立しない）

⚠️ 【3】の窓は wt_odds_snapshot(morning) がある 2026-06-08 以降のみ。
⚠️ DB 書き込みなし。
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

from src.strategy_wt import _race_zscore  # noqa: E402

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100


def market_qc(board: dict) -> dict:
    raw = {c: 1.0 / o for c, o in board.items() if o and o > 0}
    tot = sum(raw.values())
    return {c: v / tot for c, v in raw.items()} if tot > 0 else {}


def model_qc(p3: dict, cars: list[int]) -> dict:
    out = {frozenset(c): p3[c[0]] * p3[c[1]] * p3[c[2]]
           for c in itertools.combinations(sorted(cars), 3)}
    tot = sum(out.values())
    return {c: v / tot for c, v in out.items()} if tot > 0 else out


def settle(legs, top3, board):
    hit = frozenset(top3) in legs
    o = board[frozenset(top3)]
    return int(hit), len(legs) * STAKE, (round(o * 100) // 10 * 10) if hit else 0


# --------------------------------------------------------------------------
# 【1】【2】 重み w × 点数 N の掃引
# --------------------------------------------------------------------------
def sweep_w_n(races) -> pd.DataFrame:
    ws = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
    ns = [3, 5, 6, 8, 10, 12, 15]
    rows = []
    for r in races:
        board = r["board"]
        if len(board) < 35 or len(r["p3"]) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        qc = market_qc(board)
        mc = model_qc(r["p3"], sorted(r["p3"]))
        if not qc:
            continue
        for w in ws:
            sc = {c: (qc[c] ** (1 - w)) * (mc[c] ** w) for c in board}
            order = sorted(sc, key=lambda c: -sc[c])
            for n in ns:
                legs = order[:n]
                h, b, ret = settle(legs, top3, board)
                rows.append(dict(date=r["date"], w=w, n=n, hit=h, bet=b, ret=ret))
    df = pd.DataFrame(rows)
    df["win"] = np.where(df.date <= CONFIRM_END, "確認", "掃引")
    g = df.groupby(["w", "n", "win"]).agg(
        N=("hit", "size"), hit=("hit", "mean"), bet=("bet", "sum"), ret=("ret", "sum"))
    g["hit"] *= 100
    g["roi"] = 100 * g.ret / g.bet
    return g.reset_index()


# --------------------------------------------------------------------------
# 【3】朝オッズで選んで最終オッズで精算する
# --------------------------------------------------------------------------
def load_morning(keys: list[str]) -> dict:
    """{race_key: {frozenset(3車): 朝オッズ}}"""
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute("""
          SELECT race_key, combination, odds_value FROM keirin.wt_odds_snapshot
          WHERE snapshot_type='morning' AND bet_type='trio' AND race_key = ANY(%s)
        """, (keys,))
        out: dict[str, dict] = {}
        for rk, comb, o in cur.fetchall():
            if o is None or o <= 0:
                continue
            try:
                cs = frozenset(int(x) for x in comb.replace("-", "=").split("="))
            except ValueError:
                continue
            if len(cs) == 3:
                out.setdefault(rk, {})[cs] = float(o)
        return out


def morning_check(races) -> pd.DataFrame:
    tgt = [r for r in races if r["date"] >= "2026-06-08"
           and len(r["board"]) >= 35 and len(r["p3"]) == 7]
    mor = load_morning([r["rk"] for r in tgt])
    print(f"朝オッズ突合: 対象{len(tgt):,}R 中 {len(mor):,}R で朝オッズあり")
    rows = []
    for r in tgt:
        board, m = r["board"], mor.get(r["rk"])
        if not m or len(m) < 35:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        qf, qm = market_qc(board), market_qc(m)
        mc = model_qc(r["p3"], sorted(r["p3"]))
        of = sorted(qf, key=lambda c: -qf[c])
        om = sorted((c for c in qm if c in board), key=lambda c: -qm[c])
        # 朝オッズ × モデル w=0.3 のブレンド（モデルの寄与が朝でも無いかの確認用）
        bl = {c: (qm[c] ** 0.7) * (mc[c] ** 0.3) for c in qm if c in board}
        ob = sorted(bl, key=lambda c: -bl[c])
        for n in (5, 8, 10):
            for tag, order in (("最終オッズ選択", of), ("朝オッズ選択", om),
                               ("朝オッズ×モデル0.3", ob)):
                h, b, ret = settle(order[:n], top3, board)
                rows.append(dict(date=r["date"], n=n, tag=tag, hit=h, bet=b, ret=ret,
                                 topk_agree=len(set(of[:n]) & set(order[:n]))))
    df = pd.DataFrame(rows)
    g = df.groupby(["n", "tag"]).agg(
        N=("hit", "size"), hit=("hit", "mean"), bet=("bet", "sum"),
        ret=("ret", "sum"), agree=("topk_agree", "mean"))
    g["hit"] *= 100
    g["roi"] = 100 * g.ret / g.bet
    return g.reset_index()


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    print(f"読み込み: {len(races):,}R\n")

    print("=== 【1】【2】 ブレンド重み w（0=市場のみ, 1=モデルのみ）× 点数 N ===")
    g = sweep_w_n(races)
    for n in sorted(g.n.unique()):
        print(f"\n-- {n}点 --")
        print(f"{'w':>5s} | {'掃引 的中%':>9s} {'ROI%':>6s} | {'確認 的中%':>9s} {'ROI%':>6s}")
        for w in sorted(g.w.unique()):
            s = g[(g.n == n) & (g.w == w) & (g.win == "掃引")]
            c = g[(g.n == n) & (g.w == w) & (g.win == "確認")]
            if s.empty or c.empty:
                continue
            print(f"{w:5.2f} | {s.hit.iloc[0]:9.1f} {s.roi.iloc[0]:6.1f} | "
                  f"{c.hit.iloc[0]:9.1f} {c.roi.iloc[0]:6.1f}")

    print("\n\n=== 【3】朝オッズで選べるか（2026-06-08〜・最終オッズで精算）===")
    m = morning_check(races)
    print(f"{'点':>3s} {'選択方法':22s} {'n':>6s} {'的中%':>6s} {'ROI%':>6s} "
          f"{'最終選択との一致点数':>12s}")
    for _, r in m.iterrows():
        print(f"{r.n:3.0f} {r.tag:22s} {r.N:6,.0f} {r.hit:6.1f} {r.roi:6.1f} "
              f"{r.agree:12.2f}")


if __name__ == "__main__":
    main()
