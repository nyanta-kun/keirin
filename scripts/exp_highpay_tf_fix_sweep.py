#!/usr/bin/env python3
"""三連単「1着固定 × 相手プール幅」の掃引と確認窓での検証（オッズ非使用）。

## 経緯

`exp_highpay_trifecta_structures.py` でユーザー仮説「三連単も相手絞る・着固定すれば
三連複よりROIを確保できる」が**支持された**:

| 構成 | 点数 | ROI（全件36,895R） |
|---|---|---|
| 三連複 得点3-7位ボックス | 10 | 49.0% |
| 三連単 モデル1位1着固定 | 12 | 71.2% |
| 三連単 ライン先頭→番手→流し | 5 | **72.9%** |

さらに荒れレース選別（1日上位2件）と組むと
`tf_1fix_rp4w`（得点4位1着固定・相手 得点1-5位）が
**30万+ 0.69% / 100倍+的中 2.93% / ROI 74.7% / 的中時配当中央 88.9倍**。
三連複ボックス（30万+ 0.26% / 100倍+ 1.98%）を明確に上回った。

そこで設計軸を掃引する。

## 掃引する軸

- `k1` = **1着に固定する車の順位**（1〜5位）。順位は `--rank-by` で
  競走得点順 / モデル3着内率順 を切替
- `m` = **相手プールの広さ**（上位 m 車。1着車を除いた中から2車の順列）
- 点数 = 2 × C(|pool|, 2)

## 事前宣言（採否の規則）

- **主要指標 = 30万円+率**。同水準ならROIで比較する
- 掃引窓 **2025-07-01〜2026-07-15** で候補を作り、確認窓 **2024-07-01〜2025-06-30** で
  一度きり検証する。掃引窓で最良だったセルのみを確認窓で見る
- 「両窓で符号が一致し、確認窓でも三連複ボックスを上回る」ことを条件とする
  （[[keirin_7car_rank_realignment_2026_08_06]] の符号反転を踏まえる）

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402

from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, build_rows, load_entries, load_races, load_win_payouts,
)

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000

# 確認窓(2024-07〜)も honest に覆うため窓を前に伸ばす
WF = [("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
      ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
      ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
      ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
      ("2026-07-01", "2026-08-04")]
SWEEP = ("2025-07-01", "2026-07-15")
CONFIRM = ("2024-07-01", "2025-06-30")


def load_model_preds() -> dict:
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl")))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for rk, fno, p in zip(df["race_key"], df["frame_no"], df["pp3"]):
        out[rk][int(fno)] = float(p)
    return dict(out)


def bets_for(order: list[int], k1: int, m: int) -> list[str]:
    """order の k1 番目(1-indexed)を1着固定、上位m車から2・3着の順列を作る。"""
    if len(order) < max(k1, m):
        return []
    head = order[k1 - 1]
    pool = [f for f in order[:m] if f != head]
    if len(pool) < 2:
        return []
    return [f"{head}-{a}-{b}" for a, b in permutations(pool, 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-by", default="rp", choices=["rp", "model"])
    ap.add_argument("--per-day", type=int, default=2,
                    help="荒れレース選別で1日あたり採用する件数（0=全件）")
    args = ap.parse_args()

    races = load_races(7)
    ents_by_race = load_entries(sorted(races))
    rows, winners = build_rows(races, ents_by_race, 7)
    trio_pay, tf_pay = load_win_payouts(sorted(winners), winners)
    rows = [r for r in rows if r["race_key"] in trio_pay]
    for r in rows:
        r["y50"] = 1 if trio_pay[r["race_key"]] >= 50 else 0

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y50"] for r in rows])
    dates = np.array([r["race_date"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for w_from, w_to in WF:
        tr, te = dates < w_from, (dates >= w_from) & (dates <= w_to)
        if te.sum() == 0 or tr.sum() < 3000:
            continue
        mdl = lgb.train({"objective": "binary", "learning_rate": 0.05,
                         "num_leaves": 31, "min_data_in_leaf": 100,
                         "feature_fraction": 0.8, "bagging_fraction": 0.8,
                         "bagging_freq": 1, "verbose": -1, "seed": 42},
                        lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        pred[te] = mdl.predict(X[te])
    for i, r in enumerate(rows):
        r["score"] = pred[i]
    evalrows = [r for r in rows if not np.isnan(r["score"])]
    print(f"[data] honest 評価対象 {len(evalrows):,} レース", flush=True)

    pp3_all = load_model_preds()

    def order_of(rk: str) -> list[int]:
        if args.rank_by == "rp":
            e = ents_by_race[rk]
            return [int(x["frame_no"]) for x in
                    sorted(e, key=lambda z: -(float(z["race_point"] or 0)))]
        p = pp3_all.get(rk)
        return sorted(p, key=lambda f: -p[f]) if p else []

    def run(win: tuple[str, str]) -> dict:
        sub = [r for r in evalrows if win[0] <= r["race_date"] <= win[1]]
        if args.per_day:
            by_day = defaultdict(list)
            for r in sub:
                by_day[r["race_date"]].append(r)
            sub = [r for d in by_day for r in
                   sorted(by_day[d], key=lambda x: -x["score"])[:args.per_day]]
        acc = defaultdict(lambda: {"n": 0, "npt": 0, "hit": 0, "big": 0,
                                   "h100": 0, "ret": 0.0, "pays": []})
        for r in sub:
            rk = r["race_key"]
            od = order_of(rk)
            if len(od) < 7:
                continue
            w_tf, w_tr = winners[rk]["trifecta"], winners[rk]["trio"]
            o_tf, o_tr = tf_pay.get(rk), trio_pay.get(rk)
            # 比較用ベースライン: 三連複 得点3-7位ボックス
            e = ents_by_race[rk]
            rp_ord = [int(x["frame_no"]) for x in
                      sorted(e, key=lambda z: -(float(z["race_point"] or 0)))]
            base = [frozenset(c) for c in combinations(rp_ord[2:7], 3)]
            cands = {("trio_box", 0): (base, w_tr, o_tr)}
            for k1 in (1, 2, 3, 4, 5):
                for m in (3, 4, 5, 6):
                    b = bets_for(od, k1, m)
                    if b:
                        cands[(k1, m)] = (b, w_tf, o_tf)
            for key, (combos, wcomb, o) in cands.items():
                a = acc[key]
                a["n"] += 1
                a["npt"] += len(combos)
                if wcomb in combos and o:
                    pay = STAKE / len(combos) * o
                    a["hit"] += 1
                    a["ret"] += pay
                    a["pays"].append(o)
                    if o >= 100:
                        a["h100"] += 1
                    if pay >= HIGHPAY:
                        a["big"] += 1
        return acc

    label = {"rp": "競走得点順", "model": "モデル3着内率順"}[args.rank_by]
    results = {}
    for wname, win in (("掃引窓", SWEEP), ("確認窓", CONFIRM)):
        acc = run(win)
        results[wname] = acc
        n = acc[("trio_box", 0)]["n"]
        print(f"\n{'=' * 88}\n=== {wname} {win[0]}〜{win[1]}  {n:,}レース  "
              f"（1着固定の順位は{label} / 1日上位{args.per_day or '全'}件）===")
        print("  1着 相手  点数  的中%   ROI%   30万+%  件数  100倍+%  配当中央")
        for key in sorted(acc, key=lambda k: (str(k[0]), k[1])):
            a = acc[key]
            if not a["n"]:
                continue
            nm = "三連複box" if key[0] == "trio_box" else f"{key[0]}位 上位{key[1]}"
            print(f"  {nm:<10} {a['npt'] / a['n']:5.1f} "
                  f"{a['hit'] / a['n'] * 100:6.2f} "
                  f"{a['ret'] / (a['n'] * STAKE) * 100:6.1f}  "
                  f"{a['big'] / a['n'] * 100:6.2f} {a['big']:5}  "
                  f"{a['h100'] / a['n'] * 100:6.2f}  "
                  f"{np.median(a['pays']) if a['pays'] else 0:8.1f}")

    # 掃引窓で 30万+率 最良のセルを確認窓で突合
    sw = results["掃引窓"]
    best = max((k for k in sw if k[0] != "trio_box"),
               key=lambda k: (sw[k]["big"] / max(sw[k]["n"], 1),
                              sw[k]["ret"] / max(sw[k]["n"], 1)))
    cf = results["確認窓"]
    print(f"\n### 掃引窓で最良のセル: 1着{best[0]}位 × 相手上位{best[1]}車")
    for wname, acc in (("掃引窓", sw), ("確認窓", cf)):
        a, b = acc[best], acc[("trio_box", 0)]
        print(f"  {wname}: 30万+ {a['big'] / a['n'] * 100:.2f}% "
              f"({a['big']}件) / ROI {a['ret'] / (a['n'] * STAKE) * 100:.1f}%  "
              f"← 三連複box 30万+ {b['big'] / b['n'] * 100:.2f}% / "
              f"ROI {b['ret'] / (b['n'] * STAKE) * 100:.1f}%")


if __name__ == "__main__":
    main()
