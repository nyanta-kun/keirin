#!/usr/bin/env python3
"""三連単フォーメーション（1着1車固定 × 2着m2車 × 3着m3車）の掃引（オッズ非使用）。

## netkeirin 実務の観察（2026-08-06・ユーザー提示）

`https://keirin.netkeiba.com/yoso/hot/` の高額的中を実地調査したところ、
**上位予想家の高額的中は全例が同じ形**だった:

| 予想家 | レース | 券種 | 構成 | 1点 | 的中オッズ | 払戻 |
|---|---|---|---|---|---|---|
| シュウのAI指数極 | 和歌山1R(9車) | 三連単 | **1着1×2着1×3着1＝1点** | 10,000円 | 109.7倍 | 1,097,000円 |
| シュウのAI指数極 | 豊橋4R | 三連単 | **1着1車固定・2点** | 5,000円 | 300.7倍 | 1,503,500円 |
| Equine Genius | 別府6R(7車) | 三連単 | **1着1×2着2×3着5＝8点** | 1,200円 | 72.1倍 | 86,520円 |

共通点: **券種は三連単フォーメーション / 1着は必ず1車固定 / 予算は約1万円 / 均等配分 /
点数は1〜8点**。我々が `exp_highpay_tf_fix_sweep.py` で辿り着いた形と一致する。

さらに同予想家の商品成績は **回収率 52% / 73% / 75%（的中率 6% / 7% / 9%）**。
我々の実測（三連単着固定 ROI 62〜73%）と同水準で、**実務でも黒字ではない**。
つまり高額的中の実績は「回収率を犠牲にして点数を絞る」ことで作られている。

## 本スクリプトで新たに測る点

既存の掃引は「相手プールから2車の**順列**」＝2着と3着が対称だった。
実務は **2着を強く絞り3着を広げる非対称フォーメーション**（別府の 1×2×5）を使う。
これは未測定なので、ここで一般化して掃引する:

    1着 = 順位 k1 の1車固定
    2着 = 上位 m2 車（1着を除く）
    3着 = 上位 m3 車（1着・2着を除く）   ※ m3 >= m2

## 事前宣言

主要指標 = **30万円+率**、同水準ならROI。掃引窓 2025-07-01〜2026-07-15 で候補を作り、
確認窓 2024-07-01〜2025-06-30 で一度きり検証する。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_highpay_race_model import (  # noqa: E402
    build_rows, load_entries, load_races, load_win_payouts,
)

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000
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


def formation(order: list[int], k1: int, m2: int, m3: int) -> list[str]:
    """1着=order[k1-1] 固定、2着=上位m2車、3着=上位m3車。"""
    if len(order) < max(k1, m2, m3):
        return []
    head = order[k1 - 1]
    p2 = [f for f in order[:m2] if f != head]
    p3 = [f for f in order[:m3] if f != head]
    if not p2:
        return []
    return [f"{head}-{b}-{c}" for b in p2 for c in p3 if c != b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-by", default="rp", choices=["rp", "model"])
    args = ap.parse_args()

    races = load_races(7)
    ents_by_race = load_entries(sorted(races))
    rows, winners = build_rows(races, ents_by_race, 7)
    trio_pay, tf_pay = load_win_payouts(sorted(winners), winners)
    rows = [r for r in rows if r["race_key"] in tf_pay]
    pp3_all = load_model_preds()

    def order_of(rk: str) -> list[int]:
        if args.rank_by == "rp":
            e = ents_by_race[rk]
            return [int(x["frame_no"]) for x in
                    sorted(e, key=lambda z: -(float(z["race_point"] or 0)))]
        p = pp3_all.get(rk)
        return sorted(p, key=lambda f: -p[f]) if p else []

    cells = [(k1, m2, m3) for k1 in (1, 2, 3, 4, 5)
             for m2 in (2, 3, 4) for m3 in (4, 5, 6, 7) if m3 >= m2]

    def run(win):
        sub = [r for r in rows if win[0] <= r["race_date"] <= win[1]]
        acc = defaultdict(lambda: {"n": 0, "npt": 0, "hit": 0, "big": 0,
                                   "h100": 0, "ret": 0.0, "pays": []})
        for r in sub:
            rk = r["race_key"]
            od = order_of(rk)
            if len(od) < 7:
                continue
            w, o = winners[rk]["trifecta"], tf_pay.get(rk)
            # 比較用: 三連複 得点3-7位ボックス
            e = ents_by_race[rk]
            rp_ord = [int(x["frame_no"]) for x in
                      sorted(e, key=lambda z: -(float(z["race_point"] or 0)))]
            cand = {("trio_box", 0, 0):
                    ([frozenset(c) for c in combinations(rp_ord[2:7], 3)],
                     winners[rk]["trio"], trio_pay.get(rk))}
            for key in cells:
                b = formation(od, *key)
                if b:
                    cand[key] = (b, w, o)
            for key, (combos, wc, oo) in cand.items():
                a = acc[key]
                a["n"] += 1
                a["npt"] += len(combos)
                if wc in combos and oo:
                    pay = STAKE / len(combos) * oo
                    a["hit"] += 1
                    a["ret"] += pay
                    a["pays"].append(oo)
                    if oo >= 100:
                        a["h100"] += 1
                    if pay >= HIGHPAY:
                        a["big"] += 1
        return acc

    label = {"rp": "競走得点順", "model": "モデル3着内率順"}[args.rank_by]
    res = {}
    for wname, win in (("掃引窓", SWEEP), ("確認窓", CONFIRM)):
        acc = run(win)
        res[wname] = acc
        print(f"\n{'=' * 86}\n=== {wname} {win[0]}〜{win[1]}  "
              f"{acc[('trio_box', 0, 0)]['n']:,}レース  （順位は{label}）===")
        print("  1着 2着 3着  点数  的中%   ROI%   30万+%  件数  100倍+%  配当中央")
        ranked = sorted((k for k in acc if k[0] != "trio_box"),
                        key=lambda k: -(acc[k]["big"] / max(acc[k]["n"], 1)))
        for key in ranked[:14]:
            a = acc[key]
            print(f"  {key[0]}位 上{key[1]} 上{key[2]}  {a['npt'] / a['n']:5.1f} "
                  f"{a['hit'] / a['n'] * 100:6.2f} "
                  f"{a['ret'] / (a['n'] * STAKE) * 100:6.1f}  "
                  f"{a['big'] / a['n'] * 100:6.2f} {a['big']:5}  "
                  f"{a['h100'] / a['n'] * 100:6.2f}  "
                  f"{np.median(a['pays']) if a['pays'] else 0:8.1f}")
        b = acc[("trio_box", 0, 0)]
        print(f"  三連複box       {b['npt'] / b['n']:5.1f} "
              f"{b['hit'] / b['n'] * 100:6.2f} "
              f"{b['ret'] / (b['n'] * STAKE) * 100:6.1f}  "
              f"{b['big'] / b['n'] * 100:6.2f} {b['big']:5}  "
              f"{b['h100'] / b['n'] * 100:6.2f}")

    sw, cf = res["掃引窓"], res["確認窓"]
    best = max((k for k in sw if k[0] != "trio_box"),
               key=lambda k: (sw[k]["big"] / max(sw[k]["n"], 1),
                              sw[k]["ret"] / max(sw[k]["n"], 1)))
    print(f"\n### 掃引窓 最良: 1着{best[0]}位 × 2着上位{best[1]} × 3着上位{best[2]}")
    for wname, acc in (("掃引窓", sw), ("確認窓", cf)):
        a = acc[best]
        print(f"  {wname}: 点数{a['npt'] / a['n']:.1f} 的中{a['hit'] / a['n'] * 100:.2f}% "
              f"30万+ {a['big'] / a['n'] * 100:.2f}% ({a['big']}件) "
              f"ROI {a['ret'] / (a['n'] * STAKE) * 100:.1f}% "
              f"配当中央 {np.median(a['pays']) if a['pays'] else 0:.1f}倍")


if __name__ == "__main__":
    main()
