#!/usr/bin/env python3
"""三連単の着固定・相手絞りが三連複よりROIと高額到達を稼げるかを測る（オッズ非使用）。

## ユーザー仮説（2026-08-06）

> 「三連単も相手絞る、着固定することで三連複よりROI確保できると思います」

機序としては筋が通る: 7車なら三連複35通りに対し三連単210通り＝**1点あたりのオッズが
約6倍**。同じ構造的確信度でも払い戻しが30万円側に寄る。
`exp_highpay_structural_bets.py` の三連複は的中時配当中央が26〜46倍しかなく、
10点に割ると3万円前後で頭打ちだった。これを三連単で越えられるかを検証する。

## 検証する三連単の構成（すべてオッズ非依存・朝の時点で確定）

| 名前 | 1着 | 2・3着 | 点数 |
|---|---|---|---|
| `tf_box_mdl35` | モデル3着内率3-5位の3車ボックス | 6 |
| `tf_1fix_mdl1` | モデル1位固定 | モデル2-5位から2車の順列 | 12 |
| `tf_1fix_hole3` | **モデル3位固定（人気薄1着）** | モデル1,2,4,5位から2車の順列 | 12 |
| `tf_1fix_rp4` | **得点4位固定** | 得点1-3位の順列 | 6 |
| `tf_1fix_rp4w` | 得点4位固定 | 得点1-5位から2車の順列（自分除く） | 12 |
| `tf_12ax_3rd` | モデル1・2位の順列2通り | 3着=モデル3-7位 | 10 |
| `tf_3rd_hole` | モデル1・2位の順列2通り | 3着=得点下位2車 | 4 |
| `tf_line_nige` | ライン先頭固定 | 同ライン番手→残り流し | 可変 |
| 比較用 `trio_rp37` | 三連複 得点3-7位ボックス | 10 |
| 比較用 `trio_mdl35` | 三連複 モデル3-5位（1点） | 1 |

## 指標

1レース1万円・等分。的中目のオッズ o に対し払戻 = (10000/N)·o。
**ROI** / **30万円+率** / 100倍+的中率 / 的中時配当中央 を並べる。
レース選別（高配当レース選別モデルの1日上位2件）ありなしの両方。

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
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, WF_WINDOWS, build_rows, load_entries, load_races, load_win_payouts,
)

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000


def load_model_preds() -> dict:
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl")))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for rk, fno, p in zip(df["race_key"], df["frame_no"], df["pp3"]):
        out[rk][int(fno)] = float(p)
    return dict(out)


def _perm2(pool: list[int]) -> list[tuple[int, int]]:
    return [p for p in permutations(pool, 2)]


def make_bets(ents: list[dict], pp3: dict[int, float] | None) -> dict[str, tuple]:
    """(bet_type, 買い目リスト) を返す。三連単は 'a-b-c' 文字列、三連複は frozenset。"""
    out: dict[str, tuple] = {}
    by_rp = sorted(ents, key=lambda e: -(float(e["race_point"] or 0)))
    rp = [int(e["frame_no"]) for e in by_rp]
    if len(rp) < 7:
        return out

    # --- 三連複（比較用） ---
    out["trio_rp37"] = ("trio", [frozenset(c) for c in combinations(rp[2:7], 3)])

    # --- 得点順に基づく三連単 ---
    ax = rp[3]                                   # 得点4位＝人気薄寄りの1着固定
    out["tf_1fix_rp4"] = ("trifecta",
                          [f"{ax}-{a}-{b}" for a, b in _perm2(rp[0:3])])
    pool = [f for f in rp[0:5] if f != ax]
    out["tf_1fix_rp4w"] = ("trifecta",
                           [f"{ax}-{a}-{b}" for a, b in _perm2(pool)])

    if not pp3:
        return out
    by_p = sorted(pp3, key=lambda f: -pp3[f])
    if len(by_p) < 7:
        return out
    m1, m2, m3 = by_p[0], by_p[1], by_p[2]

    out["trio_mdl35"] = ("trio", [frozenset(by_p[2:5])])
    out["tf_box_mdl35"] = ("trifecta",
                           [f"{a}-{b}-{c}" for a, b, c in permutations(by_p[2:5], 3)])
    out["tf_1fix_mdl1"] = ("trifecta",
                           [f"{m1}-{a}-{b}" for a, b in _perm2(by_p[1:5])])
    hole_pool = [by_p[0], by_p[1], by_p[3], by_p[4]]
    out["tf_1fix_hole3"] = ("trifecta",
                            [f"{m3}-{a}-{b}" for a, b in _perm2(hole_pool)])
    out["tf_12ax_3rd"] = ("trifecta",
                          [f"{a}-{b}-{c}" for a, b in ((m1, m2), (m2, m1))
                           for c in by_p[2:7]])
    low2 = rp[-2:]
    out["tf_3rd_hole"] = ("trifecta",
                          [f"{a}-{b}-{c}" for a, b in ((m1, m2), (m2, m1))
                           for c in low2 if c not in (m1, m2)])

    # --- ライン構造: 先頭固定→同ライン番手→残り流し ---
    by_frame = {int(e["frame_no"]): e for e in ents}
    lg = by_frame[m1]["line_group"]
    mates = [int(e["frame_no"]) for e in ents
             if e["line_group"] is not None and e["line_group"] == lg
             and int(e["frame_no"]) != m1]
    if mates:
        mate = min(mates, key=lambda f: (by_frame[f]["line_pos"] or 9))
        rest = [f for f in by_p if f not in (m1, mate)]
        out["tf_line_nige"] = ("trifecta", [f"{m1}-{mate}-{c}" for c in rest])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    args = ap.parse_args()

    races = load_races(args.n_car)
    ents_by_race = load_entries(sorted(races))
    rows, winners = build_rows(races, ents_by_race, args.n_car)
    trio_pay, tf_pay = load_win_payouts(sorted(winners), winners)
    rows = [r for r in rows if r["race_key"] in trio_pay]
    for r in rows:
        r["y50"] = 1 if trio_pay[r["race_key"]] >= 50 else 0

    # 高配当レース選別モデル（honest walk-forward）
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y50"] for r in rows])
    dates = np.array([r["race_date"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for w_from, w_to in WF_WINDOWS:
        tr, te = dates < w_from, (dates >= w_from) & (dates <= w_to)
        if te.sum() == 0 or tr.sum() < 3000:
            continue
        m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                       "min_data_in_leaf": 100, "feature_fraction": 0.8,
                       "bagging_fraction": 0.8, "bagging_freq": 1,
                       "verbose": -1, "seed": 42},
                      lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        pred[te] = m.predict(X[te])
    for i, r in enumerate(rows):
        r["score"] = pred[i]
    evalrows = [r for r in rows if not np.isnan(r["score"])]
    ok = ~np.isnan(pred)
    print(f"\n[選別モデル] honest n={ok.sum()} AUC {roc_auc_score(y[ok], pred[ok]):.4f}")

    pp3_all = load_model_preds()
    by_day = defaultdict(list)
    for r in evalrows:
        by_day[r["race_date"]].append(r)
    top2 = {r["race_key"] for d in by_day
            for r in sorted(by_day[d], key=lambda x: -x["score"])[:2]}

    order = ["trio_rp37", "trio_mdl35", "tf_box_mdl35", "tf_1fix_mdl1",
             "tf_1fix_hole3", "tf_1fix_rp4", "tf_1fix_rp4w", "tf_12ax_3rd",
             "tf_3rd_hole", "tf_line_nige"]

    for sname, keep in (("全件", None), ("1日上位2件（荒れ選別）", top2)):
        sub = [r for r in evalrows if keep is None or r["race_key"] in keep]
        print(f"\n{'=' * 92}\n=== {sname}  {len(sub):,}レース ===")
        print("  買い目           点数  的中%   ROI%   30万+%  30万件数  "
              "100倍+的中%  的中時配当中央")
        acc = defaultdict(lambda: {"n": 0, "npt": 0, "hit": 0, "big": 0,
                                   "h100": 0, "ret": 0.0, "pays": []})
        for r in sub:
            e = ents_by_race.get(r["race_key"])
            if not e:
                continue
            bets = make_bets(e, pp3_all.get(r["race_key"]))
            w_tr = winners[r["race_key"]]["trio"]
            w_tf = winners[r["race_key"]]["trifecta"]
            for name, (bt, combos) in bets.items():
                if not combos:
                    continue
                n_pt = len(combos)
                a = acc[name]
                a["n"] += 1
                a["npt"] += n_pt
                if bt == "trio":
                    hit, o = (w_tr in combos), trio_pay.get(r["race_key"])
                else:
                    hit, o = (w_tf in combos), tf_pay.get(r["race_key"])
                if hit and o:
                    pay = STAKE / n_pt * o
                    a["hit"] += 1
                    a["ret"] += pay
                    a["pays"].append(o)
                    if o >= 100:
                        a["h100"] += 1
                    if pay >= HIGHPAY:
                        a["big"] += 1
        for name in order:
            a = acc.get(name)
            if not a or not a["n"]:
                continue
            print(f"  {name:<16} {a['npt'] / a['n']:5.1f} "
                  f"{a['hit'] / a['n'] * 100:6.2f} "
                  f"{a['ret'] / (a['n'] * STAKE) * 100:6.1f}  "
                  f"{a['big'] / a['n'] * 100:6.2f}  {a['big']:8}  "
                  f"{a['h100'] / a['n'] * 100:9.2f}  "
                  f"{np.median(a['pays']) if a['pays'] else 0:11.1f}")


if __name__ == "__main__":
    main()
