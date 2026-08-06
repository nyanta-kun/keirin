#!/usr/bin/env python3
"""「モデル軸1 ＝ WINTICKET◎ の本命が4着以下に飛ぶレース」の選別（オッズ非使用）。

## ユーザー仮説（2026-08-06）

> 「指数から1軸目に選ばれる最上位が WINTICKET の◎と一致しているが、4着以下になる
>  条件を選ぶことで、市場が間違いなく上位と目星をつけた選手が飛ぶことにより
>  高配当につながる」

これまでの選別は **「三連複配当>=50倍」を直接予測**していた（AUC 0.5573・単一特徴に負けた）。
本仮説は**目的変数を差し替える**提案で、筋が良い理由が3つある:

1. **母集団が絞られる**。モデルと公式印が一致した＝市場が確信している本命に限定するので、
   「飛べば必ず荒れる」が構造的に保証される（穴が穴として機能する条件付き）。
2. **目的変数が個人の着順**なので、レース全体の配当より予測しやすいはず。
   実測でも軸1の外れ20.7%のうち95%は「走って負け」＝予測対象になる（CLAUDE.md）。
3. **買い目と直結する**。本命が飛ぶと分かれば本命を買い目から外せる。

## 検証すること

- A: 本命バスト（4着以下）を予測できるか（honest walk-forward・AUC と十分位）
- B: バスト確率上位のレースで実際に配当が跳ねるか
- C: そこに現行最良の買い目（1着=モデル5位 × 2着上位2 × 3着上位4 の6点）を置いたとき
     30万円+率が上がるか。**本命を買い目から完全に外す版**も併せて測る

## 事前宣言

主要指標 = **30万円+率**、副次 ROI。掃引窓 2025-07-01〜2026-07-15 で候補を作り、
確認窓 2024-07-01〜2025-06-30 で一度きり検証する。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, build_rows, load_entries, load_races, load_win_payouts,
)

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000
SWEEP = ("2025-07-01", "2026-07-15")
CONFIRM = ("2024-07-01", "2025-06-30")
WF = [("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
      ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
      ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
      ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
      ("2026-07-01", "2026-08-04")]

STYLE_ENC = {"逃": 0, "両": 1, "追": 2}

# 本命（軸1）自身の属性。レース全体の特徴に上乗せする
FAV_FEATS = [
    "fav_pp3", "fav_ppw", "fav_pbad", "fav_pp3_gap12", "fav_ppw_gap12",
    "fav_rp_rank", "fav_rp", "fav_rp_gap_next", "fav_rp_gap_mean",
    "fav_frame", "fav_line_size", "fav_line_pos", "fav_is_leader", "fav_is_solo",
    "fav_style", "fav_class",
    "fav_line_rp_sum", "fav_line_rank",
    "taikou_same_line", "taikou_pp3", "taikou_rp_gap",
    "n_stronger_line",
]


def load_preds3() -> dict:
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl")))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, tuple]] = defaultdict(dict)
    for rk, f, a, b, c in zip(df["race_key"], df["frame_no"],
                              df["pp3"], df["ppw"], df["pbad"]):
        out[rk][int(f)] = (float(a), float(b), float(c))
    return dict(out)


def fav_features(ents: list[dict], pr: dict[int, tuple]) -> dict | None:
    """軸1（pred_win 最上位）が WT◎ と一致するレースだけ特徴を返す。"""
    if not pr or len(pr) < 7:
        return None
    by_frame = {int(e["frame_no"]): e for e in ents}
    fav = max(pr, key=lambda f: pr[f][1])                 # ppw 最上位＝軸1
    honmei = next((int(e["frame_no"]) for e in ents
                   if e["prediction_mark"] == 1), None)
    if honmei is None or fav != honmei:
        return None                                       # ◎と不一致は対象外
    taikou = next((int(e["frame_no"]) for e in ents
                   if e["prediction_mark"] == 2), None)

    pp3 = sorted((pr[f][0] for f in pr), reverse=True)
    ppw = sorted((pr[f][1] for f in pr), reverse=True)
    rps = sorted(((int(e["frame_no"]), float(e["race_point"] or 0))
                  for e in ents), key=lambda x: -x[1])
    rp_order = [f for f, _ in rps]
    rp_vals = [v for _, v in rps]
    fav_rank = rp_order.index(fav)
    e = by_frame[fav]

    lines = defaultdict(list)
    for x in ents:
        lines[x["line_group"] if x["line_group"] is not None
              else f"s{x['frame_no']}"].append(x)
    fav_lg = e["line_group"] if e["line_group"] is not None else f"s{fav}"
    line_sums = {k: sum(float(z["race_point"] or 0) for z in v)
                 for k, v in lines.items()}
    fav_line_sum = line_sums[fav_lg]
    line_rank = sorted(line_sums.values(), reverse=True).index(fav_line_sum)

    return {
        "fav_pp3": pr[fav][0], "fav_ppw": pr[fav][1], "fav_pbad": pr[fav][2],
        "fav_pp3_gap12": pp3[0] - pp3[1], "fav_ppw_gap12": ppw[0] - ppw[1],
        "fav_rp_rank": float(fav_rank), "fav_rp": rp_vals[fav_rank],
        "fav_rp_gap_next": (rp_vals[fav_rank] - rp_vals[fav_rank + 1]
                            if fav_rank + 1 < len(rp_vals) else 0.0),
        "fav_rp_gap_mean": rp_vals[fav_rank] - float(np.mean(rp_vals)),
        "fav_frame": float(fav),
        "fav_line_size": float(e["line_size"] or 1),
        "fav_line_pos": float(e["line_pos"] or 0),
        "fav_is_leader": float(e["is_line_leader"] or 0),
        "fav_is_solo": 1.0 if (e["line_size"] or 1) == 1 else 0.0,
        "fav_style": float(STYLE_ENC.get(e["style"], -1)),
        "fav_class": float(1 if str(e["player_class"] or "").startswith("S") else 0),
        "fav_line_rp_sum": fav_line_sum, "fav_line_rank": float(line_rank),
        "taikou_same_line": (1.0 if taikou is not None
                             and by_frame[taikou]["line_group"] == e["line_group"]
                             and e["line_group"] is not None else 0.0),
        "taikou_pp3": pr.get(taikou, (0, 0, 0))[0] if taikou else 0.0,
        "taikou_rp_gap": (rp_vals[fav_rank]
                          - float(by_frame[taikou]["race_point"] or 0)
                          if taikou else 0.0),
        "n_stronger_line": float(line_rank),
        "_fav": fav,
    }


def formation(order: list[int], k1: int, m2: int, m3: int,
              exclude: int | None = None) -> list[str]:
    od = [f for f in order if f != exclude] if exclude is not None else list(order)
    if len(od) < max(k1, m2, m3):
        return []
    head = od[k1 - 1]
    p2 = [f for f in od[:m2] if f != head]
    p3 = [f for f in od[:m3] if f != head]
    return [f"{head}-{b}-{c}" for b in p2 for c in p3 if c != b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-frac", type=float, default=0.20,
                    help="バスト確率の上位何割を選ぶか")
    args = ap.parse_args()

    races = load_races(7)
    ents_by_race = load_entries(sorted(races))
    rows, winners = build_rows(races, ents_by_race, 7)
    trio_pay, tf_pay = load_win_payouts(sorted(winners), winners)
    pr_all = load_preds3()

    data = []
    for r in rows:
        rk = r["race_key"]
        if rk not in tf_pay:
            continue
        ents = ents_by_race[rk]
        ff = fav_features(ents, pr_all.get(rk))
        if ff is None:
            continue
        fav = ff.pop("_fav")
        fo = next((x["finish_order"] for x in ents if int(x["frame_no"]) == fav), None)
        if fo is None:
            continue
        bust = 1 if (fo == 0 or fo >= 4) else 0        # 0=欠車/失格も着外
        d = {**{f: r[f] for f in FEATURES}, **ff,
             "race_key": rk, "race_date": r["race_date"],
             "bust": bust, "dnf": 1 if fo == 0 else 0,
             "trio_odds": trio_pay.get(rk, 0.0), "tf_odds": tf_pay[rk], "fav": fav}
        data.append(d)

    print(f"[data] 軸1==WT◎ のレース {len(data):,} / 全7車 {len(rows):,} "
          f"({len(data) / max(len(rows), 1) * 100:.1f}%)")
    y = np.array([d["bust"] for d in data])
    print(f"[data] 本命バスト(4着以下・欠車含む) 基準率 {y.mean() * 100:.2f}%  "
          f"うち欠車失格 {np.mean([d['dnf'] for d in data]) * 100:.2f}%")

    cols = FEATURES + FAV_FEATS
    X = np.array([[d[c] for c in cols] for d in data], dtype=float)
    dates = np.array([d["race_date"] for d in data])
    pred = np.full(len(data), np.nan)
    for w_from, w_to in WF:
        tr, te = dates < w_from, (dates >= w_from) & (dates <= w_to)
        if te.sum() == 0 or tr.sum() < 2000:
            continue
        m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                       "min_data_in_leaf": 80, "feature_fraction": 0.8,
                       "bagging_fraction": 0.8, "bagging_freq": 1,
                       "verbose": -1, "seed": 42},
                      lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        pred[te] = m.predict(X[te])
    ok = ~np.isnan(pred)
    print(f"\n=== A: 本命バスト予測 honest n={ok.sum():,}  "
          f"AUC {roc_auc_score(y[ok], pred[ok]):.4f} ===")
    for i, d in enumerate(data):
        d["score"] = pred[i]
    ev = [d for d in data if not np.isnan(d["score"])]

    # --- 十分位: バスト率と配当 ---
    sc = np.array([d["score"] for d in ev])
    bb = np.array([d["bust"] for d in ev])
    to = np.array([d["trio_odds"] for d in ev])
    order = np.argsort(-sc)
    print("  分位   件数   本命バスト%   lift   三連複配当中央  50倍+%")
    n = len(ev)
    for b in range(10):
        lo, hi = n * b // 10, n * (b + 1) // 10
        idx = order[lo:hi]
        print(f"  D{b + 1:<2}  {len(idx):6}  {bb[idx].mean() * 100:9.2f}  "
              f"{bb[idx].mean() / bb.mean():5.2f}  {np.median(to[idx]):12.1f}  "
              f"{np.mean(to[idx] >= 50) * 100:6.2f}")

    # --- 特徴量重要度 ---
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])[:12]
    print("\n  重要度上位12: " + " / ".join(f"{f}" for f, _ in imp))

    # --- C: 買い目を置いたときの高額到達 ---
    for wname, win in (("掃引窓", SWEEP), ("確認窓", CONFIRM)):
        sub = [d for d in ev if win[0] <= d["race_date"] <= win[1]]
        if not sub:
            continue
        thr = np.quantile([d["score"] for d in sub], 1 - args.top_frac)
        print(f"\n{'=' * 84}\n=== {wname} {win[0]}〜{win[1]}  "
              f"母集団{len(sub):,}レース / バスト上位{args.top_frac:.0%}を選別 ===")
        print("  選別        買い目                     点数  的中%   ROI%   "
              "30万+%  件数  配当中央")
        for sel_name, keep in (("全件", None), (f"上位{args.top_frac:.0%}", thr)):
            tgt = sub if keep is None else [d for d in sub if d["score"] >= keep]
            acc = defaultdict(lambda: {"n": 0, "npt": 0, "hit": 0, "big": 0,
                                       "ret": 0.0, "pays": []})
            for d in tgt:
                pr = pr_all[d["race_key"]]
                od = sorted(pr, key=lambda f: -pr[f][0])       # モデル3着内率順
                w = winners[d["race_key"]]["trifecta"]
                o = d["tf_odds"]
                variants = {
                    "現行(5位1着×上2×上4)": formation(od, 5, 2, 4),
                    "本命除外(4位1着×上2×上4)": formation(od, 4, 2, 4, exclude=d["fav"]),
                    "本命除外(3位1着×上2×上4)": formation(od, 3, 2, 4, exclude=d["fav"]),
                }
                for nm, combos in variants.items():
                    if not combos:
                        continue
                    a = acc[nm]
                    a["n"] += 1
                    a["npt"] += len(combos)
                    if w in combos:
                        pay = STAKE / len(combos) * o
                        a["hit"] += 1
                        a["ret"] += pay
                        a["pays"].append(o)
                        if pay >= HIGHPAY:
                            a["big"] += 1
            for nm, a in acc.items():
                if not a["n"]:
                    continue
                print(f"  {sel_name:<10} {nm:<26} {a['npt'] / a['n']:4.1f} "
                      f"{a['hit'] / a['n'] * 100:6.2f} "
                      f"{a['ret'] / (a['n'] * STAKE) * 100:6.1f}  "
                      f"{a['big'] / a['n'] * 100:6.2f} {a['big']:5}  "
                      f"{np.median(a['pays']) if a['pays'] else 0:8.1f}")


if __name__ == "__main__":
    main()
