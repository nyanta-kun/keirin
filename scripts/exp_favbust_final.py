#!/usr/bin/env python3
"""P-A 最終形: 100円単位・1万円以下の実購入規則 × 9車立て × 未使用期間検証。

## ユーザー指示（2026-08-06）

> 「最低単価は100円単位であり、1レース最大購入金額は10000円以下になるため、
>  買い目と各金額の調整をして下さい。9車立て、未使用期間での検証を進めて」

## 実購入規則（100円単位・合計 <= 10,000円）

    三連単枠 7,500円 / 三連複枠 2,500円
    各枠内で 100円単位の均等配分。端数は使わない（切り捨て）。
      unit = floor(枠 / 点数 / 100) * 100   （下限 100円。100円未満になるなら点数を削る）

7車: 三連単 8点 → 900円/点 = 7,200円 / 三連複 7.4点 → 300円/点 ≒ 2,100円
9車: 三連単 14点 → 500円/点 = 7,000円 / 三連複は上位5車BOX 10点 → 200円/点 = 2,000円

三連複の点数が増えすぎないよう **プール上位5車BOX（最大10点）に制限**する。

## 検証する3点

1. 100円単位への丸めが成績をどれだけ変えるか（理論配分との差）
2. **9車立て**（母集団・選別精度・買い目成績）
3. **未使用期間 2026-07-16〜2026-08-04**（掃引窓・確認窓のどちらにも使っていない）
   ⚠️ 検出力は極めて低い（7車で約64レース）。粗い反転の検出にしか使えない。

DB は読み取りのみ。9車のスコア済みキャッシュが無ければ作る（数分）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import scripts.exp_favbust_build as _B  # noqa: E402
from scripts.exp_favbust_roles import role_of  # noqa: E402
from scripts.exp_highpay_fav_bust import (  # noqa: E402
    FAV_FEATS, WF, fav_features, load_preds3,
)
from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, build_rows, load_entries, load_races,
)

CACHE_DIR = REPO / "data" / "exp_cache"
BUDGET_TF, BUDGET_TRIO, CAP = 7_500, 2_500, 10_000
SWEEP = ("2025-07-01", "2026-07-15")
CONFIRM = ("2024-10-01", "2025-06-30")
FRESH = ("2026-07-16", "2026-08-04")
LINE_ROLES = {"本命ライン番手", "本命ライン3番手以降"}


def load_preds9() -> dict:
    """9車の walk-forward 予測（`gen_wf_preds_9car.py` が生成）。"""
    import glob
    import pandas as pd
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds9_*.pkl")))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, tuple]] = defaultdict(dict)
    for rk, f, a, b, c in zip(df["race_key"], df["frame_no"],
                              df["pp3"], df["ppw"], df["pbad"]):
        out[rk][int(f)] = (float(a), float(b), float(c))
    return dict(out)


def scored(n_car: int) -> tuple[list, dict, dict]:
    """(スコア済みレース, entries, preds) を返す。キャッシュあり。"""
    path = CACHE_DIR / f"favbust_scored_n{n_car}.pkl"
    epath = CACHE_DIR / f"favbust_entries_n{n_car}.pkl"
    pr_all = load_preds3() if n_car == 7 else load_preds9()
    if path.exists() and epath.exists():
        with path.open("rb") as f:
            data = pickle.load(f)
        with epath.open("rb") as f:
            ents = pickle.load(f)
        print(f"[cache] n{n_car} scored={len(data):,}", flush=True)
        return data, ents, pr_all

    races = load_races(n_car)
    ents_full = load_entries(sorted(races))
    rows, _w = build_rows(races, ents_full, n_car)
    data = []
    for r in rows:
        rk = r["race_key"]
        e = ents_full[rk]
        ff = fav_features(e, pr_all.get(rk))
        if ff is None:
            continue
        fav = ff.pop("_fav")
        fo = next((x["finish_order"] for x in e if int(x["frame_no"]) == fav), None)
        if fo is None:
            continue
        data.append({**{f: r[f] for f in FEATURES}, **ff, "race_key": rk,
                     "race_date": r["race_date"], "fav": fav,
                     "bust": 1 if (fo == 0 or fo >= 4) else 0})
    cols = FEATURES + FAV_FEATS
    X = np.array([[d[c] for c in cols] for d in data], dtype=float)
    y = np.array([d["bust"] for d in data])
    dates = np.array([d["race_date"] for d in data])
    pred = np.full(len(data), np.nan)
    for w_from, w_to in WF:
        tr, te = dates < w_from, (dates >= w_from) & (dates <= w_to)
        if te.sum() == 0 or tr.sum() < (2000 if n_car == 7 else 800):
            continue
        m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                       "min_data_in_leaf": 80 if n_car == 7 else 30,
                       "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "seed": 42},
                      lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        pred[te] = m.predict(X[te])
    for i, d in enumerate(data):
        d["score"] = pred[i]
    data = [d for d in data if not np.isnan(d["score"])]
    ents = {rk: [{k: e[k] for k in ("frame_no", "line_group", "line_size", "line_pos",
                                    "is_line_leader", "race_point", "style",
                                    "prediction_mark")} for e in v]
            for rk, v in ents_full.items()}
    for obj, pth in ((data, path), (ents, epath)):
        tmp = pth.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pth)
    return data, ents, pr_all


def legs(others: list[int], roles: dict[int, str]) -> tuple[list, list]:
    pool = [f for f in others if roles.get(f) not in LINE_ROLES]
    lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)
    trio = ([frozenset(c) for c in combinations(pool[:5], 3)]
            if len(pool) >= 3 else [])          # プール上位5車BOX（最大10点）
    tf = []
    if lead is not None and len(pool) >= 3:
        rest = [f for f in pool if f != lead]
        tf = [f"{lead}-{a}-{c}" for a in rest[:2]
              for c in others if c not in (lead, a)]
    return trio, tf


def unit_of(budget: int, n: int) -> int:
    """枠内で100円単位の均等配分。100円未満になるなら0（＝買えない）。"""
    if n <= 0:
        return 0
    u = (budget // n) // 100 * 100
    return u if u >= 100 else 0


def settle(trio_legs, tf_legs, P, round100: bool) -> dict:
    if round100:
        ut = unit_of(BUDGET_TRIO, len(trio_legs))
        uf = unit_of(BUDGET_TF, len(tf_legs))
    else:
        ut = BUDGET_TRIO / len(trio_legs) if trio_legs else 0
        uf = BUDGET_TF / len(tf_legs) if tf_legs else 0
    cost = ut * len(trio_legs) + uf * len(tf_legs)
    assert cost <= CAP + 1e-6, cost
    pay = 0.0
    th = tf = 0
    if trio_legs and ut and P["trio"] in trio_legs:
        pay += ut * P["trio_odds"]
        th = 1
    if tf_legs and uf and P["tf"] in tf_legs:
        pay += uf * P["tf_odds"]
        tf = 1
    return {"pay": pay, "cost": cost, "trio_hit": th, "tf_hit": tf,
            "n_trio": len(trio_legs), "n_tf": len(tf_legs), "u_trio": ut, "u_tf": uf}


def agg(items: list[dict]) -> dict:
    if not items:
        return {}
    pays = np.array([x["pay"] for x in items])
    cost = sum(x["cost"] for x in items)
    return {"n": len(items), "購入額": cost / len(items),
            "的中%": float(np.mean(pays > 0) * 100),
            "三複%": float(np.mean([x["trio_hit"] for x in items]) * 100),
            "三単%": float(np.mean([x["tf_hit"] for x in items]) * 100),
            "ROI%": pays.sum() / cost * 100,
            "平均払戻": float(pays[pays > 0].mean()) if (pays > 0).any() else 0.0,
            "10万+": int((pays >= 100_000).sum()),
            "30万+": int((pays >= 300_000).sum())}


def line(tag, a):
    if not a:
        return f"  {tag:<22} （該当なし）"
    return (f"  {tag:<22} n={a['n']:5} 購入{a['購入額']:6.0f}円 的中{a['的中%']:6.2f}% "
            f"(三複{a['三複%']:5.2f}/三単{a['三単%']:5.2f}) ROI{a['ROI%']:7.1f}% "
            f"平均払戻{a['平均払戻']:9.0f}円 10万+{a['10万+']:4} 30万+{a['30万+']:4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()

    data, ents_all, pr_all = scored(args.n_car)
    # ⚠️ 払戻キャッシュは車数ごとに分ける（7車用を9車で使い回すと全件落ちる）
    _B.PAYCACHE = CACHE_DIR / f"favbust_payouts_n{args.n_car}.pkl"
    if args.n_car == 7 and not _B.PAYCACHE.exists():
        old = CACHE_DIR / "favbust_payouts.pkl"
        if old.exists():
            _B.PAYCACHE = old
    pay = _B.load_payouts(sorted({d["race_key"] for d in data}))
    data = [d for d in data if d["race_key"] in pay
            and pay[d["race_key"]]["trio_odds"] and pay[d["race_key"]]["tf_odds"]]
    y = np.array([d["bust"] for d in data])
    s = np.array([d["score"] for d in data])
    print(f"\n{'=' * 112}\n=== {args.n_car}車立て ===")
    print(f"母集団(軸1==WT◎) {len(data):,}R / バスト基準率 {y.mean() * 100:.2f}% / "
          f"AUC {roc_auc_score(y, s):.4f}")

    strat = [d for d in data if d["fav_ppw_gap12"] >= args.gap]
    if len(strat) < 100:
        print("抜け度条件を満たすレースが少なすぎます")
        return
    thr = np.quantile([d["score"] for d in strat], 1 - args.top_frac)
    sel = [d for d in strat if d["score"] >= thr]
    n_days = len({d["race_date"] for d in data})
    print(f"抜け度>={args.gap * 100:.0f}pt {len(strat):,}R "
          f"(バスト率 {np.mean([d['bust'] for d in strat]) * 100:.2f}%) → "
          f"上位{args.top_frac:.0%} {len(sel):,}R "
          f"({len(sel) / n_days:.2f}件/日・バスト率 "
          f"{np.mean([d['bust'] for d in sel]) * 100:.2f}%)")

    rows = []
    for d in sel:
        rk = d["race_key"]
        pr, e = pr_all.get(rk), ents_all.get(rk)
        if not pr or not e:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(e, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        tl, fl = legs(others, roles)
        if not tl or not fl:
            continue
        rows.append({"date": d["race_date"],
                     "r100": settle(tl, fl, pay[rk], True),
                     "ideal": settle(tl, fl, pay[rk], False)})
    if not rows:
        print("成立レースなし")
        return
    r0 = rows[0]["r100"]
    print(f"成立 {len(rows):,}R / 三連複 平均{np.mean([r['r100']['n_trio'] for r in rows]):.1f}点 "
          f"(1点{np.mean([r['r100']['u_trio'] for r in rows]):.0f}円) / "
          f"三連単 平均{np.mean([r['r100']['n_tf'] for r in rows]):.1f}点 "
          f"(1点{np.mean([r['r100']['u_tf'] for r in rows]):.0f}円) / "
          f"平均購入額 {np.mean([r['r100']['cost'] for r in rows]):.0f}円")

    print(f"\n--- 1. 100円単位への丸めの影響（全期間 {len(rows):,}R）---")
    print(line("理論配分(端数あり)", agg([r["ideal"] for r in rows])))
    print(line("100円単位(実購入)", agg([r["r100"] for r in rows])))

    print(f"\n--- 2. 窓別（100円単位・実購入）---")
    for nm, w in (("掃引窓", SWEEP), ("確認窓", CONFIRM), ("**未使用**", FRESH)):
        it = [r["r100"] for r in rows if w[0] <= r["date"] <= w[1]]
        print(line(f"{nm} {w[0]}〜{w[1]}", agg(it)))

    print(f"\n--- 3. 月次 ROI（100円単位）---")
    by_mo = defaultdict(list)
    for r in rows:
        by_mo[r["date"][:7]].append(r["r100"])
    vals = []
    for mo in sorted(by_mo):
        g = by_mo[mo]
        if len(g) < 10:
            continue
        a = agg(g)
        vals.append(a["ROI%"])
        print(f"  {mo} n={a['n']:4} 的中{a['的中%']:6.2f}% ROI{a['ROI%']:7.1f}% "
              f"30万+{a['30万+']:3}")
    v = np.array(vals)
    print(f"\n  月次ROI: 平均{v.mean():.1f}% 中央{np.median(v):.1f}% "
          f"100%超 {int((v > 100).sum())}/{len(v)} 最低{v.min():.1f}% 最高{v.max():.1f}%")


if __name__ == "__main__":
    main()
