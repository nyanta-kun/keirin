#!/usr/bin/env python3
"""オッズを使わない買い目構成 × 高配当レース選別（Phase 2）。

## 設計の考え方

オッズ非使用が前提なので、買い目は**構造（競走得点順・WT公式印・ライン）**だけで決める。
このとき「自分の買い目が結果的に何倍だったか」は事前に分からない。
ここで初めて**高配当レース選別モデルの役目が立つ**:

- 堅いレースで構造的な人気薄目を買うと、当たったとしても配当は付かない
  （そもそも当たらない）。
- 荒れるレースで同じ目を買うと、当たれば高配当になる。

つまりレース選別は**オッズ情報の代替**として働く。`exp_highpay_race_selection.py` で
「オッズを見て帯を固定した場合はレース選別が無効」と出たのと矛盾しない
（あちらはオッズで帯を固定済み＝選別の役目が既に果たされている状態）。

## 測る指標（ユーザーの目的に合わせる）

1レース1万円・N点等分。的中目のオッズを o とすると payout = (10000/N)·o。

- **50倍+的中率**: o >= 50 の目を的中した割合（＝見出しになる的中）
- **30万円+率**: payout >= 300,000 の割合（＝当初の「高額払い戻し」）
- 参考: 全的中率・ROI・的中時配当中央値

## 買い目（すべてオッズ非依存・朝の時点で確定）

| 名前 | 内容 | 点数 |
|---|---|---|
| `rp37_box` | 競走得点 3〜7位の5車ボックス三連複 | 10 |
| `rp47_box` | 競走得点 4〜7位の4車ボックス三連複 | 4 |
| `rp4_axis` | 得点4位を軸・相手は得点3〜7位から2車 | 6 |
| `nomark_box` | WT◎◯を外した5車ボックス三連複 | 10 |
| `nomark_ax` | WT◎を外し、モデル3着内率の2〜4位を軸2車＋残り流し | 可変 |
| `mdl35_box` | モデル3着内率 3〜5位の3車（1点） | 1 |
| `mdl37_box` | モデル3着内率 3〜7位の5車ボックス | 10 |

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

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, WF_WINDOWS, build_rows, load_entries, load_races, load_win_payouts,
)

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000


def load_model_preds() -> dict:
    """honest walk-forward の3着内率（pp3）を race_key -> {frame: pp3} で返す。"""
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl")))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for rk, fno, p in zip(df["race_key"], df["frame_no"], df["pp3"]):
        out[rk][int(fno)] = float(p)
    return dict(out)


def make_bets(ents: list[dict], pp3: dict[int, float] | None) -> dict[str, list]:
    """オッズ非依存の買い目集合（三連複の frozenset のリスト）を返す。"""
    by_rp = sorted(ents, key=lambda e: -(float(e["race_point"] or 0)))
    rp_order = [int(e["frame_no"]) for e in by_rp]
    marks = {int(e["frame_no"]): e["prediction_mark"] for e in ents}
    honmei = next((f for f, m in marks.items() if m == 1), None)
    taikou = next((f for f, m in marks.items() if m == 2), None)

    out: dict[str, list] = {}
    if len(rp_order) >= 7:
        out["rp37_box"] = [frozenset(c) for c in combinations(rp_order[2:7], 3)]
        out["rp47_box"] = [frozenset(c) for c in combinations(rp_order[3:7], 3)]
        ax = rp_order[3]
        out["rp4_axis"] = [frozenset((ax, *c)) for c in
                           combinations([f for f in rp_order[2:7] if f != ax], 2)]
    rest = [f for f in rp_order if f not in (honmei, taikou)]
    if len(rest) >= 5:
        out["nomark_box"] = [frozenset(c) for c in combinations(rest[:5], 3)]

    if pp3:
        by_p = sorted(pp3, key=lambda f: -pp3[f])
        if len(by_p) >= 7:
            out["mdl35_box"] = [frozenset(by_p[2:5])]
            out["mdl37_box"] = [frozenset(c) for c in combinations(by_p[2:7], 3)]
        rest2 = [f for f in by_p if f != honmei]
        if len(rest2) >= 6:
            a1, a2 = rest2[0], rest2[1]
            out["nomark_ax"] = [frozenset((a1, a2, o)) for o in rest2[2:]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--select", default="all,d1,top2day",
                    help="レース選別: all / d1(モデル上位10%) / top2day(1日上位2件)")
    args = ap.parse_args()

    races = load_races(args.n_car)
    ents_by_race = load_entries(sorted(races))
    rows, winners = build_rows(races, ents_by_race, args.n_car)
    trio_pay, _tf = load_win_payouts(sorted(winners), winners)
    rows = [r for r in rows if r["race_key"] in trio_pay]
    for r in rows:
        r["payout_odds"] = trio_pay[r["race_key"]]
        r["y50"] = 1 if r["payout_odds"] >= 50 else 0

    # ---- 高配当レース選別モデル（honest walk-forward・再学習） ----
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
    ok = ~np.isnan(pred)
    print(f"\n[選別モデル] honest n={ok.sum()} AUC "
          f"{roc_auc_score(y[ok], pred[ok]):.4f}  基準50倍+率 {y[ok].mean() * 100:.2f}%")

    for i, r in enumerate(rows):
        r["score"] = pred[i]

    pp3_all = load_model_preds()
    evalrows = [r for r in rows if not np.isnan(r["score"])]

    # 選別集合
    sel_sets: dict[str, set] = {"all": {r["race_key"] for r in evalrows}}
    thr = np.quantile([r["score"] for r in evalrows], 0.90)
    sel_sets["d1"] = {r["race_key"] for r in evalrows if r["score"] >= thr}
    by_day = defaultdict(list)
    for r in evalrows:
        by_day[r["race_date"]].append(r)
    sel_sets["top2day"] = {r["race_key"] for d in by_day for r in
                           sorted(by_day[d], key=lambda x: -x["score"])[:2]}

    for sname in args.select.split(","):
        keep = sel_sets[sname]
        sub = [r for r in evalrows if r["race_key"] in keep]
        base50 = np.mean([r["y50"] for r in sub]) * 100
        print(f"\n{'=' * 84}")
        print(f"=== 選別 [{sname}]  {len(sub):,}レース  "
              f"(このレース群の 50倍+発生率 {base50:.2f}%) ===")
        print("  買い目        点数  的中%   50倍+的中%  30万+%  30万件数  ROI%  "
              "的中時配当中央")
        acc = defaultdict(lambda: {"n": 0, "npt": 0, "hit": 0, "h50": 0,
                                   "big": 0, "ret": 0.0, "pays": []})
        for r in sub:
            e = ents_by_race.get(r["race_key"])
            if not e:
                continue
            bets = make_bets(e, pp3_all.get(r["race_key"]))
            w = winners[r["race_key"]]["trio"]
            o = r["payout_odds"]
            for name, combos in bets.items():
                if not combos:
                    continue
                n_pt = len(combos)
                a = acc[name]
                a["n"] += 1
                a["npt"] += n_pt
                if w in combos:
                    pay = STAKE / n_pt * o
                    a["hit"] += 1
                    a["ret"] += pay
                    a["pays"].append(o)
                    if o >= 50:
                        a["h50"] += 1
                    if pay >= HIGHPAY:
                        a["big"] += 1
        for name in ("rp37_box", "rp47_box", "rp4_axis", "nomark_box",
                     "nomark_ax", "mdl35_box", "mdl37_box"):
            a = acc.get(name)
            if not a or not a["n"]:
                continue
            print(f"  {name:<12} {a['npt'] / a['n']:5.1f} "
                  f"{a['hit'] / a['n'] * 100:6.2f}  {a['h50'] / a['n'] * 100:9.2f}  "
                  f"{a['big'] / a['n'] * 100:6.2f}  {a['big']:8}  "
                  f"{a['ret'] / (a['n'] * STAKE) * 100:5.1f}  "
                  f"{np.median(a['pays']) if a['pays'] else 0:9.1f}")


if __name__ == "__main__":
    main()
