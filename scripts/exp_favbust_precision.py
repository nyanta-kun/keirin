#!/usr/bin/env python3
"""【分類問題のみ】指数値が高く WT◎ と一致した本命が4着以下になるレースを予測できるか。

## 問い（ユーザー定義・2026-08-06）

> 「指数上の数値も高く、WINTICKET◎と一致した選手が4着以下となるレースを
>  予測できるか？ の点のみ。同レースの選別精度により、次の組み立てに
>  ステップを積んで進めていく」

**ROI・買い目・払戻は一切扱わない。** 測るのは選別精度だけ。

## 母集団の定義（2条件）

1. **一致**: 軸1（`pred_win` 最上位＝3ヘッド軸の軸1）== WINTICKET公式印 ◎
2. **指数値が高い**: 本命の指数が絶対水準で高いこと。
   前回の測定は 1 だけで切っていたため、本スクリプトで 2 を層別に加える。
   指数は 3着内率 `pp3` と 1着率 `ppw` の両方で層を切り、どちらで切るのが
   良いかも合わせて出す。

## 目的変数

    bust = 1 if 本命の finish_order >= 4 または 0（欠車・失格）

## 出す指標（選別精度のみ）

- 層ごとの **基準バスト率**（＝何もしないときの精度）
- honest walk-forward の **AUC**
- **上位k%を選んだときの実バスト率（precision）と lift**、1日あたり件数
- **月次の一貫性**（precision が基準を上回った月の割合）
- **較正**（予測確率の帯ごとの実測バスト率）

DB は読み取りのみ。スコア済みデータは pickle にキャッシュする。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.exp_highpay_fav_bust import (  # noqa: E402
    FAV_FEATS, WF, fav_features, load_preds3,
)
from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, build_rows, load_entries, load_races,
)

CACHE = REPO / "data" / "exp_cache" / "favbust_scored.pkl"


def build() -> list[dict]:
    if CACHE.exists():
        with CACHE.open("rb") as f:
            print(f"[cache] {CACHE.name}", flush=True)
            return pickle.load(f)

    races = load_races(7)
    ents_by_race = load_entries(sorted(races))
    rows, _winners = build_rows(races, ents_by_race, 7)
    pr_all = load_preds3()

    data = []
    for r in rows:
        rk = r["race_key"]
        ents = ents_by_race[rk]
        ff = fav_features(ents, pr_all.get(rk))
        if ff is None:
            continue
        fav = ff.pop("_fav")
        fo = next((x["finish_order"] for x in ents if int(x["frame_no"]) == fav), None)
        if fo is None:
            continue
        data.append({**{f: r[f] for f in FEATURES}, **ff,
                     "race_key": rk, "race_date": r["race_date"],
                     "finish": fo, "bust": 1 if (fo == 0 or fo >= 4) else 0})

    cols = FEATURES + FAV_FEATS
    X = np.array([[d[c] for c in cols] for d in data], dtype=float)
    y = np.array([d["bust"] for d in data])
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
        print(f"  [wf] {w_from}〜{w_to} test={te.sum()}", flush=True)
    for i, d in enumerate(data):
        d["score"] = pred[i]
    data = [d for d in data if not np.isnan(d["score"])]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:          # 保存失敗は握り潰さない
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE)
    return data


def precision_table(sub: list[dict], title: str, n_days: int) -> None:
    if len(sub) < 200:
        print(f"  {title}: n={len(sub)} で小さすぎるため省略")
        return
    y = np.array([d["bust"] for d in sub])
    s = np.array([d["score"] for d in sub])
    base = y.mean()
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = float("nan")
    print(f"\n  ### {title}")
    print(f"      n={len(sub):,}  基準バスト率 {base * 100:5.2f}%  AUC {auc:.4f}")
    print("      上位   件数   実バスト率   lift   1日あたり")
    order = np.argsort(-s)
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = max(int(len(sub) * frac), 1)
        idx = order[:k]
        p = y[idx].mean()
        print(f"      {frac:4.0%} {k:7}   {p * 100:8.2f}%  {p / base:5.2f}   "
              f"{k / n_days:8.2f}件")


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    data = build()
    n_days = len({d["race_date"] for d in data})
    y = np.array([d["bust"] for d in data])
    print(f"\n{'=' * 92}")
    print(f"母集団: 7車 × 軸1(pred_win最上位) == WINTICKET◎  … {len(data):,}レース "
          f"/ {n_days}日")
    print(f"目的変数: 本命が4着以下（欠車・失格を含む）… 基準率 {y.mean() * 100:.2f}%")
    fin = np.array([d["finish"] for d in data])
    print("本命の着順分布: " + " ".join(
        f"{k}着 {np.mean(fin == k) * 100:.1f}%" for k in (1, 2, 3, 4, 5, 6, 7))
        + f" / 欠車失格 {np.mean(fin == 0) * 100:.1f}%")

    # ---------- 1. 「指数値が高い」層の切り方 ----------
    print(f"\n{'=' * 92}\n=== 1. 「指数値も高い」条件を加えた層別（＝市場が確信している本命）===")
    print("  ※ 層が上がるほど本命は強い＝バスト率は下がる。予測が難しくなる代わりに"
          "飛んだときの破壊力が大きい")

    pp3 = np.array([d["fav_pp3"] for d in data])
    ppw = np.array([d["fav_ppw"] for d in data])
    print("\n  [A] 3着内率 pp3 で層別")
    for lo in (0.0, 0.55, 0.60, 0.65, 0.70, 0.75):
        sub = [d for d in data if d["fav_pp3"] >= lo]
        if len(sub) < 200:
            continue
        yy = np.array([d["bust"] for d in sub])
        print(f"    pp3 >= {lo:.2f}  n={len(sub):6,} ({len(sub) / len(data) * 100:5.1f}%)  "
              f"基準バスト率 {yy.mean() * 100:5.2f}%  "
              f"AUC {roc_auc_score(yy, [d['score'] for d in sub]):.4f}")
    print("\n  [B] 1着率 ppw で層別")
    for lo in (0.0, 0.25, 0.30, 0.35, 0.40, 0.45):
        sub = [d for d in data if d["fav_ppw"] >= lo]
        if len(sub) < 200:
            continue
        yy = np.array([d["bust"] for d in sub])
        print(f"    ppw >= {lo:.2f}  n={len(sub):6,} ({len(sub) / len(data) * 100:5.1f}%)  "
              f"基準バスト率 {yy.mean() * 100:5.2f}%  "
              f"AUC {roc_auc_score(yy, [d['score'] for d in sub]):.4f}")

    # ---------- 2. 選別精度 ----------
    print(f"\n{'=' * 92}\n=== 2. 選別精度（上位k%を選んだときの実バスト率）===")
    precision_table(data, "母集団すべて（一致のみ・指数条件なし）", n_days)
    for lo, nm in ((0.60, "pp3 >= 0.60"), (0.65, "pp3 >= 0.65"), (0.70, "pp3 >= 0.70")):
        precision_table([d for d in data if d["fav_pp3"] >= lo],
                        f"指数も高い層: {nm}", n_days)

    # ---------- 3. 較正 ----------
    print(f"\n{'=' * 92}\n=== 3. 較正（予測確率 vs 実測バスト率・母集団すべて）===")
    s = np.array([d["score"] for d in data])
    print("      予測確率帯       件数   予測平均   実測バスト率")
    for lo, hi in ((0, .10), (.10, .15), (.15, .20), (.20, .25),
                   (.25, .30), (.30, .40), (.40, 1.0)):
        m = (s >= lo) & (s < hi)
        if m.sum() < 50:
            continue
        print(f"      {lo:.2f}-{hi:.2f}   {m.sum():7,}   {s[m].mean() * 100:7.2f}%   "
              f"{y[m].mean() * 100:9.2f}%")

    # ---------- 4. 月次一貫性 ----------
    print(f"\n{'=' * 92}\n=== 4. 月次の一貫性（上位20%の実バスト率 vs その月の基準）===")
    print("   月       n     基準%   上位20%件数  実バスト%   差")
    months = sorted({d["race_date"][:7] for d in data})
    diffs = []
    for mo in months:
        sub = [d for d in data if d["race_date"][:7] == mo]
        if len(sub) < 200:
            continue
        yy = np.array([d["bust"] for d in sub])
        ss = np.array([d["score"] for d in sub])
        k = max(int(len(sub) * 0.20), 1)
        idx = np.argsort(-ss)[:k]
        p, b = yy[idx].mean(), yy.mean()
        diffs.append((p - b) * 100)
        print(f"  {mo} {len(sub):6}  {b * 100:6.2f}  {k:9}  {p * 100:9.2f}  "
              f"{(p - b) * 100:+6.2f}")
    dd = np.array(diffs)
    print(f"\n  差: 平均 {dd.mean():+.2f}pt / 中央 {np.median(dd):+.2f}pt / "
          f"**基準を上回った月 {int((dd > 0).sum())}/{len(dd)}** / 最悪 {dd.min():+.2f}pt")

    # ---------- 5. 重要な特徴 ----------
    print(f"\n{'=' * 92}\n=== 5. バストしやすい本命の条件（単変量・母集団すべて）===")
    print("      条件                          件数    バスト率   基準比")
    base = y.mean()
    conds = [
        ("本命が単騎（ライン無し）", lambda d: d["fav_is_solo"] == 1),
        ("本命がライン先頭", lambda d: d["fav_is_leader"] == 1),
        ("本命がライン番手以降", lambda d: d["fav_is_leader"] == 0 and d["fav_is_solo"] == 0),
        ("本命の脚質=逃", lambda d: d["fav_style"] == 0),
        ("本命の脚質=追", lambda d: d["fav_style"] == 2),
        ("本命の得点が1位でない", lambda d: d["fav_rp_rank"] >= 1),
        ("本命の得点が3位以下", lambda d: d["fav_rp_rank"] >= 2),
        ("本命のラインが最強でない", lambda d: d["fav_line_rank"] >= 1),
        ("◎と◯が同一ライン", lambda d: d["taikou_same_line"] == 1),
        ("◎と◯が別ライン", lambda d: d["taikou_same_line"] == 0),
        ("大敗率 pbad が上位25%", lambda d: d["fav_pbad"] >= np.quantile(
            [x["fav_pbad"] for x in data], 0.75)),
        ("逃型が2人以上", lambda d: d["n_senko"] >= 2),
        ("分戦数>=4", lambda d: d["n_lines"] >= 4),
    ]
    for nm, fn in conds:
        sel = [d for d in data if fn(d)]
        if len(sel) < 200:
            continue
        p = np.mean([d["bust"] for d in sel])
        print(f"      {nm:<28} {len(sel):7,}   {p * 100:7.2f}%   {p / base:6.2f}")


if __name__ == "__main__":
    main()
