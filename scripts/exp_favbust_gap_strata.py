#!/usr/bin/env python3
"""【分類問題のみ】「確実に抜けた人気の1車」が4着以下になるレースの選別精度。

## ユーザー指定（2026-08-06）

> 「母集団は**確実に抜けた人気の1車が飛ぶケース**を検証したい為、**一致した場合**で
>  検証して下さい」

したがって母集団は次の1条件で固定する:

    軸1（モデル pred_win 最上位）== WINTICKET 公式印 ◎

その上で「**確実に抜けた**」を **1着率の 1位−2位差（gap）** で層別する
（和歌山8Rの例で言えば 2番 47.4% に対し2位 17.8% ＝ gap 29.6pt が「抜けて強い」）。
gap が大きいほど市場とモデルの双方が確信している＝飛んだときの破壊力が大きい。

**ROI・買い目・払戻は扱わない。測るのは選別精度のみ。**

`exp_favbust_precision.py` が作った `data/exp_cache/favbust_scored.pkl`
（honest walk-forward のスコア済み）を読むだけなので即座に走る。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score  # noqa: E402

CACHE = REPO / "data" / "exp_cache" / "favbust_scored.pkl"


def prec_row(sub: list[dict], n_days: int, frac: float) -> tuple:
    y = np.array([d["bust"] for d in sub])
    s = np.array([d["score"] for d in sub])
    k = max(int(len(sub) * frac), 1)
    idx = np.argsort(-s)[:k]
    return y[idx].mean(), k, k / n_days


def main() -> None:
    with CACHE.open("rb") as f:
        data = pickle.load(f)
    n_days = len({d["race_date"] for d in data})
    y = np.array([d["bust"] for d in data])
    gap = np.array([d["fav_ppw_gap12"] for d in data])

    print(f"母集団: 7車 × 軸1 == WINTICKET◎ … {len(data):,}レース / {n_days}日 "
          f"(1日 {len(data) / n_days:.1f}件)")
    print(f"目的変数: 本命が4着以下（欠車・失格含む）… 基準率 {y.mean() * 100:.2f}%")
    print(f"\n「抜け度」= モデル1着率の 1位−2位差。分布: "
          + " ".join(f"p{q}={np.percentile(gap, q) * 100:.1f}pt"
                     for q in (10, 25, 50, 75, 90)))

    # ---------- 抜け度による層別 ----------
    print(f"\n{'=' * 96}")
    print("=== 1. 「確実に抜けた」度合いごとの バスト率と予測可能性 ===")
    print("  抜け度(1着率の1-2位差)      件数   構成比   基準バスト率    AUC")
    edges = [(0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 1.01)]
    for lo, hi in edges:
        m = (gap >= lo) & (gap < hi)
        if m.sum() < 300:
            continue
        yy, ss = y[m], np.array([d["score"] for d in data])[m]
        print(f"  {lo * 100:5.0f}pt 以上 {hi * 100:5.0f}pt 未満 {m.sum():8,} "
              f"{m.mean() * 100:6.1f}%   {yy.mean() * 100:8.2f}%   "
              f"{roc_auc_score(yy, ss):.4f}")
    print("\n  ── 累積（この値以上に絞った場合）──")
    print("  抜け度        件数   構成比   1日   基準バスト率    AUC")
    for lo in (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        m = gap >= lo
        if m.sum() < 300:
            continue
        yy, ss = y[m], np.array([d["score"] for d in data])[m]
        print(f"  >= {lo * 100:4.0f}pt {m.sum():9,} {m.mean() * 100:6.1f}% "
              f"{m.sum() / n_days:6.1f}  {yy.mean() * 100:8.2f}%   "
              f"{roc_auc_score(yy, ss):.4f}")

    # ---------- 選別精度 ----------
    print(f"\n{'=' * 96}")
    print("=== 2. 選別精度: 各層で上位k%を選んだときの実バスト率 ===")
    for lo, nm in ((0.00, "一致のみ（抜け度の条件なし）"),
                   (0.20, "抜け度 >= 20pt"),
                   (0.30, "抜け度 >= 30pt"),
                   (0.40, "抜け度 >= 40pt")):
        sub = [d for d in data if d["fav_ppw_gap12"] >= lo]
        if len(sub) < 500:
            continue
        base = np.mean([d["bust"] for d in sub])
        print(f"\n  ### {nm}   n={len(sub):,}  基準バスト率 {base * 100:.2f}%  "
              f"AUC {roc_auc_score([d['bust'] for d in sub], [d['score'] for d in sub]):.4f}")
        print("      上位    件数   実バスト率   lift   1日あたり")
        for frac in (0.05, 0.10, 0.20, 0.30):
            p, k, per = prec_row(sub, n_days, frac)
            print(f"      {frac:4.0%} {k:8,}   {p * 100:8.2f}%  {p / base:5.2f}  "
                  f"{per:7.2f}件")

    # ---------- 月次一貫性（抜け度>=20pt × 上位20%）----------
    print(f"\n{'=' * 96}")
    print("=== 3. 月次の一貫性（抜け度 >= 20pt の中で上位20%）===")
    sub = [d for d in data if d["fav_ppw_gap12"] >= 0.20]
    print("   月       n     基準%   選定  実バスト%    差")
    diffs = []
    for mo in sorted({d["race_date"][:7] for d in sub}):
        g = [d for d in sub if d["race_date"][:7] == mo]
        if len(g) < 100:
            continue
        yy = np.array([d["bust"] for d in g])
        ss = np.array([d["score"] for d in g])
        k = max(int(len(g) * 0.20), 1)
        p = yy[np.argsort(-ss)[:k]].mean()
        diffs.append((p - yy.mean()) * 100)
        print(f"  {mo} {len(g):6}  {yy.mean() * 100:6.2f}  {k:5}  {p * 100:8.2f}  "
              f"{(p - yy.mean()) * 100:+6.2f}")
    dd = np.array(diffs)
    print(f"\n  差: 平均 {dd.mean():+.2f}pt / 中央 {np.median(dd):+.2f}pt / "
          f"**基準超え {int((dd > 0).sum())}/{len(dd)}ヶ月** / 最悪 {dd.min():+.2f}pt")

    # ---------- 抜け度が大きい層で「飛ぶ」条件 ----------
    print(f"\n{'=' * 96}")
    print("=== 4. 抜け度 >= 20pt の中で、どんな本命が飛ぶか（単変量）===")
    sub = [d for d in data if d["fav_ppw_gap12"] >= 0.20]
    base = np.mean([d["bust"] for d in sub])
    q75_bad = np.quantile([d["fav_pbad"] for d in sub], 0.75)
    conds = [
        ("本命が単騎", lambda d: d["fav_is_solo"] == 1),
        ("本命がライン先頭", lambda d: d["fav_is_leader"] == 1),
        ("本命がライン番手以降", lambda d: d["fav_is_leader"] == 0 and d["fav_is_solo"] == 0),
        ("本命の脚質=逃", lambda d: d["fav_style"] == 0),
        ("本命の脚質=追", lambda d: d["fav_style"] == 2),
        ("本命の得点が1位でない", lambda d: d["fav_rp_rank"] >= 1),
        ("本命のラインが最強でない", lambda d: d["fav_line_rank"] >= 1),
        ("◎と◯が別ライン", lambda d: d["taikou_same_line"] == 0),
        ("大敗率 pbad が層内上位25%", lambda d: d["fav_pbad"] >= q75_bad),
        ("逃型が2人以上", lambda d: d["n_senko"] >= 2),
        ("単騎が3人以上", lambda d: d["n_solo"] >= 3),
        ("準決勝", lambda d: d["rt_semi"] == 1),
        ("決勝", lambda d: d["rt_final"] == 1),
    ]
    print(f"      （層内 基準バスト率 {base * 100:.2f}%・n={len(sub):,}）")
    print("      条件                          件数   バスト率   基準比")
    for nm, fn in conds:
        sel = [d for d in sub if fn(d)]
        if len(sel) < 200:
            continue
        p = np.mean([d["bust"] for d in sel])
        print(f"      {nm:<28} {len(sel):6,}   {p * 100:7.2f}%  {p / base:6.2f}")


if __name__ == "__main__":
    main()
