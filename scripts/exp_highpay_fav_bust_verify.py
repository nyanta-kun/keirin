#!/usr/bin/env python3
"""本命バスト選別の検証: 未使用期間・月次一貫性・閾値掃引（オッズ非使用）。

## 位置づけ

`exp_highpay_fav_bust.py` で「軸1==WT◎ の本命が4着以下に飛ぶ」を予測すると
AUC 0.6837 が出て、バスト確率上位20%のレースに現行6点フォーメーションを置くと
**両窓で ROI +17.3pt / +11.1pt・的中率 +42% / +51%** と改善した。
30万円到達率は動かなかった（的中時配当が下がって相殺）。

ユーザー指示（2026-08-06）:
> 「走らせて下さい。30万は目安としての提示のため、検証結果を踏まえトータルで確認します」

したがって本スクリプトは **30万円だけでなく回収の全体像**を出す。

## ⚠️ 未使用期間の検出力（結果を読む前に）

未使用期間は **2026-07-16〜2026-08-04（20日）**。7車・軸1==◎ に絞ると約900レース、
上位20%なら **約180レース**。的中率3%なら**的中は5件前後**にしかならない。

- **粗い反転（ROI 40%台など）が出れば棄却できる**
- **反転が出なくても追認したことにはならない**

という非対称な使い方しかできない。そこで**月次の一貫性**を主証拠として併記する
（walk-forward が全期間 honest なので、各月は独立に読める）。

## 出力

1. 閾値掃引（上位10/15/20/30/50%）× 掃引窓・確認窓
2. **月次の 全件 vs 上位20%**（ROI・的中率の一貫性）
3. 未使用期間 2026-07-16〜08-04 の一度きり検証
4. 払戻の階段（1万円以上=ガミ回避 / 3万 / 10万 / 30万 / 50万+）

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.exp_highpay_fav_bust import (  # noqa: E402
    FAV_FEATS, WF, fav_features, formation, load_preds3,
)
from scripts.exp_highpay_race_model import (  # noqa: E402
    FEATURES, build_rows, load_entries, load_races, load_win_payouts,
)

STAKE = 10_000
SWEEP = ("2025-07-01", "2026-07-15")
CONFIRM = ("2024-07-01", "2025-06-30")
FRESH = ("2026-07-16", "2026-08-04")
STEPS = [10_000, 30_000, 100_000, 300_000, 500_000]


def summarize(items: list[dict]) -> dict:
    """items: {'pay': 払戻(0=不的中)} のリスト。1レース1万円前提。"""
    n = len(items)
    if not n:
        return {}
    pays = np.array([d["pay"] for d in items], dtype=float)
    hits = pays > 0
    out = {"n": n, "hit%": hits.mean() * 100,
           "ROI%": pays.sum() / (n * STAKE) * 100,
           "配当中央": float(np.median(pays[hits])) if hits.any() else 0.0}
    for s in STEPS:
        out[f">={s // 10000}万"] = int((pays >= s).sum())
    return out


def line(tag: str, s: dict) -> str:
    if not s:
        return f"  {tag:<22} （該当なし）"
    return (f"  {tag:<22} n={s['n']:6} 的中{s['hit%']:5.2f}% "
            f"ROI{s['ROI%']:6.1f}% 配当中央{s['配当中央']:9.0f}円 | "
            + " ".join(f"{k}:{s[k]:4}" for k in
                       (f">={x // 10000}万" for x in STEPS)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=int, default=5)
    ap.add_argument("--m2", type=int, default=2)
    ap.add_argument("--m3", type=int, default=4)
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
        data.append({**{f: r[f] for f in FEATURES}, **ff,
                     "race_key": rk, "race_date": r["race_date"],
                     "bust": 1 if (fo == 0 or fo >= 4) else 0,
                     "tf_odds": tf_pay[rk], "fav": fav})

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
    for i, d in enumerate(data):
        d["score"] = pred[i]
    ev = [d for d in data if not np.isnan(d["score"])]
    ok = ~np.isnan(pred)
    print(f"[data] 軸1==WT◎ {len(data):,}レース / honest {len(ev):,} / "
          f"バスト基準率 {y.mean() * 100:.2f}% / AUC {roc_auc_score(y[ok], pred[ok]):.4f}")

    # 買い目を1回だけ全レースに適用して払戻を確定させる
    for d in ev:
        pr = pr_all[d["race_key"]]
        od = sorted(pr, key=lambda f: -pr[f][0])
        combos = formation(od, args.k1, args.m2, args.m3)
        w = winners[d["race_key"]]["trifecta"]
        d["npt"] = len(combos)
        d["pay"] = (STAKE / len(combos) * d["tf_odds"]) if (combos and w in combos) else 0.0

    def win_of(w):
        return [d for d in ev if w[0] <= d["race_date"] <= w[1]]

    print(f"\n買い目: 1着=モデル3着内率{args.k1}位固定 × 2着=上位{args.m2} × "
          f"3着=上位{args.m3}（{np.mean([d['npt'] for d in ev]):.0f}点・1点"
          f"{STAKE / np.mean([d['npt'] for d in ev]):.0f}円）")

    # ---- 1. 閾値掃引 ----
    print(f"\n{'=' * 100}\n=== 1. バスト確率の選別閾値 掃引 ===")
    for wname, w in (("掃引窓", SWEEP), ("確認窓", CONFIRM)):
        sub = win_of(w)
        print(f"\n[{wname}] {w[0]}〜{w[1]}")
        print(line("全件", summarize(sub)))
        for frac in (0.50, 0.30, 0.20, 0.15, 0.10):
            thr = np.quantile([d["score"] for d in sub], 1 - frac)
            print(line(f"バスト上位{frac:.0%}",
                       summarize([d for d in sub if d["score"] >= thr])))

    # ---- 2. 月次一貫性 ----
    print(f"\n{'=' * 100}\n=== 2. 月次の一貫性（全件 vs バスト上位20%）===")
    print("   月        全件 n  的中%   ROI%  |  上位20% n  的中%   ROI%   ΔROI")
    months = sorted({d["race_date"][:7] for d in ev})
    deltas = []
    for mo in months:
        sub = [d for d in ev if d["race_date"][:7] == mo]
        if len(sub) < 100:
            continue
        thr = np.quantile([d["score"] for d in sub], 0.80)
        top = [d for d in sub if d["score"] >= thr]
        a, b = summarize(sub), summarize(top)
        dl = b["ROI%"] - a["ROI%"]
        deltas.append(dl)
        print(f"  {mo}  {a['n']:6} {a['hit%']:6.2f} {a['ROI%']:6.1f}  | "
              f"{b['n']:8} {b['hit%']:6.2f} {b['ROI%']:6.1f}  {dl:+7.1f}")
    d = np.array(deltas)
    print(f"\n  ΔROI: 平均 {d.mean():+.1f}pt / 中央 {np.median(d):+.1f}pt / "
          f"**改善した月 {int((d > 0).sum())}/{len(d)}** / 最悪 {d.min():+.1f}pt")

    # ---- 3. 未使用期間 ----
    print(f"\n{'=' * 100}\n=== 3. 未使用期間 {FRESH[0]}〜{FRESH[1]} の一度きり検証 ===")
    fr = win_of(FRESH)
    print(line("全件", summarize(fr)))
    if fr:
        # (a) 掃引窓から引いた絶対閾値（運用に忠実）
        thr_abs = np.quantile([d["score"] for d in win_of(SWEEP)], 0.80)
        print(line("上位20%(掃引窓の閾値)",
                   summarize([d for d in fr if d["score"] >= thr_abs])))
        # (b) 期間内で上位20%
        thr_in = np.quantile([d["score"] for d in fr], 0.80)
        print(line("上位20%(期間内基準)",
                   summarize([d for d in fr if d["score"] >= thr_in])))

    # ---- 4. 全honest期間の総括 ----
    print(f"\n{'=' * 100}\n=== 4. 全honest期間 総括 ===")
    print(line("全件", summarize(ev)))
    thr = np.quantile([d["score"] for d in ev], 0.80)
    top = [d for d in ev if d["score"] >= thr]
    print(line("バスト上位20%", summarize(top)))
    s = summarize(top)
    print(f"\n  上位20%は 1日あたり約 {s['n'] / len({d['race_date'] for d in ev}):.1f} 件。"
          f"1レース1万円なら 投資 {s['n'] * STAKE / 1e8:.2f}億円 / "
          f"回収 {sum(d['pay'] for d in top) / 1e8:.2f}億円")

    # ---- 5. 裾依存 ----
    print(f"\n{'=' * 100}\n=== 5. 裾依存（高額配当を除いたときの ROI）===")
    print("   群            ROI%   除・上5  除・上10  除・上20  上5が回収に占める%")
    for nm, grp in (("全件", ev), ("バスト上位20%", top)):
        pays = np.sort(np.array([d["pay"] for d in grp]))[::-1]
        cost = len(grp) * STAKE
        tot = pays.sum()
        print(f"  {nm:<12} {tot / cost * 100:6.1f}  "
              f"{(tot - pays[:5].sum()) / cost * 100:7.1f}  "
              f"{(tot - pays[:10].sum()) / cost * 100:8.1f}  "
              f"{(tot - pays[:20].sum()) / cost * 100:8.1f}  "
              f"{pays[:5].sum() / tot * 100:14.1f}")

    # ---- 6. paired bootstrap（日ブロック・全件 vs 上位20%）----
    print(f"\n{'=' * 100}\n=== 6. ΔROI の日ブロック bootstrap（2,000回）===")
    days = sorted({d["race_date"] for d in ev})
    by_d = defaultdict(list)
    for d in ev:
        by_d[d["race_date"]].append(d)
    thr20 = thr
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(len(days), len(days), replace=True)
        ca = ra = ct = rt = 0.0
        for i in pick:
            for d in by_d[days[i]]:
                ca += STAKE
                ra += d["pay"]
                if d["score"] >= thr20:
                    ct += STAKE
                    rt += d["pay"]
        if ct > 0:
            diffs.append(rt / ct * 100 - ra / ca * 100)
    dd = np.array(diffs)
    print(f"  ΔROI = {dd.mean():+.1f}pt  95%CI [{np.percentile(dd, 2.5):+.1f}, "
          f"{np.percentile(dd, 97.5):+.1f}]  → "
          f"{'有意' if np.percentile(dd, 2.5) > 0 else '**有意差なし**'}")


if __name__ == "__main__":
    main()
