"""7S/7A 閾値見直しの2候補を、掃引に使っていない窓で一度きり確認する（2026-08-05）。

## 経緯

`exp_7s7a_threshold_tighten.py` で 2025-07〜2026-07 の4窓を掃引し、2候補が出た。

| 候補 | 内容 | 掃引窓での結果 |
|---|---|---|
| ① | 7S の `axis_sum <= 1.40`（現行1.50） | 4.55→2.70件/日・ROI 86.0→92.1% |
| ② | 7S+7A から `p3[軸2] < 0.566` を除外（下位30%） | 11.2→7.9件/日・ROI 82.8→85.0%・的中 41.4→43.6% |

**掃引した窓の数字をそのまま採用してはいけない**（6通り×4系列＝多重比較で
偶然良く見える帯が必ず出る）。本スクリプトは**別窓で一度きり**確認する。

## 確認窓（掃引に一度も使っていない）

    c1: 2025-04-01〜2025-06-30
    c2: 2025-01-01〜2025-03-31
    c3: 2024-10-01〜2024-12-31
    c4: 2024-07-01〜2024-09-30

学習は各窓の開始日より前のみ（TRAIN_FROM=2022-12-01＝本番のベース起点）。

## 重要な約束

**閾値は掃引窓で決めた値をそのまま持ち込む**（axis_sum<=1.40 / p3[軸2]>=0.566）。
確認窓で分位から取り直すと「その窓に合わせた閾値」になり検証にならない。
参考として窓内分位版も併記するが、**採否は絶対値版で判断する**。

⚠️ オッズは wt_odds＝最終オッズ（stale）。DB書き込みなし。
⚠️ 窓別の符号一貫性を必ず見る（平均は反転を隠す）。

使い方:
    python scripts/exp_7s7a_threshold_confirm.py
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

TRAIN_FROM = "2022-12-01"
CONFIRM = {
    "c1": ("2025-04-01", "2025-06-30", 91),
    "c2": ("2025-01-01", "2025-03-31", 90),
    "c3": ("2024-10-01", "2024-12-31", 92),
    "c4": ("2024-07-01", "2024-09-30", 92),
}
SEEDS = [42, 101, 202, 303, 404]
STAKE = 100
CACHE_DIR = REPO / "data" / "exp_cache"

# ---- 掃引窓で決めた閾値（確認窓では動かさない）--------------------------
CAND1_AXIS_SUM_MAX = 1.40
CAND2_P3A2_MIN = 0.566


def fit_predict(train, test, target):
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(train[FEATURE_COLS_WT], train[target])
        preds.append(m.predict_proba(test[FEATURE_COLS_WT])[:, 1])
    return np.mean(preds, axis=0)


def window_preds(df, tf, tt):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{TRAIN_FROM}.pkl"
    if cache.exists():
        print(f"  [cache] {cache.name}", flush=True)
        return pd.read_pickle(cache)
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
    test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
    print(f"  学習中 train {len(train):,} / test {len(test):,} ...", flush=True)
    out = test[["race_key", "frame_no"]].copy()
    out["pp3"] = fit_predict(train, test, TARGET_COL_WT)
    out["ppw"] = fit_predict(train, test, "win_flag")
    out["pbad"] = fit_predict(train, test, "bad6")
    out.to_pickle(cache)
    return out


def load_trio(keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            ch = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(p) == 3:
                        out[rk][p] = v
    return out


def settle(r, board):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in r["p3"] if x not in (a1, a2)
            and frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def agg(rows, days):
    if not rows:
        return None
    bet = sum(x[0] for x in rows)
    ret = sum(x[1] for x in rows)
    h = [x for x in rows if x[2]]
    return dict(n=len(rows), per_day=len(rows) / days,
                hit=100 * len(h) / len(rows), roi=100 * ret / bet if bet else 0,
                med=statistics.median([x[1] / x[0] for x in h]) if h else 0)


def main():
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(
        min_date=TRAIN_FROM, max_date=max(t for _, t, _ in CONFIRM.values())))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    per_window = []
    for w, (tf, tt, days) in CONFIRM.items():
        print(f"\n######## 確認窓 {w}: {tf}〜{tt} ########", flush=True)
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            window_preds(df, tf, tt), on=["race_key", "frame_no"], how="inner")
        races = []
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            fo = {int(x.frame_no): (int(x.finish_order)
                                    if x.finish_order == x.finish_order
                                    and x.finish_order is not None else 0)
                  for x in g.itertuples(index=False)}
            top3 = {f for f, v in fo.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            mk = {int(x.frame_no): x.prediction_mark for x in g.itertuples(index=False)}
            r = {"rk": rk, "top3": top3,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
                 "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
                 "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)}}
            a1 = max(r["pw"], key=lambda f: r["pw"][f])
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            cand = [f for f in r["p3"] if f != a1]
            if not cand:
                continue
            a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
            if rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"]) not in (0, 1):
                continue
            r["a1"], r["a2"] = a1, a2
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["p3a2"] = r["p3"][a2]
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        print(f"  overlap∈(0,1) の7車立て {len(races)} レース")
        per_window.append((w, races, trio, days))

    def evaluate(pred):
        per = []
        for w, races, trio, days in per_window:
            rows = [s for r in races if pred(r, races)
                    and (s := settle(r, trio[r["rk"]])) is not None]
            a = agg(rows, days)
            if a:
                per.append(a)
        if not per:
            return None, []
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}, \
               [p["roi"] for p in per]

    def show(lbl, pred, w=38):
        m, per = evaluate(pred)
        if not m:
            print(f"  {lbl:<{w}} 該当なし")
            return
        flag = "✓" if all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{w}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%{m['med']:>7.2f}倍  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    is7s = lambda r: r["asum"] <= RANK_7S_AXIS_SUM_MAX and r["ent"] <= RANK_7S_ENTROPY_MAX
    is7a = lambda r: ((r["asum"] > RANK_7S_AXIS_SUM_MAX)
                      + (r["ent"] > RANK_7S_ENTROPY_MAX)) == 1

    print("\n" + "=" * 108)
    print("【確認窓 2024-07〜2025-06・閾値は掃引窓で決めた値を固定】")
    print(f"  {'条件':<38}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}{'中央値':>8}"
          f"     窓別ROI(c1 c2 c3 c4)")
    show("現行 7S", lambda r, _: is7s(r))
    show("現行 7A", lambda r, _: is7a(r))
    show("現行 7S+7A", lambda r, _: is7s(r) or is7a(r))
    print()
    show(f"① 7S ∧ axis_sum<={CAND1_AXIS_SUM_MAX}",
         lambda r, _: r["asum"] <= CAND1_AXIS_SUM_MAX and r["ent"] <= RANK_7S_ENTROPY_MAX)
    show(f"② 7S+7A ∧ p3[軸2]>={CAND2_P3A2_MIN}",
         lambda r, _: (is7s(r) or is7a(r)) and r["p3a2"] >= CAND2_P3A2_MIN)
    show("①+② 併用",
         lambda r, _: r["asum"] <= CAND1_AXIS_SUM_MAX
         and r["ent"] <= RANK_7S_ENTROPY_MAX and r["p3a2"] >= CAND2_P3A2_MIN)

    # ---- 新しい閾値にした場合の 7A と合計（2026-08-05 追加・測り漏れの補完）----
    # axis_sum 上限を下げると 7A の中身が変わる:
    #   asum 1.40〜1.50 ∧ ent OK → 7S から 7A へ流入（良質な帯を受け取る）
    #   asum 1.40〜1.50 ∧ ent NG → 7A から除外（entropyも悪い帯を失う）
    # 7S だけ測っても全体の可否は判断できない。
    A = CAND1_AXIS_SUM_MAX
    new7s = lambda r: r["asum"] <= A and r["ent"] <= RANK_7S_ENTROPY_MAX
    new7a = lambda r: ((r["asum"] > A) + (r["ent"] > RANK_7S_ENTROPY_MAX)) == 1
    print(f"\n  ── axis_sum上限を {A} にした場合の内訳（①採用時の全体像）")
    show(f"新 7S (asum<={A} ∧ ent OK)", lambda r, _: new7s(r))
    show(f"新 7A (不合格ちょうど1つ)", lambda r, _: new7a(r))
    show("新 7S+7A 合計", lambda r, _: new7s(r) or new7a(r))
    print("     ↑ 現行 7S+7A（上記）と比較すること")

    # ---- 7A を2群に分け、各群に p3[軸2] 下限をかける（2026-08-05 追加）------
    # 掃引窓（exp_7a_split_conditions.py）で両群とも p3[軸2]>=0.566 が最良かつ
    # 4窓一貫だった。⚠️同じ条件を 7S+7A 一括にかけたときは確認窓で不合格（②）
    # だったが、7S に対して逆効果だったためで、7A に層別にかけると効く。
    # 閾値 0.566 は掃引窓で決めた値。**確認窓では動かさない**。
    print(f"\n  ── 7A を2群に分け p3[軸2]>={CAND2_P3A2_MIN} をかける（①採用前提）")
    gA = lambda r: r["asum"] > A and r["ent"] <= RANK_7S_ENTROPY_MAX
    gE = lambda r: r["asum"] <= A and r["ent"] > RANK_7S_ENTROPY_MAX
    cut = lambda r: r["p3a2"] >= CAND2_P3A2_MIN
    show("A群 axis_sum不合格（基準）", lambda r, _: gA(r))
    show(f"A群 ∧ p3[軸2]>={CAND2_P3A2_MIN}", lambda r, _: gA(r) and cut(r))
    show("E群 entropy不合格（基準）", lambda r, _: gE(r))
    show(f"E群 ∧ p3[軸2]>={CAND2_P3A2_MIN}", lambda r, _: gE(r) and cut(r))
    print("     ↓ 合計（片方だけ見ないこと）")
    show("新7S + 新7A（絞りなし・基準）",
         lambda r, _: new7s(r) or gA(r) or gE(r))
    show("新7S + 7A絞り後",
         lambda r, _: new7s(r) or ((gA(r) or gE(r)) and cut(r)))

    print("\n  参考: ②を確認窓内の分位で取り直した場合（★これは検証ではない）")
    for w, races, trio, days in per_window:
        pass
    def q30(r, races):
        thr = float(np.quantile([x["p3a2"] for x in races], 0.30))
        return (is7s(r) or is7a(r)) and r["p3a2"] >= thr
    show("② 窓内30%分位版（参考）", q30)

    print("\n  ✓ = 確認窓4つすべてで ROI>=75%")


if __name__ == "__main__":
    main()
