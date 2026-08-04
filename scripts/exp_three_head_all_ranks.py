"""3ヘッド軸選定の全ランク・9車立てへの波及検証（2026-08-04）。

exp_three_head_axis.py（7車立て・7S/7A相当のみ）で
    軸1 = z(pw) − w1·z(pbad) の最上位 / 軸2 = z(p3) − w2·z(pbad) の最上位
が **全4窓・全指標で一貫改善**（両方3着内 +2.48pt・軸2大敗率 −3.02pt・ROI +5.28pt）
したことを受け、実装前に影響範囲を確認する。

`rank_7s_select_axis` は **7S/7A/7B/9S/9A の全ランクが共有**しているため、
変更すると9車立ても同時に変わる。さらに 7B は overlap==2（◎◯完全一致）を対象と
するランクなので、軸が変われば**母集団そのものが入れ替わる**。7車立てだけを見て
採否を決めてはいけない。

測定:
  ① 7車立て: 7S / 7A / 7B の件数・軸精度・的中・ROI（現行 vs 3ヘッド合成）
  ② 9車立て: 9S / 9A の同上
  ③ ランク横断の合計（実運用の総推奨件数がどう動くか）

⚠️ オッズは wt_odds＝最終オッズ（stale）。選出はオッズ非依存。DB書き込みなし。

使い方:
    python scripts/exp_three_head_all_ranks.py [--windows w1,w2,w3,w4] [--seeds 7]
"""
from __future__ import annotations

import argparse
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

from src.database import get_connection
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (
    RANK_7B_LEGS, RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    RANK_7S_MARK3_OVERLAP_MAX, RANK_9S_ENTROPY_MAX,
    rank_7b_order_disagree, rank_7b_select_legs, rank_7s_field_entropy,
    rank_7s_select_axis, rank_7s_wt_mark3_overlap_n, rank_7s_wt_overlap_n,
)

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
    "w3": ("2025-10-01", "2025-12-31"),
    "w4": ("2025-07-01", "2025-09-30"),
}
STAKE = 100


def fit_predict(train, test, target, seeds):
    preds = []
    for seed in seeds:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(train[FEATURE_COLS_WT], train[target])
        preds.append(m.predict_proba(test[FEATURE_COLS_WT])[:, 1])
    return np.mean(preds, axis=0)


def load_trio(keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if v <= 0 or v >= 9999:
                    continue
                parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                if len(parts) == 3:
                    out[rk][parts] = v
    return out


def _z(d):
    v = np.array(list(d.values()), dtype=float)
    m, s = v.mean(), v.std()
    return {k: 0.0 for k in d} if s <= 0 else {k: (x - m) / s for k, x in d.items()}


def select_axis(r, w1, w2):
    if w1 is None:
        sel = rank_7s_select_axis(r["pw"], r["p3"])
        return (sel[0], sel[1]) if sel else None
    zw, zp, zb = _z(r["pw"]), _z(r["p3"]), _z(r["bad"])
    s1 = {f: zw[f] - w1 * zb[f] for f in zw}
    a1 = max(s1, key=lambda f: s1[f])
    s2 = {f: zp[f] - w2 * zb[f] for f in zp if f != a1}
    if not s2:
        return None
    a2 = max(s2, key=lambda f: s2[f])
    return a1, a2


def classify(r, a1, a2, ne):
    """本番と同一のランク判定。該当しなければ None。"""
    hon = next((f for f, m in r["mark"].items() if m == 1), None)
    tai = next((f for f, m in r["mark"].items() if m == 2), None)
    ana = next((f for f, m in r["mark"].items() if m == 3), None)
    ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
    ent = rank_7s_field_entropy(r["p3"])
    if ne == 7:
        if ov == 2:
            return "7B" if rank_7b_order_disagree(r["pw"], hon) is True else None
        if ov != 1 and ov != 0:
            return None
        n_fail = ((r["p3"][a1] + r["p3"][a2] > RANK_7S_AXIS_SUM_MAX)
                  + (ent > RANK_7S_ENTROPY_MAX))
        return "7S" if n_fail == 0 else ("7A" if n_fail == 1 else None)
    if ne == 9:
        if ov not in (0, 1):
            return None
        mk3 = rank_7s_wt_mark3_overlap_n(a1, a2, hon, tai, ana)
        if mk3 is None:
            return None
        n_fail = (ent > RANK_9S_ENTROPY_MAX) + (mk3 > RANK_7S_MARK3_OVERLAP_MAX)
        return "9S" if n_fail == 0 else ("9A" if n_fail == 1 else None)
    return None


def evaluate(races, trio, w1, w2):
    acc = defaultdict(lambda: {"n": 0, "a1": 0, "a2": 0, "both": 0, "a2bad": 0,
                               "hit": 0, "bet": 0, "ret": 0, "pays": []})
    for r in races:
        sel = select_axis(r, w1, w2)
        if not sel:
            continue
        a1, a2 = sel
        rk_lbl = classify(r, a1, a2, r["ne"])
        if rk_lbl is None:
            continue
        board = trio.get(r["rk"], {})
        others = [x for x in r["p3"] if x not in (a1, a2)]
        if rk_lbl == "7B":
            ana = next((f for f, m in r["mark"].items() if m == 3), None)
            legs = rank_7b_select_legs(others, r["p3"], ana, RANK_7B_LEGS)
        else:
            legs = others
        legs = [x for x in legs if frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        a = acc[rk_lbl]
        a["n"] += 1
        a["a1"] += a1 in r["top3"]
        a["a2"] += a2 in r["top3"]
        a["a2bad"] += r["fo"].get(a2, 0) >= 6
        if {a1, a2} <= r["top3"]:
            a["both"] += 1
        stake = len(legs) * STAKE
        a["bet"] += stake
        rest = r["top3"] - {a1, a2}
        if len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and rest.pop() in legs:
            a["hit"] += 1
            got = round(board[frozenset(r["top3"])] * 100) // 10 * 10
            a["ret"] += got
            a["pays"].append(got)
    out = {}
    for k, a in acc.items():
        n = max(a["n"], 1)
        out[k] = {"n": a["n"], "a1": 100*a["a1"]/n, "a2": 100*a["a2"]/n,
                  "both": 100*a["both"]/n, "a2bad": 100*a["a2bad"]/n,
                  "hit": 100*a["hit"]/n,
                  "roi": 100*a["ret"]/a["bet"] if a["bet"] else 0.0,
                  "med": statistics.median(a["pays"])/100 if a["pays"] else 0.0}
    return out


def build_races(test, p3, pw, pbad, ne_map):
    t = test.copy()
    t["pp3"], t["ppw"], t["pbad"] = p3, pw, pbad
    races = []
    for rk, g in t.groupby("race_key"):
        ne = ne_map.get(rk)
        if ne not in (7, 9) or len(g) != ne:
            continue
        fo = {int(r.frame_no): (int(r.finish_order)
                                if r.finish_order is not None and r.finish_order == r.finish_order
                                else 0) for r in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue
        races.append({
            "rk": rk, "ne": ne, "fo": fo, "top3": top3,
            "p3": {int(r.frame_no): float(r.pp3) for r in g.itertuples(index=False)},
            "pw": {int(r.frame_no): float(r.ppw) for r in g.itertuples(index=False)},
            "bad": {int(r.frame_no): float(r.pbad) for r in g.itertuples(index=False)},
            "mark": {int(r.frame_no): r.prediction_mark for r in g.itertuples(index=False)},
        })
    return races


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    ap.add_argument("--seeds", type=int, default=7)
    args = ap.parse_args()
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909][:args.seeds]

    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴 / seed×{len(seeds)})", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne_map = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne_map).isin([7, 9])].copy()
    print(f"7車/9車立て: {len(df):,}行")

    ARMS = [("現行", None, None), ("3ヘッド w1=0 w2=0.3", 0.0, 0.3),
            ("3ヘッド w1=0 w2=0.6", 0.0, 0.6)]
    acc = {a[0]: defaultdict(list) for a in ARMS}

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
        test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
        print(f"\n######## 窓 {tf}〜{tt}  train {len(train):,} / test {len(test):,} ########",
              flush=True)
        p3 = fit_predict(train, test, TARGET_COL_WT, seeds)
        pw = fit_predict(train, test, "win_flag", seeds)
        pbad = fit_predict(train, test, "bad6", seeds)
        races = build_races(test, p3, pw, pbad, ne_map)
        n7 = sum(1 for r in races if r["ne"] == 7)
        print(f"  評価対象 {len(races)} レース（7車 {n7} / 9車 {len(races)-n7}）")
        trio = load_trio(sorted({r["rk"] for r in races}))
        for name, w1, w2 in ARMS:
            res = evaluate(races, trio, w1, w2)
            for rk_lbl, s in res.items():
                acc[name][rk_lbl].append(s)

    print("\n" + "=" * 104)
    print("【全窓平均・ランク別】（n=1窓あたり件数）")
    print(f"{'ランク':6} {'案':22} {'n/窓':>6} {'軸1':>7} {'軸2':>7} {'軸2大敗':>8} "
          f"{'両方':>7} {'的中':>7} {'ROI':>8}")
    for rk_lbl in ("7S", "7A", "7B", "9S", "9A"):
        base = None
        for name, _, _ in ARMS:
            lst = acc[name].get(rk_lbl)
            if not lst:
                continue
            m = {k: float(np.mean([s[k] for s in lst]))
                 for k in ("n", "a1", "a2", "a2bad", "both", "hit", "roi")}
            if base is None:
                base = m
                d = ""
            else:
                d = (f"  (Δ両方{m['both']-base['both']:+.1f} "
                     f"Δ大敗{m['a2bad']-base['a2bad']:+.1f} "
                     f"ΔROI{m['roi']-base['roi']:+.1f})")
            print(f"{rk_lbl:6} {name:22} {m['n']:6.0f} {m['a1']:6.1f}% {m['a2']:6.1f}% "
                  f"{m['a2bad']:7.1f}% {m['both']:6.1f}% {m['hit']:6.1f}% "
                  f"{m['roi']:7.1f}%{d}")
        print()

    print("【ランク横断の合計件数（1窓あたり）】")
    for name, _, _ in ARMS:
        tot = sum(float(np.mean([s["n"] for s in lst]))
                  for lst in acc[name].values())
        det = " / ".join(f"{k} {float(np.mean([s['n'] for s in acc[name][k]])):.0f}"
                         for k in ("7S", "7A", "7B", "9S", "9A") if acc[name].get(k))
        print(f"  {name:22} 計 {tot:5.0f}   ({det})")


if __name__ == "__main__":
    main()
