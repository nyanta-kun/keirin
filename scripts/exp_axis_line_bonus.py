"""軸2の選定に「軸1と同ライン」ボーナスを入れる検証（2026-08-04）。

記述統計（exp_axis_line_relation.py・8,421件）で、**予測確率を揃えても**
同ラインの2車は別ラインより両方3着内率が **+11.31pt** 高いと判明した
（全5帯で同符号・0.00〜0.35帯 +14.0pt / 0.55〜0.65帯 +13.8pt）。

原因は、モデルが1車ずつの確率しか出しておらず
**2車が同時に来る確率（共分散）を一切表現していない**こと。競輪はライン戦なので
同ラインの2車は連れ込みで揃って上位に来るが、その構造が欠落している。
「二軸探偵」は二軸の**同時的中**が売りなので、個々の確率ではなく同時確率を
最大化すべきところ、そこが手つかずだった。

本スクリプトは軸2の選定式に同ラインボーナスを足して掃引する:

    軸1 = z(pred_win) の最上位
    軸2 = z(pred_prob) − w2·z(bad_prob) + w_line·[軸1と同ライン] の最上位

⚠️ 記述統計で差が出ても軸選定に組み込んで改善するとは限らない
   （2026-08-04 だけで「AUCは上がるが的中率は上がらない」を3回、
     「窓を増やすと消える」を1回観測している）。**4窓×5seedで窓間の
     一貫性まで確認**し、平均だけで判断しない。

⚠️ 同ラインは配当が安い（中央値 5.2倍 vs 別ライン 6.2倍）。
   的中率とROIのどちらが伸びるかは実測でしか分からない。

DB書き込みなし。

使い方:
    python scripts/exp_axis_line_bonus.py [--windows w1,w2,w3,w4]
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
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    rank_7s_field_entropy, rank_7s_select_axis, rank_7s_wt_overlap_n,
)

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
    "w3": ("2025-10-01", "2025-12-31"),
    "w4": ("2025-07-01", "2025-09-30"),
}
SEEDS = [42, 101, 202, 303, 404]
STAKE = 100
W2_BAD = 0.6          # 3ヘッド版で採用した大敗ペナルティ


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
                if 0 < v < 9999:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(parts) == 3:
                        out[rk][parts] = v
    return out


def _z(d):
    v = np.array(list(d.values()), dtype=float)
    m, s = v.mean(), v.std()
    return {k: 0.0 for k in d} if s <= 0 else {k: (x - m) / s for k, x in d.items()}


def select_axis(r, w2, w_line):
    """w2/w_line が None なら現行ロジック（win∩top3 の重なり）。"""
    if w2 is None:
        sel = rank_7s_select_axis(r["pw"], r["p3"])
        return (sel[0], sel[1]) if sel else None
    zw, zp, zb = _z(r["pw"]), _z(r["p3"]), _z(r["bad"])
    a1 = max(zw, key=lambda f: zw[f])
    g1 = r["line"].get(a1)
    s2 = {}
    for f in zp:
        if f == a1:
            continue
        same = 1.0 if (g1 is not None and r["line"].get(f) == g1) else 0.0
        s2[f] = zp[f] - w2 * zb[f] + w_line * same
    if not s2:
        return None
    return a1, max(s2, key=lambda f: s2[f])


def evaluate(races, trio, w2, w_line):
    n = hit = bet = ret = both = a2_in = same_n = 0
    pays = []
    for r in races:
        sel = select_axis(r, w2, w_line)
        if not sel:
            continue
        a1, a2 = sel
        ov = rank_7s_wt_overlap_n(
            a1, a2,
            next((f for f, m in r["mark"].items() if m == 1), None),
            next((f for f, m in r["mark"].items() if m == 2), None))
        if ov not in (0, 1):
            continue
        ent = rank_7s_field_entropy(r["p3"])
        if ((r["p3"][a1] + r["p3"][a2] > RANK_7S_AXIS_SUM_MAX)
                + (ent > RANK_7S_ENTROPY_MAX)) > 1:
            continue
        board = trio.get(r["rk"], {})
        legs = [x for x in r["p3"] if x not in (a1, a2)
                and frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        n += 1
        g1 = r["line"].get(a1)
        if g1 is not None and r["line"].get(a2) == g1:
            same_n += 1
        a2_in += a2 in r["top3"]
        if {a1, a2} <= r["top3"]:
            both += 1
        stake = len(legs) * STAKE
        bet += stake
        rest = r["top3"] - {a1, a2}
        if len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and rest.pop() in legs:
            hit += 1
            got = round(board[frozenset(r["top3"])] * 100) // 10 * 10
            ret += got
            pays.append(got)
    f = lambda x: 100 * x / n if n else 0.0            # noqa: E731
    return {"n": n, "same": f(same_n), "a2": f(a2_in), "both": f(both),
            "hit": f(hit), "roi": 100 * ret / bet if bet else 0.0,
            "med": statistics.median(pays) / 100 if pays else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    args = ap.parse_args()

    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    print(f"7車立て: {len(df):,}行")

    ARMS = [("現行（重なり）", None, 0.0),
            ("3ヘッドのみ", W2_BAD, 0.0),
            ("+ライン 0.2", W2_BAD, 0.2),
            ("+ライン 0.4", W2_BAD, 0.4),
            ("+ライン 0.6", W2_BAD, 0.6),
            ("+ライン 1.0", W2_BAD, 1.0)]
    acc = defaultdict(list)

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
        test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
        print(f"\n#### 窓 {tf}〜{tt}  train {len(train):,} / test {len(test):,} ####",
              flush=True)
        p3 = fit_predict(train, test, TARGET_COL_WT)
        pw = fit_predict(train, test, "win_flag")
        pbad = fit_predict(train, test, "bad6")
        t = test.copy()
        t["pp3"], t["ppw"], t["pbad"] = p3, pw, pbad
        races = []
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            fo_m = {int(r.frame_no): (int(r.finish_order)
                                      if r.finish_order is not None
                                      and r.finish_order == r.finish_order else 0)
                    for r in g.itertuples(index=False)}
            top3 = {f for f, v in fo_m.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            races.append({
                "rk": rk, "top3": top3,
                "p3": {int(r.frame_no): float(r.pp3) for r in g.itertuples(index=False)},
                "pw": {int(r.frame_no): float(r.ppw) for r in g.itertuples(index=False)},
                "bad": {int(r.frame_no): float(r.pbad) for r in g.itertuples(index=False)},
                "mark": {int(r.frame_no): r.prediction_mark
                         for r in g.itertuples(index=False)},
                # line_group 欠損は単騎＝車番で一意化して別ライン扱い
                "line": {int(r.frame_no): (r.line_group if r.line_group is not None
                                           and r.line_group == r.line_group
                                           else -int(r.frame_no))
                         for r in g.itertuples(index=False)},
            })
        trio = load_trio(sorted({r["rk"] for r in races}))
        print(f"  評価対象 {len(races)} レース")
        print(f"  {'案':18} {'n':>5} {'同ライン率':>9} {'軸2':>7} {'両方':>7} "
              f"{'ROI':>7} {'中央値':>7}")
        for name, w2, wl in ARMS:
            s = evaluate(races, trio, w2, wl)
            acc[name].append(s)
            print(f"  {name:18} {s['n']:5d} {s['same']:8.1f}% {s['a2']:6.1f}% "
                  f"{s['both']:6.1f}% {s['roi']:6.1f}% {s['med']:6.1f}倍")

    print("\n" + "=" * 92)
    print("【全窓平均】（現行との差・窓ごとの符号一致数）")
    print(f"  {'案':18} {'n':>5} {'同ライン率':>9} {'軸2':>7} {'両方':>7} "
          f"{'ROI':>7} {'一貫(両方/ROI)':>16}")
    base = {k: float(np.mean([s[k] for s in acc["現行（重なり）"]]))
            for k in ("n", "same", "a2", "both", "roi", "med")}
    nw = len(acc["現行（重なり）"])
    for name in [a[0] for a in ARMS]:
        lst = acc[name]
        m = {k: float(np.mean([s[k] for s in lst]))
             for k in ("n", "same", "a2", "both", "roi", "med")}
        if name == "現行（重なり）":
            print(f"  {name:18} {m['n']:5.0f} {m['same']:8.1f}% {m['a2']:6.1f}% "
                  f"{m['both']:6.1f}% {m['roi']:6.1f}%")
            continue
        okb = sum(1 for i in range(nw)
                  if lst[i]["both"] > acc["現行（重なり）"][i]["both"])
        okr = sum(1 for i in range(nw)
                  if lst[i]["roi"] > acc["現行（重なり）"][i]["roi"])
        print(f"  {name:18} {m['n']:5.0f} {m['same']:8.1f}% "
              f"{m['a2'] - base['a2']:+6.1f} {m['both'] - base['both']:+6.1f} "
              f"{m['roi'] - base['roi']:+6.1f} {okb:>8}/{nw}  {okr}/{nw}")


if __name__ == "__main__":
    main()
