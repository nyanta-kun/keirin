"""7A を2群に分け、各群で不的中を減らす条件を探す（2026-08-05・ユーザー指示）。

ユーザー指示:
  「分けた上でそれぞれについて閾値の見直し、条件追加で不的中レースの推奨を
    減らせないか検討して」

## 背景：7A は性質が正反対の2群の混合だった

`exp_7s7a_threshold_tighten.py` の分解（掃引窓・①採用 axis_sum<=1.40 前提）:

| 群 | 件/日 | 的中 | ROI | 的中中央値 |
|---|---|---|---|---|
| A群: axis_sum だけ不合格（＝軸2車が堅い） | 3.50 | 52.4% | 81.8% | 0.78倍 |
| E群: entropy だけ不合格（＝レースが荒れている） | 4.62 | 32.2% | 78.5% | 1.57倍 |

的中率が2倍近く違い、配当は逆転する。同じ「7A」で出しているため
利用者から見ると当たり方も配当も一貫しない。

## 探すもの

7S/7A は相手が総流し（残り5車）なので **的中 ⟺ 軸2車が両方3着内**。
「不的中を減らす」＝「両方3着内率を上げる」。各群で以下を掃引する。

  overlap  : 0（◎◯と全く重ならない）か 1（片方一致）か
  p3[軸1]  : 軸1の3着内確率
  p3[軸2]  : 軸2の3着内確率（本日の分解で軸2が足を引っ張ると判明）
  bad[軸2] : 軸2の大敗確率（上限を設ける）
  gap2     : 軸2 と3番手の差
  asum/ent : 群ごとの境界を動かす

⚠️ 掃引窓（2025-07〜2026-07）で候補を作るだけ。**採否は確認窓
   （2024-07〜2025-06・`exp_7s7a_threshold_confirm.py`）で一度きり**。
⚠️ 窓別の符号一貫性を必ず見る（平均は反転を隠す）。
⚠️ オッズは wt_odds＝最終オッズ。選出条件は確率のみでオッズ非依存。DB書き込みなし。

使い方:
    python scripts/exp_7a_split_conditions.py
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from scripts.exp_axis_rule_decomposition import (  # noqa: E402
    TRAIN_FROM, WINDOWS, a1_pw, a2_p3_bad, window_preds,
)
from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_ENTROPY_MAX, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

STAKE = 100
WIN_DAYS = {"w1": 94, "w2": 102, "w3": 92, "w4": 92}
NEW_AXIS_SUM_MAX = 1.40      # ①として採用済みの新しい7S境界


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
    max_to = max(t for _, t in WINDOWS.values())
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    per_window = []
    for w, (tf, tt) in WINDOWS.items():
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
            a1 = a1_pw(r)
            a2 = a2_p3_bad(r, a1)
            if a2 is None:
                continue
            ov = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
            if ov not in (0, 1):
                continue
            r["a1"], r["a2"], r["ov"] = a1, a2, ov
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["p3a1"], r["p3a2"] = r["p3"][a1], r["p3"][a2]
            r["bada1"], r["bada2"] = r["bad"][a1], r["bad"][a2]
            rest = [f for f in r["p3"] if f not in (a1, a2)]
            r["gap2"] = r["p3a2"] - max(r["p3"][f] for f in rest) if rest else 0.0
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        per_window.append((w, races, trio))
    print(f"  母集団 overlap∈(0,1) の7車立て: "
          f"{sum(len(rs) for _, rs, _ in per_window)} レース\n")

    def evaluate(pred):
        per = []
        for w, races, trio in per_window:
            rows = [s for r in races if pred(r)
                    and (s := settle(r, trio[r["rk"]])) is not None]
            a = agg(rows, WIN_DAYS[w])
            if a:
                per.append(a)
        if not per:
            return None, []
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}, \
               [p["roi"] for p in per]

    def show(lbl, pred, width=36):
        m, per = evaluate(pred)
        if not m:
            print(f"  {lbl:<{width}} 該当なし")
            return
        flag = "✓" if len(per) == 4 and all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{width}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%{m['med']:>7.2f}倍  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    HDR = (f"  {'条件':<36}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}{'中央値':>8}"
           f"     窓別ROI(w1 w2 w3 w4)")

    A = NEW_AXIS_SUM_MAX
    grpA = lambda r: r["asum"] > A and r["ent"] <= RANK_7S_ENTROPY_MAX
    grpE = lambda r: r["asum"] <= A and r["ent"] > RANK_7S_ENTROPY_MAX

    allq = {}
    for k in ("p3a1", "p3a2", "bada1", "bada2", "gap2"):
        allq[k] = np.concatenate([[r[k] for r in rs] for _, rs, _ in per_window])

    for gname, gpred in (("A群 axis_sum不合格（堅い）", grpA),
                         ("E群 entropy不合格（荒れ）", grpE)):
        print("=" * 110)
        print(f"■ {gname}")
        print(HDR)
        show("基準（群そのまま）", gpred)
        print("  ── overlap で分ける")
        for ov in (0, 1):
            show(f"overlap=={ov}", lambda r, o=ov: gpred(r) and r["ov"] == o)
        print("  ── 軸2の強さ p3[軸2] で下限")
        for q in (0.2, 0.3, 0.4, 0.5):
            thr = float(np.quantile(allq["p3a2"], q))
            show(f"p3[軸2]>={thr:.3f}（全体下位{int(q*100)}%除外）",
                 lambda r, t=thr: gpred(r) and r["p3a2"] >= t)
        print("  ── 軸2の大敗確率 bad[軸2] で上限")
        for q in (0.8, 0.7, 0.6, 0.5):
            thr = float(np.quantile(allq["bada2"], q))
            show(f"bad[軸2]<={thr:.3f}（全体上位{int((1-q)*100)}%除外）",
                 lambda r, t=thr: gpred(r) and r["bada2"] <= t)
        print("  ── 軸1の強さ p3[軸1] で下限")
        for q in (0.2, 0.3, 0.4):
            thr = float(np.quantile(allq["p3a1"], q))
            show(f"p3[軸1]>={thr:.3f}（全体下位{int(q*100)}%除外）",
                 lambda r, t=thr: gpred(r) and r["p3a1"] >= t)
        print("  ── 軸2と3番手の差 gap2 で下限")
        for q in (0.3, 0.5):
            thr = float(np.quantile(allq["gap2"], q))
            show(f"gap2>={thr:.3f}（全体下位{int(q*100)}%除外）",
                 lambda r, t=thr: gpred(r) and r["gap2"] >= t)
        if gname.startswith("A群"):
            print("  ── axis_sum の上限を設ける（A群は上に開いている）")
            for x in (1.50, 1.60, 1.70, 1.80, 1.90):
                show(f"axis_sum<={x:.2f}", lambda r, x=x: gpred(r) and r["asum"] <= x)
        else:
            print("  ── entropy の上限を設ける（E群は上に開いている）")
            for y in (1.86, 1.88, 1.90, 1.92, 1.94):
                show(f"entropy<={y:.2f}", lambda r, y=y: gpred(r) and r["ent"] <= y)
        print()

    print("=" * 110)
    print("  ✓ = 4窓すべてで ROI>=75%")
    print("  ★ 判断は「的中率が上がり かつ ROI が基準以上 かつ 4窓一貫」を満たすものだけ。")
    print("     採否は確認窓（2024-07〜2025-06）で一度きり検証してから決めること。")


if __name__ == "__main__":
    main()
