"""「◎が飛ぶレース」を事前に選び ◯△ 軸で買えるかの検証（2026-08-05・ユーザー発案）。

ユーザー指摘:
  「配当的には◎が4着以下、◯△で的中する方がオッズが取れる。この条件のレース選択を
    する方法を検討し、ROI検証してほしい」

実測（`exp_axis_rule_decomposition.py`）では確かに、7S/7A で **◎は 24.5〜28.9% 飛び**、
そのうち **33〜38% は ◯△ が3着内**。◎抜きで決まるレースは配当が跳ねる。

## ただし前提として不利な材料が2つある

1. **市場の◎は較正が良い**。[[keirin_axis_popularity_and_pool_coverage_2026_08_01]] の
   実測で 市場人気1位の3着内は 市場P3 76.0% に対し実測 77.2%（+1.2pt）。
   「人気が過大評価されている」という構造は無い
2. **同じ発想の 7SS（波乱軸選出）は 2026-08-02 に全廃済み**。live ROI 73.5% で
   控除率を割り続けた（[[keirin_7ss_abolition_2026_08_02]]）

したがって本検証は **まず「◎が飛ぶことを事前に予測できるか」から確かめる**。
予測できないなら ROI を測る意味がない（当たった後から見れば配当が高いのは当然）。

## 手順

### 1) ◎が3着外になることの予測力
候補シグナル（すべて発走前に確定・オッズ非依存）:
  S1: -p3[◎]              モデルが◎の3着内を低く見ている
  S2:  bad[◎]             モデルが◎の大敗を高く見ている
  S3:  z(bad[◎]) - z(p3[◎])  両方の合成
  S4:  ◎のp3内順位          モデル内で◎が何番手か
十分位に割り、実際の「◎が3着外」率がどれだけ動くかを見る。

### 2) ◯△軸で買った場合のROI
上位帯のレースで 三連複 軸=◯+△ を買う。相手は
  (a) 残り5車（◎を含む）… ◎が3着でも当たる
  (b) 残り4車（◎を除く）… ◎が飛ぶことに賭ける
現行 7S/7A を同じレースで買った場合と対比する。

⚠️ オッズは wt_odds＝最終オッズ（stale）。⚠️ 上位帯の選択は本スクリプトで
掃引するが、採否は別窓で一度きり確認すること。DB書き込みなし。

使い方:
    python scripts/exp_honmei_bust_selection.py
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
    TRAIN_FROM, WINDOWS, a1_pw, a2_p3_bad, classify_rank, window_preds,
)
from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import _race_zscore  # noqa: E402

STAKE = 100


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


def buy(r, board, axes, legs):
    a1, a2 = axes
    legs = [x for x in legs if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def agg(rows):
    if not rows:
        return None
    bet = sum(x[0] for x in rows)
    ret = sum(x[1] for x in rows)
    h = [x for x in rows if x[2]]
    return dict(n=len(rows), hit=100 * len(h) / len(rows),
                roi=100 * ret / bet if bet else 0,
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
            r = {"rk": rk, "top3": top3, "fo": fo, "mark": mk,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
                 "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
                 "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)}}
            if None in (r["hon"], r["tai"], r["ana"]):
                continue
            h = r["hon"]
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            r["S1"] = -r["p3"][h]
            r["S2"] = r["bad"][h]
            r["S3"] = zb[h] - zp[h]
            r["S4"] = sorted(r["p3"], key=lambda f: -r["p3"][f]).index(h)
            a1 = a1_pw(r)
            a2 = a2_p3_bad(r, a1)
            r["cur"] = (a1, a2) if a2 else None
            r["cur_rank"] = classify_rank(r, a1, a2) if a2 else None
            r["hon_out"] = h not in top3
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        print(f"  窓 {tf}〜{tt}: ◎◯△が揃う7車立て {len(races)} レース")
        per_window.append((races, trio))

    print("\n" + "=" * 80)
    print("【1】「◎が3着外」を事前に予測できるか（十分位・窓平均）")
    print("     予測できなければ配当の議論に意味がない（後から見れば高配当なのは当然）")
    base = float(np.mean([np.mean([r["hon_out"] for r in rs]) for rs, _ in per_window]))
    print(f"  基準率（全レースで◎が3着外）: {100*base:.1f}%\n")
    for sig, name in (("S1", "-p3[◎]"), ("S2", "bad[◎]"),
                      ("S3", "z(bad)-z(p3)"), ("S4", "◎のp3内順位")):
        cells = []
        for q in range(10):
            vals = []
            for rs, _ in per_window:
                v = np.array([r[sig] for r in rs], dtype=float)
                lo, hi = np.quantile(v, q / 10), np.quantile(v, (q + 1) / 10)
                sel = [r for r in rs if (lo <= r[sig] <= hi if q == 9 else lo <= r[sig] < hi)]
                if sel:
                    vals.append(np.mean([r["hon_out"] for r in sel]))
            cells.append(100 * float(np.mean(vals)) if vals else 0)
        print(f"  {name:<14}" + " ".join(f"{c:5.1f}" for c in cells)
              + f"   （D1→D10 の差 {cells[-1]-cells[0]:+.1f}pt）")

    print("\n【2】上位帯で ◯△ 軸を買った場合の ROI（窓平均）")
    print(f"  {'選択':<22}{'買い方':<20}{'n':>6}{'的中':>8}{'ROI':>8}{'中央値':>8}")
    for sig, name in (("S3", "z(bad)-z(p3)"), ("S2", "bad[◎]")):
        for topq in (0.5, 0.3, 0.2, 0.1):
            variants = defaultdict(list)
            for rs, trio in per_window:
                v = np.array([r[sig] for r in rs], dtype=float)
                thr = np.quantile(v, 1 - topq)
                sel = [r for r in rs if r[sig] >= thr]
                rows = defaultdict(list)
                for r in sel:
                    b = trio[r["rk"]]
                    others = [f for f in r["p3"] if f not in (r["tai"], r["ana"])]
                    s = buy(r, b, (r["tai"], r["ana"]), others)
                    if s:
                        rows["◯△軸 相手5車(◎込)"].append(s)
                    s = buy(r, b, (r["tai"], r["ana"]),
                            [f for f in others if f != r["hon"]])
                    if s:
                        rows["◯△軸 相手4車(◎除外)"].append(s)
                    if r["cur"] and r["cur_rank"] in ("7S", "7A"):
                        s = buy(r, b, r["cur"],
                                [f for f in r["p3"] if f not in r["cur"]])
                        if s:
                            rows["対照 現行7S/7A"].append(s)
                for k, vv in rows.items():
                    a = agg(vv)
                    if a:
                        variants[k].append(a)
            for k, vv in variants.items():
                m = {x: float(np.mean([p[x] for p in vv])) for x in vv[0]}
                print(f"  {name+' 上位'+str(int(topq*100))+'%':<22}{k:<20}"
                      f"{m['n']:>6.0f}{m['hit']:>7.1f}%{m['roi']:>7.1f}%{m['med']:>7.2f}倍")
            print()


if __name__ == "__main__":
    main()
