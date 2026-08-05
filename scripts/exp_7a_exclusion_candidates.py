"""7A の外れを減らす除外条件の探索 ①軸1の危険度 ③同一ライン（2026-08-05・ユーザー指示）。

## ① 軸1の危険度による「除外」

本日の `exp_honmei_bust_selection.py` で **「軸1(≒◎)が飛ぶ」は事前に強く予測できる**
と実証した（`-p3[◎]` の十分位で D1 3.1% → D10 41.8%・+38.6pt）。

しかし検証したのは「**◯△軸に乗り換えて買う**」であり、市場が同じ情報を織り込み
済みのため失敗した。**「危険なら買わない」は一度も試していない。**
乗り換えは配当の対価を払うが、**除外は払わない**ので構造が違う。

狙う外れ方: 軸2のみ3着内(19.3%) + 両方外し(8.7%) = 約28%

## ③ 軸1と軸2が同一ラインか別ラインか

完全に未検証。競輪の構造そのもの。`wt_entries.line_group` で直接判定できる。

- 同一ライン: 連動して来るか、ライン全体が沈むか → **分散が大きい**はず
- 別ライン: 互いに競合するがどちらかは残りやすい → **分散が小さい**はず

本日の分解で 7S/7A に**二軸の負の相関（−2.7〜−2.9pt）**が出ている。
この負相関の正体がライン構造なら、ここで切り分けられる。

## 測り方の約束

- **群別**（新7S / A群=axis_sum不合格 / E群=entropy不合格）に測る
- 本スクリプトは**掃引窓（2025-07〜2026-07）で候補を作るだけ**。
  採否は確認窓（2024-07〜2025-06）で閾値を固定して一度きり検証する
- **件数の減りも必ず見る**（7A は 13.6件/日。削りすぎると推奨が枯渇する）
- 窓別の符号一貫性を必ず見る（平均は反転を隠す）

⚠️ オッズは wt_odds＝最終オッズ。選出条件は確率のみでオッズ非依存。DB書き込みなし。

使い方:
    python scripts/exp_7a_exclusion_candidates.py
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
    RANK_7S_ENTROPY_MAX, _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

STAKE = 100
WIN_DAYS = {"w1": 94, "w2": 102, "w3": 92, "w4": 92}
NEW_AXIS_SUM_MAX = 1.40


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
    return len(legs) * STAKE, ret, hit, (a1 in r["top3"]), (a2 in r["top3"])


def agg(rows, days):
    if not rows:
        return None
    bet = sum(x[0] for x in rows)
    ret = sum(x[1] for x in rows)
    h = [x for x in rows if x[2]]
    return dict(n=len(rows), per_day=len(rows) / days,
                hit=100 * len(h) / len(rows), roi=100 * ret / bet if bet else 0,
                a1in=100 * sum(x[3] for x in rows) / len(rows),
                a2in=100 * sum(x[4] for x in rows) / len(rows),
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
            lg = {int(x.frame_no): x.line_group for x in g.itertuples(index=False)}
            r = {"rk": rk, "top3": top3, "lg": lg,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
                 "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
                 "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)}}
            a1 = a1_pw(r)
            a2 = a2_p3_bad(r, a1)
            if a2 is None:
                continue
            if rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"]) not in (0, 1):
                continue
            r["a1"], r["a2"] = a1, a2
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            # ① 軸1の危険度（大きいほど危ない）
            r["d_bad1"] = r["bad"][a1]
            r["d_np3"] = -r["p3"][a1]
            r["d_mix"] = zb[a1] - zp[a1]
            # ③ 同一ライン判定（line_group が両方あり一致なら同ライン）
            g1, g2 = lg.get(a1), lg.get(a2)
            r["same_line"] = (g1 is not None and g2 is not None
                              and str(g1) != "" and str(g1) == str(g2))
            r["line_known"] = g1 is not None and g2 is not None \
                and str(g1) != "" and str(g2) != ""
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        per_window.append((w, races, trio))
    tot = sum(len(rs) for _, rs, _ in per_window)
    known = sum(sum(r["line_known"] for r in rs) for _, rs, _ in per_window)
    print(f"  母集団 {tot} レース / ライン情報あり {known} ({100*known/tot:.1f}%)\n")

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

    def show(lbl, pred, width=34):
        m, per = evaluate(pred)
        if not m:
            print(f"  {lbl:<{width}} 該当なし")
            return
        flag = "✓" if len(per) == 4 and all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{width}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['a1in']:>7.1f}%"
              f"{m['a2in']:>7.1f}%{m['hit']:>8.1f}%{m['roi']:>8.1f}%  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    HDR = (f"  {'条件':<34}{'n':>6}{'件/日':>7}{'軸1':>8}{'軸2':>8}{'的中':>9}{'ROI':>9}"
           f"     窓別ROI(w1 w2 w3 w4)")

    A = NEW_AXIS_SUM_MAX
    groups = {
        "新7S": lambda r: r["asum"] <= A and r["ent"] <= RANK_7S_ENTROPY_MAX,
        "A群(axis_sum不合格)": lambda r: r["asum"] > A and r["ent"] <= RANK_7S_ENTROPY_MAX,
        "E群(entropy不合格)": lambda r: r["asum"] <= A and r["ent"] > RANK_7S_ENTROPY_MAX,
    }
    q = {k: np.concatenate([[r[k] for r in rs] for _, rs, _ in per_window])
         for k in ("d_bad1", "d_np3", "d_mix")}

    for gname, gp in groups.items():
        print("=" * 112)
        print(f"■ {gname}")
        print(HDR)
        show("基準", gp)

        print("  ── ① 軸1が危険なレースを除外（上位X%を切る）")
        for key, nm in (("d_bad1", "bad[軸1]"), ("d_np3", "-p3[軸1]"),
                        ("d_mix", "z(bad)-z(p3)[軸1]")):
            for cut in (0.10, 0.20, 0.30, 0.40):
                thr = float(np.quantile(q[key], 1 - cut))
                show(f"{nm} 上位{int(cut*100)}%除外",
                     lambda r, k=key, t=thr: gp(r) and r[k] < t)

        print("  ── ③ 軸1と軸2のライン関係")
        show("同一ライン", lambda r: gp(r) and r["line_known"] and r["same_line"])
        show("別ライン", lambda r: gp(r) and r["line_known"] and not r["same_line"])
        print()

    print("=" * 112)
    print("  軸1/軸2 = それぞれが3着内に入った率。的中 = 両方3着内。")
    print("  ✓ = 4窓すべてで ROI>=75%")
    print("  ★ 候補は「的中率↑ かつ ROI が基準以上 かつ 4窓一貫 かつ 件数が残る」もののみ。")
    print("     採否は確認窓（2024-07〜2025-06）で閾値固定・一度きり検証してから決める。")


if __name__ == "__main__":
    main()
