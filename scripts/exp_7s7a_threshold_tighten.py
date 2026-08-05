"""7S/7A の閾値を絞って期待値を上げられるか（2026-08-05・ユーザー依頼）。

ユーザー指摘:
  「本日は7S/7Aなど推奨が多くあった。もう少し期待値を高める上で閾値の見直しにより
    レースを絞れるか」

## 現行ゲート（`strategy_wt`）

    wt_overlap_n ∈ {0,1}
    axis_sum = p3[軸1] + p3[軸2] <= RANK_7S_AXIS_SUM_MAX (1.5)
    entropy  = field entropy      <= RANK_7S_ENTROPY_MAX (1.8329)
    7S = 両方通過 / 7A = どちらか1つだけ不合格

## 掃引するもの

1. `axis_sum` 上限 … 低いほど波乱寄り（軸2車の確率合計が小さい）
2. `entropy` 上限 … 低いほどレース全体が堅い
3. **軸2の強さ**（`p3[軸2]`）… 本日の分解で「軸2側が足を引っ張る」と判明したため
4. **軸2と3番手の差**（`p3[軸2] − p3[3番手]`）… 軸2が明確に抜けているか

7S/7A は相手が総流し（残り5車）なので **的中 ⟺ 軸2車が両方3着内**。
したがって「両方3着内率」＝的中率であり、これを上げる＝期待値を上げること。

⚠️ 閾値は本スクリプトで掃引するが、**採否は別窓で一度きり確認すること**。
⚠️ オッズは wt_odds＝最終オッズ（stale）。選出条件は確率のみでオッズ非依存。
⚠️ **窓別の符号一貫性を必ず見る**（平均は反転を隠す・[[keirin_three_head_axis_2026_08_04]]）。
DB書き込みなし。予測はキャッシュを利用。

使い方:
    python scripts/exp_7s7a_threshold_tighten.py
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
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

STAKE = 100
# 窓の日数（件数/日 に直すため）
WIN_DAYS = {"w1": 94, "w2": 102, "w3": 92, "w4": 92}


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
            if rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"]) not in (0, 1):
                continue                       # 7S/7A の母集団のみ
            r["a1"], r["a2"] = a1, a2
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["p3a2"] = r["p3"][a2]
            rest = [f for f in r["p3"] if f not in (a1, a2)]
            r["gap2"] = r["p3"][a2] - max(r["p3"][f] for f in rest) if rest else 0.0
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        print(f"  窓 {tf}〜{tt}: overlap∈(0,1) の7車立て {len(races)} レース")
        per_window.append((w, races, trio))

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
        flag = "✓" if all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{width}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%{m['med']:>7.2f}倍  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    HDR = (f"  {'条件':<34}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}{'中央値':>8}"
           f"     窓別ROI(w1 w2 w3 w4)")

    print("\n" + "=" * 108)
    print("【0】現行の基準")
    print(HDR)
    cur7s = lambda r: r["asum"] <= RANK_7S_AXIS_SUM_MAX and r["ent"] <= RANK_7S_ENTROPY_MAX
    cur7a = lambda r: ((r["asum"] > RANK_7S_AXIS_SUM_MAX)
                       + (r["ent"] > RANK_7S_ENTROPY_MAX)) == 1
    show("現行 7S", cur7s)
    show("現行 7A", cur7a)
    show("現行 7S+7A", lambda r: cur7s(r) or cur7a(r))

    print("\n【1】axis_sum 上限を動かす（entropy は現行のまま・7S定義）")
    print(HDR)
    for x in (1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55):
        mark = " ←現行" if abs(x - RANK_7S_AXIS_SUM_MAX) < 1e-9 else ""
        show(f"axis_sum<={x:.2f} ∧ ent<=1.8329{mark}",
             lambda r, x=x: r["asum"] <= x and r["ent"] <= RANK_7S_ENTROPY_MAX)

    print("\n【2】entropy 上限を動かす（axis_sum は現行のまま・7S定義）")
    print(HDR)
    for y in (1.70, 1.75, 1.78, 1.80, 1.8329, 1.86):
        mark = " ←現行" if abs(y - RANK_7S_ENTROPY_MAX) < 1e-4 else ""
        show(f"asum<=1.50 ∧ entropy<={y:.4f}{mark}",
             lambda r, y=y: r["asum"] <= RANK_7S_AXIS_SUM_MAX and r["ent"] <= y)

    print("\n【3】軸2の強さで追加カット（現行7S+7A に上乗せ）")
    print("     本日の分解で『軸2側が足を引っ張る』と判明したため")
    print(HDR)
    allv = np.concatenate([[r["p3a2"] for r in rs] for _, rs, _ in per_window])
    for q in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        thr = float(np.quantile(allv, q))
        show(f"7S+7A ∧ p3[軸2]>={thr:.3f} (下位{int(q*100)}%除外)",
             lambda r, t=thr: (cur7s(r) or cur7a(r)) and r["p3a2"] >= t)

    print("\n【4】軸2と3番手の差で追加カット（現行7S+7A に上乗せ）")
    print(HDR)
    allg = np.concatenate([[r["gap2"] for r in rs] for _, rs, _ in per_window])
    for q in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        thr = float(np.quantile(allg, q))
        show(f"7S+7A ∧ 軸2-3番手>={thr:.3f} (下位{int(q*100)}%除外)",
             lambda r, t=thr: (cur7s(r) or cur7a(r)) and r["gap2"] >= t)

    # ---- 7A の内訳（どちらが不合格かで性質が違うはず）--------------------
    # 7A は「不合格ちょうど1つ」だが、
    #   axis_sum だけ不合格 = 軸2車の確率合計が高い（堅い決着になりやすい）
    #   entropy だけ不合格  = レース全体が荒れている
    # は意味が正反対。片方が足を引っ張っているなら 7A をそこで絞れる。
    print("\n【5】7A の内訳（どちらの条件で落ちたか）")
    print(HDR)
    a_fail = lambda r: r["asum"] > RANK_7S_AXIS_SUM_MAX and r["ent"] <= RANK_7S_ENTROPY_MAX
    e_fail = lambda r: r["asum"] <= RANK_7S_AXIS_SUM_MAX and r["ent"] > RANK_7S_ENTROPY_MAX
    show("7A のうち axis_sum だけ不合格（堅い）", a_fail)
    show("7A のうち entropy だけ不合格（荒れ）", e_fail)

    print("\n【6】①採用(axis_sum<=1.40)後の 7A を同じく分解")
    print(HDR)
    A = 1.40
    a_fail2 = lambda r: r["asum"] > A and r["ent"] <= RANK_7S_ENTROPY_MAX
    e_fail2 = lambda r: r["asum"] <= A and r["ent"] > RANK_7S_ENTROPY_MAX
    show("新7A のうち axis_sum だけ不合格", a_fail2)
    show("新7A のうち entropy だけ不合格", e_fail2)

    print("\n【7】7A を絞る候補（①採用を前提）")
    print(HDR)
    show("新7A 全体（基準）", lambda r: a_fail2(r) or e_fail2(r))
    show("新7A ∧ axis_sum不合格のみ採用", a_fail2)
    show("新7A ∧ entropy不合格のみ採用", e_fail2)
    for x in (1.50, 1.60, 1.70, 1.80):
        show(f"新7A ∧ axis_sum<={x:.2f}（上限も設ける）",
             lambda r, x=x: (a_fail2(r) or e_fail2(r)) and r["asum"] <= x)
    for y in (1.86, 1.88, 1.90, 1.92):
        show(f"新7A ∧ entropy<={y:.2f}（上限も設ける）",
             lambda r, y=y: (a_fail2(r) or e_fail2(r)) and r["ent"] <= y)

    print("\n  ✓ = 4窓すべてで ROI>=75%（控除率の壁を全窓で超えた案）")


if __name__ == "__main__":
    main()
