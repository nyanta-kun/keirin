"""7B の外れ方を分解し、オッズ非依存の絞り込み条件を探す（2026-08-05・ユーザー指示）。

ユーザー方針:
  「7Bは継続で残す。オッズ判断は予想購入者には提示できるが**入稿時には正確に
    判断できない**ため、レースを絞る／外れる条件を分解し、的中する買い目・条件を
    整理するアプローチとする」

したがって**朝（入稿時）に確定している量のみ**を条件に使う。オッズは一切使わない
（ROI の計測にだけ最終オッズを使う）。

## 7B の構造と固有の負け筋

    軸2車 = ◎◯（wt_overlap_n==2 ∧ order_disagree）
    相手  = 残り5車から **WT△を除外**し pred_prob 上位3車 → 3点

7S/7A は相手が総流し（残り5車）なので **的中 ⟺ 軸2車が両方3着内**。
7B は相手を3点に絞るため、**軸が揃っても相手を外す**という固有の負け筋がある。

掃引窓の実測（`exp_axis_rule_decomposition.py`）: 7B は 軸両方3着内 48.4% に対し
的中 25.3%。**軸が揃った48.4%のうち半分近く（23.1pt）を相手選択で落としている。**
7B 全体で見ると最大の改善余地はここ。

## 測ること

1. 損失の3分解: ①軸が揃わない ②軸OKだが3着が△（＝除外した車） ③軸OKだが3着が他
2. 3着に来た車の内訳（△ / 4位以下 / 相手3車）
3. オッズ非依存の絞り込み条件の掃引:
   same_line（7A-E群で効いた）/ entropy / axis_sum / p3[△] / △と4位の差 /
   ライン数 / 軸1・軸2の bad / order_disagree の強さ

⚠️ 掃引窓（2025-07〜2026-07）で候補を作るだけ。採否は確認窓で一度きり。
DB書き込みなし。予測はキャッシュ利用。

使い方:
    python scripts/exp_7b_miss_decomposition.py
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter, defaultdict
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
    rank_7b_order_disagree, rank_7b_select_legs, rank_7s_field_entropy,
    rank_7s_wt_overlap_n, rank_7ss_same_line,
)

STAKE = 100
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


def settle(r, board, legs):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in legs if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def agg(rows, days):
    if not rows:
        return None
    bet = sum(x[0] for x in rows); ret = sum(x[1] for x in rows)
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
            lg = {int(x.frame_no): x.line_group for x in g.itertuples(index=False)}
            r = {"rk": rk, "top3": top3, "fo": fo,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "n_lines": int(g["n_lines"].iloc[0]) if "n_lines" in g else 0,
                 "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
                 "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
                 "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)}}
            a1 = a1_pw(r); a2 = a2_p3_bad(r, a1)
            if a2 is None:
                continue
            if rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"]) != 2:
                continue
            if rank_7b_order_disagree(r["pw"], r["hon"]) is not True:
                continue
            r["a1"], r["a2"] = a1, a2
            others = sorted(set(r["p3"]) - {a1, a2})
            r["others"] = others
            r["legs3"] = rank_7b_select_legs(others, r["p3"], r["ana"])
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["same_line"] = rank_7ss_same_line(a1, a2, lg)
            rest_no_ana = [f for f in others if f != r["ana"]]
            r["p3ana"] = r["p3"].get(r["ana"], 0.0)
            r["gap_ana"] = (r["p3ana"] - max(r["p3"][f] for f in rest_no_ana)
                            if rest_no_ana and r["ana"] is not None else 0.0)
            r["bad1"], r["bad2"] = r["bad"][a1], r["bad"][a2]
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        per_window.append((w, [r for r in races if trio.get(r["rk"])], trio))
    tot = sum(len(rs) for _, rs, _ in per_window)
    print(f"  7B母集団: {tot} レース\n")

    # ---- 1) 損失の分解 ------------------------------------------------
    cnt = Counter()
    third_kind = Counter()
    for _, races, _ in per_window:
        for r in races:
            both = {r["a1"], r["a2"]} <= r["top3"]
            cnt["all"] += 1
            if not both:
                cnt["軸が揃わない"] += 1
                continue
            third = next(iter(r["top3"] - {r["a1"], r["a2"]}))
            if third in r["legs3"]:
                cnt["🎯的中"] += 1
            elif third == r["ana"]:
                cnt["軸OK・3着が△(除外した車)"] += 1
            else:
                cnt["軸OK・3着が相手外(4位以下)"] += 1
            third_kind["△" if third == r["ana"] else
                        ("相手3車" if third in r["legs3"] else "その他")] += 1
    n = cnt["all"]
    print("【1】7B の損失分解（掃引窓 合算）")
    for k in ("🎯的中", "軸OK・3着が△(除外した車)", "軸OK・3着が相手外(4位以下)", "軸が揃わない"):
        print(f"  {k:<28}{cnt[k]:>6}件  {100*cnt[k]/n:>5.1f}%")
    print(f"\n  軸2車が両方3着内: {100*(n-cnt['軸が揃わない'])/n:.1f}%"
          f" → うち的中 {100*cnt['🎯的中']/(n-cnt['軸が揃わない']):.1f}%")
    print(f"  ★ 軸が揃ったのに落とした割合: "
          f"{100*(cnt['軸OK・3着が△(除外した車)']+cnt['軸OK・3着が相手外(4位以下)'])/n:.1f}%"
          f"（うち△が {100*cnt['軸OK・3着が△(除外した車)']/n:.1f}pt）")

    # ---- 2) 条件の掃引 ------------------------------------------------
    def evaluate(pred, legs_fn=None):
        per = []
        for w, races, trio in per_window:
            rows = []
            for r in races:
                if not pred(r):
                    continue
                legs = (legs_fn(r) if legs_fn else r["legs3"])
                s = settle(r, trio[r["rk"]], legs)
                if s:
                    rows.append(s)
            a = agg(rows, WIN_DAYS[w])
            if a:
                per.append(a)
        if not per:
            return None, []
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}, \
               [p["roi"] for p in per]

    def show(lbl, pred, legs_fn=None, width=34):
        m, per = evaluate(pred, legs_fn)
        if not m:
            print(f"  {lbl:<{width}} 該当なし")
            return
        flag = "✓" if len(per) == 4 and all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{width}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%{m['med']:>7.2f}倍  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    HDR = (f"  {'条件':<34}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}{'中央値':>8}"
           f"     窓別ROI(w1 w2 w3 w4)")
    q = {k: np.concatenate([[r[k] for r in rs] for _, rs, _ in per_window])
         for k in ("ent", "asum", "p3ana", "gap_ana", "bad1", "bad2")}

    print("\n【2】オッズ非依存の絞り込み（買い目は現行3点のまま）")
    print(HDR)
    show("基準（7B全件）", lambda r: True)
    show("同一ライン", lambda r: r["same_line"])
    show("別ライン", lambda r: not r["same_line"])
    for key, nm, lo in (("ent", "entropy", True), ("asum", "axis_sum", True),
                        ("p3ana", "p3[△]", True), ("gap_ana", "△と4位の差", True),
                        ("bad2", "bad[軸2]", False)):
        for cut in (0.3, 0.5, 0.7):
            thr = float(np.quantile(q[key], cut))
            if lo:
                show(f"{nm} <= {thr:.3f}（下位{int(cut*100)}%）",
                     lambda r, k=key, t=thr: r[k] <= t)
            else:
                show(f"{nm} <= {thr:.3f}", lambda r, k=key, t=thr: r[k] <= t)
    for nl in (2, 3):
        show(f"ライン数=={nl}", lambda r, x=nl: r["n_lines"] == x)

    print("\n【3】買い目の作り方を変える（絞り込みなし・全件）")
    print(HDR)
    show("現行3点（△除外）", lambda r: True)
    show("△を戻して4点", lambda r: True,
         legs_fn=lambda r: (r["legs3"] + [r["ana"]]) if r["ana"] is not None else r["legs3"])
    show("総流し5点", lambda r: True, legs_fn=lambda r: r["others"])

    print("\n  ✓ = 4窓すべてで ROI>=75%")
    print("  ※ 条件は全てオッズ非依存（朝の入稿時に確定している量のみ）。")


if __name__ == "__main__":
    main()
