"""7A 除外条件（①軸1の危険度 / ③同一ライン）を確認窓で一度きり検証する（2026-08-05）。

## 検証する条件（掃引窓 `exp_7a_exclusion_candidates.py` で候補化）

| 条件 | 掃引窓での結果 |
|---|---|
| ③ 同一ライン（軸1と軸2の `line_group` が一致） | A群 的中45.5→60.2%・ROI 77.1→87.1% ✓ / E群 的中28.7→45.0%・ROI 75.2→90.5% ✓ |
| ① 軸1の危険度で上位X%除外 | 新7S ROI 92.1→104.5%（-p3[軸1] 上位40%除外）✓ |

## 手順の約束

- **閾値は掃引窓（2025-07〜2026-07）の分位点を絶対値として算出し、確認窓では動かさない**。
  本スクリプトが掃引窓から自動計算して持ち込む（転記ミス防止）。
  確認窓で分位を取り直すと「その窓に合わせた閾値」になり検証にならない。
- 確認窓は 2024-07〜2025-06（掃引に一度も使っていない）。学習は各窓開始日より前のみ。
- **群別（新7S / A群 / E群）に測り、件数の減りも見る**。
- 窓別の符号一貫性を必ず見る（平均は反転を隠す）。

⚠️ 掃引窓と確認窓で TRAIN_FROM が違う（前者2024-04-01・後者2022-12-01）。
   これはキャッシュの都合で、どちらも「学習は窓開始日より前のみ」は満たしている。

使い方:
    python scripts/exp_7a_exclusion_confirm.py
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

from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

STAKE = 100
SWEEP = {"w1": ("2026-04-13", "2026-07-15", 94), "w2": ("2026-01-01", "2026-04-12", 102),
         "w3": ("2025-10-01", "2025-12-31", 92), "w4": ("2025-07-01", "2025-09-30", 92)}
SWEEP_TRAIN_FROM = "2024-04-01"
CONFIRM = {"c1": ("2025-04-01", "2025-06-30", 91), "c2": ("2025-01-01", "2025-03-31", 90),
           "c3": ("2024-10-01", "2024-12-31", 92), "c4": ("2024-07-01", "2024-09-30", 92)}
CONFIRM_TRAIN_FROM = "2022-12-01"
CACHE_DIR = REPO / "data" / "exp_cache"


def cached_preds(tf, tt, train_from):
    p = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{train_from}.pkl"
    if not p.exists():
        raise SystemExit(f"[FATAL] 予測キャッシュがありません: {p}\n"
                         f"  先に該当窓の掃引/確認スクリプトを実行してください。")
    return pd.read_pickle(p)


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


def build(df, spec, train_from):
    per = []
    for w, (tf, tt, days) in spec.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, train_from), on=["race_key", "frame_no"], how="inner")
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
            r["ov"] = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
            if r["ov"] not in (0, 1, 2):
                continue
            r["a1"], r["a2"] = a1, a2
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["d_bad1"], r["d_np3"], r["d_mix"] = r["bad"][a1], -r["p3"][a1], zb[a1] - zp[a1]
            g1, g2 = lg.get(a1), lg.get(a2)
            r["same_line"] = (g1 is not None and g2 is not None
                              and str(g1) != "" and str(g1) == str(g2))
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        per.append((w, [r for r in races if trio.get(r["rk"])], trio, days))
    return per


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


def main():
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=CONFIRM_TRAIN_FROM,
                                            max_date="2026-07-15"))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    print("掃引窓から閾値を算出（確認窓では動かさない）...", flush=True)
    sw = build(df, SWEEP, SWEEP_TRAIN_FROM)
    THR = {}
    for key, keep in (("d_np3", 0.60), ("d_mix", 0.60), ("d_bad1", 0.70)):
        v = np.concatenate([[r[key] for r in rs] for _, rs, _, _ in sw])
        THR[key] = float(np.quantile(v, keep))
        print(f"  {key}: 上位{int((1-keep)*100)}%除外 → 閾値 {THR[key]:.4f}")

    print("\n確認窓を構築 ...", flush=True)
    cw = build(df, CONFIRM, CONFIRM_TRAIN_FROM)
    print(f"  母集団 {sum(len(rs) for _, rs, _, _ in cw)} レース")

    def evaluate(pred):
        per = []
        for w, races, trio, days in cw:
            rows = [s for r in races if pred(r)
                    and (s := settle(r, trio[r["rk"]])) is not None]
            if not rows:
                continue
            bet = sum(x[0] for x in rows); ret = sum(x[1] for x in rows)
            h = [x for x in rows if x[2]]
            per.append(dict(n=len(rows), per_day=len(rows) / days,
                            hit=100 * len(h) / len(rows),
                            roi=100 * ret / bet if bet else 0))
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
              f"{m['roi']:>8.1f}%  {flag} " + " ".join(f"{x:5.1f}" for x in per))

    A = RANK_7S_AXIS_SUM_MAX      # 1.40（PR#9で採用済み）
    groups = {
        "新7S": lambda r: r["ov"] in (0, 1) and r["asum"] <= A and r["ent"] <= RANK_7S_ENTROPY_MAX,
        "A群(axis_sum不合格)": lambda r: r["ov"] in (0, 1) and r["asum"] > A and r["ent"] <= RANK_7S_ENTROPY_MAX,
        "E群(entropy不合格)": lambda r: r["ov"] in (0, 1) and r["asum"] <= A and r["ent"] > RANK_7S_ENTROPY_MAX,
    }
    print("\n" + "=" * 100)
    print("【確認窓 2024-07〜2025-06・閾値は掃引窓で決めた絶対値を固定】")
    for gname, gp in groups.items():
        print(f"\n■ {gname}")
        print(f"  {'条件':<36}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}"
              f"     窓別ROI(c1 c2 c3 c4)")
        show("基準", gp)
        show("③ 同一ライン", lambda r: gp(r) and r["same_line"])
        show("③ 別ライン（対照）", lambda r: gp(r) and not r["same_line"])
        show(f"① -p3[軸1] < {THR['d_np3']:.3f}", lambda r: gp(r) and r["d_np3"] < THR["d_np3"])
        show(f"① z(bad)-z(p3) < {THR['d_mix']:.3f}", lambda r: gp(r) and r["d_mix"] < THR["d_mix"])
        show(f"① bad[軸1] < {THR['d_bad1']:.3f}", lambda r: gp(r) and r["d_bad1"] < THR["d_bad1"])
        show("③+① 同一ライン ∧ -p3[軸1]",
             lambda r: gp(r) and r["same_line"] and r["d_np3"] < THR["d_np3"])

    print("\n■ 合計（片方の群だけ見ないこと）")
    # 7B は overlap==2 ∧ order_disagree。件数の全体像を出すため母集団に含めてある。
    from src.strategy_wt import rank_7b_order_disagree
    is7b = lambda r: r["ov"] == 2 and rank_7b_order_disagree(r["pw"], r["hon"]) is True
    print(f"  {'条件':<36}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}"
          f"     窓別ROI(c1 c2 c3 c4)")
    any_g = lambda r: any(f(r) for f in groups.values())
    is7a = lambda r: groups["A群(axis_sum不合格)"](r) or groups["E群(entropy不合格)"](r)
    show("新7S+7A（基準）", any_g)
    show("新7S + 7A同一ラインのみ",
         lambda r: groups["新7S"](r) or (is7a(r) and r["same_line"]))
    show("新7S+7A すべて同一ラインのみ", lambda r: any_g(r) and r["same_line"])
    show("★新7S + A群全件 + E群同一ラインのみ",
         lambda r: groups["新7S"](r) or groups["A群(axis_sum不合格)"](r)
         or (groups["E群(entropy不合格)"](r) and r["same_line"]))
    print("  ── 7B（件数の全体像用・上の合計には含まれない）")
    show("7B", is7b)
    print("\n  ✓ = 確認窓4つすべてで ROI>=75%")


if __name__ == "__main__":
    main()
