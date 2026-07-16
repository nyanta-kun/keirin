"""条件別 ROI 深掘り分析

doc45 で浮上した有力条件（gap12・grade・bank_length）を深掘りする。
- TRAIN を 2022-12-01 まで拡張（追加 7 ヶ月）
- gap12 閾値を 0.04〜0.20 でスイープ
- grade × gap12 交差の最適点を探索
- bank_length × grade 交差
- 各セルで n・ROI・的中率・平均払戻を出力

Usage:
  python3 scripts/exp_conditional_deep_wt.py
"""
import sys, re
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT
from src.database import get_connection
from exp_segment_first_wt import VAL, HOLD, LGB_PARAMS

# TRAIN を 2022-12 に拡張（+7 ヶ月）
TRAIN_EXT = ("2022-12-01", "2025-06-30")

GAMI_THRESHOLD = 5.0


# ── データロード ──────────────────────────────────────────────────────
def load_n_entries() -> dict:
    with get_connection() as conn:
        rows = conn.execute("SELECT race_key, n_entries FROM wt_races").fetchall()
    return {r[0]: r[1] for r in rows}


def load_trio_map() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()
    trio_map: dict = {}
    for rk, comb, ov in rows:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)
    return trio_map


def _period_of(rd: str, train_end: str):
    if rd <= train_end:                   return "TRAIN"
    if VAL[0] <= rd <= VAL[1]:            return "VAL"
    if HOLD[0] <= rd <= HOLD[1]:          return "HOLD"
    return None


# ── レース単位集計 ────────────────────────────────────────────────────
def build_race_df(df: pd.DataFrame, n_entries_map: dict, trio_map: dict, train_end: str) -> pd.DataFrame:
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    records = []
    for rk, grp in df.groupby("race_key"):
        n_ent = n_entries_map.get(rk)
        if n_ent is None or n_ent > 6:
            continue

        g = grp.sort_values("pred", ascending=False).reset_index(drop=True)
        if len(g) < 3:
            continue
        period = _period_of(str(g["race_date"].iloc[0]), train_end)
        if period is None:
            continue

        preds = g["pred"].values
        gap12 = float(preds[0] - preds[1])
        pred1_val = float(preds[0])

        p1 = int(g.iloc[0]["frame_no"])
        p2 = int(g.iloc[1]["frame_no"])
        thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]

        bd = trio_map.get(rk, {})
        combos = [frozenset({p1, p2, t}) for t in thirds]
        min_odds = min((bd.get(k, 0) for k in combos if bd.get(k, 0) > 0), default=0)
        if min_odds < GAMI_THRESHOLD:
            continue

        actual = actual_trio.get(rk, frozenset())
        pay = 0.0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual == k:
                pay = bd.get(k, 0) * 100
                break
        cost = len(thirds) * 100

        grade_raw = g["grade"].iloc[0] if "grade" in g.columns else None
        if grade_raw in ("S級", "SA混合"):
            grade_bin = "S級"
        elif grade_raw == "A級":
            grade_bin = "A級"
        else:
            grade_bin = "その他"

        bank = g["bank_length"].iloc[0] if "bank_length" in g.columns else None
        try:
            bank_int = int(bank) if bank and not pd.isna(bank) else None
        except (ValueError, TypeError):
            bank_int = None
        bank_bin = {333: "333m", 400: "400m", 500: "500m"}.get(bank_int, "other")

        records.append({
            "race_key":  rk,
            "period":    period,
            "n_entries": int(n_ent),
            "grade_bin": grade_bin,
            "bank_bin":  bank_bin,
            "gap12":     gap12,
            "pred1":     pred1_val,
            "pay":       pay,
            "cost":      cost,
            "hit":       int(pay > 0),
        })

    return pd.DataFrame(records)


# ── 統計出力 ──────────────────────────────────────────────────────────
def stats(sub: pd.DataFrame):
    """(n, roi, hit_rate, avg_pay) を返す。n=0 は nan。"""
    n = len(sub)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    total_cost = sub["cost"].sum()
    total_pay  = sub["pay"].sum()
    roi = total_pay / total_cost * 100 if total_cost > 0 else float("nan")
    hit_rate = sub["hit"].mean() * 100
    avg_pay  = sub[sub["hit"] == 1]["pay"].mean() if sub["hit"].sum() > 0 else 0.0
    return n, roi, hit_rate, avg_pay


def fmt_roi(roi, n):
    if n == 0 or np.isnan(roi): return "    -   "
    mk = "★" if roi >= 100 else " "
    return f"{roi:6.1f}%{mk}"


def print_row(label, sub, width=28):
    parts = [f"  {label:<{width}}"]
    for p in ["TRAIN", "VAL", "HOLD"]:
        n, roi, hr, ap = stats(sub[sub["period"] == p])
        parts.append(f" {fmt_roi(roi, n):>9}({n:3d})")
    print("".join(parts))


def print_row_detail(label, sub, width=28):
    """ROI + 的中率 + 平均払戻を出力"""
    parts = [f"  {label:<{width}}"]
    for p in ["TRAIN", "VAL", "HOLD"]:
        n, roi, hr, ap = stats(sub[sub["period"] == p])
        if n == 0:
            parts.append(f" {'    -':>14}")
        else:
            mk = "★" if roi >= 100 else " "
            parts.append(f" {roi:5.0f}%{mk}/{hr:4.1f}%/{ap:6.0f}({n})")
    print("".join(parts))


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("条件別ROI深掘り分析\n")
    print(f"TRAIN拡張: {TRAIN_EXT[0]} 〜 {TRAIN_EXT[1]}")
    print(f"VAL  : {VAL[0]} 〜 {VAL[1]}")
    print(f"HOLD : {HOLD[0]} 〜 {HOLD[1]}\n")

    print("データ準備中（2022-12〜HOLD）...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN_EXT[0], max_date=HOLD[1]))
    df = df_all[df_all["finish_order"] >= 1].copy()
    print(f"  全行数: {len(df):,}  レース数: {df['race_key'].nunique():,}")

    n_entries_map = load_n_entries()
    trio_map = load_trio_map()

    # TRAIN期間のみで学習（リーク無し・拡張データ使用）
    print("モデル学習中（TRAIN拡張期間のみ）...", flush=True)
    fit = df[df["race_date"] <= TRAIN_EXT[1]]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"].values)
    df["pred"] = m.predict_proba(prepare_X(df))[:, 1]
    print(f"  学習行数: {len(fit):,}")

    print("レース単位集計中...", flush=True)
    rdf = build_race_df(df, n_entries_map, trio_map, TRAIN_EXT[1])
    for p in ["TRAIN", "VAL", "HOLD"]:
        sub = rdf[rdf["period"] == p]
        print(f"  {p}: {len(sub)}R")

    HEADER = f"\n  {'条件':<28} {'TRAIN(ROI/n)':>13} {'VAL(ROI/n)':>13} {'HOLD(ROI/n)':>13}"
    HEADER_D = f"\n  {'条件':<28} {'TRAIN(ROI/的中率/avg払)':>20} {'VAL':>20} {'HOLD':>20}"

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 1: 全体・単一条件ベースライン")
    print("=" * 75)
    print(HEADER)
    r = rdf
    print_row("ALL",    r)
    print_row("S級",    r[r["grade_bin"] == "S級"])
    print_row("A級",    r[r["grade_bin"] == "A級"])
    print_row("n=5車",  r[r["n_entries"] == 5])
    print_row("n=6車",  r[r["n_entries"] == 6])
    print_row("333m",   r[r["bank_bin"] == "333m"])
    print_row("400m",   r[r["bank_bin"] == "400m"])
    print_row("500m",   r[r["bank_bin"] == "500m"])

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 2: gap12 閾値スイープ（ALL・S級・A級）")
    print("=" * 75)
    thresholds = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20]

    print(HEADER)
    print("  --- ALL ---")
    for thr in thresholds:
        sub = r[r["gap12"] > thr]
        print_row(f"  gap12 >{thr:.2f}", sub)

    print("\n  --- S級のみ ---")
    s_rdf = r[r["grade_bin"] == "S級"]
    for thr in thresholds:
        sub = s_rdf[s_rdf["gap12"] > thr]
        print_row(f"  S × gap12 >{thr:.2f}", sub)

    print("\n  --- A級のみ ---")
    a_rdf = r[r["grade_bin"] == "A級"]
    for thr in thresholds:
        sub = a_rdf[a_rdf["gap12"] > thr]
        print_row(f"  A × gap12 >{thr:.2f}", sub)

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 3: bank_length × grade 詳細（的中率・平均払戻付き）")
    print("=" * 75)
    print(HEADER_D)
    cells = [
        ("333m × S級",  (r["bank_bin"]=="333m") & (r["grade_bin"]=="S級")),
        ("333m × A級",  (r["bank_bin"]=="333m") & (r["grade_bin"]=="A級")),
        ("400m × S級",  (r["bank_bin"]=="400m") & (r["grade_bin"]=="S級")),
        ("400m × A級",  (r["bank_bin"]=="400m") & (r["grade_bin"]=="A級")),
        ("500m × S級",  (r["bank_bin"]=="500m") & (r["grade_bin"]=="S級")),
        ("500m × A級",  (r["bank_bin"]=="500m") & (r["grade_bin"]=="A級")),
        ("333m 全体",   r["bank_bin"]=="333m"),
        ("400m 全体",   r["bank_bin"]=="400m"),
        ("500m 全体",   r["bank_bin"]=="500m"),
    ]
    for label, mask in cells:
        print_row_detail(label, r[mask])

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 4: 有望 2条件の gap12 閾値スイープ（的中率・平均払戻付き）")
    print("=" * 75)
    print(f"  （ROI = 払戻合計/賭け金合計、的中率 = 的中R/全R、avg払 = 的中時の平均払戻）")

    print(HEADER_D)
    print("  --- 400m × S級 ---")
    s400 = r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級")]
    for thr in [0.0, 0.05, 0.07, 0.10, 0.12, 0.15]:
        sub = s400[s400["gap12"] > thr] if thr > 0 else s400
        print_row_detail(f"  400mS gap12>{thr:.2f}", sub)

    print("\n  --- S級 all banklength ---")
    s_all = r[r["grade_bin"] == "S級"]
    for thr in [0.0, 0.05, 0.07, 0.10, 0.12, 0.15]:
        sub = s_all[s_all["gap12"] > thr] if thr > 0 else s_all
        print_row_detail(f"  S × gap12>{thr:.2f}", sub)

    print("\n  --- n=6 × S級 ---")
    n6s = r[(r["n_entries"]==6) & (r["grade_bin"]=="S級")]
    for thr in [0.0, 0.05, 0.07, 0.10, 0.12, 0.15]:
        sub = n6s[n6s["gap12"] > thr] if thr > 0 else n6s
        print_row_detail(f"  n6S gap12>{thr:.2f}", sub)

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 5: 最有力セル — 月別ROI推移（VAL+HOLD 期間）")
    print("=" * 75)

    best_conditions = [
        ("ALL C0",          rdf),
        ("S級",              r[r["grade_bin"]=="S級"]),
        ("gap12>0.10",       r[r["gap12"]>0.10]),
        ("S × gap12>0.10",   r[(r["grade_bin"]=="S級") & (r["gap12"]>0.10)]),
        ("400m × S",         r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級")]),
        ("400m × S × g>0.10",r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級") & (r["gap12"]>0.10)]),
    ]

    val_hold = rdf[rdf["period"].isin(["VAL", "HOLD"])].copy()
    val_hold["ym"] = val_hold["race_key"].apply(
        lambda rk: df.loc[df["race_key"]==rk, "race_date"].iloc[0][:7]
        if (df["race_key"]==rk).any() else "?"
    ) if "race_date" in rdf.columns else "?"

    # race_date を rdf に追加
    rd_map = df.groupby("race_key")["race_date"].first().to_dict()
    rdf["race_date"] = rdf["race_key"].map(rd_map)
    rdf["ym"] = rdf["race_date"].str[:7]

    months = sorted(rdf[rdf["period"].isin(["VAL","HOLD"])]["ym"].dropna().unique())

    for label, sub_base in best_conditions:
        print(f"\n  【{label}】")
        vals = []
        for ym in months:
            s = sub_base[sub_base["ym"] == ym] if "ym" in sub_base.columns else pd.DataFrame()
            n = len(s)
            if n == 0:
                vals.append(f"{ym}: -(0R)")
            else:
                roi = s["pay"].sum() / s["cost"].sum() * 100
                mk = "★" if roi >= 100 else " "
                vals.append(f"{ym}: {roi:5.0f}%{mk}({n}R)")
        print("    " + "  ".join(vals))

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("▶ Section 6: 総合まとめ — VAL×HOLD 両通過セル（n≥8/4）")
    print("=" * 75)

    candidates = [
        ("ALL",                  r),
        ("S級",                  r[r["grade_bin"]=="S級"]),
        ("A級",                  r[r["grade_bin"]=="A級"]),
        ("n=6",                  r[r["n_entries"]==6]),
        ("333m",                 r[r["bank_bin"]=="333m"]),
        ("400m",                 r[r["bank_bin"]=="400m"]),
        ("400m×S",               r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級")]),
        ("400m×A",               r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="A級")]),
        ("333m×S",               r[(r["bank_bin"]=="333m") & (r["grade_bin"]=="S級")]),
        ("n6×S",                 r[(r["n_entries"]==6) & (r["grade_bin"]=="S級")]),
        ("n6×A",                 r[(r["n_entries"]==6) & (r["grade_bin"]=="A級")]),
    ]
    for thr in [0.07, 0.08, 0.09, 0.10, 0.12, 0.15]:
        candidates += [
            (f"gap12>{thr:.2f}",        r[r["gap12"]>thr]),
            (f"S×gap12>{thr:.2f}",      r[(r["grade_bin"]=="S級") & (r["gap12"]>thr)]),
            (f"A×gap12>{thr:.2f}",      r[(r["grade_bin"]=="A級") & (r["gap12"]>thr)]),
            (f"400mS×gap12>{thr:.2f}",  r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級") & (r["gap12"]>thr)]),
            (f"n6×gap12>{thr:.2f}",     r[(r["n_entries"]==6) & (r["gap12"]>thr)]),
            (f"n6S×gap12>{thr:.2f}",    r[(r["n_entries"]==6) & (r["grade_bin"]=="S級") & (r["gap12"]>thr)]),
        ]

    print(f"\n  {'条件':<30} {'VAL-ROI':>9} {'HOLD-ROI':>9}  n(VA/HO)")
    print("  " + "-" * 65)
    passed = []
    for label, sub in candidates:
        _, vr, _, _ = stats(sub[sub["period"]=="VAL"])
        vn = len(sub[sub["period"]=="VAL"])
        _, hr, _, _ = stats(sub[sub["period"]=="HOLD"])
        hn = len(sub[sub["period"]=="HOLD"])
        if vn >= 8 and hn >= 4 and not np.isnan(vr) and vr >= 100 and not np.isnan(hr) and hr >= 100:
            passed.append((label, vr, vn, hr, hn))

    if passed:
        for lbl, vr, vn, hr, hn in sorted(passed, key=lambda x: -(x[1]+x[3])):
            print(f"  ★ {lbl:<30} {vr:7.1f}%   {hr:7.1f}%   {vn}/{hn}")
    else:
        print("  （VAL≥100% かつ HOLD≥100% のセルなし）")
    print()


if __name__ == "__main__":
    main()
