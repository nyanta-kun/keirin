"""条件別 ROI 体系スキャン

既存モデル（TRAIN期間のみで学習・リーク無し）の予測をベースに、
レース属性（n_entries, grade, bank_length）および
モデル出力（gap12, pred1）の条件分けを体系的にスキャンし、
C0戦略（trio・ガミ≥5倍・≤6車）の ROI を評価する。

多重比較の懸念があるため HOLD での再現を必須とする。

Usage:
  python3 scripts/exp_conditional_split_wt.py
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
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0


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


def _period_of(rd: str):
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def build_race_records(df: pd.DataFrame, n_entries_map: dict, trio_map: dict) -> pd.DataFrame:
    """レース単位で条件・ROI を集計した DataFrame を返す。"""
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
        period = _period_of(str(g["race_date"].iloc[0]))
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
            "race_key":   rk,
            "period":     period,
            "n_entries":  int(n_ent),
            "grade_bin":  grade_bin,
            "bank_bin":   bank_bin,
            "gap12":      gap12,
            "pred1":      pred1_val,
            "pay":        pay,
            "cost":       cost,
        })

    return pd.DataFrame(records)


def roi_cell(sub: pd.DataFrame, period: str):
    s = sub[sub["period"] == period]
    if len(s) == 0:
        return float("nan"), 0
    roi = s["pay"].sum() / s["cost"].sum() * 100
    return roi, len(s)


def fmt(roi, n):
    if n == 0 or np.isnan(roi):
        return "    -    "
    mk = "★" if roi >= 100 else " "
    return f"{roi:6.1f}%{mk}"


def print_table(title, rows):
    """rows: list of (label, mask_or_sub_df)"""
    print(f"\n  [{title}]")
    print(f"  {'条件':<30} {'TRAIN':>10} {'VAL':>10} {'HOLD':>10}  n(TR/VA/HO)")
    print("  " + "-" * 72)
    for label, sub in rows:
        tr_roi, tr_n = roi_cell(sub, "TRAIN")
        va_roi, va_n = roi_cell(sub, "VAL")
        ho_roi, ho_n = roi_cell(sub, "HOLD")
        line = (f"  {label:<30}"
                f" {fmt(tr_roi, tr_n):>10}"
                f" {fmt(va_roi, va_n):>10}"
                f" {fmt(ho_roi, ho_n):>10}"
                f"  {tr_n}/{va_n}/{ho_n}")
        print(line)


def main():
    print("条件別ROI体系スキャン\n")

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    df = df_all[df_all["finish_order"] >= 1].copy()
    print(f"  全行数: {len(df):,}  レース数: {df['race_key'].nunique():,}")

    n_entries_map = load_n_entries()
    trio_map = load_trio_map()

    # TRAIN 期間のみで学習（リーク無し）
    print("モデル学習中（TRAIN期間のみ）...", flush=True)
    fit = df[df["race_date"] <= TRAIN[1]]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"].values)
    df["pred"] = m.predict_proba(prepare_X(df))[:, 1]

    print("レース単位集計中...", flush=True)
    race_df = build_race_records(df, n_entries_map, trio_map)
    tr_n = len(race_df[race_df["period"] == "TRAIN"])
    va_n = len(race_df[race_df["period"] == "VAL"])
    ho_n = len(race_df[race_df["period"] == "HOLD"])
    print(f"  C0対象レース: TRAIN={tr_n} / VAL={va_n} / HOLD={ho_n}")

    # ── binning ─────────────────────────────────────────────────────────
    race_df["gap12_bin"] = pd.cut(
        race_df["gap12"],
        bins=[-np.inf, 0.04, 0.07, 0.10, np.inf],
        labels=["<0.04", "0.04-0.07", "0.07-0.10", ">0.10"]
    )
    q25, q50, q75 = race_df["pred1"].quantile([0.25, 0.50, 0.75])

    race_df["pred1_bin"] = pd.cut(
        race_df["pred1"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["Q1(低)", "Q2", "Q3", "Q4(高)"]
    )

    print("\n" + "=" * 75)
    print("条件別 ROI スキャン（C0・trio・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 75)

    # ── 全体ベースライン ──────────────────────────────────────────────
    print_table("全体", [("ALL", race_df)])

    # ── 単一条件 ─────────────────────────────────────────────────────
    print_table("n_entries", [
        ("n=5車",  race_df[race_df["n_entries"] == 5]),
        ("n=6車",  race_df[race_df["n_entries"] == 6]),
    ])

    print_table("grade", [
        ("S級",    race_df[race_df["grade_bin"] == "S級"]),
        ("A級",    race_df[race_df["grade_bin"] == "A級"]),
        ("その他", race_df[race_df["grade_bin"] == "その他"]),
    ])

    print_table("bank_length", [
        ("333m",  race_df[race_df["bank_bin"] == "333m"]),
        ("400m",  race_df[race_df["bank_bin"] == "400m"]),
        ("500m",  race_df[race_df["bank_bin"] == "500m"]),
    ])

    print_table("gap12 帯", [
        ("gap12 <0.04",      race_df[race_df["gap12_bin"] == "<0.04"]),
        ("gap12 0.04-0.07",  race_df[race_df["gap12_bin"] == "0.04-0.07"]),
        ("gap12 0.07-0.10",  race_df[race_df["gap12_bin"] == "0.07-0.10"]),
        ("gap12 >0.10",      race_df[race_df["gap12_bin"] == ">0.10"]),
    ])

    print_table("pred1 分位数", [
        (f"Q1(低)  ≤{q25:.3f}",  race_df[race_df["pred1_bin"] == "Q1(低)"]),
        (f"Q2    ≤{q50:.3f}",    race_df[race_df["pred1_bin"] == "Q2"]),
        (f"Q3    ≤{q75:.3f}",    race_df[race_df["pred1_bin"] == "Q3"]),
        (f"Q4(高) >{q75:.3f}",   race_df[race_df["pred1_bin"] == "Q4(高)"]),
    ])

    # ── 2条件交差 ─────────────────────────────────────────────────────
    r = race_df
    cross2 = [
        ("n5 × S級",         r[(r["n_entries"]==5) & (r["grade_bin"]=="S級")]),
        ("n5 × A級",         r[(r["n_entries"]==5) & (r["grade_bin"]=="A級")]),
        ("n6 × S級",         r[(r["n_entries"]==6) & (r["grade_bin"]=="S級")]),
        ("n6 × A級",         r[(r["n_entries"]==6) & (r["grade_bin"]=="A級")]),
        ("S級 × gap12>0.07", r[(r["grade_bin"]=="S級") & (r["gap12"]>0.07)]),
        ("A級 × gap12>0.07", r[(r["grade_bin"]=="A級") & (r["gap12"]>0.07)]),
        ("S級 × gap12>0.10", r[(r["grade_bin"]=="S級") & (r["gap12"]>0.10)]),
        ("A級 × gap12>0.10", r[(r["grade_bin"]=="A級") & (r["gap12"]>0.10)]),
        ("S級 × pred1 Q4",   r[(r["grade_bin"]=="S級") & (r["pred1_bin"]=="Q4(高)")]),
        ("A級 × pred1 Q4",   r[(r["grade_bin"]=="A級") & (r["pred1_bin"]=="Q4(高)")]),
        ("n5 × gap12>0.07",  r[(r["n_entries"]==5) & (r["gap12"]>0.07)]),
        ("n6 × gap12>0.07",  r[(r["n_entries"]==6) & (r["gap12"]>0.07)]),
        ("n5 × gap12>0.10",  r[(r["n_entries"]==5) & (r["gap12"]>0.10)]),
        ("n6 × gap12>0.10",  r[(r["n_entries"]==6) & (r["gap12"]>0.10)]),
        ("333m × S級",       r[(r["bank_bin"]=="333m") & (r["grade_bin"]=="S級")]),
        ("400m × S級",       r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級")]),
        ("500m × S級",       r[(r["bank_bin"]=="500m") & (r["grade_bin"]=="S級")]),
        ("333m × A級",       r[(r["bank_bin"]=="333m") & (r["grade_bin"]=="A級")]),
        ("400m × A級",       r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="A級")]),
        ("500m × A級",       r[(r["bank_bin"]=="500m") & (r["grade_bin"]=="A級")]),
    ]
    print_table("2条件交差", cross2)

    # ── 3条件交差 ─────────────────────────────────────────────────────
    cross3 = [
        ("n5 × S × gap12>0.07",  r[(r["n_entries"]==5) & (r["grade_bin"]=="S級") & (r["gap12"]>0.07)]),
        ("n6 × S × gap12>0.07",  r[(r["n_entries"]==6) & (r["grade_bin"]=="S級") & (r["gap12"]>0.07)]),
        ("n5 × A × gap12>0.07",  r[(r["n_entries"]==5) & (r["grade_bin"]=="A級") & (r["gap12"]>0.07)]),
        ("n6 × A × gap12>0.07",  r[(r["n_entries"]==6) & (r["grade_bin"]=="A級") & (r["gap12"]>0.07)]),
        ("n5 × S × gap12>0.10",  r[(r["n_entries"]==5) & (r["grade_bin"]=="S級") & (r["gap12"]>0.10)]),
        ("n6 × S × gap12>0.10",  r[(r["n_entries"]==6) & (r["grade_bin"]=="S級") & (r["gap12"]>0.10)]),
        ("n5 × A × gap12>0.10",  r[(r["n_entries"]==5) & (r["grade_bin"]=="A級") & (r["gap12"]>0.10)]),
        ("n6 × A × gap12>0.10",  r[(r["n_entries"]==6) & (r["grade_bin"]=="A級") & (r["gap12"]>0.10)]),
        ("333m × S × gap12>0.07",r[(r["bank_bin"]=="333m") & (r["grade_bin"]=="S級") & (r["gap12"]>0.07)]),
        ("400m × S × gap12>0.07",r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="S級") & (r["gap12"]>0.07)]),
        ("400m × A × gap12>0.07",r[(r["bank_bin"]=="400m") & (r["grade_bin"]=="A級") & (r["gap12"]>0.07)]),
    ]
    print_table("3条件交差", cross3)

    # ── 総合判定 ──────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("総合判定（VAL ≥100% かつ HOLD ≥100% のセル）")
    print("=" * 75)

    all_rows = [("ALL", race_df)] + cross2 + cross3
    # single-dimension rows
    for cond_col, labels_vals in [
        ("n_entries", [(5, "n=5車"), (6, "n=6車")]),
        ("grade_bin",  [("S級", "S級"), ("A級", "A級")]),
        ("bank_bin",   [("333m", "333m"), ("400m", "400m"), ("500m", "500m")]),
    ]:
        for val, lbl in labels_vals:
            all_rows.append((lbl, race_df[race_df[cond_col] == val]))
    for bin_val in ["<0.04", "0.04-0.07", "0.07-0.10", ">0.10"]:
        all_rows.append((f"gap12 {bin_val}", race_df[race_df["gap12_bin"] == bin_val]))
    for bin_val in ["Q1(低)", "Q2", "Q3", "Q4(高)"]:
        all_rows.append((f"pred1 {bin_val}", race_df[race_df["pred1_bin"] == bin_val]))

    passed_both = []
    passed_val = []
    for label, sub in all_rows:
        va_roi, va_n = roi_cell(sub, "VAL")
        ho_roi, ho_n = roi_cell(sub, "HOLD")
        if va_n >= 10 and ho_n >= 5 and not np.isnan(va_roi) and va_roi >= 100:
            if not np.isnan(ho_roi) and ho_roi >= 100:
                passed_both.append((label, va_roi, va_n, ho_roi, ho_n))
            else:
                passed_val.append((label, va_roi, va_n, ho_roi, ho_n))

    if passed_both:
        print(f"\n  VAL かつ HOLD 両方 ≥100%（n≥10/5）: {len(passed_both)} セル")
        for lbl, vr, vn, hr, hn in passed_both:
            print(f"    ★ {lbl:<30} VAL {vr:.1f}%({vn}R) / HOLD {hr:.1f}%({hn}R)")
    else:
        print("\n  VAL かつ HOLD 両方 ≥100% のセル: なし")

    if passed_val:
        print(f"\n  VAL のみ ≥100%（HOLDは不通過）: {len(passed_val)} セル")
        for lbl, vr, vn, hr, hn in passed_val:
            hr_str = f"{hr:.1f}%" if not np.isnan(hr) else "-"
            print(f"    △ {lbl:<30} VAL {vr:.1f}%({vn}R) / HOLD {hr_str}({hn}R)")
    print()


if __name__ == "__main__":
    main()
