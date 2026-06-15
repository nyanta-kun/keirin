"""grade_enc バグ修正の影響検証

バグ内容:
  feature_wt.py の grade_map が {"GP":7, "G1":6, ...} と keirin-station 形式になっており
  wt の実際の値（S級/A級/L級）に全く一致しない。
  結果として grade_enc が全レースで fillna(1) = 1 に固定され、グレード情報がゼロ。

修正内容:
  grade_map = {"S級": 3, "SA混合": 3, "A級": 2, "L級": 1}

検証:
  Phase1: AUC 比較（バグ版 vs 修正版）
  Phase2: ROI 比較（C0戦略・リーク無しプロトコル）
  グレード別 ROI 内訳（S/A/L 別の数字）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15
"""

import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0

# WT 用の正しいグレードマッピング
GRADE_MAP_FIX = {"S級": 3, "SA混合": 3, "A級": 2, "L級": 1}


def build_fixed(df_base: pd.DataFrame) -> pd.DataFrame:
    """grade_enc を正しいマッピングで上書きする。"""
    df = df_base.copy()
    if "grade" not in df.columns:
        with get_connection() as conn:
            grade_df = pd.read_sql("SELECT race_key, grade FROM wt_races", conn)
        df = df.merge(grade_df, on="race_key", how="left", suffixes=("", "_r"))
        if "grade_r" in df.columns:
            df["grade"] = df.pop("grade_r")
    df["grade_enc"] = df["grade"].map(GRADE_MAP_FIX).fillna(2).astype(int)
    return df


def period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def roi_records(df_pred: pd.DataFrame, trio_map: dict, actual_trio: dict,
                n_entries_map: dict, grade_map_rk: dict) -> list[dict]:
    records = []
    for rk, grp in df_pred.groupby("race_key"):
        period = period_of(str(grp["race_date"].iloc[0]))
        if period is None:
            continue
        if n_entries_map.get(rk, 99) > 6:
            continue
        g = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 3:
            continue
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
        grade = grade_map_rk.get(rk, "?")
        records.append({
            "period": period, "race_key": rk,
            "pay": pay, "cost": len(thirds) * 100,
            "hit": int(pay > 0), "grade": grade,
        })
    return records


def print_roi_table(rec_b: list, rec_f: list, grades: list) -> None:
    rec_b_df = pd.DataFrame(rec_b)
    rec_f_df = pd.DataFrame(rec_f)

    print(f"\n  {'期間':<8} {'バグ版':>10} {'修正版':>10} {'差分':>8}  {'n(bug)':>7} {'n(fix)':>7}")
    print("  " + "-" * 58)
    for period in ["TRAIN", "VAL", "HOLD"]:
        b = rec_b_df[rec_b_df["period"] == period]
        f = rec_f_df[rec_f_df["period"] == period]
        roi_b = b["pay"].sum() / b["cost"].sum() * 100 if b["cost"].sum() > 0 else float("nan")
        roi_f = f["pay"].sum() / f["cost"].sum() * 100 if f["cost"].sum() > 0 else float("nan")
        mk_b = "★" if roi_b >= 100 else ""
        mk_f = "★" if roi_f >= 100 else ""
        print(f"  {period:<8} {roi_b:>9.1f}%{mk_b}  {roi_f:>9.1f}%{mk_f}  {roi_f-roi_b:>+7.1f}pp  {len(b):>7} {len(f):>7}")

    print(f"\n  {'グレード':>8} {'期間':<7} {'修正版 ROI':>10}  n")
    print("  " + "-" * 35)
    for grade in grades:
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec_f_df[(rec_f_df["grade"] == grade) & (rec_f_df["period"] == period)]
            if len(sub) == 0:
                continue
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if sub["cost"].sum() > 0 else float("nan")
            mk = "★" if roi >= 100 else ""
            print(f"  {grade:>8} {period:<7} {roi:>9.1f}%{mk}  {len(sub)}")


def main():
    print("grade_enc バグ修正の影響検証")
    print(f"  バグ: grade_map が全wt値をfillna(1)にする → grade_enc=1 固定")
    print(f"  修正: S級→3 / A級→2 / L級→1")
    print()

    print("データ準備中...", flush=True)
    df_bug = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    df_fix = build_fixed(df_bug)
    print(f"  バグ版 grade_enc 分布: {df_bug['grade_enc'].value_counts().to_dict()}")
    print(f"  修正版 grade_enc 分布: {df_fix['grade_enc'].value_counts().to_dict()}")
    print()

    # 学習
    fit_b = df_bug[(df_bug["race_date"] >= TRAIN[0]) & (df_bug["race_date"] <= TRAIN[1]) & (df_bug["finish_order"] >= 1)]
    fit_f = df_fix[(df_fix["race_date"] >= TRAIN[0]) & (df_fix["race_date"] <= TRAIN[1]) & (df_fix["finish_order"] >= 1)]

    m_bug = lgb.LGBMClassifier(**LGB_PARAMS)
    m_bug.fit(prepare_X(fit_b), fit_b["top3_flag"].values)
    m_fix = lgb.LGBMClassifier(**LGB_PARAMS)
    m_fix.fit(prepare_X(fit_f), fit_f["top3_flag"].values)
    print(f"  両モデル学習完了 (TRAIN {len(fit_b):,} rows)")

    # ─── Phase1: AUC ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較")
    print("=" * 70)
    print(f"  {'期間':<10} {'バグ版 AUC':>12} {'修正版 AUC':>12} {'差分':>10}")
    print("  " + "-" * 46)
    for period, (s, e) in [("VAL", VAL), ("HOLD", HOLD), ("VAL+HOLD", (VAL[0], HOLD[1]))]:
        sub_b = df_bug[(df_bug["race_date"].between(s, e)) & (df_bug["finish_order"] >= 1)]
        sub_f = df_fix[(df_fix["race_date"].between(s, e)) & (df_fix["finish_order"] >= 1)]
        auc_b = roc_auc_score(sub_b["top3_flag"], m_bug.predict_proba(prepare_X(sub_b))[:, 1])
        auc_f = roc_auc_score(sub_f["top3_flag"], m_fix.predict_proba(prepare_X(sub_f))[:, 1])
        diff  = auc_f - auc_b
        mark  = "★ PASS" if (period == "VAL+HOLD" and diff >= 0.001) else ""
        print(f"  {period:<10} {auc_b:>12.4f} {auc_f:>12.4f} {diff:>+10.4f}  {mark}")

    # ─── Phase2: ROI ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍）")
    print("=" * 70)

    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries, grade FROM wt_races", conn)
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()

    trio_map: dict[str, dict] = {}
    for rk, comb, ov in trio_raw:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)

    actual_trio = (
        df_bug[df_bug["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )
    n_entries_map = dict(zip(races_info["race_key"], races_info["n_entries"]))
    grade_map_rk  = dict(zip(races_info["race_key"], races_info["grade"]))

    df_bug["pred_prob"] = m_bug.predict_proba(prepare_X(df_bug))[:, 1]
    df_fix["pred_prob"] = m_fix.predict_proba(prepare_X(df_fix))[:, 1]

    rec_b = roi_records(df_bug, trio_map, actual_trio, n_entries_map, grade_map_rk)
    rec_f = roi_records(df_fix, trio_map, actual_trio, n_entries_map, grade_map_rk)

    print_roi_table(rec_b, rec_f, ["S級", "A級", "L級"])

    # ─── 特徴量重要度変化 ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("grade_enc の重要度比較（バグ版 vs 修正版）")
    print("=" * 70)
    cols = FEATURE_COLS_WT
    imp_b = pd.Series(m_bug.feature_importances_, index=cols)
    imp_f = pd.Series(m_fix.feature_importances_, index=cols)
    rank_b = imp_b.rank(ascending=False).astype(int)
    rank_f = imp_f.rank(ascending=False).astype(int)
    print(f"  grade_enc: バグ版 重要度 {imp_b['grade_enc']:.0f} (全体{rank_b['grade_enc']}位) "
          f"→ 修正版 {imp_f['grade_enc']:.0f} (全体{rank_f['grade_enc']}位)")
    print(f"  全体合計: バグ版 {imp_b.sum():.0f} / 修正版 {imp_f.sum():.0f}")

    # top5の変化
    print("\n  修正版 重要度 Top10:")
    for feat, score in imp_f.sort_values(ascending=False).head(10).items():
        pct = score / imp_f.sum() * 100
        print(f"    {feat:<28} {score:>6.0f}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
