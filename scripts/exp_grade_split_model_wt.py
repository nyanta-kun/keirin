"""グレード別モデル分離実験（S級 vs A級）

仮説:
  S級（選手クラス SS/S1/S2）と A級（A1/A2/A3）は競争構造が異なる。
  同一モデルで両者を扱うと「平均的なパターン」しか学べず、
  グレード固有のシグナルを取りこぼしている可能性がある。

実験内容:
  arm Base: 全グレード統合モデル（現行）
  arm S:    S級レースのみで学習・S級レースのみで評価
  arm A:    A級レースのみで学習・A級レースのみで評価

  Phase1: AUC 比較
  Phase2: ROI 比較（C0戦略: pred1+pred2→thirds・ガミ≥5倍）

注: L級（ガールズ）はレース数が少なく別カテゴリのため除外。

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


def period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def train_model(df: pd.DataFrame, grade_filter: str | None = None) -> lgb.LGBMClassifier:
    """TRAIN 期間のみで学習。grade_filter が None なら全グレード。"""
    mask = (df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)
    if grade_filter:
        mask &= (df["grade_group"] == grade_filter)
    fit = df[mask]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"].values)
    return m, len(fit)


def compute_roi_records(df: pd.DataFrame, trio_map: dict, actual_trio: dict,
                        n_entries_map: dict, grade_rk: dict) -> list[dict]:
    records = []
    for rk, grp in df.groupby("race_key"):
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
        records.append({
            "period": period, "race_key": rk,
            "grade": grade_rk.get(rk, "?"),
            "pay": pay, "cost": len(thirds) * 100,
            "hit": int(pay > 0),
        })
    return records


def roi_str(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return f"  {'—':>9}   {'—':>4}"
    roi = sub["pay"].sum() / sub["cost"].sum() * 100
    mark = "★" if roi >= 100 else " "
    return f"  {roi:>8.1f}%{mark}  {len(sub):>4}"


def main():
    print("グレード別モデル分離実験（S級 / A級）")
    print()

    print("データ準備中...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    # grade_group 付与
    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries, grade FROM wt_races", conn)
    grade_map_rk = dict(zip(races_info["race_key"], races_info["grade"]))
    df["grade_group"] = df["race_key"].map(grade_map_rk).map(
        {"S級": "S", "SA混合": "S", "A級": "A", "L級": "L"}
    ).fillna("A")
    n_entries_map = dict(zip(races_info["race_key"], races_info["n_entries"]))

    print(f"  grade_group 分布 (全行): {df['grade_group'].value_counts().to_dict()}")

    # モデル学習（3種）
    print("\nモデル学習中...", flush=True)
    m_all, n_all = train_model(df, None)
    m_s,   n_s   = train_model(df, "S")
    m_a,   n_a   = train_model(df, "A")
    print(f"  Base (全グレード): {n_all:,} rows")
    print(f"  S-model          : {n_s:,} rows")
    print(f"  A-model          : {n_a:,} rows")

    # 予測
    df["pred_all"] = m_all.predict_proba(prepare_X(df))[:, 1]
    df["pred_s"]   = m_s.predict_proba(prepare_X(df))[:, 1]
    df["pred_a"]   = m_a.predict_proba(prepare_X(df))[:, 1]
    # 専用モデル: S级レースにはS-model、A级レースにはA-model
    df["pred_grade"] = np.where(df["grade_group"] == "S", df["pred_s"], df["pred_a"])

    # ─── Phase1: AUC ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較（リーク無し）")
    print("=" * 70)

    for grade in ["ALL", "S", "A"]:
        print(f"\n  グレード: {grade}")
        print(f"  {'期間':<10} {'Base AUC':>10} {'専用 AUC':>10} {'差分':>8}")
        print("  " + "-" * 40)
        for period, s, e in [("VAL", VAL[0], VAL[1]), ("HOLD", HOLD[0], HOLD[1]),
                              ("VAL+HOLD", VAL[0], HOLD[1])]:
            mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
            if grade != "ALL":
                mask &= (df["grade_group"] == grade)
            sub = df[mask]
            if len(sub) < 10:
                continue
            auc_b = roc_auc_score(sub["top3_flag"], sub["pred_all"])
            pred_col = "pred_s" if grade == "S" else ("pred_a" if grade == "A" else "pred_grade")
            auc_g = roc_auc_score(sub["top3_flag"], sub[pred_col])
            diff = auc_g - auc_b
            mark = "★" if (period == "VAL+HOLD" and diff >= 0.001) else ""
            print(f"  {period:<10} {auc_b:>10.4f} {auc_g:>10.4f} {diff:>+8.4f}  {mark}")

    # ─── Phase2: ROI ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車）")
    print("=" * 70)

    with get_connection() as conn:
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()

    trio_map: dict = {}
    for rk, comb, ov in trio_raw:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)

    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    # Base モデル
    df_b = df.rename(columns={"pred_all": "pred_prob"}).copy()
    rec_b = compute_roi_records(df_b, trio_map, actual_trio, n_entries_map, grade_map_rk)

    # 専用モデル（S→S-model, A→A-model）
    df_g = df.rename(columns={"pred_grade": "pred_prob"}).copy()
    rec_g = compute_roi_records(df_g, trio_map, actual_trio, n_entries_map, grade_map_rk)

    rec_b_df = pd.DataFrame(rec_b)
    rec_g_df = pd.DataFrame(rec_g)

    print(f"\n  全体比較:")
    print(f"  {'期間':<8} {'Base ROI':>10} {'専用 ROI':>10} {'差分':>8}  n(base)/n(grade)")
    print("  " + "-" * 55)
    for period in ["TRAIN", "VAL", "HOLD"]:
        b = rec_b_df[rec_b_df["period"] == period]
        g = rec_g_df[rec_g_df["period"] == period]
        roi_b = b["pay"].sum() / b["cost"].sum() * 100 if b["cost"].sum() > 0 else float("nan")
        roi_g = g["pay"].sum() / g["cost"].sum() * 100 if g["cost"].sum() > 0 else float("nan")
        mk_b = "★" if roi_b >= 100 else " "
        mk_g = "★" if roi_g >= 100 else " "
        print(f"  {period:<8} {roi_b:>9.1f}%{mk_b}  {roi_g:>9.1f}%{mk_g}  {roi_g-roi_b:>+7.1f}pp  {len(b):>5}/{len(g)}")

    for grade in ["S級", "A級"]:
        g_key = "S" if grade == "S級" else "A"
        print(f"\n  {grade} 内訳（専用モデル）:")
        print(f"  {'期間':<8} {'専用 ROI':>10}  n")
        print("  " + "-" * 25)
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec_g_df[(rec_g_df["period"] == period) & (rec_g_df["grade"] == grade)]
            if len(sub) == 0:
                continue
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if sub["cost"].sum() > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            print(f"  {period:<8} {roi:>9.1f}%{mk}  {len(sub)}")

    # ─── 特徴量重要度（グレード別） ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("特徴量重要度 比較（Base vs S-model vs A-model）")
    print("=" * 70)
    imp_all = pd.Series(m_all.feature_importances_, index=FEATURE_COLS_WT)
    imp_s   = pd.Series(m_s.feature_importances_,   index=FEATURE_COLS_WT)
    imp_a   = pd.Series(m_a.feature_importances_,   index=FEATURE_COLS_WT)

    top_feats = imp_all.sort_values(ascending=False).head(15).index
    print(f"\n  {'特徴量':<28} {'Base':>8} {'S-model':>9} {'A-model':>9}")
    print("  " + "-" * 56)
    for feat in top_feats:
        v_all = imp_all[feat] / imp_all.sum() * 100
        v_s   = imp_s[feat]   / imp_s.sum()   * 100
        v_a   = imp_a[feat]   / imp_a.sum()   * 100
        print(f"  {feat:<28} {v_all:>7.1f}%  {v_s:>8.1f}%  {v_a:>8.1f}%")


if __name__ == "__main__":
    main()
