"""S级専用モデル ハイパーパラメータ最適化実験（doc39）

背景 (doc35-B):
  S级 × S-model: TRAIN 150%★ / VAL 107%★ / HOLD 88%（15R）
  HOLD 15R は小標本のため統計的結論不可。
  hyperparameter の変更で HOLD を 100% 超えにできるか検証する。

設計:
  S级専用モデル（S-class TRAIN のみ学習・S-class レースのみ評価）に対して
  4つのパラメータセットを比較する。

NOTE: HOLD 15R は標本が小さすぎるため、いずれの結果も統計的に確定できない。
     方向確認と「より安全なパラメータの特定」が目的。
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from exp_segment_first_wt import TRAIN, VAL, HOLD

GAMI_THRESHOLD = 5.0
GRADE_MAP = {"S級": "S", "SA混合": "S", "A級": "A", "L級": "L"}

PARAM_SETS = {
    "A: 現行(Base)": dict(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
    ),
    "B: 保守的": dict(
        objective="binary", n_estimators=400, learning_rate=0.03,
        num_leaves=15, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
    ),
    "C: 積極的": dict(
        objective="binary", n_estimators=800, learning_rate=0.03,
        num_leaves=63, min_child_samples=10, subsample=0.7,
        colsample_bytree=0.7, random_state=42, verbose=-1,
    ),
    "D: 軽量": dict(
        objective="binary", n_estimators=200, learning_rate=0.08,
        num_leaves=15, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
    ),
}


def period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def compute_roi_records(df, trio_map, actual_trio, n_entries_map):
    records = []
    for rk, grp in df.groupby("race_key"):
        if n_entries_map.get(rk, 99) > 6:
            continue
        period = period_of(str(grp["race_date"].iloc[0]))
        if period is None:
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
            "pay": pay, "cost": len(thirds) * 100,
        })
    return records


def main():
    print("S级専用モデル ハイパーパラメータ最適化実験（doc39）")
    print()

    print("データ準備中...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries, grade FROM wt_races", conn)
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()

    grade_map_rk = dict(zip(races_info["race_key"], races_info["grade"]))
    df["grade_group"] = df["race_key"].map(grade_map_rk).map(GRADE_MAP).fillna("A")
    n_entries_map = dict(zip(races_info["race_key"], races_info["n_entries"]))

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

    # TRAIN 分布確認
    train_fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    print(f"TRAIN 全体: {len(train_fit):,} rows")
    print(f"TRAIN S级: {(train_fit['grade_group'] == 'S').sum():,} rows")
    print()

    print("=" * 75)
    print(f"{'パラメータセット':<20}  {'TRAIN':>10}  {'VAL':>10}  {'HOLD':>10}  n(T/V/H)")
    print("=" * 75)

    for name, params in PARAM_SETS.items():
        # S级専用モデル
        s_fit = train_fit[train_fit["grade_group"] == "S"]
        m_s = lgb.LGBMClassifier(**params)
        m_s.fit(prepare_X(s_fit), s_fit["top3_flag"].values)

        # S级レースにS-model、A级にはbase model（全体）
        m_base = lgb.LGBMClassifier(**dict(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        ))
        m_base.fit(prepare_X(train_fit), train_fit["top3_flag"].values)

        pred_s = m_s.predict_proba(prepare_X(df))[:, 1]
        pred_b = m_base.predict_proba(prepare_X(df))[:, 1]

        df["pred_prob"] = np.where(df["grade_group"] == "S", pred_s, pred_b)

        # S级レースのみ評価
        df_s = df[df["grade_group"] == "S"].copy()
        rec = pd.DataFrame(compute_roi_records(df_s, trio_map, actual_trio, n_entries_map))

        roi_parts = []
        ns = []
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec[rec["period"] == period] if len(rec) else pd.DataFrame()
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            roi_parts.append(f"{roi:>9.1f}%{mk}")
            ns.append(str(len(sub)))

        print(f"  {name:<20}  {'  '.join(roi_parts)}  {'/'.join(ns)}")

    print()
    print("⚠ HOLD は 15R 程度（小標本）：結果は統計的ノイズの範囲内。")
    print("  VAL（38R）での一貫した改善があれば参考値として採用を検討。")


if __name__ == "__main__":
    main()
