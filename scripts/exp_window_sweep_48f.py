"""学習開始日を複数点でスイープし、sb_dyn特徴の効果と学習窓の関係を精査する。

exp_window_ab_48f.py の結果（2022-12開始でもΔAUC+0.0004・2024-04開始で+0.013）は
「0埋め希釈」仮説（S/Bラベルが2024-01以前に存在しないため）を反証した
（バックフィル完了後の2022-12開始でも改善しなかったため）。
真の要因（データ量 vs 直近性/concept drift）を切り分けるため、学習開始日を
2022-12/2023-07/2024-01/2024-04/2024-10 の5点でスイープする。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.database import get_connection
from exp_sb_dyn_ab import SB_COLS, race_metrics

TEST_FROM, TEST_TO = "2026-04-13", "2026-07-15"
SEEDS = [42, 101, 202, 303, 404]
COLS44 = [c for c in FEATURE_COLS_WT if c not in SB_COLS]

TRAIN_FROMS = ["2022-12-01", "2023-07-01", "2024-01-01", "2024-04-01", "2024-10-01"]


def main() -> None:
    print("データ読み込み（2022-12〜） ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date="2022-12-01", max_date=TEST_TO))
    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (TEST_FROM, TEST_TO)))
    test = df[(df["race_date"] >= TEST_FROM) & (df["race_date"] <= TEST_TO)]

    from sklearn.metrics import roc_auc_score
    results = []
    for train_from in TRAIN_FROMS:
        train = df[(df["race_date"] >= train_from) & (df["race_date"] < TEST_FROM)]
        row = {"train_from": train_from, "n_rows": len(train)}
        for arm, cols in (("44f", COLS44), ("48f", list(FEATURE_COLS_WT))):
            aucs, wins, top3s = [], [], []
            n = 0
            for seed in SEEDS:
                m = lgb.LGBMClassifier(
                    objective="binary", n_estimators=500, learning_rate=0.05,
                    num_leaves=31, min_child_samples=20, subsample=0.8,
                    colsample_bytree=0.8, random_state=seed,
                    deterministic=True, force_row_wise=True, verbose=-1)
                m.fit(train[cols], train[TARGET_COL_WT])
                p = m.predict_proba(test[cols])[:, 1]
                aucs.append(roc_auc_score(test[TARGET_COL_WT], p))
                w, t3, n = race_metrics(test, p, ne_map)
                wins.append(w)
                top3s.append(t3)
            row[f"{arm}_auc"] = np.mean(aucs)
            row[f"{arm}_win"] = np.mean(wins) * 100
            row[f"{arm}_top3"] = np.mean(top3s) * 100
            print(f"  [{train_from}] {arm} ({len(cols)}特徴・train {len(train):,}行): "
                  f"AUC={np.mean(aucs):.5f} 勝率={np.mean(wins)*100:.2f}% "
                  f"3着内={np.mean(top3s)*100:.2f}%", flush=True)
        row["d_auc"] = row["48f_auc"] - row["44f_auc"]
        row["d_top3"] = row["48f_top3"] - row["44f_top3"]
        results.append(row)
        print(f"  → Δ: AUC{row['d_auc']:+.5f} 3着内{row['d_top3']:+.2f}pt\n", flush=True)

    print("\n===== まとめ =====")
    print(f"{'開始日':<12} {'行数':>9} {'44fAUC':>8} {'48fAUC':>8} {'ΔAUC':>8} {'Δ3着内':>8}")
    for r in results:
        print(f"{r['train_from']:<12} {r['n_rows']:>9,} {r['44f_auc']:>8.5f} "
              f"{r['48f_auc']:>8.5f} {r['d_auc']:>+8.5f} {r['d_top3']:>+7.2f}pt")


if __name__ == "__main__":
    main()
