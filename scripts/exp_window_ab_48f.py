"""学習窓×特徴セットの切り分け実験（sb_dyn 4特徴の希釈問題）。

観測: A/B（学習2024-04〜）では ΔAUC+0.013 だが、本番方式（学習2022-12〜）の
train-wt では holdout +0.0007 に縮んだ。仮説: 2024-01以前は S/B ラベルが存在せず
sb_dyn 特徴が全行 0 埋めのため、長窓学習で分割が汚染され信号が希釈される。

アーム（テスト=2026-04-13〜07-15・5seed・A/Bと同一LGB設定）:
  44f × 2022-12〜 : 旧本番相当
  48f × 2022-12〜 : 新本番相当（希釈あり）
  ※ 44f/48f × 2024-04〜 は exp_sb_dyn_ab 実測済み（45.71/78.87・46.22/79.80）
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


def main() -> None:
    print("データ読み込み（2022-12〜） ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date="2022-12-01", max_date=TEST_TO))
    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (TEST_FROM, TEST_TO)))
    test = df[(df["race_date"] >= TEST_FROM) & (df["race_date"] <= TEST_TO)]

    from sklearn.metrics import roc_auc_score
    for train_from in ("2022-12-01",):
        train = df[(df["race_date"] >= train_from) & (df["race_date"] < TEST_FROM)]
        for arm, cols in ((f"44f×{train_from}〜", COLS44),
                          (f"48f×{train_from}〜", list(FEATURE_COLS_WT))):
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
            print(f"== {arm} ({len(cols)}特徴・train {len(train):,}行) ==")
            print(f"  AUC      : {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
            print(f"  1位勝率  : {np.mean(wins)*100:.2f}% ± {np.std(wins)*100:.2f} (n={n})")
            print(f"  1位3着内 : {np.mean(top3s)*100:.2f}% ± {np.std(top3s)*100:.2f}")
    print("\n[参考・実測済み(exp_sb_dyn_ab w1)] 44f×2024-04〜: AUC0.76911 勝率45.71% 3着内78.87%")
    print("[参考・実測済み(exp_sb_dyn_ab w1)] 48f×2024-04〜: AUC0.78205 勝率46.22% 3着内79.80%")


if __name__ == "__main__":
    main()
