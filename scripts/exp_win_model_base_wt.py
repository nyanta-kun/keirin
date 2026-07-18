"""Phase B: 1着率モデルの基礎分析（正規プロトコル・検証期間のみ）。

3着内モデル(lgbm_wt・target=top3_flag)とは別に、1着率モデル
（target=finish_order==1）を学習し、以下を検証する:
  1. 1着モデル単体のAUC・7車レースでの指数1位の勝率
  2. 3着内モデルの1位と1着モデルの1位が一致する率、および一致/不一致それぞれの
     実際の勝率・3着内率（＝複合推奨の軸選定に使えるかの基礎データ）
  3. 「3着内モデルで上位だが1着モデルで下位」（差される本命候補）の検出力

学習は2025-03-31まで（正規プロトコル・学習=〜2025-03-31）。
検証は2025-04-01〜2026-03-31（分析・条件選定はここだけ）。
テストデータ（2026-04-01〜）には一切触れない。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt, prepare_X,
)
from src.database import get_connection

TRAIN_FROM = "2022-12-01"
TRAIN_TO = "2025-03-31"
VAL_FROM, VAL_TO = "2025-04-01", "2026-03-31"
SEED = 42


def main():
    print("データ読み込み（学習〜検証期間）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=VAL_TO))
    df["win_flag"] = (df["finish_order"] == 1).astype(int)
    df["top3_flag_"] = df["finish_order"].between(1, 3).astype(int)

    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] <= TRAIN_TO)
               & df["finish_order"].notna()]
    val = df[(df["race_date"] >= VAL_FROM) & (df["race_date"] <= VAL_TO)]
    print(f"train {len(train):,}行 / val {len(val):,}行", flush=True)

    X_tr = prepare_X(train)
    X_val = prepare_X(val)

    print("\n=== 1着モデル学習 ===", flush=True)
    win_model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)
    win_model.fit(X_tr, train["win_flag"])

    print("=== 3着内モデル学習（比較用・同一特徴・同一seed） ===", flush=True)
    top3_model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)
    top3_model.fit(X_tr, train["top3_flag_"])

    val = val.copy()
    val["p_win"] = win_model.predict_proba(X_val)[:, 1]
    val["p_top3"] = top3_model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import roc_auc_score
    val_labeled = val[val["finish_order"].notna()]
    auc_win = roc_auc_score(val_labeled["win_flag"], val_labeled["p_win"])
    auc_top3 = roc_auc_score(val_labeled["top3_flag_"], val_labeled["p_top3"])
    print(f"\n検証期間 AUC: 1着モデル={auc_win:.5f}  3着内モデル={auc_top3:.5f}", flush=True)

    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (VAL_FROM, VAL_TO)))

    # 7車レースのみで分析
    rows = []
    for rk, g in val.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        fo = g["finish_order"]
        if not (fo.notna() & (fo >= 1)).sum() >= 3:
            continue
        top3_top = g.loc[g["p_top3"].idxmax()]
        win_top = g.loc[g["p_win"].idxmax()]
        agree = int(top3_top["frame_no"]) == int(win_top["frame_no"])
        rows.append({
            "race_key": rk, "agree": agree,
            "top3_axis_frame": int(top3_top["frame_no"]),
            "win_axis_frame": int(win_top["frame_no"]),
            "top3_axis_fo": top3_top["finish_order"],
            "win_axis_fo": win_top["finish_order"],
            "top3_axis_p_win": top3_top["p_win"],
            "win_axis_p_top3": win_top["p_top3"],
        })
    rdf = pd.DataFrame(rows)
    print(f"\n7車完全確定レース数: {len(rdf)}", flush=True)

    print("\n=== 一致率 ===")
    agree_rate = rdf["agree"].mean()
    print(f"3着内1位 == 1着モデル1位 の一致率: {agree_rate*100:.1f}% (n={len(rdf)})")

    print("\n=== 軸（3着内モデル1位）の成績: 一致 vs 不一致 ===")
    for label, sub in (("一致", rdf[rdf["agree"] == 1]), ("不一致", rdf[rdf["agree"] == 0])):
        if len(sub) == 0:
            continue
        win_rate = (sub["top3_axis_fo"] == 1).mean()
        top3_rate = sub["top3_axis_fo"].between(1, 3).mean()
        print(f"  [{label}] n={len(sub)}  軸1着率={win_rate*100:.1f}%  軸3着内率={top3_rate*100:.1f}%")

    print("\n=== 1着モデル1位（=win_axis）の成績: 一致 vs 不一致 ===")
    for label, sub in (("一致", rdf[rdf["agree"] == 1]), ("不一致", rdf[rdf["agree"] == 0])):
        if len(sub) == 0:
            continue
        win_rate = (sub["win_axis_fo"] == 1).mean()
        top3_rate = sub["win_axis_fo"].between(1, 3).mean()
        print(f"  [{label}] n={len(sub)}  軸1着率={win_rate*100:.1f}%  軸3着内率={top3_rate*100:.1f}%")

    # 「差される本命」検出: 3着内モデル1位だが1着モデルでは順位が低い馬
    print("\n=== 「差される本命」候補: 3着内1位 かつ 1着モデル内の相対順位が低い ===")
    val_ranked = val.copy()
    val_ranked["win_rank_in_race"] = val_ranked.groupby("race_key")["p_win"].rank(
        ascending=False, method="first")
    val7 = val_ranked[val_ranked["race_key"].isin(rdf["race_key"])]
    # 3着内モデル1位の行だけ抽出
    top3_axis_rows = val7.loc[val7.groupby("race_key")["p_top3"].idxmax()]
    for wr_min in (2, 3, 4):
        sub = top3_axis_rows[top3_axis_rows["win_rank_in_race"] >= wr_min]
        if len(sub) < 20:
            continue
        win_rate = (sub["finish_order"] == 1).mean()
        top3_rate = sub["finish_order"].between(1, 3).mean()
        print(f"  win_rank_in_race>={wr_min}: n={len(sub)}  1着率={win_rate*100:.1f}%  "
              f"3着内率={top3_rate*100:.1f}%（全体は上記「一致」行参照）")

    # モデル保存（後続スクリプトで再利用）
    import pickle
    out_dir = REPO / "data" / "models"
    with open(out_dir / "lgbm_wt_winmodel_val.pkl", "wb") as f:
        pickle.dump(win_model, f)
    print(f"\n[保存] {out_dir / 'lgbm_wt_winmodel_val.pkl'}（学習〜2025-03-31・検証用）")


if __name__ == "__main__":
    main()
