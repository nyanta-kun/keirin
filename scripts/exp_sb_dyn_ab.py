"""レース単位S/B取得・上がりタイム由来の4特徴のA/B検証。

仮説: 「誰がレースを支配するか（B取り）」「脚の直接測定値（上がり）」は
集計値（s_count/b_count・勝率系）に平滑化されない選手状態の情報を持つ。
res_standing/res_back/final_half（2024-01〜バックフィル済み）から
point-in-time ローリング特徴を作り 44特徴に追加して A/B する。

新特徴（全て過去レースのみ・closed=left 90D・レース内相対化済み）:
  b_rate_90       : 直近90日の B取り率
  s_rate_90       : 直近90日の S取り率
  fh_rel_90       : 直近90日の上がり相対値平均（レース内中央値との差・負=速い）
  fh_best_rate_90 : 直近90日の「レース内上がり最速」率

検証: exp_rp_trend_ab.py と同一方法論。
  窓1: 学習 2024-04-01〜2026-04-12 / テスト 2026-04-13〜2026-07-15
  窓2: 学習 2024-04-01〜2025-12-31 / テスト 2026-01-01〜2026-04-12
  （学習開始はラベル充足後の 2024-04。5 seeds × {44特徴, +4特徴}・
   deterministic=True・AUC / 指数1位の勝率・3着内率で比較）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.database import get_connection

TRAIN_FROM = "2024-04-01"  # S/Bラベル(2024-01〜)の90D窓が充足する時点
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
}
SEEDS = [42, 101, 202, 303, 404]

SB_COLS = ["b_rate_90", "s_rate_90", "fh_rel_90", "fh_best_rate_90"]


def add_sb_dyn(df: pd.DataFrame) -> pd.DataFrame:
    """S/B取得・上がり由来の point-in-time ローリング特徴を付与する。"""
    with get_connection() as conn:
        H = pd.read_sql_query(
            "SELECT e.race_key, e.player_id, e.res_standing, e.res_back, "
            "e.final_half, r.race_date "
            "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key "
            "WHERE e.res_back IS NOT NULL AND e.finish_order >= 1", conn)
    H["_dt"] = pd.to_datetime(H["race_date"])

    # レース内相対化: fh_rel = 自上がり − レース中央値（負=速い）・fh_best = レース内最速
    fh = H["final_half"].where(H["final_half"] > 0)
    H["_fh"] = fh
    med = H.groupby("race_key")["_fh"].transform("median")
    mn = H.groupby("race_key")["_fh"].transform("min")
    H["fh_rel"] = H["_fh"] - med
    H["fh_best"] = (H["_fh"] == mn).astype(float).where(H["_fh"].notna())

    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)

    def _rm(col):
        return (H.set_index("_dt").groupby("player_id")[col]
                .rolling("90D", closed="left").mean()
                .reset_index(level=0, drop=True).values)

    H["b_rate_90"] = _rm("res_back")
    H["s_rate_90"] = _rm("res_standing")
    H["fh_rel_90"] = _rm("fh_rel")
    H["fh_best_rate_90"] = _rm("fh_best")

    key = H[["race_key", "player_id"] + SB_COLS]
    out = df.merge(key, on=["race_key", "player_id"], how="left")
    # 履歴不足は 0.0（本番 prepare_X の NaN→0 統一と同一表現）
    for c in SB_COLS:
        out[c] = out[c].fillna(0.0)
    return out


def race_metrics(test: pd.DataFrame, prob: np.ndarray, ne_map: dict) -> tuple[float, float, int]:
    """7車レースの指数1位の勝率・3着内率。"""
    t = test.copy()
    t["p"] = prob
    win = top3 = n = 0
    for rk, g in t.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        fo = g["finish_order"]
        if not (fo.notna() & (fo >= 1)).sum() >= 3:
            continue
        top = g.loc[g["p"].idxmax()]
        f = top["finish_order"]
        if f is None or not f == f:
            f = 99
        n += 1
        win += 1 if f == 1 else 0
        top3 += 1 if 1 <= f <= 3 else 0
    return (win / n if n else 0.0, top3 / n if n else 0.0, n)


def run_window(df: pd.DataFrame, test_from: str, test_to: str) -> None:
    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (test_from, test_to)))
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < test_from)]
    test = df[(df["race_date"] >= test_from) & (df["race_date"] <= test_to)]
    print(f"\n######## 窓 test={test_from}〜{test_to}  "
          f"train {len(train):,}行 / test {len(test):,}行 ########", flush=True)

    print("== 新特徴の分布（test） ==")
    print(test[SB_COLS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    q = pd.qcut(test["b_rate_90"], 5, duplicates="drop")
    print("b_rate_90 五分位別 top3率:")
    print(test.groupby(q, observed=True)[TARGET_COL_WT].mean().round(4).to_string())

    from sklearn.metrics import roc_auc_score
    results = {}
    n = 0
    for arm, cols in (("baseline", FEATURE_COLS_WT),
                      ("+sb_dyn", FEATURE_COLS_WT + SB_COLS)):
        aucs, wins, top3s = [], [], []
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
        results[arm] = (aucs, wins, top3s)
        print(f"== {arm} ({len(cols)}特徴) ==")
        print(f"  AUC      : {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
        print(f"  1位勝率  : {np.mean(wins)*100:.2f}% ± {np.std(wins)*100:.2f} (n={n})")
        print(f"  1位3着内 : {np.mean(top3s)*100:.2f}% ± {np.std(top3s)*100:.2f}")
        if arm == "+sb_dyn":
            imp = pd.Series(m.feature_importances_, index=cols)
            ranks = imp.rank(ascending=False).astype(int)
            print("  新特徴の重要度（最終seed・順位/全特徴中）:")
            for c in SB_COLS:
                print(f"    {c:<16} imp={imp[c]:4d}  順位 {ranks[c]}/{len(cols)}")

    b, a = results["baseline"], results["+sb_dyn"]
    print("== 差分（+sb_dyn − baseline） ==")
    print(f"  ΔAUC      : {np.mean(a[0])-np.mean(b[0]):+.5f} (seed std b={np.std(b[0]):.5f})")
    print(f"  Δ1位勝率  : {(np.mean(a[1])-np.mean(b[1]))*100:+.2f}pt")
    print(f"  Δ1位3着内 : {(np.mean(a[2])-np.mean(b[2]))*100:+.2f}pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2", help="実行する窓（カンマ区切り）")
    args = ap.parse_args()

    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    df = add_sb_dyn(df)

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt)


if __name__ == "__main__":
    main()
