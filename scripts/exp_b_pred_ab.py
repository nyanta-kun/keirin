"""B取り予測モデル（展開想定）のOOF予測を3着内モデルへ追加するA/B。

2段構成:
  1段目: res_back（このレースでBを取るか）を48特徴（44+sb_dyn4）で予測する
         LGBモデル。拡張窓OOF（四半期ブロックごとに「それ以前の全データ」で学習）
         により、全行の b_pred が point-in-time（未来情報なし）で得られる。
  2段目: b_pred（生値）と b_pred_share（レース内シェア=Σ正規化・展開の相対想定）を
         3着内モデルの特徴に追加して A/B。

ベースライン = 48特徴（exp_sb_dyn_ab の +sb_dyn アーム）。
検証: 5 seeds × 2独立窓（exp_sb_dyn_ab と同一）・deterministic=True。
"""
import argparse
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
from exp_sb_dyn_ab import SB_COLS, add_sb_dyn, race_metrics

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
}
SEEDS = [42, 101, 202, 303, 404]
BASE_COLS = FEATURE_COLS_WT + SB_COLS
BP_COLS = ["b_pred", "b_pred_share"]

# 拡張窓OOFブロック（各ブロックは「それ以前の全データ」で学習したモデルで予測）
OOF_BLOCKS = [
    ("2024-04-01", "2024-06-30"), ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"), ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-07-15"),
]


def add_b_pred(df: pd.DataFrame) -> pd.DataFrame:
    """拡張窓OOFで b_pred / b_pred_share を付与する。

    B取りラベル（res_back）は 2024-01〜。最初のブロック(2024-04〜)は
    2024-01〜03 学習となる。ラベルを DB から引き直す（df には res_back 列がない）。
    """
    with get_connection() as conn:
        lab = pd.read_sql_query(
            "SELECT e.race_key, e.player_id, e.res_back "
            "FROM wt_entries e WHERE e.res_back IS NOT NULL", conn)
    d = df.merge(lab, on=["race_key", "player_id"], how="left")
    d["b_pred"] = np.nan

    for bf, bt in OOF_BLOCKS:
        tr = d[(d["race_date"] < bf) & d["res_back"].notna()]
        bl = (d["race_date"] >= bf) & (d["race_date"] <= bt)
        if tr.empty or not bl.any():
            continue
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(tr[BASE_COLS], tr["res_back"].astype(int))
        d.loc[bl, "b_pred"] = m.predict_proba(d.loc[bl, BASE_COLS])[:, 1]
        # ブロック内のB予測品質（参考）: レース内b_pred1位が実際にBを取った率
        blk = d[bl & d["res_back"].notna()]
        hit = n = 0
        for rk, g in blk.groupby("race_key"):
            if g["res_back"].sum() != 1:
                continue
            n += 1
            hit += int(g.loc[g["b_pred"].idxmax(), "res_back"] == 1)
        print(f"  [b_pred] block {bf}〜{bt}: train {len(tr):,}行 "
              f"B的中率 {hit/n*100 if n else 0:.1f}% (n={n:,})", flush=True)

    d["b_pred_share"] = (d["b_pred"] /
                         d.groupby("race_key")["b_pred"].transform("sum"))
    for c in BP_COLS:
        d[c] = d[c].fillna(0.0)
    return d.drop(columns=["res_back"])


def run_window(df: pd.DataFrame, test_from: str, test_to: str) -> None:
    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (test_from, test_to)))
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < test_from)]
    test = df[(df["race_date"] >= test_from) & (df["race_date"] <= test_to)]
    print(f"\n######## 窓 test={test_from}〜{test_to}  "
          f"train {len(train):,}行 / test {len(test):,}行 ########", flush=True)

    from sklearn.metrics import roc_auc_score
    results = {}
    n = 0
    for arm, cols in (("base48", BASE_COLS), ("+b_pred", BASE_COLS + BP_COLS)):
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
        if arm == "+b_pred":
            imp = pd.Series(m.feature_importances_, index=cols)
            ranks = imp.rank(ascending=False).astype(int)
            for c in BP_COLS:
                print(f"    {c:<14} imp={imp[c]:4d}  順位 {ranks[c]}/{len(cols)}")

    b, a = results["base48"], results["+b_pred"]
    print("== 差分（+b_pred − base48） ==")
    print(f"  ΔAUC      : {np.mean(a[0])-np.mean(b[0]):+.5f} (seed std b={np.std(b[0]):.5f})")
    print(f"  Δ1位勝率  : {(np.mean(a[1])-np.mean(b[1]))*100:+.2f}pt")
    print(f"  Δ1位3着内 : {(np.mean(a[2])-np.mean(b[2]))*100:+.2f}pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2")
    args = ap.parse_args()

    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    # B取りモデルの初回ブロック（2024-04〜）の学習データとして 2024-01〜03 も読む
    # （3着内モデルの train は run_window 内で TRAIN_FROM 以降に絞る）
    df = build_features_wt(load_raw_data_wt(min_date="2024-01-01", max_date=max_to))
    df = add_sb_dyn(df)
    print("B取りOOF予測を構築 ...", flush=True)
    df = add_b_pred(df)

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt)


if __name__ == "__main__":
    main()
