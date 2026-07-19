"""S1再検討: 1着モデル(win model)のレース内順位別 実際の1着率/3着内率を計測する。

ユーザー質問: 「1着選出モデルにおけるレースの1位、2位は実際の1着率は何%か」に回答する
ための基礎データ。正規プロトコル（学習〜2025-03-31・検証2025-04-01〜2026-03-31・
テスト2026-04-01〜07-15）で、学習に使っていない期間で honest に計測する。

6車・7車それぞれで集計する（S1は歴史的に6車/7車の両方が検討対象だったため）。
"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

TRAIN_FROM, TRAIN_TO = "2022-12-01", "2025-03-31"
VAL_FROM, VAL_TO = "2025-04-01", "2026-03-31"
TEST_FROM, TEST_TO = "2026-04-01", "2026-07-15"
SEED = 42


def train_win_model():
    print("学習データ読み込み...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=TRAIN_TO))
    df = df[df["finish_order"].notna()]
    X = prepare_X(df)
    win_y = (df["finish_order"] == 1).astype(int)
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)
    print("1着モデル学習...", flush=True)
    m.fit(X, win_y)
    return m


def collect(tf, tt, model, n_riders):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_win"] = model.predict_proba(X)[:, 1]
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
    rows = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != n_riders or len(g) != n_riders:
            continue
        fo = g["finish_order"]
        if (fo.notna() & (fo >= 1)).sum() < 3:
            continue
        g_sorted = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        for rank_idx, fo_val in enumerate(g_sorted["finish_order"].tolist(), start=1):
            rows.append({"win_rank": rank_idx, "finish_order": fo_val})
    return pd.DataFrame(rows)


def summarize(label, rdf, n_riders):
    print(f"\n=== {label}（{n_riders}車・n_races={rdf['win_rank'].eq(1).sum():,}） ===")
    for rk in range(1, min(4, n_riders) + 1):
        sub = rdf[rdf["win_rank"] == rk]
        if len(sub) == 0:
            continue
        win_rate = (sub["finish_order"] == 1).mean()
        top2_rate = sub["finish_order"].between(1, 2).mean()
        top3_rate = sub["finish_order"].between(1, 3).mean()
        print(f"  win_rank={rk}: n={len(sub):,}  1着率={win_rate*100:.1f}%  "
              f"2着内率={top2_rate*100:.1f}%  3着内率={top3_rate*100:.1f}%")


def main():
    model = train_win_model()
    for n_riders in (6, 7):
        print(f"\n{'='*60}\n{n_riders}車レース\n{'='*60}", flush=True)
        val = collect(VAL_FROM, VAL_TO, model, n_riders)
        test = collect(TEST_FROM, TEST_TO, model, n_riders)
        summarize(f"検証期間 {VAL_FROM}〜{VAL_TO}", val, n_riders)
        summarize(f"テスト期間 {TEST_FROM}〜{TEST_TO}（純OOS）", test, n_riders)


if __name__ == "__main__":
    main()
