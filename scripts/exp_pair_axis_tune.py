"""ペア軸モデルの詰め: 学習設定と定式化の小掃引（2026-08-07）。

v2 は best_iter=99（lr=0.05）で早期に飽和した。p3_prod 以外の情報が本当に薄いのか、
それとも学習設定・検証分割の都合なのかを切り分ける。

  A. 検証分割: 時系列末尾2ヶ月 vs 学習窓内ランダム10%
  B. 学習率・容量の掃引
  C. 定式化: 生ラベル vs 「p3積で説明できる分を差し引いた残差」を学習
  D. 学習窓を延ばす（2024-07〜2025-06 の12ヶ月しかない制約の確認）

評価は常に 2025-07-01〜 の二軸的中率（＝三連複2軸総流しの的中率）。
⚠️ 読み取りのみ・オッズ不使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

CACHE = REPO / "data" / "exp_cache" / "pair_axis_dataset_v2.pkl"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
CATS = ["style_hi", "style_lo", "style_pair", "style_pair_same",
        "cls_hi", "cls_lo", "grade_c", "rtype_c"]


def acc(te: pd.DataFrame, s: np.ndarray) -> float:
    t = te.assign(_s=s)
    return 100 * t.loc[t.groupby("rk")._s.idxmax()].label.mean()


def main() -> None:
    df = pd.read_pickle(CACHE)
    for c in CATS:
        df[c] = df[c].astype("category")
    feats = [c for c in df.columns if c not in {"rk", "date", "hi", "lo", "label"}]
    trall = df[df.date <= TRAIN_END]
    te = df[df.date >= TEST_START]
    print(f"学習 {len(trall):,} / 評価 {len(te):,} / 特徴量 {len(feats)}")
    print("基準: 現行3ヘッド軸 53.48% / p3積 最大 "
          f"{acc(te, te.p3_prod.values):.2f}%\n")

    rng = np.random.default_rng(7)
    keys = trall.rk.unique()
    va_keys = set(rng.choice(keys, size=int(len(keys) * 0.12), replace=False))
    splits = {
        "時系列末尾2ヶ月": (trall[trall.date < "2025-05-01"],
                            trall[trall.date >= "2025-05-01"]),
        "ランダム12%(レース単位)": (trall[~trall.rk.isin(va_keys)],
                                    trall[trall.rk.isin(va_keys)]),
    }

    grid = [dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100),
            dict(learning_rate=0.02, num_leaves=255, min_data_in_leaf=50),
            dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=300),
            dict(learning_rate=0.01, num_leaves=127, min_data_in_leaf=200)]

    print(f"{'検証分割':22s} {'lr':>5s} {'leaves':>7s} {'minleaf':>8s} "
          f"{'iter':>5s} {'二軸的中%':>9s}")
    best = (0.0, None)
    for sname, (tr, va) in splits.items():
        for g in grid:
            p = dict(objective="binary", feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, verbose=-1, seed=42, **g)
            m = lgb.train(
                p, lgb.Dataset(tr[feats], tr.label, categorical_feature=CATS),
                num_boost_round=6000,
                valid_sets=[lgb.Dataset(va[feats], va.label, categorical_feature=CATS)],
                callbacks=[lgb.early_stopping(200, verbose=False)])
            a = acc(te, m.predict(te[feats]))
            print(f"{sname:22s} {g['learning_rate']:5.3f} {g['num_leaves']:7d} "
                  f"{g['min_data_in_leaf']:8d} {m.best_iteration:5d} {a:9.2f}")
            if a > best[0]:
                best = (a, (sname, g, m.best_iteration))

    print(f"\n最良: {best[0]:.2f}%  {best[1]}")

    # --- C. 残差定式化: p3積のロジットを init_score に置き、上乗せ分だけ学習する ---
    tr = trall[trall.date < "2025-05-01"]
    va = trall[trall.date >= "2025-05-01"]

    def logit(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))

    # p3積を 3着内2車同時の素朴な確率とみなし較正してから init_score にする
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(max_iter=1000).fit(
        logit(tr.p3_prod.values).reshape(-1, 1), tr.label.values)

    def init_of(d):
        return cal.decision_function(logit(d.p3_prod.values).reshape(-1, 1))

    dtr = lgb.Dataset(tr[feats], tr.label, categorical_feature=CATS,
                      init_score=init_of(tr))
    dva = lgb.Dataset(va[feats], va.label, categorical_feature=CATS,
                      init_score=init_of(va))
    m = lgb.train(dict(objective="binary", learning_rate=0.02, num_leaves=127,
                       min_data_in_leaf=200, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42),
                  dtr, num_boost_round=6000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    a = acc(te, init_of(te) + m.predict(te[feats], raw_score=True))
    print(f"C. 残差学習（p3積を init_score）: {a:.2f}%  (iter={m.best_iteration})")

    # 参考: 較正済み p3積 単独
    print(f"   参考 較正済みp3積 単独        : {acc(te, init_of(te)):.2f}%")


if __name__ == "__main__":
    main()
