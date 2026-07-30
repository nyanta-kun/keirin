"""race_type（レース種目: 予選/準決勝/決勝/特選/チャレンジ系/ガールズ系...）特徴の追加検証
（netkeirin未活用データ調査の一環・2026-07-28）。

netkeirinのデータ分析タブ「レース種目別」（選手×レース種目の勝率）に相当する軸だが、
`wt_races.race_type` は生カラムとして既にDBに存在するにもかかわらず FEATURE_COLS_WT の
どこにも使われていないことが判明した（スクレイピング不要・生カラムをエンコードするだけ）。
値は「予選/一般/準決勝/特選/決勝/チャレンジ*/ガールズ*/...」など100種以上の自由記述に近い
文字列（イベント名や表記ゆれを含む）で、綺麗な段階順序を人手で作るのはノイズが多いため、
まずは frequency-based な単純ラベルエンコード（pd.factorize、学習データ全体で固定した
category→id辞書）で試す。木モデルなのでオーディナリティは問わない。

検証は既存の exp_day_index_wt.py と同一ハーネス・分割・指標。
baseline / +race_type_id を複数seed・クリーンOOSで比較。
本番 FEATURE_COLS_WT / lgbm_wt.pkl は変更しない。

クリーン分割: TRAIN 2022-12-01〜2026-03-31 / TEST(未使用OOS) 2026-04-01〜2026-06-30 /
FWD 2026-07-01〜2026-07-10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)

TR_FROM, TR_TO = "2022-12-01", "2026-03-31"
TE_FROM, TE_TO = "2026-04-01", "2026-06-30"
FW_FROM, FW_TO = "2026-07-01", "2026-07-10"
SEEDS = [42, 7, 123, 2024, 99]

PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)


def load_race_type():
    # 注意: PG接続時 get_connection().execute().fetchall() は sqlite3.Row 相当の
    # _PgRow を返し、__iter__ を持たないため pd.DataFrame(rows, columns=[...]) の
    # 位置アンパックが壊れる（"Shape of passed values is (n,1)..."エラー）。
    # dict風アクセスで明示的にタプル化する（exp_day_index_wt.py はSQLite時代に
    # 書かれたコードのままなので、PG運用下で再実行すると同じ理由で壊れる）。
    with get_connection() as c:
        rows = c.execute("SELECT race_key, race_type FROM wt_races").fetchall()
        rows = [(r["race_key"], r["race_type"]) for r in rows]
    return pd.DataFrame(rows, columns=["race_key", "race_type"])


def race_metrics(df_ev):
    both_top3, win1, top3_1, n = 0, 0, 0, 0
    for _, g in df_ev.groupby("race_key"):
        g = g[g["finish_order"] >= 1]
        if g["race_key"].isna().any() or len(g) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        fo = g["finish_order"].astype(float).tolist()
        n += 1
        if fo[0] == 1:
            win1 += 1
        if 1 <= fo[0] <= 3:
            top3_1 += 1
        if (1 <= fo[0] <= 3) and (1 <= fo[1] <= 3):
            both_top3 += 1
    if n == 0:
        return dict(n=0, ss=0.0, win1=0.0, top3_1=0.0)
    return dict(n=n, ss=both_top3 / n, win1=win1 / n, top3_1=top3_1 / n)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date=TR_FROM, max_date=FW_TO)
    rt = load_race_type()
    raw = raw.merge(rt, on="race_key", how="left")
    df = build_features_wt(raw)

    # frequency-based ラベルエンコード（データ全体で固定した辞書。ターゲット非依存なのでリークなし）
    df["race_type"] = df["race_type"].fillna("(不明)")
    freq = df["race_type"].value_counts()
    df["race_type_id"] = df["race_type"].map({v: i for i, v in enumerate(freq.index)}).astype(int)
    print(f"race_type ユニーク数: {df['race_type'].nunique()}")
    df = df[df["finish_order"] >= 1].copy()

    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)
    df7 = df[df["_ne"] == 7].copy()

    tr = df[df["race_date"] <= TR_TO].copy()
    te = df7[(df7["race_date"] >= TE_FROM) & (df7["race_date"] <= TE_TO)].copy()
    fw = df7[(df7["race_date"] >= FW_FROM) & (df7["race_date"] <= FW_TO)].copy()
    print(f"TRAIN {tr['race_key'].nunique()}R / TEST(7車) {te['race_key'].nunique()}R / "
          f"FWD(7車) {fw['race_key'].nunique()}R")

    variants = {
        "baseline": list(FEATURE_COLS_WT),
        "+race_type": list(FEATURE_COLS_WT) + ["race_type_id"],
    }

    agg = {v: {"auc_te": [], "auc_fw": [], "ss_te": [], "win1_te": [], "top3_te": [],
               "ss_fw": [], "win1_fw": [], "top3_fw": []} for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            Xtr = tr[cols].fillna(0).values
            ytr = tr[TARGET_COL_WT].values
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(Xtr, ytr)
            for tag, ev in (("te", te), ("fw", fw)):
                ev = ev.copy()
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                auc = roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"])
                mt = race_metrics(ev)
                agg[vname][f"auc_{tag}"].append(auc)
                agg[vname][f"ss_{tag}"].append(mt["ss"])
                agg[vname][f"win1_{tag}"].append(mt["win1"])
                agg[vname][f"top3_{tag}"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    def ms(a):
        return np.mean(a), np.std(a)

    print("\n================ 結果（seed平均 ± std, n_seeds=%d）================" % len(SEEDS))
    for tag, label in (("te", "TEST 2026-04〜06 (クリーンOOS)"), ("fw", "FWD 2026-07")):
        print(f"\n--- {label} ---")
        print(f"{'variant':<14}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>14}{'1位複勝率':>14}")
        base = agg["baseline"]
        for v in variants:
            a = agg[v]
            auc_m, auc_s = ms(a[f"auc_{tag}"])
            ss_m, ss_s = ms(a[f"ss_{tag}"])
            w_m, w_s = ms(a[f"win1_{tag}"])
            t_m, t_s = ms(a[f"top3_{tag}"])
            dss = ss_m - np.mean(base[f"ss_{tag}"])
            mark = ""
            if v != "baseline":
                mark = "  ★" if dss > ss_s else ("  ×" if dss < -ss_s else "  ~")
            print(f"{v:<14}{auc_m:>7.4f}±{auc_s:.4f}{ss_m:>13.1%}±{ss_s:.1%}"
                  f"{w_m:>8.1%}±{w_s:.1%}{t_m:>8.1%}±{t_s:.1%}{mark}")
    print("\n判定: SSΔが seed std を超えれば ★(採用候補) / 範囲内 ~ / 悪化 ×。")
    print("採用は TEST・FWD 双方で非悪化かつ TEST で ★ が条件。")


if __name__ == "__main__":
    main()
