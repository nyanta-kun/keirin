"""day_index（開催日進行）特徴の追加検証（netkeirin 予想家ファクター取り込み・2026-07-15）。

発見(大津昌広 プロフィール): 「選手の近況や前日の調子。特に開催3日目・4日目を重視」。
競輪は3〜4日開催で、準決・決勝(3-4日目)は勝ち上がり後の疲労/上積みが着に効く。
day_index(初日=1/2日目/3日目/4日目) は wt_races に既にあるが FEATURE_COLS_WT に未採用。

検証: baseline(現特徴) vs 拡張(+day_index [+days_since]) を同一データ・同一分割で
複数seed学習し、クリーンOOSで
  ① holdout AUC
  ② 上位2車ともに3着内=SS的中率（二軸厳選ラボの核心指標）
  ③ 指数1位の勝率 / 複勝率
を seed平均±std で比較。改善が seed std を超えるものだけ採用候補とする。
**本番の FEATURE_COLS_WT / lgbm_wt.pkl は変更しない**（採用判断後に正式反映）。

クリーン分割（memory keirin-r-rank-race-gami に準拠）:
  TRAIN 2022-12-01〜2026-03-31 / TEST(未使用OOS) 2026-04-01〜2026-06-30 / FWD 2026-07-01〜2026-07-10
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


def load_day_index():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, day_index FROM wt_races").fetchall()
    return pd.DataFrame(rows, columns=["race_key", "day_index"])


def race_metrics(df_ev):
    """7車・完走レース単位で SS的中(上位2車とも3着内)/1位勝率/1位複勝率 を返す。"""
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
    di = load_day_index()
    raw = raw.merge(di, on="race_key", how="left")
    df = build_features_wt(raw)
    # 拡張特徴
    df["day_index"] = pd.to_numeric(df["day_index"], errors="coerce").fillna(1).clip(1, 4)
    df["is_day34"] = (df["day_index"] >= 3).astype(int)
    df = df[df["finish_order"] >= 1].copy()

    # 7車ちょうど限定（本番母集団）
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)
    df7 = df[df["_ne"] == 7].copy()

    tr = df[df["race_date"] <= TR_TO].copy()                       # 学習は全頭数（現行と同じ）
    te = df7[(df7["race_date"] >= TE_FROM) & (df7["race_date"] <= TE_TO)].copy()
    fw = df7[(df7["race_date"] >= FW_FROM) & (df7["race_date"] <= FW_TO)].copy()
    print(f"TRAIN {tr['race_key'].nunique()}R / TEST(7車) {te['race_key'].nunique()}R / "
          f"FWD(7車) {fw['race_key'].nunique()}R")
    print(f"day_index分布(TRAIN): {tr['day_index'].value_counts().sort_index().to_dict()}")

    variants = {
        "baseline": list(FEATURE_COLS_WT),
        "+day_index": list(FEATURE_COLS_WT) + ["day_index", "is_day34"],
        "+is_day34only": list(FEATURE_COLS_WT) + ["is_day34"],
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
        print(f"{'variant':<18}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>14}{'1位複勝率':>14}")
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
            print(f"{v:<18}{auc_m:>7.4f}±{auc_s:.4f}{ss_m:>13.1%}±{ss_s:.1%}"
                  f"{w_m:>8.1%}±{w_s:.1%}{t_m:>8.1%}±{t_s:.1%}{mark}")
    print("\n判定: SSΔが seed std を超えれば ★(採用候補) / 範囲内 ~ / 悪化 ×。")
    print("採用は TEST・FWD 双方で非悪化かつ TEST で ★ が条件。")


if __name__ == "__main__":
    main()
