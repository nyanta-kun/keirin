"""same-meet form（同開催の前日までの実着順）特徴の追加検証（2026-07-15）。

大津昌広ファクターの忠実版: day_index(日番号)単体は無効(exp_day_index_wt)だったが、
本来の主張は「前日の"実際の"調子」。同一開催(cup_id)内で、その選手の
**strictly 前日までの着順実績**を point-in-time に集計して特徴化する。

追加特徴(同cup・より小さいday_indexの行のみで集計＝未来漏洩なし):
  sm_n_prev       … 今節の既走レース数
  sm_prev_top3    … 今節前日までの3着内率
  sm_prev_best    … 今節前日までの最高着順(小さいほど良・未走は8)
  sm_prev_win     … 今節前日までに1着があったか(勝ち上がり proxy)

baseline vs +samemeet を複数seed・クリーンOOSで比較(指標は exp_day_index_wt と同一)。
本番 FEATURE_COLS_WT / lgbm_wt.pkl は変更しない。
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

TR_TO = "2026-03-31"
TE_FROM, TE_TO = "2026-04-01", "2026-06-30"
FW_FROM, FW_TO = "2026-07-01", "2026-07-10"
SEEDS = [42, 7, 123, 2024, 99]
PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)
SM_COLS = ["sm_n_prev", "sm_prev_top3", "sm_prev_best", "sm_prev_win"]


def load_meta():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, cup_id, day_index, n_entries FROM wt_races").fetchall()
    return pd.DataFrame(rows, columns=["race_key", "cup_id", "day_index", "n_entries"])


def add_samemeet(df):
    """同cup・strictly前日(day_index小)の着順を point-in-time 集計。"""
    d = df[["race_key", "cup_id", "day_index", "player_id", "finish_order"]].copy()
    d["fin"] = pd.to_numeric(d["finish_order"], errors="coerce")
    d["is_top3"] = ((d["fin"] >= 1) & (d["fin"] <= 3)).astype(float)
    d["is_win"] = (d["fin"] == 1).astype(float)
    d["fin_valid"] = d["fin"].where(d["fin"] >= 1, np.nan)
    d = d.sort_values(["cup_id", "player_id", "day_index"])
    g = d.groupby(["cup_id", "player_id"], sort=False)
    # shift(1) 以前のみ集計 = 当該レースを含めない expanding
    d["sm_n_prev"] = g.cumcount()
    d["sm_prev_top3"] = g["is_top3"].apply(lambda s: s.shift(1).expanding().mean()).values
    d["sm_prev_win"] = g["is_win"].apply(lambda s: s.shift(1).expanding().max()).values
    d["sm_prev_best"] = g["fin_valid"].apply(lambda s: s.shift(1).expanding().min()).values
    d["sm_prev_top3"] = d["sm_prev_top3"].fillna(0.0)
    d["sm_prev_win"] = d["sm_prev_win"].fillna(0.0)
    d["sm_prev_best"] = d["sm_prev_best"].fillna(8.0).clip(1, 9)
    key = ["race_key", "player_id"]
    return df.merge(d[key + SM_COLS], on=key, how="left")


def race_metrics(df_ev):
    both_top3, win1, top3_1, n = 0, 0, 0, 0
    for _, g in df_ev.groupby("race_key"):
        g = g[g["finish_order"] >= 1]
        if len(g) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        fo = g["finish_order"].astype(float).tolist()
        n += 1
        win1 += fo[0] == 1
        top3_1 += 1 <= fo[0] <= 3
        both_top3 += (1 <= fo[0] <= 3) and (1 <= fo[1] <= 3)
    if n == 0:
        return dict(ss=0.0, win1=0.0, top3_1=0.0)
    return dict(ss=both_top3 / n, win1=win1 / n, top3_1=top3_1 / n)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=FW_TO)
    meta = load_meta()
    raw = raw.merge(meta, on="race_key", how="left")
    df = build_features_wt(raw)
    df = add_samemeet(df)
    df = df[df["finish_order"] >= 1].copy()
    df7 = df[df["n_entries"] == 7].copy()

    tr = df[df["race_date"] <= TR_TO].copy()
    te = df7[(df7["race_date"] >= TE_FROM) & (df7["race_date"] <= TE_TO)].copy()
    fw = df7[(df7["race_date"] >= FW_FROM) & (df7["race_date"] <= FW_TO)].copy()
    print(f"TRAIN {tr['race_key'].nunique()}R / TEST(7車) {te['race_key'].nunique()}R / "
          f"FWD(7車) {fw['race_key'].nunique()}R")
    print(f"sm_prev_top3 mean(TRAIN, n_prev>0)="
          f"{tr[tr['sm_n_prev']>0]['sm_prev_top3'].mean():.3f} / "
          f"既走率={ (tr['sm_n_prev']>0).mean():.2f}")

    variants = {"baseline": list(FEATURE_COLS_WT),
                "+samemeet": list(FEATURE_COLS_WT) + SM_COLS}
    agg = {v: {f"{m}_{t}": [] for m in ("auc", "ss", "win1", "top3") for t in ("te", "fw")}
           for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(tr[cols].fillna(0).values, tr[TARGET_COL_WT].values)
            for tag, ev in (("te", te), ("fw", fw)):
                ev = ev.copy()
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                agg[vname][f"auc_{tag}"].append(roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"]))
                mt = race_metrics(ev)
                agg[vname][f"ss_{tag}"].append(mt["ss"])
                agg[vname][f"win1_{tag}"].append(mt["win1"])
                agg[vname][f"top3_{tag}"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    ms = lambda a: (np.mean(a), np.std(a))
    print("\n============= 結果（seed平均 ± std, n=%d）=============" % len(SEEDS))
    for tag, label in (("te", "TEST 2026-04〜06 (クリーンOOS)"), ("fw", "FWD 2026-07")):
        print(f"\n--- {label} ---")
        print(f"{'variant':<14}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>13}{'1位複勝率':>13}")
        base = agg["baseline"]
        for v in variants:
            a = agg[v]
            am, asd = ms(a[f"auc_{tag}"]); sm, ssd = ms(a[f"ss_{tag}"])
            wm, wsd = ms(a[f"win1_{tag}"]); tm, tsd = ms(a[f"top3_{tag}"])
            dss = sm - np.mean(base[f"ss_{tag}"])
            mk = "" if v == "baseline" else ("  ★" if dss > ssd else ("  ×" if dss < -ssd else "  ~"))
            print(f"{v:<14}{am:>7.4f}±{asd:.4f}{sm:>13.1%}±{ssd:.1%}{wm:>7.1%}±{wsd:.1%}{tm:>7.1%}±{tsd:.1%}{mk}")
    print("\n判定: SSΔ>seed std で ★(採用候補)。採用は TEST・FWD 双方で非悪化かつ TEST で ★。")


if __name__ == "__main__":
    main()
