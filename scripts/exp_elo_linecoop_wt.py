"""Elo レーティング + ライン連携実績 特徴の追加検証（2026-07-15）。

netkeirin 予想家ファクター調査の続き（day_index/same-meet は不採用 → memory
keirin-netkeirin-factor-import）。今回はスクレイプ不要で自前計算できる2候補:

① player Elo（Kドリームス「Kレーティング」相当）
   競走得点(4ヶ月平均)と違い「対戦相手の強さ」を加味し毎走更新される。
   point-in-time: 各レースの特徴は必ずそのレース"前"のレートで作る。
   ペア比較方式: レース内の全ペア (i beat j) について標準 Elo 更新。

② ライン連携実績（対戦成績の連携版）
   過去に同ライン(line_group)を組んだ相手との「回数」「両者3着内率」を
   point-in-time 累積し、当該レースのライン構成員との親密度として特徴化。

baseline / +elo / +linecoop / +both を複数seed・クリーンOOSで比較。
指標: SS的中(上位2車とも3着内)率・1位勝率/複勝率・AUC。
本番 FEATURE_COLS_WT / lgbm_wt.pkl は変更しない。
"""
import sys
from collections import defaultdict
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

ELO_K = 24.0
ELO_SCALE = 400.0
ELO_INIT = 1500.0

ELO_COLS = ["elo", "elo_rank", "elo_z"]
LC_COLS = ["lc_n_prev", "lc_rate", "lc_max_n"]


def compute_elo_and_linecoop(df):
    """レース時系列で point-in-time の Elo とライン連携特徴を付与する。

    df: build_features 前の生データ相当（race_key, race_date, start_at,
        player_id, line_group, finish_order を含む）。
    """
    d = df[["race_key", "race_date", "start_at", "player_id",
            "line_group", "finish_order"]].copy()
    d["fin"] = pd.to_numeric(d["finish_order"], errors="coerce")

    # レースを時系列順に（同日内は start_at → race_key で安定ソート）
    race_order = (d.groupby("race_key")
                  .agg(race_date=("race_date", "first"), start_at=("start_at", "first"))
                  .sort_values(["race_date", "start_at"])
                  .index.tolist())

    rating: dict = defaultdict(lambda: ELO_INIT)
    pair_n: dict = defaultdict(int)      # (pid_a,pid_b) → 同ライン回数
    pair_hit: dict = defaultdict(int)    # (pid_a,pid_b) → 両者3着内回数

    groups = {rk: g for rk, g in d.groupby("race_key", sort=False)}
    out_elo, out_lc = {}, {}

    for rk in race_order:
        g = groups[rk]
        pids = g["player_id"].tolist()
        fins = g["fin"].tolist()
        lines = g["line_group"].tolist()

        # --- 特徴（レース前の状態で） ---
        pre = {p: rating[p] for p in pids}
        for p, lg in zip(pids, lines):
            mates = [q for q, ql in zip(pids, lines)
                     if q != p and ql is not None and ql == lg]
            ns, rates = [], []
            for q in mates:
                key = (p, q) if p < q else (q, p)
                n = pair_n[key]
                if n > 0:
                    ns.append(n)
                    rates.append(pair_hit[key] / n)
            out_lc[(rk, p)] = (
                float(sum(ns)),
                float(np.average(rates, weights=ns)) if ns else 0.0,
                float(max(ns)) if ns else 0.0,
            )
            out_elo[(rk, p)] = pre[p]

        # --- 更新（レース後） ---
        # Elo: 完走者間の全ペア比較
        finished = [(p, f) for p, f in zip(pids, fins) if f is not None and f >= 1]
        delta = defaultdict(float)
        for i in range(len(finished)):
            for j in range(i + 1, len(finished)):
                pa, fa = finished[i]
                pb, fb = finished[j]
                if fa == fb:
                    continue
                ea = 1.0 / (1.0 + 10 ** ((pre[pb] - pre[pa]) / ELO_SCALE))
                sa = 1.0 if fa < fb else 0.0
                delta[pa] += ELO_K * (sa - ea)
                delta[pb] += ELO_K * ((1.0 - sa) - (1.0 - ea))
        for p, dl in delta.items():
            rating[p] += dl
        # ライン連携: 同ラインペアの実績更新
        fin_map = dict(zip(pids, fins))
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if lines[i] is None or lines[i] != lines[j]:
                    continue
                key = (pids[i], pids[j]) if pids[i] < pids[j] else (pids[j], pids[i])
                pair_n[key] += 1
                fi, fj = fin_map[pids[i]], fin_map[pids[j]]
                if fi and fj and 1 <= fi <= 3 and 1 <= fj <= 3:
                    pair_hit[key] += 1

    df = df.copy()
    key = list(zip(df["race_key"], df["player_id"]))
    df["elo"] = [out_elo.get(k, ELO_INIT) for k in key]
    lc = [out_lc.get(k, (0.0, 0.0, 0.0)) for k in key]
    df["lc_n_prev"] = [x[0] for x in lc]
    df["lc_rate"] = [x[1] for x in lc]
    df["lc_max_n"] = [x[2] for x in lc]
    # レース内相対
    grp = df.groupby("race_key")["elo"]
    df["elo_rank"] = grp.rank(ascending=False)
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(1.0).replace(0.0, 1.0)
    df["elo_z"] = ((df["elo"] - mean) / std).clip(-5, 5)
    return df


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
    print(f"  raw rows={len(raw)}")
    raw = compute_elo_and_linecoop(raw)
    print("  Elo/ライン連携 計算完了 "
          f"(elo range {raw['elo'].min():.0f}-{raw['elo'].max():.0f}, "
          f"lc_n_prev>0 率={(raw['lc_n_prev']>0).mean():.2f})")
    df = build_features_wt(raw)
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
        "+elo": list(FEATURE_COLS_WT) + ELO_COLS,
        "+linecoop": list(FEATURE_COLS_WT) + LC_COLS,
        "+both": list(FEATURE_COLS_WT) + ELO_COLS + LC_COLS,
    }
    agg = {v: defaultdict(list) for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(tr[cols].fillna(0).values, tr[TARGET_COL_WT].values)
            for tag, ev in (("te", te), ("fw", fw)):
                ev = ev.copy()
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                agg[vname][f"auc_{tag}"].append(
                    roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"]))
                mt = race_metrics(ev)
                agg[vname][f"ss_{tag}"].append(mt["ss"])
                agg[vname][f"win1_{tag}"].append(mt["win1"])
                agg[vname][f"top3_{tag}"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    ms = lambda a: (np.mean(a), np.std(a))
    print("\n============= 結果（seed平均 ± std, n=%d）=============" % len(SEEDS))
    for tag, label in (("te", "TEST 2026-04〜06 (クリーンOOS)"), ("fw", "FWD 2026-07")):
        print(f"\n--- {label} ---")
        print(f"{'variant':<12}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>13}{'1位複勝率':>13}")
        base = agg["baseline"]
        for v in variants:
            a = agg[v]
            am, asd = ms(a[f"auc_{tag}"]); sm, ssd = ms(a[f"ss_{tag}"])
            wm, wsd = ms(a[f"win1_{tag}"]); tm, tsd = ms(a[f"top3_{tag}"])
            dss = sm - np.mean(base[f"ss_{tag}"])
            mk = "" if v == "baseline" else ("  ★" if dss > ssd else ("  ×" if dss < -ssd else "  ~"))
            print(f"{v:<12}{am:>7.4f}±{asd:.4f}{sm:>13.1%}±{ssd:.1%}"
                  f"{wm:>7.1%}±{wsd:.1%}{tm:>7.1%}±{tsd:.1%}{mk}")
    print("\n判定: SSΔ>seed std で ★(採用候補)。採用は TEST・FWD 双方で非悪化かつ TEST で ★。")


if __name__ == "__main__":
    main()
