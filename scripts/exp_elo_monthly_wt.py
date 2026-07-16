"""Elo 特徴の月別一貫性検証（exp_elo_linecoop_wt の追試・2026-07-15）。

前段の結果: Elo は AUC を TEST/FWD 両窓で +0.003 改善（初の実質ゲイン）だが、
FWD(7月・654R) のレース単位指標(SS的中/1位勝率)が悪化。
7月の悪化が「小標本ノイズ」か「実劣化」かを、月別(4/5/6/7月)の
baseline vs +elo 差分で切り分ける。全月で改善→7月はノイズの公算、
月によって符号が暴れる→不安定として不採用を確定する。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)
from exp_elo_linecoop_wt import (
    compute_elo_and_linecoop, race_metrics, ELO_COLS, PARAMS, SEEDS, TR_TO,
)

MONTHS = [("2026-04", "2026-04-01", "2026-04-30"),
          ("2026-05", "2026-05-01", "2026-05-31"),
          ("2026-06", "2026-06-01", "2026-06-30"),
          ("2026-07", "2026-07-01", "2026-07-10")]


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date="2026-07-10")
    raw = compute_elo_and_linecoop(raw)
    df = build_features_wt(raw)
    df = df[df["finish_order"] >= 1].copy()
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)
    df7 = df[df["_ne"] == 7].copy()
    tr = df[df["race_date"] <= TR_TO].copy()

    variants = {"baseline": list(FEATURE_COLS_WT),
                "+elo": list(FEATURE_COLS_WT) + ELO_COLS}
    # agg[variant][month][metric] = list over seeds
    agg = {v: {m[0]: defaultdict(list) for m in MONTHS} for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(tr[cols].fillna(0).values, tr[TARGET_COL_WT].values)
            for mon, d_from, d_to in MONTHS:
                ev = df7[(df7["race_date"] >= d_from) & (df7["race_date"] <= d_to)].copy()
                if ev.empty:
                    continue
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                agg[vname][mon]["auc"].append(
                    roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"]))
                mt = race_metrics(ev)
                agg[vname][mon]["ss"].append(mt["ss"])
                agg[vname][mon]["win1"].append(mt["win1"])
                agg[vname][mon]["top3"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    print("\n===== 月別: baseline → +elo（seed平均, Δ=elo−base）=====")
    print(f"{'月':<9}{'nR':>5}{'AUC Δ':>10}{'SS的中 base→elo (Δ)':>28}{'1位勝率 Δ':>12}{'1位複勝 Δ':>12}")
    for mon, d_from, d_to in MONTHS:
        nr = df7[(df7["race_date"] >= d_from) & (df7["race_date"] <= d_to)]["race_key"].nunique()
        b, e = agg["baseline"][mon], agg["+elo"][mon]
        if not b["ss"]:
            continue
        dauc = np.mean(e["auc"]) - np.mean(b["auc"])
        bss, ess = np.mean(b["ss"]), np.mean(e["ss"])
        dw = np.mean(e["win1"]) - np.mean(b["win1"])
        dt3 = np.mean(e["top3"]) - np.mean(b["top3"])
        # レース数由来の二項ノイズ目安（1σ）
        bin_sd = (bss * (1 - bss) / max(nr, 1)) ** 0.5
        print(f"{mon:<9}{nr:>5}{dauc:>+10.4f}{bss:>12.1%}→{ess:.1%} ({ess-bss:+.1%})"
              f"{dw:>+11.1%}{dt3:>+11.1%}   (二項1σ≈{bin_sd:.1%})")
    print("\n読み方: 全月で SSΔ・勝率Δ が正→7月悪化はノイズ濃厚。符号が月で暴れる→不安定・不採用。")


if __name__ == "__main__":
    main()
