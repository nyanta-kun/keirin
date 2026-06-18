"""is_lone_nige 特徴の追加検証（形×脚質検証 P4-3 の帰結・2026-06-10）。

発見(docs/analysis/13): 拮抗レースで唯一の逃げ車が指数下位(4位-)のとき、その車のtop3を
モデルが系統的に過小評価（較正残差 TRAIN+7.1%/n=244・TEST+27.6%/n=20）。
n_senko(人数)はあるが「誰が単騎逃げか」の選手レベル交互作用が不足している仮説。

検証: `is_lone_nige = (style==逃) & (n_senko==1)` を選手レベル特徴として追加し、
同一データ・同一分割で baseline(現40特徴) vs 拡張(41特徴) を学習して
  ① holdout AUC ② ≤6車 SS/S/A層別ROI(本番戦略) ③ P4-3セルの較正残差の解消
を比較。n_senko採用時の手順(AUC 0.7778→0.7784・ROI 393→404%で採用)に準拠。
**本番の FEATURE_COLS_WT / lgbm_wt.pkl は変更しない**（採用判断後に正式反映）。
TRAIN 2023-07-01〜2026-02-28 / TEST 2026-03-01〜2026-06-08・払戻=最終オッズ上限値。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _load_payouts_wt
from roi_robustness_wt import roi_summary

TR_FROM, TE_FROM, TE_TO = "2023-07-01", "2026-03-01", "2026-06-08"


def tiered_roi(df_te, label):
    """≤6車 SS/S/A 本番戦略のROI（pred_prob列を使用）。"""
    sizes = df_te.groupby("race_key")["frame_no"].count()
    sub = df_te[df_te["race_key"].isin(sizes[sizes <= 6].index)]
    pm = _load_payouts_wt(sub["race_key"].unique().tolist())
    pays, bets = [], []
    tier_stats = {"SS": [], "S": [], "A": []}
    for rk, g in sub.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        ratio = p[0] / (3.0 / n)
        if gap12 < 0.06 or (gap12 >= 0.15 and ratio >= 1.6):
            continue
        tier = "SS" if (gap12 >= 0.15 and ratio < 1.3) else ("S" if gap12 >= 0.15 else "A")
        fr = g["frame_no"].astype(int).tolist()
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order)
        rp = pm.get(rk, {})
        thirds = fr[2:5]
        pay = 0
        for x in thirds:
            if tier == "SS":
                if order == (fr[0], fr[1], x):
                    pay = rp.get(("trifecta", (fr[0], fr[1], x)), 0); break
            else:
                if frozenset((fr[0], fr[1], x)) == top3:
                    pay = rp.get(("trio", frozenset((fr[0], fr[1], x))), 0); break
        bet = len(thirds) * 100
        pays.append(pay); bets.append(bet)
        tier_stats[tier].append((pay, bet))
    s = roi_summary(pays, bets)
    print(f"  {label}: 計{s['n']}R 的中{s['hit_rate']:.1%} ROI{s['roi']:.0%} [{s['ci_lo']:.0%},{s['ci_hi']:.0%}] 最大除{s['roi_ex_max']:.0%}")
    for t in ["SS", "S", "A"]:
        ts = roi_summary([p for p, _ in tier_stats[t]], [b for _, b in tier_stats[t]])
        print(f"    {t:<3} {ts['n']:>4}R 的中{ts['hit_rate']:>6.1%} ROI{ts['roi']:>6.0%}")
    return s


def p43_residual(df_te, label):
    """P4-3セル（拮抗っぽい代理=単騎逃げで指数下位）の較正残差。"""
    sizes = df_te.groupby("race_key")["frame_no"].count()
    sub = df_te[df_te["race_key"].isin(sizes[sizes <= 6].index)]
    acts, preds = [], []
    for rk, g in sub.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 4:
            continue
        styles = [str(s) for s in g["style"].tolist()]
        if styles.count("逃") != 1:
            continue
        idx = styles.index("逃")
        if idx < 3:                       # 指数4位以下の単騎逃げのみ
            continue
        fo = g.loc[idx, "finish_order"]
        if not (isinstance(fo, (int, float)) and fo == fo and fo >= 1):
            continue
        acts.append(1 if 1 <= fo <= 3 else 0)
        preds.append(float(g.loc[idx, "pred_prob"]))
    if acts:
        a, p = np.mean(acts), np.mean(preds)
        print(f"  {label}: n={len(acts)} 実績{a:.1%} 予測{p:.1%} 残差{a-p:+.1%}")


print("データ構築中...")
df = build_features_wt(load_raw_data_wt(min_date=TR_FROM, max_date=TE_TO))
df["is_lone_nige"] = ((df["style"] == "逃") & (df["n_senko"] == 1)).astype(int)
df = df[df["finish_order"] >= 1]          # M-2: DNS負例除去（本番学習と同一）
df_tr = df[df["race_date"] < TE_FROM]
df_te = df[df["race_date"] >= TE_FROM].copy()
print(f"TRAIN {df_tr['race_key'].nunique()}R / TEST {df_te['race_key'].nunique()}R  lone_nige率={df['is_lone_nige'].mean():.3f}")

COLS_BASE = FEATURE_COLS_WT
COLS_EXT = FEATURE_COLS_WT + ["is_lone_nige"]

results = {}
for name, cols in [("baseline(40)", COLS_BASE), ("+is_lone_nige(41)", COLS_EXT)]:
    print(f"\n=== {name} 学習 ===")
    model = train_lgbm(df_tr.copy(), n_splits=3, feature_cols=cols, target_col=TARGET_COL_WT)
    proba = model.predict_proba(df_te.reindex(columns=cols).fillna(0))[:, 1]
    auc = roc_auc_score(df_te[TARGET_COL_WT], proba)
    print(f"  holdout(TEST) AUC = {auc:.4f}")
    df_te["pred_prob"] = proba
    s = tiered_roi(df_te, f"{name} 層別ROI")
    p43_residual(df_te, f"{name} P4-3残差")
    results[name] = (auc, s["roi"])

print(f"\n=== 判定 ===")
(a0, r0), (a1, r1) = results["baseline(40)"], results["+is_lone_nige(41)"]
print(f"  AUC: {a0:.4f} → {a1:.4f} ({a1-a0:+.4f})")
print(f"  層別ROI: {r0:.0%} → {r1:.0%}")
print("  採用基準(n_senko前例): AUC非劣化＋層別ROI改善＋P4-3残差縮小。劣化なら見送り。")
