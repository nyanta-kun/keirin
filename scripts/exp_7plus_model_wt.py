"""7車以上 専用モデルの構築・検証（回収率100%超を狙う）

プールモデル(全車)の7+実績は全条件75-80%(控除率の壁)。
7+のみで学習した専用モデルが、いずれかの条件でOOS再現的に100%超を出せるか検証。
- 学習: 7+レース・2023-07〜2025-11（train）。再現性確認のため val(2025-12〜2026-02)/test(2026-03〜)の2期間OOSで評価。
- 戦略: std3/wide5/box4/box5（三連複）。条件: ALL/gap12閾値/top3_sum下位四分位(7+内)。
- payout=最終オッズ上限値。FEATURE_COLS_WT(40)。
"""
import sys, itertools, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT, prepare_X
from src.models.trainer import train_lgbm, load_model
from src.evaluation.backtest_wt import _load_payouts_wt, _assign_tier
from sklearn.metrics import roc_auc_score
from roi_robustness_wt import roi_summary


def build(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()
    return df


print("[load] train/val/test (7+) ...")
tr = build("2023-07-01", "2025-11-30")
va = build("2025-12-01", "2026-02-28")
te = build("2026-03-01", "2026-06-08")
trf = tr[tr["finish_order"] >= 1]
print(f"  train {trf['race_key'].nunique()}R / val {va['race_key'].nunique()}R / test {te['race_key'].nunique()}R")

print("[train] 7+専用モデル ...")
model7 = train_lgbm(trf, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT)
pool = load_model("lgbm_wt")   # 比較用プールモデル

# top3_sum 7+四分位(train)
def races(df, model):
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    out = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 7: continue
        p = g["pred_prob"].tolist()
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        top3 = frozenset(int(x) for x in fin["frame_no"]); rp = pm.get(rk, {})
        def mk(combos):
            h = top3 in combos; return (h, rp.get(("trio", top3), 0) if h else 0, len(combos)*100)
        out.append({"gap12": p[0]-p[1], "top3_sum": p[0]+p[1]+p[2],
                    "std3": mk([frozenset((fr[0], fr[1], x)) for x in fr[2:5]]),
                    "wide5": mk([frozenset((fr[0], fr[1], x)) for x in fr[2:7]]),
                    "box4": mk([frozenset(c) for c in itertools.combinations(fr[:4], 3)]),
                    "box5": mk([frozenset(c) for c in itertools.combinations(fr[:5], 3)])})
    return out, df

# AUC
for lab, df in [("val", va), ("test", te)]:
    d = df[df["finish_order"] >= 1]
    a7 = roc_auc_score(d[TARGET_COL_WT], model7.predict_proba(prepare_X(d))[:, 1])
    ap = roc_auc_score(d[TARGET_COL_WT], pool.predict_proba(prepare_X(d))[:, 1])
    print(f"  AUC({lab}/7+): 専用={a7:.4f}  プール={ap:.4f}")

rv, _ = races(va, model7); rte, _ = races(te, model7)
med = statistics.median([r["top3_sum"] for r in rv]) if rv else 0
CONDS = {"ALL": lambda r: True, "gap12>=0.10": lambda r: r["gap12"] >= 0.10,
         "gap12>=0.15": lambda r: r["gap12"] >= 0.15, "gap12>=0.20": lambda r: r["gap12"] >= 0.20,
         "top3_sum<中央": lambda r: r["top3_sum"] < med}

def agg(rows, cond, k):
    s = [r for r in rows if cond(r)];
    return roi_summary([r[k][1] for r in s if r[k][2] > 0], [r[k][2] for r in s if r[k][2] > 0]), len([r for r in s if r[k][2] > 0])

print(f"\n{'='*96}\n  7+専用モデル 戦略×条件 ROI（VAL/TEST とも OOS・最終オッズ上限値）\n{'='*96}")
print(f"  {'条件':<16}{'戦略':<7}{'VAL_R':>6}{'VAL_ROI':>8}{'TEST_R':>7}{'TEST_ROI':>9}{'TEST_CI':>18}")
for cn, cond in CONDS.items():
    for k in ["std3", "wide5", "box4", "box5"]:
        sv, nv = agg(rv, cond, k); stt, nt = agg(rte, cond, k)
        print(f"  {cn:<16}{k:<7}{nv:>6}{sv['roi']:>7.0%}{nt:>7}{stt['roi']:>8.0%} [{stt['ci_lo']:>4.0%},{stt['ci_hi']:>5.0%}]")
    print(f"  {'-'*94}")
print("\n  ※ VAL/TEST とも 100%超で再現する条件のみ採用候補。プールモデル7+は全条件75-80%。")
