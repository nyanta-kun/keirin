"""利益条件マイニング（過学習耐性つき）

本番の2軸流し三連複(pivot1-pivot2-各third, 3点)を、各種条件で層別し、
訓練期間でROI>閾値の条件を抽出 → テスト期間(out-of-sample)で検証する。
両期間で100%超＋十分サンプルの条件のみを「頑健な利益条件」とする。

使い方:
  PYTHONPATH=. .venv/bin/python3 scripts/mine_profitable_conditions_wt.py \
      --model lgbm_wt_interim --from 2025-06-01 --split 2026-02-01
"""
import argparse, itertools, numpy as np, pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lgbm_wt_interim")
ap.add_argument("--from", dest="from_date", default="2025-06-01")
ap.add_argument("--split", dest="split", default="2026-02-01")  # train < split <= test
ap.add_argument("--min-n", type=int, default=30)
args = ap.parse_args()

model = load_model(args.model)
raw = load_raw_data_wt(min_date=args.from_date)
df = build_features_wt(raw)
df = df[df["finish_order"].notna()].copy()
df = _apply_pred_prob_wt(model, df)
pm = _load_payouts_wt(df["race_key"].unique().tolist())

# レース単位の特徴と本番ベットの収支を構築
g = df.groupby("race_key")["pred_prob"]
df["zc"] = (df["pred_prob"] - g.transform("mean")) / g.transform("std").replace(0, 1)

rows = []
for rk, grp in df.groupby("race_key"):
    grp = grp.sort_values("pred_prob", ascending=False)
    n = len(grp)
    if n < 3:
        continue
    fr = grp["frame_no"].astype(int).tolist()
    probs = grp["pred_prob"].tolist()
    pivot1, pivot2 = fr[0], fr[1]
    thirds = fr[2:5]
    fin = grp[grp["finish_order"] <= 3]
    top3 = frozenset(fin["frame_no"].astype(int).tolist())
    if len(top3) < 3:
        continue
    rp = pm.get(rk, {})
    bet = len(thirds) * 100
    ret = 0
    for t in thirds:
        combo = frozenset((pivot1, pivot2, t))
        if combo == top3:
            ret += rp.get(("trio", combo), 0)
    gap12 = probs[0] - probs[1]
    ratio = probs[0] / (3.0 / n)
    row0 = grp.iloc[0]
    rows.append({
        "race_key": rk,
        "race_date": row0["race_date"],
        "n_riders": n,
        "grade": row0.get("grade"),
        "bank": int(row0.get("bank_length_enc", 0) * 100) if pd.notna(row0.get("bank_length_enc")) else 0,
        "gap12": gap12,
        "ratio": ratio,
        "z1": probs and (probs[0] - np.mean(probs)) / (np.std(probs) or 1),
        "p1_line_leader": int(row0.get("is_line_leader", 0)),
        "is_indoor": int(row0.get("is_indoor", 0)),
        "bet": bet, "ret": ret, "hit": int(ret > 0),
    })

R = pd.DataFrame(rows)
tr = R[R["race_date"] < args.split]
te = R[R["race_date"] >= args.split]
print(f"全{len(R)}R  train={len(tr)}R(〜{args.split})  test={len(te)}R")
print(f"全体ROI: train={tr['ret'].sum()/tr['bet'].sum():.1%}  test={te['ret'].sum()/te['bet'].sum():.1%}\n")

def roi(d):
    return d["ret"].sum() / d["bet"].sum() if d["bet"].sum() else 0

def evaluate_condition(name, mask_fn):
    """条件マスク関数で train/test 別に集計"""
    mt, me = mask_fn(tr), mask_fn(te)
    nt, ne = mt.sum(), me.sum()
    if nt < args.min_n or ne < args.min_n:
        return None
    rt = roi(tr[mt]); re = roi(te[me])
    ht = tr[mt]["hit"].mean(); he = te[me]["hit"].mean()
    return dict(cond=name, n_tr=int(nt), roi_tr=rt, hit_tr=ht,
               n_te=int(ne), roi_te=re, hit_te=he)

results = []
# 単変数条件のグリッド
for nr in [5, 6, 7, 8, 9]:
    results.append(evaluate_condition(f"n_riders=={nr}", lambda d, nr=nr: d["n_riders"] == nr))
for lo, hi in [(0, .06), (.06, .10), (.10, .15), (.15, .25), (.25, 1)]:
    results.append(evaluate_condition(f"gap12 [{lo},{hi})", lambda d, lo=lo, hi=hi: (d["gap12"] >= lo) & (d["gap12"] < hi)))
for lo, hi in [(0, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 9)]:
    results.append(evaluate_condition(f"ratio [{lo},{hi})", lambda d, lo=lo, hi=hi: (d["ratio"] >= lo) & (d["ratio"] < hi)))
for lo, hi in [(0, 1.0), (1.0, 1.3), (1.3, 1.6), (1.6, 3)]:
    results.append(evaluate_condition(f"z1 [{lo},{hi})", lambda d, lo=lo, hi=hi: (d["z1"] >= lo) & (d["z1"] < hi)))
results.append(evaluate_condition("p1_line_leader==1", lambda d: d["p1_line_leader"] == 1))
results.append(evaluate_condition("is_indoor==1", lambda d: d["is_indoor"] == 1))
for gr in R["grade"].dropna().unique():
    results.append(evaluate_condition(f"grade=={gr}", lambda d, gr=gr: d["grade"] == gr))
# 2変数: 6車以下 × gap/ratio/z
results.append(evaluate_condition("n<=6 & gap12>=0.15", lambda d: (d["n_riders"] <= 6) & (d["gap12"] >= 0.15)))
results.append(evaluate_condition("n<=6 & ratio<1.3", lambda d: (d["n_riders"] <= 6) & (d["ratio"] < 1.3)))
results.append(evaluate_condition("n<=6 & z1>=1.3", lambda d: (d["n_riders"] <= 6) & (d["z1"] >= 1.3)))
results.append(evaluate_condition("n<=6 & gap12>=0.15 & ratio<1.3", lambda d: (d["n_riders"] <= 6) & (d["gap12"] >= 0.15) & (d["ratio"] < 1.3)))

res = pd.DataFrame([r for r in results if r])
res = res.sort_values("roi_te", ascending=False)
pd.set_option("display.width", 200)
print("=== 条件別 train/test ROI（min_n={}）===".format(args.min_n))
print(res.to_string(index=False, formatters={
    "roi_tr": "{:.0%}".format, "roi_te": "{:.0%}".format,
    "hit_tr": "{:.0%}".format, "hit_te": "{:.0%}".format}))
print("\n=== 頑健候補（train>105% かつ test>100%）===")
robust = res[(res["roi_tr"] > 1.05) & (res["roi_te"] > 1.00)]
print(robust.to_string(index=False, formatters={
    "roi_tr": "{:.0%}".format, "roi_te": "{:.0%}".format,
    "hit_tr": "{:.0%}".format, "hit_te": "{:.0%}".format}) if len(robust) else "  (該当なし)")
