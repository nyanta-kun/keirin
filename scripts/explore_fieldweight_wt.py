"""頭数重み付け(1/n_riders)の効果検証
baseline vs field-weighted で ①6車以下AUC ②EVバケット別実ROI を比較
"""
import itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _load_payouts_wt

TRAIN_FROM, TEST_FROM = "2025-06-01", "2026-03-01"

df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM))
df = df[df["finish_order"].notna()].copy()
df["race_size"] = df.groupby("race_key")["frame_no"].transform("count")
df["w_field"] = 1.0 / df["race_size"]

tr = df[df["race_date"] < TEST_FROM].copy()
te = df[df["race_date"] >= TEST_FROM].copy()
print(f"train {tr['race_key'].nunique()}R / test {te['race_key'].nunique()}R")

def pl_trio(fp):
    fr=list(fp.keys()); s={k:max(v,1e-6) for k,v in fp.items()}; out={}
    for a,b,c in itertools.combinations(fr,3):
        tot=sum(s.values()); p=0.0
        for x,y,z in itertools.permutations((a,b,c)):
            d1=tot;d2=tot-s[x];d3=tot-s[x]-s[y]
            if d2>0 and d3>0: p+=s[x]/d1*s[y]/d2*s[z]/d3
        out[frozenset((a,b,c))]=p
    return out

pm = _load_payouts_wt(te["race_key"].unique().tolist())

def evaluate(model, tag):
    X = te[FEATURE_COLS_WT].fillna(0)
    te2 = te.copy()
    te2["pred_prob"] = model.predict_proba(X)[:, 1]
    # AUC 全体 & 6車以下
    auc_all = roc_auc_score(te2[TARGET_COL_WT], te2["pred_prob"])
    sub6 = te2[te2["race_size"] <= 6]
    auc_6 = roc_auc_score(sub6[TARGET_COL_WT], sub6["pred_prob"])
    print(f"\n[{tag}] AUC 全体={auc_all:.4f}  6車以下={auc_6:.4f}  (6車以下{sub6['race_key'].nunique()}R)")
    # EVバケット診断
    rows=[]
    for rk,grp in te2.groupby("race_key"):
        grp=grp.sort_values("pred_prob",ascending=False); n=len(grp)
        if n<3: continue
        fp=dict(zip(grp["frame_no"].astype(int),grp["pred_prob"]))
        cp=pl_trio(fp); rp=pm.get(rk,{})
        fin=grp[grp["finish_order"]<=3]; top3=frozenset(fin["frame_no"].astype(int).tolist())
        ok=len(top3)==3
        for combo,p in cp.items():
            o=rp.get(("trio",combo))
            if not o: continue
            rows.append((p*(o/100), 1 if (ok and combo==top3) else 0, o))
    d=pd.DataFrame(rows,columns=["ev","hit","payout"])
    bins=[0,0.8,1.0,1.2,1.5,2.0,100]
    d["b"]=pd.cut(d["ev"],bins)
    g=d.groupby("b",observed=True).apply(
        lambda x: pd.Series({"n":len(x),"hits":x["hit"].sum(),
                             "ROI":x.loc[x["hit"]==1,"payout"].sum()/(len(x)*100)}))
    print(g.to_string())

m_base = train_lgbm(tr, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT)
evaluate(m_base, "baseline")
m_w = train_lgbm(tr, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT, weight_col="w_field")
evaluate(m_w, "field-weighted 1/n")
