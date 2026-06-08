"""本命バイアス検証: prediction_mark / winticket表示勝率 を外すと妙味を拾うか
A: 全特徴+rolling(現行)  B: -prediction_mark  C: -mark-表示勝率(純粋実力)
各構成で A層の 的中率・当たり平均配当・ROI を比較。
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.database import get_connection
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _load_payouts_wt

TRAIN_FROM,VAL_FROM,TEST_FROM,TEST_TO="2024-06-01","2025-09-01","2026-03-01","2026-07-01"
ROLL=["win_3m","top3_3m","quin_3m","win_6m","top3_6m","quin_6m","venue_wr","days_since","wr_trend"]

def compute_rolling():
    with get_connection() as conn:
        h=pd.read_sql_query("SELECT e.race_key,e.player_id,e.finish_order,r.race_date,r.venue_id FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key WHERE e.finish_order IS NOT NULL",conn)
    h["dt"]=pd.to_datetime(h["race_date"])
    for c,cond in [("win",h["finish_order"]==1),("top3",h["finish_order"]<=3),("quin",h["finish_order"]<=2)]:
        h[c]=cond.astype(float)
    h=h.sort_values(["player_id","dt"]).reset_index(drop=True)
    def rm(c,w): return (h.set_index("dt").groupby("player_id")[c].rolling(w,closed="left").mean().reset_index(level=0,drop=True).values)
    for c in ["win","top3","quin"]: h[f"{c}_3m"]=rm(c,"90D"); h[f"{c}_6m"]=rm(c,"180D")
    h["venue_wr"]=(h.sort_values(["player_id","venue_id","dt"]).groupby(["player_id","venue_id"])["win"].apply(lambda s:s.expanding().mean().shift(1)).reset_index(level=[0,1],drop=True))
    h["days_since"]=h.groupby("player_id")["dt"].diff().dt.days
    h["wr_trend"]=h["win_3m"]-h["win_6m"]
    return h[["race_key","player_id"]+ROLL]

print("構築中...")
raw=load_raw_data_wt(min_date=TRAIN_FROM); df=build_features_wt(raw); df=df[df["finish_order"].notna()].copy()
df["player_id"]=raw.loc[df.index,"player_id"].values
df=df.merge(compute_rolling(),on=["race_key","player_id"],how="left")
for c in ROLL: df[c]=df[c].fillna(df[c].median())
df["race_size"]=df.groupby("race_key")["frame_no"].transform("count"); df["w_field"]=1.0/df["race_size"]
tr=df[df["race_date"]<VAL_FROM]
pm=_load_payouts_wt(df["race_key"].unique().tolist())

def atier(s):
    nA=hit=bet=ret=0; pays=[]
    for rk,grp in s.groupby("race_key"):
        grp=grp.sort_values("pred_prob",ascending=False); n=len(grp)
        if n>6 or n<3: continue
        p=grp["pred_prob"].tolist(); gap=p[0]-p[1]
        if not(0.06<=gap<0.15): continue
        fr=grp["frame_no"].astype(int).tolist(); p1,p2=fr[0],fr[1]; thirds=fr[2:5]
        fin=grp[grp["finish_order"]<=3]; top3=frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3)<3: continue
        nA+=1; rp=pm.get(rk,{})
        for t in thirds:
            bet+=100; c=frozenset((p1,p2,t))
            if c==top3: pay=rp.get(("trio",c),0); ret+=pay; hit+=1; pays.append(pay)
    roi=ret/bet if bet else 0
    return nA,hit,(np.mean(pays) if pays else 0),roi

configs={
 "A:全+roll(現行)": FEATURE_COLS_WT+ROLL,
 "B:-AI印": [c for c in FEATURE_COLS_WT if c!="prediction_mark"]+ROLL,
 "C:-AI印-表示勝率": [c for c in FEATURE_COLS_WT if c not in ("prediction_mark","first_rate_norm","third_rate_norm","wr_rank","top3r_rank")]+ROLL,
}
print(f"\n{'構成':<18}{'期':<5}{'A層R':>6}{'的中率':>7}{'当たり平均':>10}{'ROI':>7}")
for name,cols in configs.items():
    m=train_lgbm(tr,feature_cols=cols,target_col=TARGET_COL_WT,weight_col="w_field")
    d2=df.copy(); d2["pred_prob"]=m.predict_proba(d2[cols].fillna(0))[:,1]
    for sp,a,b in [("val",VAL_FROM,TEST_FROM),("test",TEST_FROM,TEST_TO)]:
        s=d2[(d2["race_date"]>=a)&(d2["race_date"]<b)]
        nA,hit,avgp,roi=atier(s)
        print(f"{name:<18}{sp:<5}{nA:>6}{(hit/nA if nA else 0):>7.0%}{avgp:>9.0f}円{roi:>7.0%}")
