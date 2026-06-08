"""wtデータを「ks同等24特徴量だけ」で学習し233%を再現するか検証
再現→wtの追加特徴が害。非再現→同等とみなした特徴に実データ差。
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.database import get_connection
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _load_payouts_wt

VAL_FROM, TEST_FROM, TEST_TO = "2025-09-01", "2026-03-01", "2026-07-01"
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
raw=load_raw_data_wt(min_date="2023-07-01"); df=build_features_wt(raw); df=df[df["finish_order"].notna()].copy()
df["player_id"]=raw.loc[df.index,"player_id"].values
df=df.merge(compute_rolling(),on=["race_key","player_id"],how="left")
for c in ROLL: df[c]=df[c].fillna(df[c].median())

# ks24特徴に対応するwt列（得点・勝率は同一値, 脚質=style_enc, ローリングはwt計算）
KS_EQUIV = ["race_point","gear_ratio","win_3m","top3_3m","style_enc","frame_no",
            "score_rank","score_z","wr_rank","top3r_rank","is_inner","is_outer",
            "grade_enc","win_6m","top3_6m","wr_trend","venue_wr","days_since",
            "bank_length_enc","is_indoor","quin_6m","period_norm","player_class_enc","is_home"]

tr=df[df["race_date"]<VAL_FROM]
pm=_load_payouts_wt(df["race_key"].unique().tolist())

def tiered(model, cols, tag):
    d2=df.copy(); d2["pred_prob"]=model.predict_proba(d2[cols].fillna(0))[:,1]
    for sp,a,b in [("検証",VAL_FROM,TEST_FROM),("テスト",TEST_FROM,TEST_TO)]:
        s=d2[(d2["race_date"]>=a)&(d2["race_date"]<b)]
        auc=roc_auc_score(s[TARGET_COL_WT],s["pred_prob"])
        rec={t:{"r":0,"bet":0,"ret":0,"hit":0} for t in ("SS","S","A")}
        for rk,grp in s.groupby("race_key"):
            grp=grp.sort_values("pred_prob",ascending=False); n=len(grp)
            if n>6 or n<3: continue
            p=grp["pred_prob"].tolist(); gap=p[0]-p[1]; ratio=p[0]/(3.0/n)
            if gap<0.06: continue
            tg="SS" if(gap>=0.15 and ratio<1.3)else("S" if(gap>=0.15 and ratio<1.6)else("A" if gap<0.15 else None))
            if tg is None: continue
            fr=grp["frame_no"].astype(int).tolist(); p1,p2=fr[0],fr[1]; thirds=fr[2:5]
            fin=grp[grp["finish_order"]<=3].sort_values("finish_order"); order=fin["frame_no"].astype(int).tolist()
            if len(order)<3: continue
            top3=frozenset(order); rp=pm.get(rk,{}); rec[tg]["r"]+=1
            for t in thirds:
                rec[tg]["bet"]+=100
                if tg=="SS":
                    if order[:3]==[p1,p2,t]: rec[tg]["ret"]+=rp.get(("trifecta",(p1,p2,t)),0); rec[tg]["hit"]+=1
                else:
                    if frozenset((p1,p2,t))==top3: rec[tg]["ret"]+=rp.get(("trio",frozenset((p1,p2,t))),0); rec[tg]["hit"]+=1
        line=f"  [{tag}/{sp}] AUC={auc:.4f}"
        for t in ("SS","S","A"):
            x=rec[t]; roi=x["ret"]/x["bet"] if x["bet"] else 0
            line+=f"  {t}={roi*100:.0f}%({x['r']}R)"
        print(line)

print("\n=== wt: ks同等24特徴のみ ===")
m1=train_lgbm(tr,feature_cols=KS_EQUIV,target_col=TARGET_COL_WT); tiered(m1,KS_EQUIV,"wt-ks特徴")
print("\n=== wt: 全39特徴(現行) ===")
m2=train_lgbm(tr,feature_cols=FEATURE_COLS_WT+ROLL,target_col=TARGET_COL_WT); tiered(m2,FEATURE_COLS_WT+ROLL,"wt-全特徴")
