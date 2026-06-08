"""winticketデータに ks流ローリング特徴量を実装・検証

wt_entries(player_id, finish_order) + wt_races(race_date, venue_id) から
point-in-time(現レースより過去のみ)で:
  - 直近3m/6m 勝率・top3率, quinella率(2着以内)
  - 場別勝率, 前走からの日数, 勝率トレンド(3m-6m)
を計算し、FEATURE_COLS_WT に追加して 9/6/3 分割でtier ROI を比較。
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.database import get_connection
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _load_payouts_wt

TRAIN_FROM, VAL_FROM, TEST_FROM, TEST_TO = "2025-01-01","2025-10-01","2026-04-01","2026-07-01"

# 全wt履歴(選手の過去成績計算用に最初期から)
print("ローリング特徴量を計算中...")
with get_connection() as conn:
    hist = pd.read_sql_query("""
        SELECT e.race_key, e.player_id, e.finish_order, r.race_date, r.venue_id
        FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key
        WHERE e.finish_order IS NOT NULL
    """, conn)
hist["dt"] = pd.to_datetime(hist["race_date"])
hist["win"] = (hist["finish_order"]==1).astype(float)
hist["top3"] = (hist["finish_order"]<=3).astype(float)
hist["quin"] = (hist["finish_order"]<=2).astype(float)
hist = hist.sort_values(["player_id","dt"]).reset_index(drop=True)

def roll_mean(col, window):
    # closed='left' で現レースを除外(リーク防止)
    return (hist.set_index("dt").groupby("player_id")[col]
            .rolling(window, closed="left").mean().reset_index(level=0, drop=True).values)

for col in ["win","top3","quin"]:
    hist[f"{col}_3m"] = roll_mean(col, "90D")
    hist[f"{col}_6m"] = roll_mean(col, "180D")
# 場別勝率(過去, 同選手×同場)
hist["venue_wr"] = (hist.sort_values(["player_id","venue_id","dt"])
                    .groupby(["player_id","venue_id"])["win"]
                    .apply(lambda s: s.expanding().mean().shift(1)).reset_index(level=[0,1],drop=True))
# 前走からの日数
hist["days_since"] = hist.groupby("player_id")["dt"].diff().dt.days
# 勝率トレンド
hist["wr_trend"] = hist["win_3m"] - hist["win_6m"]

roll_cols = ["win_3m","top3_3m","quin_3m","win_6m","top3_6m","quin_6m","venue_wr","days_since","wr_trend"]
hist_feat = hist[["race_key","player_id"]+roll_cols]

# 学習データ構築
raw = load_raw_data_wt(min_date=TRAIN_FROM)
df = build_features_wt(raw)
df = df[df["finish_order"].notna()].copy()
df["player_id"] = raw.loc[df.index, "player_id"].values
df = df.merge(hist_feat, on=["race_key","player_id"], how="left")
for c in roll_cols:
    df[c] = df[c].fillna(df[c].median())
df["race_size"] = df.groupby("race_key")["frame_no"].transform("count")
df["w_field"] = 1.0/df["race_size"]

NEW_FEATS = FEATURE_COLS_WT + roll_cols
tr = df[df["race_date"] < VAL_FROM]
print(f"学習 {tr['race_key'].nunique()}R")

def tiered_roi(s, pm):
    """スコア済みdf(pred_prob列)から SS/S/A 層別ROIを直接計算(6車以下)"""
    rec = {t: {"races":0,"bet":0,"ret":0,"hits":0} for t in ("SS","S","A")}
    for rk, grp in s.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False); n=len(grp)
        if n>6 or n<3: continue
        p = grp["pred_prob"].tolist(); gap12=p[0]-p[1]; ratio=p[0]/(3.0/n)
        if gap12<0.06: continue
        tg = "SS" if (gap12>=0.15 and ratio<1.3) else ("S" if (gap12>=0.15 and ratio<1.6) else ("A" if gap12<0.15 else None))
        if tg is None: continue
        fr=grp["frame_no"].astype(int).tolist(); p1,p2=fr[0],fr[1]; thirds=fr[2:5]
        if not thirds: continue
        fin=grp[grp["finish_order"]<=3]; top3=frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3)<3: continue
        order=tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp=pm.get(rk,{}); rec[tg]["races"]+=1
        for t in thirds:
            rec[tg]["bet"]+=100
            if tg=="SS":
                if order==(p1,p2,t): rec[tg]["ret"]+=rp.get(("trifecta",(p1,p2,t)),0); rec[tg]["hits"]+=1
            else:
                c=frozenset((p1,p2,t))
                if c==top3: rec[tg]["ret"]+=rp.get(("trio",c),0); rec[tg]["hits"]+=1
    out={}
    tb=tr_=0
    for t in ("SS","S","A"):
        r=rec[t]; roi=r["ret"]/r["bet"] if r["bet"] else 0
        out[t]=(r["races"],roi); tb+=r["bet"]; tr_+=r["ret"]
    out["合計"]=(sum(rec[t]["races"] for t in rec), tr_/tb if tb else 0)
    return out

def evaluate(cols, tag):
    m = train_lgbm(tr, feature_cols=cols, target_col=TARGET_COL_WT, weight_col="w_field")
    d2 = df.copy()
    d2["pred_prob"] = m.predict_proba(d2[cols].fillna(0))[:,1]
    pm = _load_payouts_wt(d2["race_key"].unique().tolist())
    for sp, a, b in [("val",VAL_FROM,TEST_FROM),("test",TEST_FROM,TEST_TO)]:
        s = d2[(d2["race_date"]>=a)&(d2["race_date"]<b)].copy()
        auc = roc_auc_score(s[TARGET_COL_WT], s["pred_prob"])
        o = tiered_roi(s, pm)
        print(f"  [{tag}/{sp}] AUC={auc:.4f}  "
              f"SS={o['SS'][1]:.0%}({o['SS'][0]}R) S={o['S'][1]:.0%}({o['S'][0]}R) "
              f"A={o['A'][1]:.0%}({o['A'][0]}R) 合計={o['合計'][1]:.0%}")

print("\n=== baseline (元の30特徴) ===")
evaluate(FEATURE_COLS_WT, "base")
print("\n=== +ks流ローリング特徴 ===")
evaluate(NEW_FEATS, "roll")
