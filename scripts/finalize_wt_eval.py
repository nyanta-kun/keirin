"""【収集完了後に実行】winticket 本番評価: ks流ローリング特徴 + ライン情報

全wt履歴から point-in-time ローリング特徴を構築し、
FEATURE_COLS_WT + rolling で学習 → 学習/検証/テストの3分割OOSで
SS/S/A層別ROIを算出。ksの実績(A238%/S185%)と比較しks超えを判定。

使い方:
  PYTHONPATH=. .venv/bin/python3 scripts/finalize_wt_eval.py \
      --train-from 2023-07-01 --val-from 2025-09-01 --test-from 2026-03-01 \
      --save-as lgbm_wt_v1
"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.database import get_connection
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT
from src.models.trainer import train_lgbm, save_model
from src.evaluation.backtest_wt import _load_payouts_wt

ap = argparse.ArgumentParser()
ap.add_argument("--train-from", default="2023-07-01")
ap.add_argument("--val-from",   default="2025-09-01")
ap.add_argument("--test-from",  default="2026-03-01")
ap.add_argument("--test-to",    default="2026-07-01")
ap.add_argument("--save-as",    default="lgbm_wt_v1")
ap.add_argument("--no-rolling", action="store_true", help="ローリング特徴なし(baseline比較用)")
args = ap.parse_args()

ROLL_COLS = ["win_3m","top3_3m","quin_3m","win_6m","top3_6m","quin_6m","venue_wr","days_since","wr_trend"]

def compute_rolling():
    """全wt履歴から point-in-time ローリング特徴を計算"""
    with get_connection() as conn:
        h = pd.read_sql_query("""
            SELECT e.race_key, e.player_id, e.finish_order, r.race_date, r.venue_id
            FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key
            WHERE e.finish_order IS NOT NULL
        """, conn)
    h["dt"] = pd.to_datetime(h["race_date"])
    h["win"]=(h["finish_order"]==1).astype(float)
    h["top3"]=(h["finish_order"].between(1,3)).astype(float)
    h["quin"]=(h["finish_order"].between(1,2)).astype(float)
    h=h.sort_values(["player_id","dt"]).reset_index(drop=True)
    def rm(col,w):
        return (h.set_index("dt").groupby("player_id")[col]
                .rolling(w,closed="left").mean().reset_index(level=0,drop=True).values)
    for c in ["win","top3","quin"]:
        h[f"{c}_3m"]=rm(c,"90D"); h[f"{c}_6m"]=rm(c,"180D")
    h["venue_wr"]=(h.sort_values(["player_id","venue_id","dt"])
                   .groupby(["player_id","venue_id"])["win"]
                   .apply(lambda s:s.expanding().mean().shift(1)).reset_index(level=[0,1],drop=True))
    h["days_since"]=h.groupby("player_id")["dt"].diff().dt.days
    h["wr_trend"]=h["win_3m"]-h["win_6m"]
    return h[["race_key","player_id"]+ROLL_COLS]

print("データ構築中...")
raw = load_raw_data_wt(min_date=args.train_from)
df = build_features_wt(raw)
df = df[df["finish_order"].notna() & (df["finish_order"]>=1)].copy()
df["player_id"] = raw.loc[df.index,"player_id"].values
df["race_size"] = df.groupby("race_key")["frame_no"].transform("count")
df["w_field"] = 1.0/df["race_size"]

cols = list(FEATURE_COLS_WT)
if not args.no_rolling:
    print("ローリング特徴を計算中...")
    rf = compute_rolling()
    df = df.merge(rf, on=["race_key","player_id"], how="left")
    for c in ROLL_COLS:
        df[c]=df[c].fillna(df[c].median())
    cols += ROLL_COLS

tr = df[df["race_date"] < args.val_from]
print(f"学習 {tr['race_key'].nunique()}R ({args.train_from}〜{args.val_from})  特徴量{len(cols)}個")
model = train_lgbm(tr, feature_cols=cols, target_col=TARGET_COL_WT, weight_col="w_field")
save_model(model, args.save_as)

df["pred_prob"] = model.predict_proba(df[cols].fillna(0))[:,1]
pm = _load_payouts_wt(df["race_key"].unique().tolist())

def tiered(s):
    rec={t:{"races":0,"bet":0,"ret":0,"hits":0} for t in ("SS","S","A")}
    for rk,grp in s.groupby("race_key"):
        grp=grp.sort_values("pred_prob",ascending=False); n=len(grp)
        if n>6 or n<3: continue
        p=grp["pred_prob"].tolist(); gap=p[0]-p[1]; ratio=p[0]/(3.0/n)
        if gap<0.06: continue
        tg="SS" if(gap>=0.15 and ratio<1.3)else("S" if(gap>=0.15 and ratio<1.6)else("A" if gap<0.15 else None))
        if tg is None: continue
        fr=grp["frame_no"].astype(int).tolist(); p1,p2=fr[0],fr[1]; thirds=fr[2:5]
        if not thirds: continue
        fin=grp[grp["finish_order"].between(1,3)]; top3=frozenset(fin["frame_no"].astype(int).tolist())
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
    return rec

print(f"\n{'='*64}\nwinticket 本番評価  (特徴量: {'rolling込み' if not args.no_rolling else 'baseline'})\n{'='*64}")
KS = {"A":238,"S":185,"SS":1321}  # ks 真のOOS実績(参考)
for sp,a,b in [("検証",args.val_from,args.test_from),("テスト",args.test_from,args.test_to)]:
    s=df[(df["race_date"]>=a)&(df["race_date"]<b)].copy()
    auc=roc_auc_score(s[TARGET_COL_WT],s["pred_prob"])
    rec=tiered(s)
    print(f"\n[{sp}] {s['race_key'].nunique()}R  AUC={auc:.4f}")
    print(f"  {'層':<4}{'対象R':>6}{'的中率':>7}{'ROI':>7}{'  ks参考':>9}")
    tb=tr_=0
    for t in("SS","S","A"):
        r=rec[t]; roi=r['ret']/r['bet'] if r['bet'] else 0; hr=r['hits']/r['races'] if r['races'] else 0
        tb+=r['bet']; tr_+=r['ret']
        mark=" ★ks超" if (sp=="テスト" and roi*100>KS.get(t,9999)) else ""
        print(f"  {t:<4}{r['races']:>6}{hr:>7.0%}{roi:>7.0%}{KS.get(t,'-'):>8}%{mark}")
    print(f"  合計ROI: {tr_/tb if tb else 0:.0%}")
