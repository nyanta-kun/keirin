"""ks 238%の正体: 特徴量由来か v6チューニング由来か

新規ksモデル(wtと同じtrain_lgbm既定パラメータ・同じ2023-07学習)を作り、
同じtiered評価で lgbm_v6 / 新規ks / (参考)wt と比較する。
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS, TARGET_COL
from src.models.trainer import train_lgbm, load_model
from src.evaluation.backtest import _load_payouts

VAL_FROM, TEST_FROM, TEST_TO = "2025-09-01", "2026-03-01", "2026-07-01"

df = build_features(load_raw_data(min_date="2023-07-01"))
df = df[df["finish_position"].notna()].dropna(subset=FEATURE_COLS).copy()
tr = df[df["race_date"] < VAL_FROM]
print(f"学習 {tr['race_key'].nunique()}R (2023-07〜{VAL_FROM})")

fresh = train_lgbm(tr, feature_cols=FEATURE_COLS, target_col=TARGET_COL)

def tiered(model, tag):
    d2 = df.copy()
    d2["pred_prob"] = model.predict_proba(d2[FEATURE_COLS])[:, 1]
    pm = _load_payouts(d2["race_key"].unique().tolist())
    for sp, a, b in [("検証", VAL_FROM, TEST_FROM), ("テスト", TEST_FROM, TEST_TO)]:
        s = d2[(d2["race_date"] >= a) & (d2["race_date"] < b)]
        auc = roc_auc_score(s[TARGET_COL], s["pred_prob"])
        rec = {t: {"r": 0, "bet": 0, "ret": 0, "hit": 0} for t in ("SS", "S", "A")}
        for rk, grp in s.groupby("race_key"):
            grp = grp.sort_values("pred_prob", ascending=False); n = len(grp)
            if n > 6 or n < 3: continue
            p = grp["pred_prob"].tolist(); gap = p[0]-p[1]; ratio = p[0]/(3.0/n)
            if gap < 0.06: continue
            tg = "SS" if (gap>=0.15 and ratio<1.3) else ("S" if (gap>=0.15 and ratio<1.6) else ("A" if gap<0.15 else None))
            if tg is None: continue
            fr = grp["frame_no"].astype(int).tolist(); p1,p2=fr[0],fr[1]; thirds=fr[2:5]
            fin = grp[grp["finish_position"]<=3].sort_values("finish_position")
            order = fin["frame_no"].astype(int).tolist()
            if len(order) < 3: continue
            top3 = frozenset(order); rp = pm.get(rk, {}); rec[tg]["r"] += 1
            for t in thirds:
                rec[tg]["bet"] += 100
                if tg == "SS":
                    if order[:3]==[p1,p2,t]: rec[tg]["ret"]+=rp.get(("trifecta",f"{p1}-{p2}-{t}"),0); rec[tg]["hit"]+=1
                else:
                    if frozenset((p1,p2,t))==top3: rec[tg]["ret"]+=rp.get(("trifecta_box","=".join(map(str,sorted(top3)))),0); rec[tg]["hit"]+=1
        line = f"  [{tag}/{sp}] AUC={auc:.4f}"
        for t in ("SS","S","A"):
            x=rec[t]; roi=x["ret"]/x["bet"] if x["bet"] else 0
            line += f"  {t}={roi*100:.0f}%({x['r']}R)"
        print(line)

print("\n=== 新規ksモデル (wtと同一パラメータ・2023-07学習) ===")
tiered(fresh, "新規ks")
print("\n=== 本番 lgbm_v6 (反復改良済み) ===")
tiered(load_model("lgbm"), "v6")
