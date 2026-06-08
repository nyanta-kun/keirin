"""ksルート ROI 再監査

ks本番のSS/S/A層別戦略を、報告ホールドアウト(2025-06〜2026-02)と
真のOOS(2026-03〜)で再現し、分散・jackknife(最大払戻除外)で
報告ROI(A215%/SS3944%)が本物の優位性か分散/期間アーティファクトかを判定。
"""
import argparse, numpy as np, pandas as pd
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model
from src.evaluation.backtest import _load_payouts

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lgbm")
args = ap.parse_args()

model = load_model(args.model)
df = build_features(load_raw_data(min_date="2025-06-01"))
df = df[df["finish_position"].notna()].copy()
df = df.dropna(subset=FEATURE_COLS).copy()
df["pred_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

PERIODS = {
    "報告HO(2025-06〜2026-02)": ("2025-06-01", "2026-03-01"),
    "真のOOS(2026-03〜06)":     ("2026-03-01", "2026-07-01"),
}

def tier(gap12, ratio):
    if gap12 < 0.06: return None
    if gap12 >= 0.15:
        if ratio < 1.3: return "SS"
        if ratio < 1.6: return "S"
        return None
    return "A"

def run(d):
    pm = _load_payouts(d["race_key"].unique().tolist())
    rec = {t: {"races":0,"bet":0,"ret":0,"hits":0,"pays":[]} for t in ("SS","S","A")}
    for rk, grp in d.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n > 6 or n < 3: continue
        probs = grp["pred_prob"].tolist()
        gap12 = probs[0]-probs[1]; ratio = probs[0]/(3.0/n)
        tg = tier(gap12, ratio)
        if tg is None: continue
        fr = grp["frame_no"].astype(int).tolist()
        p1,p2 = fr[0],fr[1]; thirds = fr[2:5]
        if not thirds: continue
        fin = grp[grp["finish_position"]<=3]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3)<3: continue
        order = tuple(fin.sort_values("finish_position")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {})
        rec[tg]["races"] += 1
        for t in thirds:
            rec[tg]["bet"] += 100
            if tg == "SS":
                if order == (p1,p2,t):
                    pay = rp.get(("trifecta", f"{p1}-{p2}-{t}"), 0)
                    rec[tg]["ret"] += pay; rec[tg]["hits"] += 1; rec[tg]["pays"].append(pay)
            else:
                combo = frozenset((p1,p2,t))
                if combo == top3:
                    key = "=".join(map(str, sorted(combo)))
                    pay = rp.get(("trifecta_box", key), 0)
                    rec[tg]["ret"] += pay; rec[tg]["hits"] += 1; rec[tg]["pays"].append(pay)
    return rec

for pname,(a,b) in PERIODS.items():
    d = df[(df["race_date"]>=a)&(df["race_date"]<b)]
    rec = run(d)
    print(f"\n===== {pname}  ({d['race_key'].nunique()}R) =====")
    print(f"{'層':<4}{'対象R':>6}{'的中':>5}{'的中率':>7}{'投資':>8}{'回収':>9}{'ROI':>8}{'最大払戻':>9}{'ROI(最大除外)':>14}")
    for t in ("SS","S","A"):
        r = rec[t]
        if r["races"]==0:
            print(f"{t:<4}{0:>6}"); continue
        roi = r["ret"]/r["bet"] if r["bet"] else 0
        hr = r["hits"]/r["races"]
        maxpay = max(r["pays"]) if r["pays"] else 0
        # jackknife: 最大払戻1件を除外したROI
        ret_jk = r["ret"] - maxpay
        roi_jk = ret_jk/r["bet"] if r["bet"] else 0
        print(f"{t:<4}{r['races']:>6}{r['hits']:>5}{hr:>7.0%}{r['bet']:>8}{r['ret']:>9}{roi:>8.0%}{maxpay:>9}{roi_jk:>13.0%}")
