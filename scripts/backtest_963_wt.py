"""9ヶ月学習 → 6ヶ月検証 → 3ヶ月テスト の3分割バックテスト

- 学習(train)でモデルfit
- 検証(val)で戦略/条件を選択
- テスト(test)で最終確認（out-of-sample）
"""
import itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT
from src.models.trainer import train_lgbm
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt

TRAIN_FROM, VAL_FROM, TEST_FROM, TEST_TO = "2025-01-01", "2025-10-01", "2026-04-01", "2026-07-01"

print("データロード...")
df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM))
df = df[df["finish_order"].notna()].copy()
df["race_size"] = df.groupby("race_key")["frame_no"].transform("count")
df["w_field"] = 1.0 / df["race_size"]

tr = df[df["race_date"] < VAL_FROM]
print(f"学習: {tr['race_key'].nunique()}R ({TRAIN_FROM}〜{VAL_FROM})")
model = train_lgbm(tr, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT, weight_col="w_field")

df = _apply_pred_prob_wt(model, df)
g = df.groupby("race_key")["pred_prob"]
df["zc"] = (df["pred_prob"] - g.transform("mean")) / g.transform("std").replace(0, 1)
pm = _load_payouts_wt(df["race_key"].unique().tolist())

splits = {
    "検証(val 6mo)":  df[(df["race_date"] >= VAL_FROM) & (df["race_date"] < TEST_FROM)],
    "テスト(test 3mo)": df[(df["race_date"] >= TEST_FROM) & (df["race_date"] < TEST_TO)],
}
for tag, d in splits.items():
    auc = roc_auc_score(d[TARGET_COL_WT], d["pred_prob"])
    print(f"  {tag}: {d['race_key'].nunique()}R  AUC={auc:.4f}")

# レース単位の本番2軸流し三連複ベット収支 + 条件
def build_race_table(d):
    rows = []
    for rk, grp in d.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3: continue
        fr = grp["frame_no"].astype(int).tolist(); probs = grp["pred_prob"].tolist()
        p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        fin = grp[grp["finish_order"] <= 3]; top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3: continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {})
        # 3連複2軸流し
        bp = len(thirds)*100; rp_ret = 0
        for t in thirds:
            c = frozenset((p1,p2,t))
            if c == top3: rp_ret += rp.get(("trio", c), 0)
        # 3連単2軸流し(p1->p2->t)
        bt = len(thirds)*100; rt_ret = 0
        for t in thirds:
            if order == (p1,p2,t): rt_ret += rp.get(("trifecta",(p1,p2,t)),0)
        rows.append(dict(n_riders=n, gap12=probs[0]-probs[1], ratio=probs[0]/(3.0/n),
                         z1=(probs[0]-np.mean(probs))/(np.std(probs) or 1),
                         puku_bet=bp, puku_ret=rp_ret, tan_bet=bt, tan_ret=rt_ret))
    return pd.DataFrame(rows)

RT = {tag: build_race_table(d) for tag, d in splits.items()}

def roi(d, col):
    b=d[f"{col}_bet"].sum(); return d[f"{col}_ret"].sum()/b if b else 0
def hit(d, col):
    return (d[f"{col}_ret"]>0).mean() if len(d) else 0

print("\n=== 全体ROI（2軸流し3点）===")
for tag, t in RT.items():
    print(f"  {tag}: 三連複 {roi(t,'puku'):.1%}(的中{hit(t,'puku'):.0%})  三連単 {roi(t,'tan'):.1%}(的中{hit(t,'tan'):.0%})  {len(t)}R")

# 条件グリッド: 検証で>105%を抽出 → テストで確認
conds = []
for nr in [5,6,7,8,9]:
    conds.append((f"n=={nr}", lambda d,nr=nr: d["n_riders"]==nr))
for lo,hi in [(.15,.25),(.25,1)]:
    conds.append((f"gap12[{lo},{hi})", lambda d,lo=lo,hi=hi:(d["gap12"]>=lo)&(d["gap12"]<hi)))
for lo,hi in [(0,1.1),(1.1,1.3)]:
    conds.append((f"ratio[{lo},{hi})", lambda d,lo=lo,hi=hi:(d["ratio"]>=lo)&(d["ratio"]<hi)))
for lo,hi in [(1.3,1.6),(1.6,3)]:
    conds.append((f"z1[{lo},{hi})", lambda d,lo=lo,hi=hi:(d["z1"]>=lo)&(d["z1"]<hi)))
conds.append(("n<=6&ratio<1.3", lambda d:(d["n_riders"]<=6)&(d["ratio"]<1.3)))
conds.append(("n<=6&gap12>=0.15", lambda d:(d["n_riders"]<=6)&(d["gap12"]>=0.15)))
conds.append(("n<=6&z1>=1.3", lambda d:(d["n_riders"]<=6)&(d["z1"]>=1.3)))

val, test = RT["検証(val 6mo)"], RT["テスト(test 3mo)"]
print("\n=== 条件別 検証→テスト（三連複2軸流し, min_n=30）===")
print(f"{'条件':<20}{'検証n':>6}{'検証ROI':>8}{'検証的中':>8}{'テストn':>7}{'テストROI':>9}{'テスト的中':>9}")
robust=[]
for name, fn in conds:
    mv, mt = fn(val), fn(test)
    if mv.sum()<30 or mt.sum()<30: continue
    rv, rt_ = roi(val[mv],'puku'), roi(test[mt],'puku')
    hv, ht_ = hit(val[mv],'puku'), hit(test[mt],'puku')
    flag = " ★" if rv>1.05 and rt_>1.0 else ""
    print(f"{name:<20}{int(mv.sum()):>6}{rv:>8.0%}{hv:>8.0%}{int(mt.sum()):>7}{rt_:>9.0%}{ht_:>9.0%}{flag}")
    if rv>1.05 and rt_>1.0: robust.append(name)
print(f"\n頑健条件(検証>105%&テスト>100%): {robust if robust else '該当なし'}")
