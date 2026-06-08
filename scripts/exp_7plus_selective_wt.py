"""7車以上: 選別を極端に強め＋買い目を絞って回収率100%超を探す（再現性ファースト）

的中率でなくROIが目的。「購入レースを絞る×買い目を絞る」で控除率(〜75%)を超える
ポケットが7+に存在するかを、極端な閾値スイープ＋ナロー買いで探索。
- 選別軸: gap12(本命の支配度) / top3_sum(レースの開き具合・7+内percentile) / ratio
- 買い目: t1=三連複1点(指数top3) / t2=2点 / std3=3点 / tri1=三連単1点(指数1→2→3)
- pooled model(lgbm_wt・7+でAUC高)。payout=最終オッズ上限値。train→test(OOS)。
- 小標本(test<30)はノイズとして区別。両期間100%超のみ採用候補。
"""
import sys, itertools, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _load_payouts_wt
from roi_robustness_wt import roi_summary

model = load_model("lgbm_wt")


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 7: continue
        p = g["pred_prob"].tolist()
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order); rp = pm.get(rk, {})
        def trio(c): return rp.get(("trio", c), 0)
        def tri(o): return rp.get(("trifecta", o), 0)
        t1c = [frozenset((fr[0], fr[1], fr[2]))]
        t2c = [frozenset((fr[0], fr[1], fr[2])), frozenset((fr[0], fr[1], fr[3]))]
        s3c = [frozenset((fr[0], fr[1], x)) for x in fr[2:5]]
        def mk(combos):
            h = top3 in combos; return (h, trio(top3) if h else 0, len(combos)*100)
        tri1_hit = order == (fr[0], fr[1], fr[2])
        rows.append({"gap12": p[0]-p[1], "ratio": p[0]/(3/n), "top3_sum": p[0]+p[1]+p[2],
                     "t1": mk(t1c), "t2": mk(t2c), "std3": mk(s3c),
                     "tri1": (tri1_hit, tri(order) if tri1_hit else 0, 100)})
    return rows


tr = collect("2023-07-01", "2026-02-28"); te = collect("2026-03-01", "2026-06-08")
# top3_sum percentile cut（trainで算出）
ts = sorted(r["top3_sum"] for r in tr)
def pct(q): return ts[int(len(ts)*q)]
P = {"p25": pct(0.25), "p10": pct(0.10), "p05": pct(0.05)}

def agg(rows, cond, k):
    s = [r for r in rows if cond(r)]
    pays = [r[k][1] for r in s if r[k][2] > 0]; bets = [r[k][2] for r in s if r[k][2] > 0]
    return roi_summary(pays, bets), len(pays)

def line(label, cond):
    print(f"\n  ▼ {label}")
    for k in ["std3", "t2", "t1", "tri1"]:
        s1, n1 = agg(tr, cond, k); s2, n2 = agg(te, cond, k)
        flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
        print(f"    {k:<6} TR {n1:>5}R {s1['roi']:>5.0%} | TE {n2:>5}R {s2['roi']:>5.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}] {flag}")

print(f"\n{'='*78}\n  7+ 選別×ナロー買い ROI探索  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）")
print(f"  買い目: std3=三連複3点/t2=2点/t1=1点(指数top3)/tri1=三連単1点(1→2→3)\n{'='*78}")
# 本命支配度を極端に
for g in [0.20, 0.25, 0.30, 0.40]:
    line(f"gap12>={g}", lambda r, g=g: r["gap12"] >= g)
# レースの開き(波乱)を極端に
for q, v in P.items():
    line(f"top3_sum<={q}({v:.2f})", lambda r, v=v: r["top3_sum"] <= v)
# 複合: 強本命 × 1点 / 開き × 1点
line("gap12>=0.30 & top3_sum>=中央(堅い)", lambda r: r["gap12"] >= 0.30 and r["top3_sum"] >= pct(0.5))
line("top3_sum<=p10 & gap12>=0.15", lambda r: r["top3_sum"] <= P["p10"] and r["gap12"] >= 0.15)
print("\n  ※ ★再現 = TR/TE とも100%超 かつ test≥30R。無ければ7+は選別/絞り込みでも控除率を超えないと確定。")
