"""7車以上: 二車複・ワイド での収益化確認（三連複が全滅のため2車券も検証）"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from roi_robustness_wt import roi_summary


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 7: continue
        p = g["pred_prob"].tolist(); gap12 = p[0]-p[1]
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        order = [int(x) for x in fin.sort_values("finish_order")["frame_no"]]
        pair12 = frozenset(order[:2]); top3 = frozenset(order)
        rp = pm.get(rk, {})
        # 二車複: 指数top2（1点）/ top3 box（3点）
        _qt2_hit = frozenset((fr[0], fr[1])) == pair12
        q_top2 = (_qt2_hit, rp.get(("quinella", pair12), 0) if _qt2_hit else 0, 100)
        qbox = [frozenset(c) for c in itertools.combinations(fr[:3], 2)]
        q_box3 = (pair12 in qbox, rp.get(("quinella", pair12), 0) if pair12 in qbox else 0, 300)
        # ワイド: 指数top2（1点・両者3着内）/ top3 box（3点・的中分合算）
        w_t2 = frozenset((fr[0], fr[1]))
        w_top2 = (w_t2.issubset(top3), rp.get(("quinellaPlace", w_t2), 0) if w_t2.issubset(top3) else 0, 100)
        wpay = sum(rp.get(("quinellaPlace", frozenset(c)), 0) for c in itertools.combinations(fr[:3], 2)
                   if frozenset(c).issubset(top3))
        w_box3 = (wpay > 0, wpay, 300)
        rows.append({"gap12": gap12, "q_top2": q_top2, "q_box3": q_box3, "w_top2": w_top2, "w_box3": w_box3})
    return rows


def agg(rows, cond, key):
    sub = [r for r in rows if cond(r)]
    pays = [r[key][1] for r in sub]; bets = [r[key][2] for r in sub]
    return roi_summary(pays, bets), len(pays)


tr = collect("2023-07-01", "2026-02-28"); te = collect("2026-03-01", "2026-06-08")
CONDS = {"ALL(7+)": lambda r: True, "gap12>=0.15": lambda r: r["gap12"] >= 0.15,
         "gap12>=0.20": lambda r: r["gap12"] >= 0.20}
print(f"\n7車以上 2車券  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）")
print(f"  {'条件':<14}{'券種':<10}{'TR_ROI':>8}{'TR_的中':>8}{'TE_ROI':>8}{'TE_的中':>8}{'TE_CI':>18}")
for cn, cond in CONDS.items():
    for key, lab in [("q_top2", "二車複top2"), ("q_box3", "二車複box3"), ("w_top2", "ワイドtop2"), ("w_box3", "ワイドbox3")]:
        s1, _ = agg(tr, cond, key); s2, _ = agg(te, cond, key)
        print(f"  {cn:<14}{lab:<10}{s1['roi']:>7.0%}{s1['hit_rate']:>8.0%}{s2['roi']:>7.0%}{s2['hit_rate']:>8.0%}"
              f" [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]")
    print(f"  {'-'*70}")
