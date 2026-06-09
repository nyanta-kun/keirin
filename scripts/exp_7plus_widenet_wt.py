"""7車以上: 「波乱レース選別 × 広め三連単(20点級)で高配当狙い」の検証

ユーザー戦略: 低配当(鉄板)レースは見送り、波乱含みレースに20点程度広げ、
≥5000(レースとして~200%)/狙いは≥10000 を取りトータル回収率100%超を目指す。

選別: 鉄板(高gap12/堅い)を避け、波乱含み(open)レースのみ。
買い目(三連単フォーメーション):
  box4   : 上位4車の三連単総流し C(4,3)*6=24点
  f_3x4x5: 1着∈top3 / 2着∈top4 / 3着∈top5（distinct・~30点）
  f_h2   : 1着∈{top1,top2} / 2-3着=top5のperm（~24点）
pooled model・7+のみ・最終オッズ上限値・train→test(OOS)。
ROI＋的中率＋的中時の配当(≥5000/≥10000)捕捉率を見る。
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
        po = pm.get(rk, {})
        def tri(o): return po.get(("trifecta", o), 0)

        def form(heads, seconds, thirds):
            cs = [(a, b, c) for a in heads for b in seconds for c in thirds if len({a, b, c}) == 3]
            cs = list(dict.fromkeys(cs))
            hit = order in cs
            return (hit, tri(order) if hit else 0, len(cs) * 100)
        box4 = form(fr[:4], fr[:4], fr[:4])
        f345 = form(fr[:3], fr[:4], fr[:5])
        fh2 = form(fr[:2], fr[:5], fr[:5])
        rows.append({"gap12": p[0]-p[1], "top3_sum": p[0]+p[1]+p[2],
                     "box4": box4, "f345": f345, "fh2": fh2})
    return rows


tr = collect("2023-07-01", "2026-02-28"); te = collect("2026-03-01", "2026-06-08")

def agg(rows, cond, k):
    s = [r for r in rows if cond(r)]
    pays = [r[k][1] for r in s]; bets = [r[k][2] for r in s]
    res = roi_summary(pays, bets)
    hi5 = sum(1 for x in pays if x >= 5000); hi10 = sum(1 for x in pays if x >= 10000)
    ppr = (sum(bets)/len(bets)/100) if bets else 0
    return res, len(s), ppr, hi5, hi10

def line(label, cond):
    print(f"\n  ▼ {label}")
    for k, nm in [("box4", "box4(24)"), ("f345", "1着3x2着4x3着5"), ("fh2", "1着2頭xBOX5")]:
        s1, n1, pp1, _, _ = agg(tr, cond, k)
        s2, n2, pp2, h5, h10 = agg(te, cond, k)
        flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
        print(f"    {nm:<14}{pp2:>4.0f}点 TR {n1:>5}R {s1['roi']:>5.0%} | TE {n2:>5}R {s2['roi']:>5.0%} "
              f"[{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}] 的中{s2['hit_rate']:>4.0%}(≥5k:{h5}/≥10k:{h10}) {flag}")

print(f"\n{'='*94}\n  7+ 波乱選別×広め三連単 ROI  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）\n{'='*94}")
line("ALL(7+)", lambda r: True)
line("波乱: gap12<0.10(本命弱)", lambda r: r["gap12"] < 0.10)
line("波乱: gap12<0.06", lambda r: r["gap12"] < 0.06)
import statistics
med = statistics.median([r["top3_sum"] for r in tr])
line(f"波乱: top3_sum<中央({med:.2f})", lambda r, m=med: r["top3_sum"] < m)
p25 = sorted(r["top3_sum"] for r in tr)[len(tr)//4]
line(f"波乱: top3_sum<=p25({p25:.2f})", lambda r, v=p25: r["top3_sum"] <= v)
print("\n  ※ 20点級でも ROI=Σ払戻/Σ投資 は控除率に支配される。★再現(TR/TE>100%&TE≥30R)が出るかが焦点。")
