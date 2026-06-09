"""7車以上: 構造条件で勝てるレースを絞れるか（実効車数/1位抜け/会場別）

ユーザー仮説:
  ①実効車数: 下位がpred_probで離れ、contender(≥τ)が≤6なら実質小フィールド＝≤6車の妙味が出る?
  ②1位抜け: race_point(競走得点)の1-2位差が大きい本命圧倒レース
  ③会場別: 特定の競輪場が再現的に黒字か
pooled model・7+のみ・std3(三連複3点)。train(2023-07〜2026-02)→test(2026-03〜)。
両期間100%超かつ十分Rのみ採用候補。最終オッズ上限値。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict
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
        rp = sorted(g["race_point"].tolist(), reverse=True)
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        top3 = frozenset(int(x) for x in fin["frame_no"])
        po = pm.get(rk, {})
        s3 = [frozenset((fr[0], fr[1], x)) for x in fr[2:5]]
        hit = top3 in s3
        rows.append({
            "venue": rk.split("_")[1],
            "gap12": p[0]-p[1],
            "eff15": sum(1 for x in p if x >= 0.15),
            "eff20": sum(1 for x in p if x >= 0.20),
            "eff25": sum(1 for x in p if x >= 0.25),
            "rp_gap": (rp[0]-rp[1]) if len(rp) > 1 else 0,   # 競走得点 1-2位差
            "pay": (po.get(("trio", top3), 0) if hit else 0), "bet": len(s3)*100,
        })
    return rows


tr = collect("2023-07-01", "2026-02-28"); te = collect("2026-03-01", "2026-06-08")

def agg(rows, cond):
    s = [r for r in rows if cond(r)]
    return roi_summary([r["pay"] for r in s], [r["bet"] for r in s]), len(s)

def line(label, cond):
    s1, n1 = agg(tr, cond); s2, n2 = agg(te, cond)
    flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
    print(f"  {label:<24} TR {n1:>5}R {s1['roi']:>5.0%} | TE {n2:>5}R {s2['roi']:>5.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}] {flag}")

print(f"\n{'='*82}\n  7+ 構造条件で勝てるレース絞り込み  TRAIN {len(tr)}R / TEST {len(te)}R (std3・上限値)\n{'='*82}")
print("\n  ① 実効車数（pred_prob≥τ の contender数）が小さい＝実質小フィールド")
for τ, key in [(0.15, "eff15"), (0.20, "eff20"), (0.25, "eff25")]:
    for thr in [6, 5, 4]:
        line(f"contender(≥{τ})<={thr}", lambda r, k=key, thr=thr: r[k] <= thr)
print("\n  ② 1位抜け（競走得点 1-2位差 rp_gap）")
for thr in [5, 10, 15, 20]:
    line(f"race_point差>={thr}", lambda r, thr=thr: r["rp_gap"] >= thr)
print("\n  ③ 実効≤5 かつ 1位抜け の複合")
line("eff20<=5 & rp_gap>=10", lambda r: r["eff20"] <= 5 and r["rp_gap"] >= 10)

print("\n  ③ 会場別 std3 ROI（test上位・train併記。再現性チェック）")
vt = defaultdict(lambda: {"tr": [], "te": []})
for r in tr: vt[r["venue"]]["tr"].append(r)
for r in te: vt[r["venue"]]["te"].append(r)
res = []
for v, dd in vt.items():
    if len(dd["te"]) < 30: continue
    s1, _ = roi_summary([r["pay"] for r in dd["tr"]], [r["bet"] for r in dd["tr"]]), 0
    s1 = roi_summary([r["pay"] for r in dd["tr"]], [r["bet"] for r in dd["tr"]])
    s2 = roi_summary([r["pay"] for r in dd["te"]], [r["bet"] for r in dd["te"]])
    res.append((v, len(dd["tr"]), s1["roi"], len(dd["te"]), s2["roi"], s2["ci_lo"], s2["ci_hi"]))
res.sort(key=lambda x: -x[4])
print(f"  {'会場':<6}{'TR_R':>6}{'TR_ROI':>8}{'TE_R':>6}{'TE_ROI':>8}{'TE_CI':>16}{'再現':>6}")
for v, n1, r1, n2, r2, lo, hi in res[:10]:
    flag = "★" if (r1 > 1.0 and r2 > 1.0) else ""
    print(f"  {v:<6}{n1:>6}{r1:>7.0%}{n2:>6}{r2:>7.0%} [{lo:>4.0%},{hi:>4.0%}]{flag:>5}")
print("\n  ※ ★再現/★ = train・test とも100%超。無ければ7+はこれら条件でも黒字化せず。")
