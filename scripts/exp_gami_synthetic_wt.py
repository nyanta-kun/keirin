"""ガミ判定: 最安レグ vs 合成オッズ どちらで振り分けるべきか（≤6車）

現行ガミ3段階: 3点(SS=3連単/S・A=3連複)のうち**最安1点**の朝オッズで振り分け
  最安<3倍→見送り / 3〜5倍→Bランク / ≥5倍→推奨。
ユーザー提案: 束全体の **合成オッズ = 1/Σ(1/oᵢ)** で判断すべきか。
  合成オッズは数学的に常に 最安レグ以下で、3点全体の市場的中確率(×控除率)を反映する。

検証(lgbm_wt_eval・OOS・払戻=最終オッズ=上限値・TR 2023-07〜2026-02 / TE 2026-03〜):
  A. 最安レグ と 合成オッズ の相関・現行バケット×合成の分布。
  B. ROI を「最安レグ四分位」vs「合成オッズ四分位」で層別＝どちらが ROI をきれいに単調分離するか。
  C. 決定的: 両者が食い違うレースの実ROI。
     - 現行推奨(最安≥5)だが合成低 → 降格すべきか
     - 現行見送り(最安<3)だが合成高 → 過剰スキップか
     - 現行B(最安∈[3,5))を合成で分割 → 高合成Bは推奨に上げるべきか
  D. 合成オッズ閾値スイープ: 推奨=合成≥thr の ROI/購入R を現行(最安≥5)と比較。
規律: roi_summary(bootstrap CI・最大払戻除去)。3点が揃うレース(thirds≥3)のみ。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)
from roi_robustness_wt import roi_summary

MODEL = "lgbm_wt_eval"


def collect(f, t):
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist(); gap = p[0] - p[1]; ratio = p[0] / (3 / n)
        tier = _assign_tier(gap, ratio)
        if tier is None:
            continue
        fr = g["frame_no"].astype(int).tolist(); p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        if len(thirds) < 3:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3:
            continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {})
        odds_list, pay, hit = [], 0, False
        for x in thirds:
            if tier == "SS":
                o = rp.get(("trifecta", (p1, p2, x)))
                if order == (p1, p2, x):
                    pay = o or 0; hit = True
            else:
                combo = frozenset((p1, p2, x))
                o = rp.get(("trio", combo))
                if combo == top3:
                    pay = o or 0; hit = True
            if o:
                odds_list.append(o / 100.0)
        if len(odds_list) < 3:        # 全レグのオッズが揃う場合のみ（公平比較）
            continue
        min_leg = min(odds_list)
        syn = 1.0 / sum(1.0 / o for o in odds_list)   # 合成オッズ
        rows.append({"tier": tier, "min_leg": min_leg, "syn": syn,
                     "pay": float(pay), "bet": len(odds_list) * 100, "hit": hit})
    return rows


def _roi(rows):
    return roi_summary([r["pay"] for r in rows], [r["bet"] for r in rows]), len(rows)


def partA(te):
    ml = np.array([r["min_leg"] for r in te]); sy = np.array([r["syn"] for r in te])
    print(f"\n{'='*92}\n  A. 最安レグ vs 合成オッズ（TEST {len(te)}R）\n{'='*92}")
    print(f"  相関(min_leg, syn) = {np.corrcoef(ml, sy)[0,1]:.3f}   "
          f"合成は常に最安以下: syn≤min_leg 成立率 {(sy<=ml+1e-9).mean():.1%}")
    print(f"  min_leg 中央値 {np.median(ml):.2f} / 合成 中央値 {np.median(sy):.2f}")
    # 現行バケット × 合成三分位の分布
    print(f"\n  現行(最安レグ)バケット内の 合成オッズ 分布:")
    print(f"    {'現行バケット':<16}{'R':>5}{'合成 中央':>10}{'合成 25%':>10}{'合成 75%':>10}")
    for lab, lo, hi in [("見送り <3", 0, 3), ("B [3,5)", 3, 5), ("推奨 ≥5", 5, 1e9)]:
        s = sy[(ml >= lo) & (ml < hi)]
        if len(s):
            print(f"    {lab:<16}{len(s):>5}{np.median(s):>10.2f}{np.percentile(s,25):>10.2f}{np.percentile(s,75):>10.2f}")


def partB(tr, te):
    print(f"\n{'='*92}\n  B. ROI層別: 最安レグ四分位 vs 合成オッズ四分位（どちらがROIを単調分離するか）\n{'='*92}")
    for key, name in [("min_leg", "最安レグ"), ("syn", "合成オッズ")]:
        vals = np.array([r[key] for r in tr])
        cuts = np.quantile(vals, [0.25, 0.5, 0.75])
        print(f"\n  ◆ {name} 四分位（TRAINカット {cuts[0]:.2f}/{cuts[1]:.2f}/{cuts[2]:.2f} を TEST 適用）")
        print(f"    {'帯':<10}{'R':>5}{'的中':>7}{'ROI':>7}{'95%CI':>18}{'最大除':>8}{'中央払戻':>10}")
        labels = ["Q1_安", "Q2", "Q3", "Q4_高"]
        for i, lab in enumerate(labels):
            lo = cuts[i-1] if i > 0 else -1e9
            hi = cuts[i] if i < 3 else 1e9
            sub = [r for r in te if (r[key] > lo if i > 0 else True) and (r[key] <= hi if i < 3 else True)]
            if not sub:
                continue
            s, nR = _roi(sub)
            print(f"    {lab:<10}{nR:>5}{s['hit_rate']:>7.0%}{s['roi']:>7.0%} "
                  f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>8.0%}{s['median_hit']:>8,.0f}円")


def partC(te):
    print(f"\n{'='*92}\n  C. 決定的: 現行ルールと合成オッズが食い違うレースの実ROI（TEST）\n{'='*92}")
    syn_med = np.median([r["syn"] for r in te])
    sets = [
        ("現行=推奨(最安≥5) 全体", lambda r: r["min_leg"] >= 5),
        ("  └ うち 合成<中央値→降格候補", lambda r, m=syn_med: r["min_leg"] >= 5 and r["syn"] < m),
        ("  └ うち 合成≥中央値", lambda r, m=syn_med: r["min_leg"] >= 5 and r["syn"] >= m),
        ("現行=見送り(最安<3) 全体", lambda r: r["min_leg"] < 3),
        ("  └ うち 合成≥中央値→過剰スキップ?", lambda r, m=syn_med: r["min_leg"] < 3 and r["syn"] >= m),
        ("現行=B(最安∈[3,5)) 全体", lambda r: 3 <= r["min_leg"] < 5),
        ("  └ うち 合成≥中央値→昇格候補", lambda r, m=syn_med: 3 <= r["min_leg"] < 5 and r["syn"] >= m),
        ("  └ うち 合成<中央値", lambda r, m=syn_med: 3 <= r["min_leg"] < 5 and r["syn"] < m),
    ]
    print(f"  (合成オッズ TEST中央値 = {syn_med:.2f})")
    print(f"    {'集合':<36}{'R':>5}{'的中':>7}{'ROI':>7}{'95%CI':>18}{'最大除':>8}")
    for name, cond in sets:
        sub = [r for r in te if cond(r)]
        if not sub:
            print(f"    {name:<36}{'0':>5}")
            continue
        s, nR = _roi(sub)
        print(f"    {name:<36}{nR:>5}{s['hit_rate']:>7.0%}{s['roi']:>7.0%} "
              f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>8.0%}")


def partD(tr, te):
    print(f"\n{'='*92}\n  D. 推奨=合成オッズ≥thr スイープ（現行=最安≥5 と比較・TR→TE）\n{'='*92}")
    cur_tr = [r for r in tr if r["min_leg"] >= 5]; cur_te = [r for r in te if r["min_leg"] >= 5]
    s1, n1 = _roi(cur_tr); s2, n2 = _roi(cur_te)
    print(f"    {'方式':<22}{'TR_ROI(R)':>14}{'TE_ROI(R)':>14}{'TE的中':>7}{'TE_CI':>16}{'TE最大除':>9}")
    print(f"    {'[現行]最安≥5':<22}{s1['roi']:>6.0%}({n1:>5}){s2['roi']:>7.0%}({n2:>4})"
          f"{s2['hit_rate']:>7.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{s2['roi_ex_max']:>9.0%}")
    for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        a = [r for r in tr if r["syn"] >= thr]; b = [r for r in te if r["syn"] >= thr]
        if not a or not b:
            continue
        sa, na = _roi(a); sb, nb = _roi(b)
        print(f"    {'合成≥'+str(thr):<22}{sa['roi']:>6.0%}({na:>5}){sb['roi']:>7.0%}({nb:>4})"
              f"{sb['hit_rate']:>7.0%} [{sb['ci_lo']:>4.0%},{sb['ci_hi']:>5.0%}]{sb['roi_ex_max']:>9.0%}")
    print("\n  ※ 同程度の購入Rで合成≥thr が 最安≥5 よりROI高なら、合成オッズ判定が優位。")


if __name__ == "__main__":
    print("collecting TRAIN...", flush=True)
    tr = collect("2023-07-01", "2026-02-28")
    print(f"  TRAIN {len(tr)}R", flush=True)
    print("collecting TEST...", flush=True)
    te = collect("2026-03-01", "2026-06-08")
    print(f"  TEST {len(te)}R", flush=True)
    partA(te)
    partB(tr, te)
    partC(te)
    partD(tr, te)
