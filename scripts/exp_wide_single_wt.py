"""≤6車 ワイド1点推奨の検討（的中率最大化の観点）。

ユーザー方針: ワイド1点はガミを度外視（低オッズ見送りは別途）。
純粋に「的中率を求めた1点推奨」が成立するかを検証する。

ワイド = 2車が共に3着以内なら的中（1レースに当たりペアは C(3,2)=3 通り）。
候補1点:
  W12 = 指数1位+2位（pred_prob top2 ＝ 同時top3確率が最大と期待）
  W13 = 指数1位+3位 / W23 = 指数2位+3位
選別シグナル: top2_sum(=p1+p2)・joint(=p1*p2)・gap12・top3_sum(波乱)・n_riders。

モデル=lgbm_wt_eval（OOS評価用）。TRAIN 2023-07〜2026-02 / TEST 2026-03〜。
payout=quinellaPlace 最終オッズ×100＝上限値。的中率は最終データでもlive再現性高
（欠車は事後だが top3 判定は between(1,3)）。
"""
import sys
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from roi_robustness_wt import roi_summary


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes.between(2, 6)].index)]  # ≤6車（2車未満除外）
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3:   # 着順欠損/欠車で3着内が確定しないレースは除外
            continue
        rp = pm.get(rk, {})

        def wide(a, b):
            pair = frozenset((a, b))
            hit = pair.issubset(top3)
            pay = rp.get(("quinellaPlace", pair), 0) if hit else 0
            return (hit, pay, 100)

        p1, p2, p3 = p[0], p[1], p[2]
        # W12ペアのワイドオッズ（事前確定・的中有無に関係なく取得＝足切り検討用）
        w12_odds = rp.get(("quinellaPlace", frozenset((fr[0], fr[1]))), None)
        rows.append({
            "w12_odds": (w12_odds / 100.0) if w12_odds else None,
            "n_riders": n,
            "gap12": p1 - p2,
            "ratio": p1 / (3.0 / n),
            "top2_sum": p1 + p2,
            "top3_sum": p1 + p2 + p3,
            "joint12": p1 * p2,
            "W12": wide(fr[0], fr[1]),
            "W13": wide(fr[0], fr[2]),
            "W23": wide(fr[1], fr[2]),
            # 参考: 指数1位を含む2点流し（1点ではないがhit率上限の目安）
            "p1_top3": fr[0] in top3,
        })
    return rows


def agg(rows, cond, key):
    sub = [r for r in rows if cond(r)]
    pays = [r[key][1] for r in sub]
    bets = [r[key][2] for r in sub]
    return roi_summary(pays, bets), len(sub)


def line(label, s, n):
    return (f"  {label:<22}{n:>6}{s['hit_rate']:>9.1%}{s['roi']:>8.0%}"
            f" [{s['ci_lo']:>5.0%},{s['ci_hi']:>5.0%}]{s['median_hit']:>8,.0f}円")


tr = collect("2023-07-01", "2026-02-28")
te = collect("2026-03-01", "2026-06-08")
print(f"\n≤6車 ワイド1点  TRAIN {len(tr)}R / TEST {len(te)}R（payout=最終オッズ上限値）")

# ---- Part1: どの1点が最も的中率が高いか（全レース）----
print(f"\n【Part1】候補1点の的中率（全≤6車・TEST）")
print(f"  {'1点':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
for key, lab in [("W12", "指数1-2位"), ("W13", "指数1-3位"), ("W23", "指数2-3位")]:
    s, n = agg(te, lambda r: True, key)
    print(line(lab, s, n))
sp1 = sum(r["p1_top3"] for r in te) / len(te)
print(f"  参考: 指数1位のtop3率 = {sp1:.1%}")

# ---- Part2: 的中率でレース選別が可能か（top2_sum 四分位・TRAINカットをTEST適用）----
print(f"\n【Part2】W12を選別シグナルで層別（TRAINカット→TEST適用・★=的中率で選別可）")
for sig in ["top2_sum", "joint12", "gap12", "top3_sum"]:
    cuts = np.quantile([r[sig] for r in tr], [0.25, 0.5, 0.75])
    print(f"\n  ◆ signal={sig}  cuts={[round(c,3) for c in cuts]}")
    print(f"  {'帯':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
    bins = [("Q1_low", lambda r, c=cuts: r[sig] < c[0]),
            ("Q2", lambda r, c=cuts: c[0] <= r[sig] < c[1]),
            ("Q3", lambda r, c=cuts: c[1] <= r[sig] < c[2]),
            ("Q4_high", lambda r, c=cuts: r[sig] >= c[2])]
    for lab, cond in bins:
        s, n = agg(te, cond, "W12")
        print(line(lab, s, n))

# ---- Part3: 車数別（小フィールドほどワイドは当たりやすい）----
print(f"\n【Part3】W12 車数別（TEST）")
print(f"  {'車数':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
for nn in [4, 5, 6]:
    s, n = agg(te, lambda r, k=nn: r["n_riders"] == k, "W12")
    print(line(f"{nn}車立て", s, n))

# ---- Part4: 高的中率ゾーンの抽出（top2_sm 上位 + 車数）----
print(f"\n【Part4】高的中率ゾーン抽出（W12・TEST・的中率優先）")
print(f"  {'条件':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
t2_hi = np.quantile([r["top2_sum"] for r in tr], 0.75)
t2_top10 = np.quantile([r["top2_sum"] for r in tr], 0.90)
conds = [
    ("top2_sum≥Q3", lambda r: r["top2_sum"] >= t2_hi),
    ("top2_sum≥P90", lambda r: r["top2_sum"] >= t2_top10),
    ("top2_sum≥Q3 & ≤5車", lambda r: r["top2_sum"] >= t2_hi and r["n_riders"] <= 5),
    ("gap12≥0.15", lambda r: r["gap12"] >= 0.15),
    ("top3_sum≥Q4_chalk", lambda r: r["top3_sum"] >= np.quantile([x["top3_sum"] for x in tr], 0.75)),
]
for lab, cond in conds:
    s, n = agg(te, cond, "W12")
    print(line(lab, s, n))

# ---- Part5: W12のワイドオッズ帯別 的中率（低オッズ足切り後に残る的中率）----
print(f"\n【Part5】W12をワイドオッズ帯で層別（オッズは事前確定・足切り後の的中率を見る・TEST）")
print(f"  {'オッズ帯':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
bands = [("<1.5倍", lambda r: r["w12_odds"] is not None and r["w12_odds"] < 1.5),
         ("1.5-2.0倍", lambda r: r["w12_odds"] is not None and 1.5 <= r["w12_odds"] < 2.0),
         ("2.0-3.0倍", lambda r: r["w12_odds"] is not None and 2.0 <= r["w12_odds"] < 3.0),
         ("3.0-5.0倍", lambda r: r["w12_odds"] is not None and 3.0 <= r["w12_odds"] < 5.0),
         ("≥5.0倍", lambda r: r["w12_odds"] is not None and r["w12_odds"] >= 5.0)]
for lab, cond in bands:
    s, n = agg(te, cond, "W12")
    print(line(lab, s, n))
# 足切り（≥X倍のみ買う）後の累積的中率・ROI
print(f"\n  ◆ 「オッズ≥X倍のみ購入」した場合の残存的中率・ROI（TEST）")
print(f"  {'足切り':<22}{'R':>6}{'的中率':>9}{'ROI':>8}{'95%CI':>14}{'中央払戻':>9}")
for x in [1.0, 1.5, 2.0, 2.5, 3.0]:
    s, n = agg(te, lambda r, xx=x: r["w12_odds"] is not None and r["w12_odds"] >= xx, "W12")
    print(line(f"≥{x}倍", s, n))
print()
