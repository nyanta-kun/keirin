"""波乱スコア(top3_sum/upset_tier)によるステーク傾斜の検証（方針A）

フラット(全レース同額) vs 波乱帯で傾斜配分 を TRAIN→TEST(OOS) で比較。
ROI・総損益・ブートストラップ95%CI を併記し、傾斜が頑健にROIを改善するか判定する。

honest OOS のため holdout 評価モデル lgbm_wt_eval（train<2026-03 / test>=2026-03）を使用。
SS/S/A をプール。投資はレース可変(点数×100)。払戻=最終オッズ=上限値。
傾斜は「TRAINで決めた帯別倍率」をTESTに適用（policyのtest漏洩なし）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.strategy_wt import upset_tier, UPSET_TIERS
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)
from roi_robustness_wt import roi_summary

MODEL = "lgbm_wt_eval"   # holdout評価モデル（test>=2026-03 はOOS）


def collect(date_from, date_to):
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 3: continue
        p = g["pred_prob"].tolist()
        tier = _assign_tier(p[0]-p[1], p[0]/(3/n))
        if tier is None: continue
        fr = g["frame_no"].astype(int).tolist(); p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        if not thirds: continue
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3: continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {}); pay = 0
        for x in thirds:
            if tier == "SS":
                if order == (p1, p2, x): pay = rp.get(("trifecta", (p1, p2, x)), 0); break
            else:
                if frozenset((p1, p2, x)) == top3: pay = rp.get(("trio", frozenset((p1, p2, x))), 0); break
        rows.append({"utier": upset_tier(p[0]+p[1]+p[2]),
                     "pay": float(pay), "bet": len(thirds) * 100.0})
    return rows


POLICIES = {
    "flat(現行)":            {"Q1_loose": 1, "Q2": 1, "Q3": 1, "Q4_chalk": 1},
    "gate Q1+Q2のみ":        {"Q1_loose": 1, "Q2": 1, "Q3": 0, "Q4_chalk": 0},
    "tilt 2/1/0.5/0":        {"Q1_loose": 2, "Q2": 1, "Q3": 0.5, "Q4_chalk": 0},
    "推奨 2/1/0/0":          {"Q1_loose": 2, "Q2": 1, "Q3": 0, "Q4_chalk": 0},
    "tilt 3/1/0/0":          {"Q1_loose": 3, "Q2": 1, "Q3": 0, "Q4_chalk": 0},
    "Q1_looseのみ":          {"Q1_loose": 1, "Q2": 0, "Q3": 0, "Q4_chalk": 0},
}


def apply_policy(rows, mult):
    pays, bets = [], []
    for r in rows:
        m = mult[r["utier"]]
        if m <= 0:
            continue
        pays.append(r["pay"] * m)
        bets.append(r["bet"] * m)
    return pays, bets


def show(name, rows):
    print(f"\n{'='*96}\n  【{name}】 対象 {len(rows)}R（SS/S/A・eval model OOS）\n{'='*96}")
    print(f"  {'方針':<18}{'購入R':>6}{'投資単位':>9}{'ROI':>8}{'95%CI':>20}{'最大除ROI':>11}{'相対損益指数':>12}")
    print(f"  {'-'*94}")
    # 相対損益指数: flat の総損益を1.0とした比（同一“基準額”比較のため bet 合計で正規化した損益率）
    base = None
    for pname, mult in POLICIES.items():
        pays, bets = apply_policy(rows, mult)
        s = roi_summary(pays, bets)
        pl_per_unit = (sum(pays) - sum(bets)) / (sum(bets) if sum(bets) else 1)  # =ROI-1
        n_eff = len(pays)
        bet_units = sum(bets) / 100
        print(f"  {pname:<18}{n_eff:>6}{bet_units:>8.0f}点{s['roi']:>7.0%} "
              f"[{s['ci_lo']:>5.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>10.0%}{pl_per_unit:>11.0%}")


train = collect("2023-07-01", "2026-02-28")
test = collect("2026-03-01", "2026-06-08")
show("TRAIN 2023-07〜2026-02", train)
show("TEST/OOS 2026-03〜", test)
print("\n  ※ ROIは最終オッズ=上限値。傾斜は帯別倍率(TRAINで設計)をTESTに適用。")
print("    『購入R』はstake>0のレース数、『投資単位』は総点数(=資金量の代理)。")
