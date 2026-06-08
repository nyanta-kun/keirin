"""小標本ROIの頑健提示（H-4）: 層別ROIをブートストラップ95%CI＋最大払戻除去後ROIで併記。

点推定ROI（特にSS等 N<100 層）は単発万車券に支配されCIが極端に広い。
本スクリプトは本番 SS/S/A 層別で per-race の払戻を集め、
  - 点推定ROI
  - ブートストラップ95%CI（レース再標本化）
  - 最大払戻1本/2本 除去後ROI（単発依存の確認）
  - 的中レース数・的中払戻中央値
を併記する。採否は CI・順序安定性・中央値で判断すべき（点推定の絶対値を訴求しない）。
払戻=wt_odds最終オッズ＝上限値。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)

def roi_summary(payouts: list[float], bets: list[float],
                n_boot: int = 2000, seed: int = 42) -> dict:
    """per-race (払戻, 投資) から ROI 統計を計算（純粋関数）。

    bet はレースごとに可変（小フィールドは点数<3＝投資<300円。run_tiered と整合）。
    ROI = Σ払戻 / Σ投資。ブートストラップは (払戻,投資) ペアをレース単位で再標本化。
    """
    pay = np.asarray(payouts, dtype=float)
    bet = np.asarray(bets, dtype=float)
    n = len(pay)
    if n == 0 or bet.sum() == 0:
        return {"n": n, "hits": 0, "hit_rate": 0.0, "roi": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "roi_ex_max": 0.0, "roi_ex_top2": 0.0,
                "median_hit": 0.0}
    roi = pay.sum() / bet.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = pay[idx].sum(axis=1) / bet[idx].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    # 払戻の大きい順にレースを除去（ペアで除く）
    order = np.argsort(pay)
    def _roi_drop(k):
        keep = order[:-k] if k > 0 else order
        return pay[keep].sum() / bet[keep].sum() if bet[keep].sum() > 0 else 0.0
    roi_ex_max = _roi_drop(1) if n > 1 else 0.0
    roi_ex_top2 = _roi_drop(2) if n > 2 else 0.0
    hits_arr = pay[pay > 0]
    return {
        "n": n, "hits": int((pay > 0).sum()), "hit_rate": (pay > 0).mean(),
        "roi": roi, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "roi_ex_max": float(roi_ex_max), "roi_ex_top2": float(roi_ex_top2),
        "median_hit": float(np.median(hits_arr)) if len(hits_arr) else 0.0,
    }


def collect_tier_payouts(date_from: str, date_to: str | None, model_name="lgbm_wt"):
    model = load_model(model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    # run_tiered_backtest_wt と同一条件: thirds≥1（4-5車も含む）・投資=点数×100
    tiers: dict[str, list] = {"SS": [], "S": [], "A": []}
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 3: continue
        p = g["pred_prob"].tolist(); tier = _assign_tier(p[0]-p[1], p[0]/(3/n))
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
        bet = len(thirds) * 100   # 点数×100（小フィールドは<300）
        tiers[tier].append((pay, bet))
    return tiers


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default="2026-03-01")
    ap.add_argument("--to", dest="date_to", default=None)
    args = ap.parse_args()

    tiers = collect_tier_payouts(args.date_from, args.date_to)

    def summ(pairs):
        return roi_summary([p for p, _ in pairs], [b for _, b in pairs])

    print(f"\n{'='*94}\n  層別ROI 頑健提示（{args.date_from}〜{args.date_to or 'latest'}・最終オッズ=上限値）\n{'='*94}")
    print(f"  {'層':<5}{'R':>5}{'的中率':>8}{'ROI':>9}{'95%CI':>22}{'ROI(最大除)':>12}{'ROI(上位2除)':>13}{'中央払戻':>10}")
    print(f"  {'-'*92}")
    allpairs = []
    for t in ["SS", "S", "A"]:
        s = summ(tiers[t]); allpairs += tiers[t]
        flag = "  ⚠N<100=暫定" if s["n"] < 100 else ""
        print(f"  {t:<5}{s['n']:>5}{s['hit_rate']:>8.1%}{s['roi']:>8.0%} "
              f"[{s['ci_lo']:>6.0%},{s['ci_hi']:>6.0%}]{s['roi_ex_max']:>11.0%}{s['roi_ex_top2']:>12.0%}"
              f"{s['median_hit']:>9,.0f}円{flag}")
    sa = summ(tiers['S'] + tiers['A'])
    al = summ(allpairs)
    print(f"  {'-'*92}")
    print(f"  {'S+A':<5}{sa['n']:>5}{sa['hit_rate']:>8.1%}{sa['roi']:>8.0%} "
          f"[{sa['ci_lo']:>6.0%},{sa['ci_hi']:>6.0%}]{sa['roi_ex_max']:>11.0%}{sa['roi_ex_top2']:>12.0%}{sa['median_hit']:>9,.0f}円")
    print(f"  {'計':<5}{al['n']:>5}{al['hit_rate']:>8.1%}{al['roi']:>8.0%} "
          f"[{al['ci_lo']:>6.0%},{al['ci_hi']:>6.0%}]{al['roi_ex_max']:>11.0%}{al['roi_ex_top2']:>12.0%}{al['median_hit']:>9,.0f}円")
    print(f"{'='*94}")
    print("  ※ ROI絶対値ではなく CI・最大払戻除去後・的中率・中央払戻で採否判断。N<100層は暫定。")


if __name__ == "__main__":
    main()
