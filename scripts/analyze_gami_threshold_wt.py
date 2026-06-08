"""ガミ・スキップ閾値スイープ（winticket）

「3点(SS=3連単 / S・A=3連複)のうち1点でも最終オッズ < N倍 を含むレースは推奨から外す」
を N=3,4,5,6,7,8 で比較。ROI・購入R・1日R・総損益・的中率の変化を train→test(OOS) で確認。
落車等の下振れは過去結果に内包済（=この ROI は落車込み）。払戻は最終オッズ=上限値。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)

MODEL = "lgbm_wt"


def build(f, t):
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 3: continue
        p = g["pred_prob"].tolist(); gap = p[0]-p[1]; ratio = p[0]/(3/n)
        tier = _assign_tier(gap, ratio)
        if tier is None: continue
        fr = g["frame_no"].astype(int).tolist(); p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        if len(thirds) < 3: continue
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3: continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {})

        legs = []
        for x in thirds:
            if tier == "SS":   # 3連単 pivot1→pivot2→x
                odds = rp.get(("trifecta", (p1, p2, x)))
                hit = (order == (p1, p2, x))
            else:              # 3連複
                combo = frozenset((p1, p2, x))
                odds = rp.get(("trio", combo))
                hit = (combo == top3)
            legs.append({"odds": (odds/100.0 if odds else None),
                         "payout": odds or 0, "hit": hit})
        rows.append({"tier": tier, "legs": legs, "date": rk[:8]})
    return rows


def roi_for(rows, thr):
    """thr=None: 全件3点。thr>0: 3点中1点でも<thr倍ならレース見送り。"""
    bet = ret = hit = nR = 0
    days = set()
    for r in rows:
        legs = r["legs"]
        if thr is not None:
            has_low = any((L["odds"] is not None and L["odds"] < thr) for L in legs)
            if has_low:
                continue
        nR += 1; days.add(r["date"])
        for L in legs:
            bet += 100
            if L["hit"]:
                ret += L["payout"]; hit += 1
    return {"R": nR, "days": len(days), "roi": ret/bet if bet else 0,
            "pl": ret-bet, "hit_rate": hit/nR if nR else 0,
            "rpd": nR/len(days) if days else 0}


def show(name, rows):
    print(f"\n{'='*82}\n  【{name}】 SS/S/A 計 {len(rows)}R\n{'='*82}")
    print(f"  {'閾値':<16}{'購入R':>6}{'R/日':>6}{'的中率':>8}{'ROI':>9}{'総損益':>12}")
    print(f"  {'-'*70}")
    for thr in [None, 3, 4, 5, 6, 7, 8]:
        a = roi_for(rows, thr)
        label = "全件(カット無)" if thr is None else f"<{thr}倍含むR除外"
        print(f"  {label:<16}{a['R']:>6}{a['rpd']:>6.1f}{a['hit_rate']:>8.1%}"
              f"{a['roi']:>9.1%}{a['pl']:>+12,}")


for name, (f, t) in {"TRAIN 2023-07〜2026-02": ("2023-07-01", "2026-02-28"),
                     "TEST/OOS 2026-03〜": ("2026-03-01", "2026-06-08")}.items():
    show(name, build(f, t))
