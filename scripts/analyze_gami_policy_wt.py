"""ガミ対策: 「低オッズ点カット」vs「レーススキップ」比較（winticket・3連複 S/A）

各 S/A レースの3連複2軸流し3点について、各点(pivot1-pivot2-third)の trio オッズを取得し、
3倍未満(=払戻<300円)の点が存在するレースで以下を比較:
  baseline : 3点そのまま(300円)
  cut      : <300円の点だけ外科的にカット(残り点×100円)。全滅ならそのレースは0円
  skip     : 3点中に<300円が1つでもあればレースごと降りる(0円)
TRAIN→TEST(OOS)。払戻は wt_odds 最終オッズ=上限値（朝オッズではない点に注意）。
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
LOW = 300   # 払戻<300円(=オッズ<3.0)を「ガミ点」とする

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
        if tier not in ("S", "A"): continue       # 3連複のS/Aのみ（SSは3連単で別）
        fr = g["frame_no"].astype(int).tolist(); p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        if len(thirds) < 3: continue
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3: continue
        rp = pm.get(rk, {})
        legs = []
        for x in thirds:
            combo = frozenset((p1, p2, x))
            odds_payout = rp.get(("trio", combo))   # 100円賭けの払戻(=odds×100)。Noneは盤面欠損
            legs.append({"combo": combo, "payout": odds_payout,
                         "hit": (combo == top3)})
        rows.append({"tier": tier, "legs": legs, "rk": rk})
    return rows

def evaluate(rows):
    res = {}
    for pol in ("baseline", "cut", "skip"):
        bet = ret = hit = nraces = 0
        for r in rows:
            legs = r["legs"]
            # オッズ盤面が1点でも欠損ならガミ判定不能→baseline同様に3点(保守)
            known = [L for L in legs if L["payout"] is not None]
            has_low = any((L["payout"] is not None and L["payout"] < LOW) for L in legs)
            if pol == "baseline":
                use = legs
            elif pol == "cut":
                # 既知かつ<LOW の点を除外。未知(None)は残す。
                use = [L for L in legs if not (L["payout"] is not None and L["payout"] < LOW)]
            else:  # skip
                use = [] if has_low else legs
            if not use:
                continue
            nraces += 1
            for L in use:
                bet += 100
                if L["hit"]:
                    ret += (L["payout"] or 0)
                    hit += 1
        roi = ret/bet if bet else 0
        res[pol] = {"R": nraces, "bet": bet, "ret": ret, "hit": hit,
                    "roi": roi, "pl": ret-bet,
                    "hit_rate": hit/nraces if nraces else 0}
    return res

def show(name, rows):
    # ガミ点を含むレースの割合
    n = len(rows)
    with_low = sum(1 for r in rows
                   if any((L["payout"] is not None and L["payout"] < LOW) for L in r["legs"]))
    print(f"\n{'='*70}\n  【{name}】 S/A 3連複 {n}R  (うち<300円点を含む {with_low}R = {with_low/n:.1%})\n{'='*70}")
    res = evaluate(rows)
    print(f"  {'方針':<10}{'購入R':>6}{'点/R':>6}{'的中率':>8}{'ROI':>9}{'損益':>11}")
    for pol, label in [("baseline","現行(3点)"),("cut","低oddsカット"),("skip","レーススキップ")]:
        a = res[pol]
        ppr = a["bet"]/a["R"]/100 if a["R"] else 0
        print(f"  {label:<10}{a['R']:>6}{ppr:>6.1f}{a['hit_rate']:>8.1%}{a['roi']:>9.1%}{a['pl']:>+11,}")

for name, (f, t) in {"TRAIN 2023-07〜2026-02": ("2023-07-01","2026-02-28"),
                     "TEST/OOS 2026-03〜": ("2026-03-01","2026-06-08")}.items():
    show(name, build(f, t))
