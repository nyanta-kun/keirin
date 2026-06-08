"""月×ランク別バックテスト（winticket・ガミ3段階込み）

本番 wave-picks-wt と同条件:
  SS=gap12≥0.15&ratio<1.3(3連単) / S=gap12≥0.15&ratio[1.3,1.6)(3連複) / A=gap12[0.06,0.15)(3連複)
  6車以下・3点300円。最安目の最終オッズで:
    <3倍→見送り / 3〜5倍未満→Bランク(別枠) / ≥5倍→通常(SS/S/A)
払戻=wt_odds最終オッズ(=上限値)。2026-06 は 06-08 まで（部分月）。
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

SKIP, BMAX = 3.0, 5.0   # <3倍見送り / 3〜5倍未満B

def collect(f, t):
    model = load_model("lgbm_wt")
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
        is_ss = (tier == "SS")
        legs, hit, pay = [], False, 0
        for x in thirds:
            if is_ss:
                o = rp.get(("trifecta", (p1, p2, x)))
                h = (order == (p1, p2, x))
            else:
                o = rp.get(("trio", frozenset((p1, p2, x))))
                h = (frozenset((p1, p2, x)) == top3)
            legs.append(o/100.0 if o else None)
            if h: hit, pay = True, (o or 0)
        known = [o for o in legs if o is not None]
        min_leg = min(known) if known else None
        # 仕分け
        if min_leg is not None and min_leg < SKIP:
            disp = "見送り(<3倍)"
        elif min_leg is not None and min_leg < BMAX:
            disp = "B"
        else:
            disp = tier   # SS/S/A
        rows.append({"ym": rk[:6], "tier": tier, "disp": disp,
                     "hit": hit, "pay": pay, "bet": 300})
    return pd.DataFrame(rows)

dfs = [collect(f"2026-{m:02d}-01", f"2026-{m:02d}-{d}")
       for m, d in [(4, "30"), (5, "31"), (6, "08")]]
allr = pd.concat(dfs, ignore_index=True)

ORDER = ["SS", "S", "A", "B", "見送り(<3倍)"]
def agg(sub):
    bet, ret, hit, n = sub["bet"].sum(), sub["pay"].sum(), int(sub["hit"].sum()), len(sub)
    avg = ret/hit if hit else 0
    return n, hit, (hit/n if n else 0), bet, ret, (ret/bet if bet else 0), ret-bet, avg

for ym in ["202604", "202605", "202606"]:
    m = allr[allr["ym"] == ym]
    label = f"{ym[:4]}-{ym[4:]}" + ("（〜06-08・部分月）" if ym == "202606" else "")
    print(f"\n{'='*86}\n  【{label}】  対象 {len(m)}R\n{'='*86}")
    print(f"  {'ランク':<14}{'R':>5}{'的中':>5}{'的中率':>8}{'投資':>9}{'回収':>10}{'ROI':>9}{'損益':>11}{'avg配当':>9}")
    print(f"  {'-'*82}")
    # 推奨(SS/S/A)計
    rec = m[m["disp"].isin(["SS", "S", "A"])]
    for d in ORDER:
        sub = m[m["disp"] == d]
        if sub.empty:
            continue
        n, hit, hr, bet, ret, roi, pl, avg = agg(sub)
        tag = "  ＊各自判断" if d == "B" else ("  ＊投資せず参考" if d.startswith("見送り") else "")
        print(f"  {d:<14}{n:>5}{hit:>5}{hr:>8.1%}{bet:>9,}{ret:>10,}{roi:>9.1%}{pl:>+11,}{avg:>8,.0f}円{tag}")
    if not rec.empty:
        n, hit, hr, bet, ret, roi, pl, avg = agg(rec)
        print(f"  {'-'*82}")
        print(f"  {'推奨計(SS/S/A)':<14}{n:>5}{hit:>5}{hr:>8.1%}{bet:>9,}{ret:>10,}{roi:>9.1%}{pl:>+11,}{avg:>8,.0f}円")

# 全期間まとめ
print(f"\n{'='*86}\n  【4〜6月 合計】\n{'='*86}")
print(f"  {'ランク':<14}{'R':>5}{'的中':>5}{'的中率':>8}{'ROI':>9}{'損益':>11}")
for d in ORDER + ["推奨計"]:
    sub = allr[allr["disp"].isin(["SS","S","A"])] if d == "推奨計" else allr[allr["disp"] == d]
    if sub.empty: continue
    n, hit, hr, bet, ret, roi, pl, avg = agg(sub)
    print(f"  {d:<14}{n:>5}{hit:>5}{hr:>8.1%}{roi:>9.1%}{pl:>+11,}")
