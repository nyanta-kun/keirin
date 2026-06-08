"""#7 波乱ゲート(top3_sum・オッズ不要) vs ガミ3段階(最終オッズ基準) の重複検証

問い: オッズ不要の top3_sum 波乱ゲートで、オッズ基準のガミ仕分けを代替できるか？
代替できれば「朝→直前オッズドリフト」依存を外せる。

各 SS/S/A レースに対し:
  - gami_class : 3点の最安目の最終オッズで <3倍=skip / 3〜5倍未満=B / ≥5倍=rec
  - upset_tier : top3_sum で Q1_loose/Q2/Q3/Q4_chalk（strategy_wt のカット）
を付与し、クロス集計・順位相関・推奨セットのROI/重複(Jaccard)を比較。
払戻=最終オッズ(上限値)。TRAIN 2023-07〜2026-02 / TEST 2026-03〜。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.strategy_wt import upset_tier, UPSET_TIERS
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)

model = load_model("lgbm_wt")


def build(f, t):
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
        fin = g[g["finish_order"].between(1,3)]; top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3: continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {}); is_ss = (tier == "SS")
        legs, hit, pay = [], False, 0
        for x in thirds:
            o = rp.get(("trifecta",(p1,p2,x))) if is_ss else rp.get(("trio",frozenset((p1,p2,x))))
            legs.append(o/100.0 if o else None)
            h = (order==(p1,p2,x)) if is_ss else (frozenset((p1,p2,x))==top3)
            if h: hit, pay = True, (o or 0)
        known = [o for o in legs if o is not None]
        min_leg = min(known) if known else None
        top3_sum = p[0]+p[1]+(p[2] if n>2 else 0)
        if min_leg is None:
            gami = "unknown"
        elif min_leg < 3:
            gami = "skip<3"
        elif min_leg < 5:
            gami = "B(3-5)"
        else:
            gami = "rec>=5"
        rows.append({"rk": rk, "tier": tier, "min_leg": min_leg, "gami": gami,
                     "top3_sum": top3_sum, "utier": upset_tier(top3_sum),
                     "hit": hit, "pay": pay})
    return pd.DataFrame(rows)


def roi(sub):
    bet = len(sub)*300; ret = sub["pay"].sum()
    return len(sub), (sub["hit"].mean() if len(sub) else 0), (ret/bet if bet else 0), ret-bet


def analyze(name, df):
    df = df[df["gami"] != "unknown"].copy()
    print(f"\n{'='*80}\n  【{name}】 SS/S/A {len(df)}R（オッズ既知）\n{'='*80}")

    # 1) クロス集計 gami × upset_tier
    print("\n  ① クロス集計（行=ガミ仕分け / 列=波乱ゲート帯）件数")
    ct = pd.crosstab(df["gami"], df["utier"]).reindex(
        index=["skip<3","B(3-5)","rec>=5"], columns=list(UPSET_TIERS), fill_value=0)
    print(ct.to_string())
    print("\n  行%（各ガミ区分が波乱帯にどう分布するか）")
    print((ct.div(ct.sum(axis=1), axis=0)*100).round(0).astype(int).to_string())

    # 2) 相関・帯別オッズ
    rho = df[["min_leg","top3_sum"]].rank().corr().iloc[0,1]
    print(f"\n  ② min_leg(最安オッズ) と top3_sum の順位相関: {rho:.3f}（負=looseほど高オッズ）")
    print("     波乱帯別の最安オッズ中央値 / 見送り(<3倍)率:")
    for u in UPSET_TIERS:
        s = df[df["utier"]==u]
        if len(s)==0: continue
        print(f"       {u:<9} n={len(s):>4}  min_leg中央={s['min_leg'].median():>5.1f}倍  "
              f"<3倍率={ (s['min_leg']<3).mean():>5.0%}  ≥5倍率={(s['min_leg']>=5).mean():>5.0%}")

    # 3) 推奨セット比較（ROIと重複）
    print("\n  ③ 推奨セット比較（オッズ基準 vs オッズ不要ゲート）")
    sets = {
        "現行ガミ推奨(≥5倍)":          set(df[df["gami"]=="rec>=5"]["rk"]),
        "ゲート Q1_loose":             set(df[df["utier"]=="Q1_loose"]["rk"]),
        "ゲート Q1+Q2":                set(df[df["utier"].isin(["Q1_loose","Q2"])]["rk"]),
    }
    base = sets["現行ガミ推奨(≥5倍)"]
    print(f"  {'セット':<22}{'R':>5}{'的中率':>8}{'ROI':>9}{'損益':>11}{'vs現行Jaccard':>14}")
    for name2, s in sets.items():
        sub = df[df["rk"].isin(s)]
        n, hr, r, pl = roi(sub)
        jac = len(s & base)/len(s | base) if (s|base) else 0
        print(f"  {name2:<22}{n:>5}{hr:>8.1%}{r:>9.1%}{pl:>+11,}{jac:>13.2f}")
    # 交差: ガミ推奨のうちゲートでも残る割合 / ゲートが拾う非≥5倍
    g12 = sets["ゲート Q1+Q2"]
    print(f"\n  現行ガミ推奨(≥5倍) {len(base)}R のうち ゲートQ1+Q2 にも入る: "
          f"{len(base & g12)}R ({len(base&g12)/len(base) if base else 0:.0%})")
    print(f"  ゲートQ1+Q2 {len(g12)}R のうち ガミ的に<3倍(本来見送り): "
          f"{len(g12 & set(df[df['gami']=='skip<3']['rk']))}R "
          f"({len(g12 & set(df[df['gami']=='skip<3']['rk']))/len(g12) if g12 else 0:.0%})")


tr = build("2023-07-01","2026-02-28")
te = build("2026-03-01","2026-06-08")
analyze("TRAIN 2023-07〜2026-02", tr)
analyze("TEST/OOS 2026-03〜", te)
