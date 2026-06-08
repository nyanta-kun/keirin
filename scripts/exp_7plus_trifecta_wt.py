"""7車以上: 三連単フォーメーションで回収率を確保できるか（再現性ファースト）

三連複(1位含む)が7+で控除率の壁(75-80%)→三連単で点数を絞り高配当を狙う。
指数順 p1,p2,...。検証フォーメーション:
  trio3   : 三連複 軸p1,p2 流し3-5位（baseline・3点）
  F1a     : 三連単 1着=p1固定, 2-3着=perm{p2,p3,p4}（6点・本命頭）
  F1b     : 三連単 1着=p1固定, 2-3着=perm{p3,p4,p5}（6点・本命頭+穴相手）
  F2      : 三連単 波乱頭=1着{p2,p3}, p1を2 or 3着, 残り{p2,p3,p4}（〜8点・本命2-3着）
payout: trifecta/trio 最終オッズ上限値・eval OOS。条件別 train→test。
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from roi_robustness_wt import roi_summary


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < 7: continue
        p = g["pred_prob"].tolist(); gap12 = p[0]-p[1]
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order)
        rp = pm.get(rk, {})
        def trio(s): return rp.get(("trio", s), 0)
        def tri(o):  return rp.get(("trifecta", o), 0)

        def mk_tri(combos):
            combos = list(dict.fromkeys(combos))   # 重複除去
            hit = order in combos
            return (hit, tri(order) if hit else 0, len(combos)*100)

        p1, p2, p3, p4, p5 = fr[0], fr[1], fr[2], fr[3], fr[4]
        trio3 = (top3 in [frozenset((p1, p2, x)) for x in fr[2:5]],
                 trio(top3) if top3 in [frozenset((p1, p2, x)) for x in fr[2:5]] else 0, 300)
        F1a = mk_tri([(p1, a, b) for a, b in itertools.permutations([p2, p3, p4], 2)])
        F1b = mk_tri([(p1, a, b) for a, b in itertools.permutations([p3, p4, p5], 2)])
        f2 = []
        for h in [p2, p3]:
            for o in [x for x in [p2, p3, p4] if x != h]:
                f2 += [(h, p1, o), (h, o, p1)]
        F2 = mk_tri(f2)
        rows.append({"gap12": gap12, "top3_sum": p[0]+p[1]+p[2], "p1_1st": order[0] == p1,
                     "trio3": trio3, "F1a": F1a, "F1b": F1b, "F2": F2})
    return rows


def agg(rows, cond, key):
    sub = [r for r in rows if cond(r)]
    pays = [r[key][1] for r in sub if r[key][2] > 0]
    bets = [r[key][2] for r in sub if r[key][2] > 0]
    return roi_summary(pays, bets), len(pays)


tr = collect("2023-07-01", "2026-02-28"); te = collect("2026-03-01", "2026-06-08")
import statistics
med = statistics.median([r["top3_sum"] for r in tr])
print(f"\n7車以上 三連単フォーメーション  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）")
print(f"  指数1位の1着率(test): {sum(r['p1_1st'] for r in te)/len(te):.0%}")
CONDS = {
    "ALL(7+)":       lambda r: True,
    "gap12<0.10弱本命": lambda r: r["gap12"] < 0.10,
    "gap12>=0.20強本命": lambda r: r["gap12"] >= 0.20,
    "top3_sum<中央":  lambda r: r["top3_sum"] < med,
}
LAB = {"trio3": "三連複3点", "F1a": "F1a本命頭234", "F1b": "F1b本命頭345穴", "F2": "F2本命2-3着波乱"}
print(f"\n  {'条件':<18}{'戦略':<16}{'点/R':>5}{'TR_ROI':>8}{'TR_的中':>8}{'TE_ROI':>8}{'TE_的中':>8}{'TE_CI':>18}")
for cn, cond in CONDS.items():
    for key in ["trio3", "F1a", "F1b", "F2"]:
        s1, _ = agg(tr, cond, key); s2, nte = agg(te, cond, key)
        ppr = {"trio3": 3, "F1a": 6, "F1b": 6, "F2": 8}[key]
        print(f"  {cn:<18}{LAB[key]:<16}{ppr:>5}{s1['roi']:>7.0%}{s1['hit_rate']:>8.0%}"
              f"{s2['roi']:>7.0%}{s2['hit_rate']:>8.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]")
    print(f"  {'-'*92}")
print("\n  ※ train/test とも 100%超(理想120%超)で再現する条件のみ採用候補。再現しなければ破棄。")
