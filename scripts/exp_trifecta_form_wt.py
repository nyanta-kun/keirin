"""三連複 vs 三連単フォーメーション の条件別ROI比較（回収率向上の検討）

着想: 三連複2軸流し=三連単の軸2頭マルチ（同一カバー）。三連単で頭固定/形式にすれば
点数を絞れ高配当でROI向上の可能性。鍵は着順(特に1着)の予測可能性。

戦略（指数順 p1=1位,p2=2位,thirds=3-5位）:
  A trio_3pt      : 三連複 {p1,p2,t} for t in thirds（現行・3点）
  B tri_12fix_3pt : 三連単 p1→p2→{t}（1-2着固定・3点・SS式）
  C tri_1fix_6pt  : 三連単 1着=p1固定, 2-3着=perm{p2,t3,t4}（6点）
  D tri_1fix_2nd  : 三連単 1着=p1, 2着=p2, 3着流し{thirds} = Bと同一 → 省略
payout: trio=三連複オッズ, trifecta=三連単オッズ（順序）。最終オッズ=上限値・eval modelOOS。
条件別 TRAIN→TEST。再現しないものは破棄。
"""
import sys
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.strategy_wt import upset_tier
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt
from roi_robustness_wt import roi_summary


def collect(date_from, date_to):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3: continue
        p = g["pred_prob"].tolist(); gap12 = p[0]-p[1]
        if gap12 < 0.06: continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist(); p1, p2 = fr[0], fr[1]; thirds = fr[2:5]
        if not thirds: continue
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order)
        rp = pm.get(rk, {})
        ratio = p[0] / (3.0 / n)

        def trio_pay(s):  return rp.get(("trio", s), 0)
        def tri_pay(t):   return rp.get(("trifecta", t), 0)

        # A: 三連複 3点
        A_combos = [frozenset((p1, p2, t)) for t in thirds]
        A_hit = top3 in A_combos
        A = (A_hit, trio_pay(top3) if A_hit else 0, len(A_combos)*100)
        # B: 三連単 p1→p2→t 3点
        B_combos = [(p1, p2, t) for t in thirds]
        B_hit = order in B_combos
        B = (B_hit, tri_pay(order) if B_hit else 0, len(B_combos)*100)
        # C: 三連単 1着=p1, 2-3着= perm{p2,t3,t4}（最大6点）
        second_set = [p2] + thirds[:2]            # p2,t3,t4
        C_combos = [(p1, a, b) for a, b in itertools.permutations(second_set, 2)]
        C_hit = order in C_combos
        C = (C_hit, tri_pay(order) if C_hit else 0, len(C_combos)*100)

        rows.append({
            "ut": upset_tier(p[0]+p[1]+p[2]), "gap12": gap12, "ratio": ratio,
            "p1_1st": order[0] == p1,                    # 指数1位が1着
            "p1_top3": p1 in top3,
            "A": A, "B": B, "C": C,
        })
    return rows


CONDS = {
    "ALL":            lambda r: True,
    "gap12>=0.15":    lambda r: r["gap12"] >= 0.15,
    "gap12>=0.25":    lambda r: r["gap12"] >= 0.25,
    "ratio<1.3(競合)": lambda r: r["ratio"] < 1.3,
    "Q1_loose(波乱)":  lambda r: r["ut"] == "Q1_loose",
    "Q4_chalk(鉄板)":  lambda r: r["ut"] == "Q4_chalk",
}


def agg(rows, cond, key):
    sub = [r for r in rows if cond(r)]
    pays = [r[key][1] for r in sub if r[key][2] > 0]
    bets = [r[key][2] for r in sub if r[key][2] > 0]
    return roi_summary(pays, bets), len(pays)


def main():
    train = collect("2023-07-01", "2026-02-28")
    test = collect("2026-03-01", "2026-06-08")

    print(f"\n【基礎率（TEST {len(test)}R）】")
    for cn, cond in CONDS.items():
        s = [r for r in test if cond(r)]
        if not s: continue
        p1_1st = sum(r["p1_1st"] for r in s)/len(s)
        p1_t3 = sum(r["p1_top3"] for r in s)/len(s)
        print(f"  {cn:<16} n={len(s):>4}  指数1位が1着={p1_1st:>4.0%}  指数1位3着内={p1_t3:>4.0%}")

    print(f"\n{'='*100}")
    print("  三連複(A) vs 三連単形式(B:1-2着固定3点 / C:1着固定6点)  条件別 ROI（最終オッズ上限値）")
    print(f"{'='*100}")
    print(f"  {'条件':<16}{'戦略':<14}{'TRAIN_R':>8}{'TR_ROI':>8}{'TR_的中':>8}{'TEST_R':>7}{'TE_ROI':>8}{'TE_的中':>8}{'TE_CI':>18}")
    print(f"  {'-'*98}")
    LAB = {"A": "A三連複3点", "B": "B三単1-2固定", "C": "C三単1着固定6点"}
    for cn, cond in CONDS.items():
        for key in ["A", "B", "C"]:
            str_, ntr = agg(train, cond, key)
            ste, nte = agg(test, cond, key)
            print(f"  {cn:<16}{LAB[key]:<14}{ntr:>8}{str_['roi']:>7.0%}{str_['hit_rate']:>8.0%}"
                  f"{nte:>7}{ste['roi']:>7.0%}{ste['hit_rate']:>8.0%} [{ste['ci_lo']:>4.0%},{ste['ci_hi']:>5.0%}]")
        print(f"  {'-'*98}")
    print("\n  ※ B/Cが A(三連複) を train/test とも上回る条件のみ採用候補。順序予測が効く＝高gap12で有利な想定。")


if __name__ == "__main__":
    main()
