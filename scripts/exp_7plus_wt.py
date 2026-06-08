"""7車以上レースの収益化検討（再現性ファースト）

現行は≤6車のみ。7+(主に7/9車)は組合せ数が多く従来「採算不可」とされたが、
特定条件で再現的に黒字化できるかを train→OOS で検証する。

戦略（指数順 p1,p2,...）:
  std3  : 三連複 軸p1,p2 流し{3-5位}（3点・現行構造）
  wide5 : 三連複 軸p1,p2 流し{3-7位}（5点）
  box4  : 三連複 上位4車BOX（C(4,3)=4点・軸なし）
  box5  : 三連複 上位5車BOX（C(5,3)=10点）
条件: 全7+ / gap12閾値 / top3_sum下位（7+内で算出）。payout=最終オッズ上限値・eval OOS。
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


def collect(date_from, date_to):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    keep = sizes[sizes >= 7].index            # 7車以上
    df = df[df["race_key"].isin(keep)]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 7: continue
        p = g["pred_prob"].tolist(); gap12 = p[0]-p[1]
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        top3 = frozenset(int(x) for x in fin["frame_no"])
        rp = pm.get(rk, {})
        def tp(s): return rp.get(("trio", s), 0)

        def make(combos):
            hit = top3 in combos
            return (hit, tp(top3) if hit else 0, len(combos) * 100)
        std3 = make([frozenset((fr[0], fr[1], t)) for t in fr[2:5]])
        wide5 = make([frozenset((fr[0], fr[1], t)) for t in fr[2:7]])
        box4 = make([frozenset(c) for c in itertools.combinations(fr[:4], 3)])
        box5 = make([frozenset(c) for c in itertools.combinations(fr[:5], 3)])
        rows.append({"n": n, "gap12": gap12, "top3_sum": p[0]+p[1]+p[2],
                     "std3": std3, "wide5": wide5, "box4": box4, "box5": box5})
    return rows


def agg(rows, cond, key):
    sub = [r for r in rows if cond(r)]
    pays = [r[key][1] for r in sub if r[key][2] > 0]
    bets = [r[key][2] for r in sub if r[key][2] > 0]
    return roi_summary(pays, bets), len(pays)


def main():
    train = collect("2023-07-01", "2026-02-28")
    test = collect("2026-03-01", "2026-06-08")

    # 7+内 top3_sum 中央値（下位=波乱寄りの代理）
    import statistics
    med = statistics.median([r["top3_sum"] for r in train]) if train else 0
    print(f"\n【7車以上 規模】 TRAIN {len(train)}R / TEST {len(test)}R")
    from collections import Counter
    print("  車数分布(TEST):", dict(sorted(Counter(r["n"] for r in test).items())))
    print(f"  7+内 top3_sum 中央値(train)={med:.3f}")

    CONDS = {
        "ALL(7+)":        lambda r: True,
        "gap12>=0.10":    lambda r: r["gap12"] >= 0.10,
        "gap12>=0.15":    lambda r: r["gap12"] >= 0.15,
        "gap12>=0.20":    lambda r: r["gap12"] >= 0.20,
        "top3_sum<中央":   lambda r: r["top3_sum"] < med,
        "gap12>=0.15&loose": lambda r: r["gap12"] >= 0.15 and r["top3_sum"] < med,
    }
    print(f"\n{'='*100}")
    print("  7車以上 戦略×条件 ROI（最終オッズ上限値・eval OOS）")
    print(f"{'='*100}")
    print(f"  {'条件':<18}{'戦略':<8}{'点/R':>5}{'TR_R':>6}{'TR_ROI':>8}{'TR_的中':>7}{'TE_R':>6}{'TE_ROI':>8}{'TE_的中':>7}{'TE_CI':>18}")
    print(f"  {'-'*98}")
    for cn, cond in CONDS.items():
        for key in ["std3", "wide5", "box4", "box5"]:
            str_, ntr = agg(train, cond, key)
            ste, nte = agg(test, cond, key)
            ppr = {"std3": 3, "wide5": 5, "box4": 4, "box5": 10}[key]
            print(f"  {cn:<18}{key:<8}{ppr:>5}{ntr:>6}{str_['roi']:>7.0%}{str_['hit_rate']:>7.0%}"
                  f"{nte:>6}{ste['roi']:>7.0%}{ste['hit_rate']:>7.0%} [{ste['ci_lo']:>4.0%},{ste['ci_hi']:>5.0%}]")
        print(f"  {'-'*98}")
    print("\n  ※ train/test とも ROI>100%(理想>120%) かつ test十分R の条件のみ採用候補。再現しなければ破棄。")


if __name__ == "__main__":
    main()
