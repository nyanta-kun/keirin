"""波乱専用「裏軸」買い目の条件別検証（再現性ファースト）

解剖(04)の知見: 波乱でも指数1位は83%来る／伏兵は非本命ライン先頭・指数3-4位。
→ 検証する戦略:
  current : 三連複 軸=指数1位,指数2位 / 流し=指数3-5位（本番A/S相当）
  uraziku : 三連複 軸=指数1位, 裏軸=「非本命ラインの最高指数選手」/ 流し=残り指数上位
  各3点。payout=最終オッズ(上限値)。

特定条件(ALL/n_lines≥4/≥5/Q1_loose/Q1+Q2)別に TRAIN→TEST(OOS) で
ROI・的中率・CI・中央払戻を出し、**両期間で再現し currentを上回る条件だけ採用候補**、
再現しない/小標本は破棄する。指数=eval model(OOS)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < 0.06:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        frames = g["frame_no"].astype(int).tolist()
        fav_line = g.loc[0, "line_group"]
        top3 = frozenset(int(f) for f in fin["frame_no"])
        rp = pm.get(rk, {})

        # current: 軸 指数1,2 / 流し 指数3-5
        cur_third = frames[2:5]
        cur_combos = [frozenset((frames[0], frames[1], t)) for t in cur_third]
        # uraziku: 軸 指数1位 + 非本命ライン最高指数 / 流し 残り指数上位3
        nonfav = [int(r.frame_no) for r in g.itertuples() if r.line_group != fav_line]
        uz = nonfav[0] if nonfav else None
        ur_combos = []
        if uz is not None:
            flow = [f for f in frames if f not in (frames[0], uz)][:3]
            ur_combos = [frozenset((frames[0], uz, t)) for t in flow]

        def eval_combos(combos):
            for c in combos:
                if c == top3:
                    return True, rp.get(("trio", c), 0)
            return False, 0
        cur_hit, cur_pay = eval_combos(cur_combos)
        ur_hit, ur_pay = eval_combos(ur_combos)

        rows.append({
            "ut": upset_tier(p[0]+p[1]+p[2]),
            "n_lines": int(g.loc[0, "n_lines"]) if "n_lines" in g else 0,
            "gap12": gap12,
            "uz_diff": (uz is not None and uz != frames[1]),   # 裏軸が指数2位と異なる=現行と別物
            "cur_pay": float(cur_pay), "cur_bet": float(len(cur_combos) * 100),
            "ur_pay": float(ur_pay), "ur_bet": float(len(ur_combos) * 100),
        })
    return rows


CONDS = {
    "ALL":            lambda r: True,
    "n_lines>=4":     lambda r: r["n_lines"] >= 4,
    "n_lines>=5":     lambda r: r["n_lines"] >= 5,
    "Q1_loose":       lambda r: r["ut"] == "Q1_loose",
    "Q1+Q2":          lambda r: r["ut"] in ("Q1_loose", "Q2"),
    "Q1 & n_lines>=4": lambda r: r["ut"] == "Q1_loose" and r["n_lines"] >= 4,
}


def agg(rows, cond, which):
    sub = [r for r in rows if cond(r)]
    pays = [r[f"{which}_pay"] for r in sub if r[f"{which}_bet"] > 0]
    bets = [r[f"{which}_bet"] for r in sub if r[f"{which}_bet"] > 0]
    s = roi_summary(pays, bets)
    return len(pays), s


def show(train, test):
    print(f"\n{'='*104}")
    print(f"  波乱専用『裏軸』 vs 現行(指数1-2位軸)  条件別 TRAIN→TEST(OOS)  ※ROI=最終オッズ上限値")
    print(f"{'='*104}")
    print(f"  {'条件':<16}{'戦略':<8}{'TRAIN_R':>8}{'TRAIN_ROI':>10}{'TEST_R':>7}{'TEST_ROI':>9}{'TEST_CI':>20}{'再現':>6}")
    print(f"  {'-'*102}")
    for cname, cond in CONDS.items():
        for which, label in [("cur", "現行"), ("ur", "裏軸")]:
            ntr, str_ = agg(train, cond, which)
            nte, ste = agg(test, cond, which)
            # 再現判定: train/test とも ROI>120% かつ test n>=30
            repro = "○" if (str_["roi"] > 1.2 and ste["roi"] > 1.2 and nte >= 30) else ("小標本" if nte < 30 else "×")
            print(f"  {cname:<16}{label:<8}{ntr:>8}{str_['roi']:>9.0%}{nte:>7}{ste['roi']:>8.0%}"
                  f" [{ste['ci_lo']:>4.0%},{ste['ci_hi']:>5.0%}]{repro:>6}")
        print(f"  {'-'*102}")


train = collect("2023-07-01", "2026-02-28")
test = collect("2026-03-01", "2026-06-08")
show(train, test)
print("\n  再現○ = train/test とも ROI>120% かつ test≥30R。×/小標本 は破棄候補。")
print("  ※裏軸が指数2位と同一になるケースもあり（その時 現行≒裏軸）。条件で『非本命ライン先頭が来る波乱』を絞れているかを見る。")
