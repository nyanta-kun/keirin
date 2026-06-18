"""予想家活動の商品メニュー候補を統一条件で計測（2026-06-11）。

ユーザーの方針: 現行システムをベースに予想家として活動。運営戦略上必要な
「的中率を上げる/高配当を当てる/多くのレースを予想する」の各商品候補について、
TEST期間の 件数/日・点数・的中率・ROI(上限値)・配当分布 を同一テーブルで出す。

※これは新レバー探索ではなく既検証レバーの商品化棚卸し（多重比較の対象外）。
  各候補は docs/analysis/05(7+的中率選別)/10(S3系)/12(ワイド)/14(高配当S3) の
  検証済み構造のみ。ROIは参考値（最終オッズ上限値・商品の訴求軸は別）。

商品候補:
  ≤6車: W12ワイド1点(ALL/top2_sum帯) / 二車複top2 1点 / 現行SS/S/A(3点) /
        S3 10点(RANK∩UPSET=万車券狙い)
  7+ : 三連複2軸流しr3-6(4点・ALL/gap23≥p75/r2_prob≥p75=的中率商品) /
        W12ワイド / 二車複top2
model=lgbm_wt_eval(OOS)・TRAIN 2023-07〜2026-02(カット算出のみ)・TEST 2026-03〜06-08。
"""
import itertools
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt, _assign_tier
from roi_robustness_wt import roi_summary

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 5:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order)
        rp = pm.get(rk, {})
        p1, p2 = fr[0], fr[1]
        thirds3 = fr[2:5]
        tier = _assign_tier(p[0] - p[1], p[0] / (3.0 / n))
        m = DATE_RE.search(str(rk))

        # --- 各商品の (pay, bet) ---
        # W12 ワイド1点（wt市場名は quinellaPlace）
        w12 = (rp.get(("quinellaPlace", frozenset((p1, p2))), 0)
               if frozenset((p1, p2)) <= top3 else 0, 100)
        # 二車複 top2 1点
        qn = (rp.get(("quinella", frozenset((p1, p2))), 0)
              if frozenset((p1, p2)) == frozenset(order[:2]) else 0, 100)
        # 現行 SS/S/A（SS=三連単3点・S/A=三連複3点）
        tpay = 0
        if tier:
            for x in thirds3:
                if tier == "SS":
                    if order == (p1, p2, x):
                        tpay = rp.get(("trifecta", (p1, p2, x)), 0); break
                else:
                    if frozenset((p1, p2, x)) == top3:
                        tpay = rp.get(("trio", frozenset((p1, p2, x))), 0); break
        tier_sc = (tpay, len(thirds3) * 100)
        # S3 三連単10点（3頭BOX＋p1⇄p2→{4,5位}）
        c3 = list(itertools.permutations((fr[0], fr[1], fr[2])))
        c3 += [(a, b, x) for a, b in ((p1, p2), (p2, p1)) for x in fr[3:5]]
        s3pay = 0
        for c in c3:
            if order == c:
                s3pay = rp.get(("trifecta", c), 0); break
        s3 = (s3pay, len(c3) * 100)
        # 7+向け 三連複2軸流し r3-6（4点）
        th4 = fr[2:6]
        t4pay = 0
        for x in th4:
            if frozenset((p1, p2, x)) == top3:
                t4pay = rp.get(("trio", frozenset((p1, p2, x))), 0); break
        trio46 = (t4pay, len(th4) * 100)

        rows.append({
            "date": m.group(1) if m else None, "n": n, "tier": tier,
            "top2_sum": p[0] + p[1], "top3_sum": p[0] + p[1] + p[2],
            "gap23": p[1] - p[2], "r2_prob": p[1],
            "w12": w12, "qn": qn, "tier_sc": tier_sc, "s3": s3, "trio46": trio46,
        })
    return rows


def seg(rows, key, cond):
    sub = [r for r in rows if cond(r)]
    s = roi_summary([r[key][0] for r in sub], [r[key][1] for r in sub])
    pays = [r[key][0] for r in sub if r[key][0] > 0]
    return s, len(sub), (np.median(pays) if pays else 0), (max(pays) if pays else 0)


def main():
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    days = len({r["date"] for r in te if r["date"]}) or 99
    le6 = lambda r: r["n"] <= 6
    p7 = lambda r: r["n"] >= 7
    print(f"\n予想家メニュー候補 統一計測  TEST {len(te)}R / {days}日（最終オッズ上限値・5車以上）")
    print(f"  TEST レース数/日: ≤6車 {sum(le6(r) for r in te)/days:.1f}R / 7+ {sum(p7(r) for r in te)/days:.1f}R")

    # TRAINカット
    t2 = [r["top2_sum"] for r in tr if le6(r)]
    cut_t2 = {q: np.quantile(t2, q) for q in (0.5, 0.75, 0.9)}
    t3le6 = np.quantile([r["top3_sum"] for r in tr if le6(r)], 0.25)
    g23 = np.quantile([r["gap23"] for r in tr if p7(r)], 0.75)
    r2p = np.quantile([r["r2_prob"] for r in tr if p7(r)], 0.75)

    menu = [
        # (ラベル, 層, score key, 条件)
        ("≤6 W12ワイド1点 ALL",            le6, "w12", lambda r: True),
        ("≤6 W12 top2_sum≥p50",            le6, "w12", lambda r: r["top2_sum"] >= cut_t2[0.5]),
        ("≤6 W12 top2_sum≥p75",            le6, "w12", lambda r: r["top2_sum"] >= cut_t2[0.75]),
        ("≤6 W12 top2_sum≥p90",            le6, "w12", lambda r: r["top2_sum"] >= cut_t2[0.9]),
        ("≤6 二車複top2 1点 ALL",          le6, "qn",  lambda r: True),
        ("≤6 現行SS/S/A(3点)",             le6, "tier_sc", lambda r: r["tier"] is not None),
        ("≤6 S3 10点 RANK",                le6, "s3",  lambda r: r["tier"] is not None),
        ("≤6 S3 10点 RANK∩UPSET",          le6, "s3",  lambda r: r["tier"] is not None and r["top3_sum"] <= t3le6),
        ("7+ 三連複2軸r3-6(4点) ALL",      p7,  "trio46", lambda r: True),
        ("7+ 三連複2軸r3-6 gap23≥p75",     p7,  "trio46", lambda r: r["gap23"] >= g23),
        ("7+ 三連複2軸r3-6 r2_prob≥p75",   p7,  "trio46", lambda r: r["r2_prob"] >= r2p),
        ("7+ 三連複2軸r3-6 両方≥p75",      p7,  "trio46", lambda r: r["gap23"] >= g23 and r["r2_prob"] >= r2p),
        ("7+ W12ワイド1点 ALL",            p7,  "w12", lambda r: True),
        ("7+ W12 gap23≥p75",               p7,  "w12", lambda r: r["gap23"] >= g23),
        ("7+ 二車複top2 1点 ALL",          p7,  "qn",  lambda r: True),
        ("7+ 二車複top2 gap23≥p75",        p7,  "qn",  lambda r: r["gap23"] >= g23),
    ]
    print(f"\n  {'商品候補':<32}{'R':>5}{'R/日':>6}{'的中率':>8}{'ROI':>7}{'中央配当':>9}{'最大':>9}")
    for lab, sz, key, cond in menu:
        s, nn, med, mx = seg(te, key, lambda r, sz=sz, cond=cond: sz(r) and cond(r))
        print(f"  {lab:<32}{nn:>5}{nn/days:>6.2f}{s['hit_rate']:>8.1%}{s['roi']:>7.0%}"
              f"{med:>8,.0f}円{mx:>8,.0f}円")
    print("\n  ※的中率商品はROI<100%が正常（的中率↔オッズ逆連動の実証済み構造）。"
          "訴求軸は商品ごとに片方のみ。")
    # 連敗リスク（公開実績の見栄え設計用）
    print("\n  連敗確率の目安（的中率h → k連敗が起きる確率 (1-h)^k）:")
    for h in (0.8, 0.65, 0.55, 0.38, 0.2):
        ks = {k: (1 - h) ** k for k in (5, 10)}
        print(f"    h={h:.0%}: 5連敗 {ks[5]:.1%} / 10連敗 {ks[10]:.2%}")


if __name__ == "__main__":
    main()
