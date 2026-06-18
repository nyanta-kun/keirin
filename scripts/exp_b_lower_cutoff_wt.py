"""ランクB閾値緩和 × 買い目オッズ足切り — 推奨0件日対策でROI>100%を維持できるか。

背景:
  ガミ3段階(確立レバーD)は最安目朝オッズで <3倍=見送り / 3〜5倍=B / ≥5倍=推奨 とレース単位振り分け。
  推奨0件日が続く（現行≥5帯は購入可能日~3割）ため、ユーザー問い=「Bの閾値を下げて見送り帯を取り込み、
  買い目単位のオッズ足切りで鉄板点だけ除けば ROI100% を超えられるか」。

定義は本番に忠実（analyze_gami_threshold_wt.py と同一のレッグ構成）:
  - `_assign_tier(gap12, ratio)` で SS/S/A のみ対象（None=対象外）
  - SS = 3連単 p1→p2→x (3点) / S・A = 3連複 {p1,p2,x} (3点)
  - レース帯 = その3点の最安オッズ（wt_odds最終=上限値。実運用の朝オッズ判定とはドリフト差あり）
  - 足切り = 点単位のオッズ下限（または[10,80]中間帯=docs/analysis/06のレバーB）

既知の前提:
  - D: <3倍点を含むレースは集団で収支ゼロ。skip(レース単位) > cut(点単位) （analyze_gami_policy_wt.py）
  - B帯(3-5)は +EV(~116%) とされてきた（量を増やす候補）
  - 中間オッズ[10,80]点集中=★再現 100-104%（docs/analysis/06・運用未反映）

判定規律: ★=TRAIN/TEST両方>100%かつTEST≥30R。「除」=最大払戻1R除去後ROI（単発依存チェック）。
TRAIN 2023-07〜2026-02 / TEST 2026-03〜2026-06-11。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)
from roi_robustness_wt import roi_summary

MODEL = "lgbm_wt"
INF = float("inf")

RACE_BANDS = [
    ("<2        (見送り内・超鉄板)", 0, 2),
    ("2-3       (見送り内・鉄板)",   2, 3),
    ("3-5       (現行B)",            3, 5),
    (">=5       (現行推奨)",         5, INF),
    ("<3        (現行見送り全体)",   0, 3),
    ("2-5       (B下限を2へ緩和)",   2, 5),
    ("<5        (見送り+B 全取込)",  0, 5),
    ("全レース  (帯制限なし)",       0, INF),
]
CUTOFFS = [
    ("足切りなし", 0, INF),
    (">=2倍",     2, INF),
    (">=3倍",     3, INF),
    (">=5倍",     5, INF),
    (">=10倍",   10, INF),
    ("[10,80]",  10, 80),
]


def collect(f, t):
    """本番と同一のレッグ構成で per-race {tier, legs[(odds倍, payout円, hit)], date} を構築。"""
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, 6)                       # ≤6車
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        tier = _assign_tier(p[0] - p[1], p[0] / (3 / n))
        if tier is None:
            continue
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]
        if len(thirds) < 3:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        if len(top3) < 3:
            continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        rp = pm.get(rk, {})
        legs = []
        for x in thirds:
            if tier == "SS":
                pay = rp.get(("trifecta", (p1, p2, x)))
                hit = (order == (p1, p2, x))
            else:
                pay = rp.get(("trio", frozenset((p1, p2, x))))
                hit = (frozenset((p1, p2, x)) == top3)
            if pay:
                legs.append((pay / 100.0, float(pay), hit))
        if not legs:
            continue
        rows.append({"tier": tier, "legs": legs, "date": rk[:8],
                     "min_odds": min(o for o, _, _ in legs)})
    return rows


def cell_roi(rows, b_lo, b_hi, c_lo, c_hi, tiers=None):
    pays, bets, dates = [], [], set()
    for r in rows:
        if tiers and r["tier"] not in tiers:
            continue
        if not (b_lo <= r["min_odds"] < b_hi):
            continue
        sub = [(pay, hit) for o, pay, hit in r["legs"] if c_lo <= o < c_hi]
        if not sub:
            continue
        pays.append(sum(pay for pay, hit in sub if hit))
        bets.append(len(sub) * 100)
        dates.add(r["date"])
    return roi_summary(pays, bets), len(pays), dates


def fmt(s, n):
    if n == 0:
        return f"{0:>5}R  --"
    return (f"{n:>5}R {s['roi']:>5.0%} 的{s['hit_rate']:>4.0%} "
            f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] 除{s['roi_ex_max']:>4.0%}")


def matrix(tr, te, tiers=None, label="SS/S/A 全層"):
    te_days_all = len({r["date"] for r in te})
    print(f"\n{'='*118}")
    print(f"  ◇ {label}  TRAIN {sum(1 for r in tr if not tiers or r['tier'] in tiers)}R / "
          f"TEST {sum(1 for r in te if not tiers or r['tier'] in tiers)}R（TEST営業日 {te_days_all}日）")
    print(f"{'='*118}")
    for blab, b_lo, b_hi in RACE_BANDS:
        _, n_te0, days_te0 = cell_roi(te, b_lo, b_hi, 0, INF, tiers)
        rpd = n_te0 / te_days_all if te_days_all else 0.0
        cov = len(days_te0) / te_days_all if te_days_all else 0.0
        print(f"\n  ◆ レース帯: 最安目 {blab}   TEST量: {rpd:.1f}R/日・購入可能日 {cov:.0%}")
        print(f"    {'足切り':<10}{'TRAIN':<44}{'TEST':<44}{'判定':>5}")
        for clab, c_lo, c_hi in CUTOFFS:
            s1, n1, _ = cell_roi(tr, b_lo, b_hi, c_lo, c_hi, tiers)
            s2, n2, _ = cell_roi(te, b_lo, b_hi, c_lo, c_hi, tiers)
            flag = "★再現" if (n1 > 0 and n2 >= 30 and s1["roi"] > 1.0 and s2["roi"] > 1.0) else \
                   ("小標本" if 0 < n2 < 30 else "")
            print(f"    {clab:<10}{fmt(s1,n1):<44}{fmt(s2,n2):<44}{flag:>5}")


def main():
    print("collecting TRAIN (2023-07〜2026-02)...", flush=True)
    tr = collect("2023-07-01", "2026-02-28")
    print(f"  TRAIN {len(tr)}R", flush=True)
    print("collecting TEST (2026-03〜2026-06-11)...", flush=True)
    te = collect("2026-03-01", "2026-06-11")
    print(f"  TEST {len(te)}R", flush=True)

    matrix(tr, te)                                   # 本番推奨セット全体
    matrix(tr, te, tiers=("S", "A"), label="S/A のみ（3連複・B運用の主対象）")

    print(f"\n{'='*118}")
    print("  読み方: 「除」=最大払戻1R除去後ROI。★再現でも『除』<100%なら単発依存=非頑健。")
    print("  現行運用=『>=5帯×足切りなし』。緩和の採否は ★かつ除>100% のセルが現行に量を上積みできるかで判断。")
    print(f"{'='*118}")


if __name__ == "__main__":
    main()
