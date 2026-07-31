"""7S/7A: 3列目(残り流し)を5点→2点に絞った場合のROI検証（2026-07-31）。

現行: 軸2車+残り5車全流し（5点均等買い・2,000円/点=10,000円）。
本検証: 5点のうち2点だけに絞る場合、3つの絞り込み基準を比較する:
  - ModelTop2: 3列目のpred_top3_pct上位2点（オッズ不使用・事前情報のみ）
  - LowOdds2 : trioオッズが低い(=市場的中確率が高い)2点
  - HighOdds2: trioオッズが高い(=穴)2点

母集団はS7+7A（2026-07-31改定後の現行2ゲート版）。月次凍結vintageモデルに
よるhonest walk-forward。読み取り専用・DB書き込みなし。均等買い(2,000円/点)。
"""
import statistics
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import s7_evening_reselect, s7a_daily_select
from src.wt_vintage_config import monthly_windows
from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

STAKE_PER_POINT = 2000.0
SCHEMES = ["Full5", "ModelTop2", "LowOdds2", "HighOdds2"]


def pick_points(scheme, others, bf, odds_map):
    if scheme == "Full5":
        return list(others)
    if scheme == "ModelTop2":
        order = sorted(others, key=lambda x: -bf[x]["p"])
    elif scheme == "LowOdds2":
        order = sorted(others, key=lambda x: odds_map[x])
    elif scheme == "HighOdds2":
        order = sorted(others, key=lambda x: -odds_map[x])
    else:
        raise ValueError(scheme)
    return order[:2]


def main(date_from_filter=None, date_to_filter=None, label="全期間"):
    windows = monthly_windows()
    if date_from_filter:
        windows = [w for w in windows if w[1] >= date_from_filter and w[0] <= date_to_filter]
    print(f"[main] {label}: 月次窓数={len(windows)}", flush=True)

    totals = {s: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0} for s in SCHEMES}
    monthly_roi = {s: [] for s in SCHEMES}

    for date_from, date_to, eval_model, win_model in windows:
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            continue
        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        selected = []
        for _d, day_cands in by_day.items():
            selected.extend(s7_evening_reselect(day_cands, [], set()))
            selected.extend(s7a_daily_select(day_cands))
        if not selected:
            continue

        m = {s: {"bet": 0.0, "ret": 0.0} for s in SCHEMES}
        for c_ in selected:
            a, b = c_["axis1"], c_["axis2"]
            others = c_["others"]
            trio = c_["trio"]
            combos = {x: trio[frozenset({a, b, x})] for x in others if frozenset({a, b, x}) in trio}
            if len(combos) < 2:
                continue
            actual_top3 = c_["actual_top3"]
            hit_x = next((x for x, key in [(x, frozenset({a, b, x})) for x in combos] if key == actual_top3), None)

            for scheme in SCHEMES:
                pts = pick_points(scheme, list(combos.keys()), c_["bf"], combos)
                bet = STAKE_PER_POINT * len(pts)
                ret = STAKE_PER_POINT * combos[hit_x] if (hit_x is not None and hit_x in pts) else 0.0
                m[scheme]["bet"] += bet
                m[scheme]["ret"] += ret
                totals[scheme]["n"] += 1
                totals[scheme]["bet"] += bet
                totals[scheme]["ret"] += ret
                if ret > 0:
                    totals[scheme]["hit"] += 1

        line = f"[{date_from}〜{date_to}] n={len(selected)}"
        for s in SCHEMES:
            roi = m[s]["ret"] / m[s]["bet"] * 100 if m[s]["bet"] else 0.0
            monthly_roi[s].append(roi if m[s]["bet"] else None)
            line += f"  {s}={roi:.1f}%"
        print(line, flush=True)

    print("\n" + "=" * 100)
    print(f"全期間合計（{label}）")
    print("=" * 100)
    for s in SCHEMES:
        t = totals[s]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        hitrate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        vals = [v for v in monthly_roi[s] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{s:<10}: {t['n']}R 的中{t['hit']} ({hitrate:.1f}%) "
              f"投資{t['bet']:,.0f} → 回収{t['ret']:,.0f} ROI {roi:.1f}%  月次標準偏差={sd:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--label", default="全期間")
    args = ap.parse_args()
    main(args.date_from, args.date_to, args.label)
