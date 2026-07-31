"""【C・D・F レジームの全期間honest検証】単月スクリーニングの本確認（2026-07-31）。

`exp_s7_single_month_gate_test.py`（2025-02単月）で以下を確認済み:
  C: mark3ゲート撤廃（axis_sum<=1.3・entropy<=1.8329・overlap∈{0,1}のみ）
     n=45・1.61件/日・的中33.3%・ROI105.0%
  D: mark3撤廃+axis_sum<=1.5
     n=159・5.68件/日・的中36.5%・ROI75.2%
  F: 全緩和（axis_sum<=1.5・entropy<=1.8329のみ・overlap/mark3撤廃）
     n=332・11.86件/日・的中48.8%・ROI75.7%
単月はROI水準の判断には使えない（分散が大きい）ため、全期間honestで確認する。

配分はuniform固定。DB書き込みなし・読み取り専用。
"""
import statistics
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import S7_DAILY_CAP, S7_STAKE
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

REGIMES = [
    ("A:現行(参考)", lambda c: c["axis_sum"] <= 1.3 and c.get("entropy", 9) <= 1.8329
                              and c.get("wt_overlap_n") in (0, 1)
                              and c.get("wt_mark3_overlap_n", 2) <= 1),
    ("C:mark3撤廃", lambda c: c["axis_sum"] <= 1.3 and c.get("entropy", 9) <= 1.8329
                             and c.get("wt_overlap_n") in (0, 1)),
    ("D:mark3撤廃+axis1.5", lambda c: c["axis_sum"] <= 1.5 and c.get("entropy", 9) <= 1.8329
                                       and c.get("wt_overlap_n") in (0, 1)),
    ("F:全緩和(axis1.5)", lambda c: c["axis_sum"] <= 1.5 and c.get("entropy", 9) <= 1.8329),
]


def select(candidates, pred, daily_cap=S7_DAILY_CAP):
    pool = [c for c in candidates if pred(c)]
    by_day = defaultdict(list)
    for c_ in pool:
        by_day[c_["race_date"]].append(c_)
    selected = []
    for _d, day_cands in by_day.items():
        day_sorted = sorted(day_cands, key=lambda c: c["entropy"])
        selected.extend(day_sorted[:daily_cap])
    return selected


def score(selected, pm):
    bet = ret = hit = 0.0
    payouts = []
    for c_ in selected:
        axis1, axis2 = c_["axis1"], c_["axis2"]
        trio = c_["trio"]
        combos = [frozenset({axis1, axis2, x}) for x in c_["others"]
                  if frozenset({axis1, axis2, x}) in trio]
        if not combos:
            continue
        rk = c_["race_key"]
        actual_top3 = c_["actual_top3"]
        is_hit = actual_top3 in combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        bet += len(combos) * S7_STAKE
        if is_hit:
            hit += 1
            ret += trio_pay * S7_STAKE // 100
            payouts.append(trio_pay)
    n = len(selected)
    return {"n": n, "bet": bet, "ret": ret, "hit": hit,
            "hit_rate": hit / n * 100 if n else 0.0,
            "roi": ret / bet * 100 if bet else 0.0,
            "avg_payout": statistics.mean(payouts) if payouts else 0.0}


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}")

    monthly_roi = {name: [] for name, _ in REGIMES}
    totals = {name: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0.0} for name, _ in REGIMES}
    total_days = 0

    for date_from, date_to, eval_model, win_model in windows:
        n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
        total_days += n_days
        print(f"\n[main] {date_from}〜{date_to}（{n_days}日）", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue
        for name, pred in REGIMES:
            selected = select(candidates, pred)
            s = score(selected, pm)
            monthly_roi[name].append(s["roi"] if s["bet"] else None)
            for k in ("n", "bet", "ret", "hit"):
                totals[name][k] += s[k]
            per_day = s["n"] / n_days
            print(f"  [{name:<20}] n={s['n']:>4} 1日平均={per_day:>5.2f}件 "
                  f"的中率={s['hit_rate']:>5.1f}% ROI={s['roi']:>6.1f}% "
                  f"平均払戻(的中時)={s['avg_payout']:>7.0f}円")

    print("\n" + "=" * 112)
    print("全期間合計（月次vintageモデル・honest walk-forward・配分はuniform固定）")
    print("=" * 112)
    print(f"{'レジーム':<20}{'総R数':>8}{'真の1日平均':>12}{'的中率':>8}{'ROI':>8}"
          f"{'月次ROI標準偏差':>16}{'月次0%回数':>10}")
    for name, _ in REGIMES:
        t = totals[name]
        hit_rate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in monthly_roi[name] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        n_zero = sum(1 for v in vals if v == 0.0)
        per_day_true = t["n"] / total_days if total_days else 0.0
        print(f"{name:<20}{t['n']:>8}{per_day_true:>12.2f}{hit_rate:>7.1f}%{roi:>7.1f}%"
              f"{sd:>16.1f}{n_zero:>10}")


if __name__ == "__main__":
    main()
