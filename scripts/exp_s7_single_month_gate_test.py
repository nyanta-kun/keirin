"""【単月テスト】mark3ゲート緩和・axis_sum緩和の組み合わせを1ヶ月で高速検証（2026-07-31）。

entropy緩和は的中率・ROIを悪化させるトレードオフと判明済み
（[[keirin_s7_ev_filter_favorite_exclusion_bug_2026_07_30]]系の続き）。
ゲート別通過数診断で真のボトルネックは`wt_mark3_overlap_n`（◎◯△との重なり
判定・生候補の5-7%しか通過しない）と判明。ユーザー指示により、いきなり
全期間で検証せず、まず1ヶ月で高速に複数レジームを比較する。

対象月: 2025-02（既存の部分ログで生候補数を把握済みの月）。

テストするレジーム（axis_sum/entropy/overlap/mark3 の組み合わせ）:
  A. 現行そのまま（axis_sum<=1.3, entropy<=1.8329, overlap∈{0,1}, mark3<=1）
  B. axis_sum<=1.5（前回honestで的中率・ROI同時改善を確認済み）+ 他は現行
  C. mark3制限撤廃（mark3<=2 実質無制限）+ 他は現行
  D. mark3制限撤廃 + axis_sum<=1.5
  E. mark3<=1のまま + overlap制限も撤廃（wt_overlap_nの値を問わない）
  F. axis_sum<=1.5 + mark3撤廃 + overlap撤廃（最も緩い組み合わせ）

honest: 単月のみ（月次vintageモデル使用）。配分はuniform固定。
DB書き込みなし・読み取り専用。
"""
import statistics
import sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_DAILY_CAP, RANK_7S_STAKE
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

TARGET_MONTH = "2025-02"

REGIMES = [
    ("A:現行", lambda c: c["axis_sum"] <= 1.3 and c.get("entropy", 9) <= 1.8329
                          and c.get("wt_overlap_n") in (0, 1)
                          and c.get("wt_mark3_overlap_n", 2) <= 1),
    ("B:axis_sum<=1.5", lambda c: c["axis_sum"] <= 1.5 and c.get("entropy", 9) <= 1.8329
                                  and c.get("wt_overlap_n") in (0, 1)
                                  and c.get("wt_mark3_overlap_n", 2) <= 1),
    ("C:mark3撤廃", lambda c: c["axis_sum"] <= 1.3 and c.get("entropy", 9) <= 1.8329
                             and c.get("wt_overlap_n") in (0, 1)),
    ("D:mark3撤廃+axis1.5", lambda c: c["axis_sum"] <= 1.5 and c.get("entropy", 9) <= 1.8329
                                       and c.get("wt_overlap_n") in (0, 1)),
    ("E:overlap撤廃", lambda c: c["axis_sum"] <= 1.3 and c.get("entropy", 9) <= 1.8329
                                and c.get("wt_mark3_overlap_n", 2) <= 1),
    ("F:全緩和(axis1.5)", lambda c: c["axis_sum"] <= 1.5 and c.get("entropy", 9) <= 1.8329),
]


def select(candidates, pred, daily_cap=RANK_7S_DAILY_CAP):
    pool = [c for c in candidates if pred(c)]
    pool.sort(key=lambda c: c.get("entropy", 9))
    from collections import defaultdict
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
        bet += len(combos) * RANK_7S_STAKE
        if is_hit:
            hit += 1
            ret += trio_pay * RANK_7S_STAKE // 100
            payouts.append(trio_pay)
    n = len(selected)
    return {"n": n, "bet": bet, "ret": ret, "hit": hit,
            "hit_rate": hit / n * 100 if n else 0.0,
            "roi": ret / bet * 100 if bet else 0.0,
            "avg_payout": statistics.mean(payouts) if payouts else 0.0}


def main():
    windows = monthly_windows()
    window = next((w for w in windows if w[0].startswith(TARGET_MONTH)), None)
    if window is None:
        print(f"対象月 {TARGET_MONTH} が見つかりません")
        return
    date_from, date_to, eval_model, win_model = window
    n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
    print(f"[main] {date_from}〜{date_to}（{n_days}日）eval={eval_model}", flush=True)

    candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
    print(f"[main] 生候補: {len(candidates)}件\n")

    print(f"{'レジーム':<22}{'n':>6}{'1日平均':>9}{'的中率':>8}{'ROI':>8}{'平均払戻(的中時)':>16}")
    for name, pred in REGIMES:
        selected = select(candidates, pred)
        s = score(selected, pm)
        per_day = s["n"] / n_days
        print(f"{name:<22}{s['n']:>6}{per_day:>9.2f}{s['hit_rate']:>7.1f}%"
              f"{s['roi']:>7.1f}%{s['avg_payout']:>16.0f}円")


if __name__ == "__main__":
    main()
