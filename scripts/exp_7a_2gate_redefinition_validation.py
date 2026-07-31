"""【7A再定義の妥当性検証】S7のD案(mark3撤廃+axis_sum<=1.5)適用に伴う7Aの2ゲート化（2026-07-31）。

## 背景

S7を「axis_sum<=1.5・entropy<=1.8329・overlap∈{0,1}」（mark3ゲート撤廃＝D案）に
変更すると、旧7A定義（S7の3ゲート=axis_sum/entropy/mark3のうちちょうど1つだけ
不合格）との間で重複選出が起きる（mark3のみ不合格のレースが新S7にも旧7Aにも
該当してしまう）。ユーザー判断により7Aも2ゲート化（mark3を排除し
axis_sum<=1.5・entropy<=1.8329の2条件のうちちょうど1つ不合格）して整合を取る。

9Aは対象外（S9はaxis_sum非導入のためS9側のmark3ゲートは今回のS7変更と無関係。
9A自体もS9のmark3ゲートをそのまま使い続けるため、変更不要）。

## 検証内容

- 旧7A（現行本番）: axis_sum<=1.3, entropy<=1.8329, mark3<=1 のうち exactly 1 fail
- 新7A（2ゲート化）: axis_sum<=1.5, entropy<=1.8329 のうち exactly 1 fail
- 新S7・新7Aが相互排他になっている（重複ゼロ）ことも直接検算する

honest: 月次凍結vintageモデル。配分はuniform固定。DB書き込みなし・読み取り専用。
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


def old_7a(c):
    if c.get("wt_overlap_n") not in (0, 1):
        return False
    mark3 = c.get("wt_mark3_overlap_n")
    if mark3 is None:
        return False
    axis_ok = c["axis_sum"] <= 1.3
    ent_ok = c.get("entropy", float("inf")) <= 1.8329
    mark3_ok = mark3 <= 1
    return (not axis_ok) + (not ent_ok) + (not mark3_ok) == 1


def new_7a(c):
    if c.get("wt_overlap_n") not in (0, 1):
        return False
    axis_ok = c["axis_sum"] <= 1.5
    ent_ok = c.get("entropy", float("inf")) <= 1.8329
    return (not axis_ok) + (not ent_ok) == 1


def new_s7(c):
    return (c["axis_sum"] <= 1.5 and c.get("entropy", float("inf")) <= 1.8329
            and c.get("wt_overlap_n") in (0, 1))


def select_daily(candidates, pred, daily_cap=S7_DAILY_CAP):
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
    n = len(selected)
    return {"n": n, "bet": bet, "ret": ret, "hit": hit,
            "hit_rate": hit / n * 100 if n else 0.0,
            "roi": ret / bet * 100 if bet else 0.0}


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}")

    names = ["旧7A(現行)", "新7A(2ゲート化)"]
    monthly_roi = {n: [] for n in names}
    totals = {n: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0.0} for n in names}
    total_days = 0
    overlap_count = 0  # 新S7 と 新7A の重複件数（検算・0であるべき）

    for date_from, date_to, eval_model, win_model in windows:
        n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
        total_days += n_days
        print(f"\n[main] {date_from}〜{date_to}（{n_days}日）", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue

        # 検算: 新S7と新7Aの重複がゼロであることを確認
        for c_ in candidates:
            if new_s7(c_) and new_7a(c_):
                overlap_count += 1

        for name, pred in (("旧7A(現行)", old_7a), ("新7A(2ゲート化)", new_7a)):
            selected = select_daily(candidates, pred)
            s = score(selected, pm)
            monthly_roi[name].append(s["roi"] if s["bet"] else None)
            for k in ("n", "bet", "ret", "hit"):
                totals[name][k] += s[k]
            per_day = s["n"] / n_days
            print(f"  [{name:<16}] n={s['n']:>4} 1日平均={per_day:>5.2f}件 "
                  f"的中率={s['hit_rate']:>5.1f}% ROI={s['roi']:>6.1f}%")

    print("\n" + "=" * 100)
    print(f"検算: 新S7 と 新7A の重複件数 = {overlap_count}（0であるべき）")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("全期間合計（月次vintageモデル・honest walk-forward）")
    print("=" * 100)
    print(f"{'レジーム':<18}{'総R数':>8}{'真の1日平均':>12}{'的中率':>8}{'ROI':>8}"
          f"{'月次ROI標準偏差':>16}{'月次0%回数':>10}")
    for name in names:
        t = totals[name]
        hit_rate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in monthly_roi[name] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        n_zero = sum(1 for v in vals if v == 0.0)
        per_day_true = t["n"] / total_days if total_days else 0.0
        print(f"{name:<18}{t['n']:>8}{per_day_true:>12.2f}{hit_rate:>7.1f}%{roi:>7.1f}%"
              f"{sd:>16.1f}{n_zero:>10}")


if __name__ == "__main__":
    main()
