"""【S7 5件/日・10件/日達成検証】entropyゲートを含む各ゲートのボトルネック切り分け（2026-07-30）。

## 経緯

[[keirin_netkeirin_race_selection_verification_2026_07_30]]系の続き。
`exp_s7_axis_sum_regime_sweep.py` の全期間honest検証で、axis_sumの上限を
撤廃しても「目標10件/日」ですら実日数ベースで約3.0件/日にしか届かないと判明
（総R数2,858 ÷ 実日数約942日 ≈ 3.03件/日）。ユーザー要望によりentropyゲートを
含めた真のボトルネックを特定し、緩めた場合の到達件数・的中率・ROIを検証する。

## Part 1: ゲート別の通過数診断（どのゲートが真のボトルネックか）

`s7_select_axis()`自体（win_probs∩top3_probs重なり>=1が必要）で既に一部の
レースが候補にすらならない。そこから先の3ゲート
（entropy<=1.8329 / wt_overlap_n∈{0,1} / wt_mark3_overlap_n<=1）を
個別に・組み合わせて通過数を数え、どのゲートが最も絞り込んでいるかを
月次で集計する。

## Part 2: entropy閾値 × 日次capターゲットの掃引

entropyの理論上限は7車・一様分布でln(7)≈1.9459。現行1.8329は理論上限の
94.2%に相当し、それ自体は緩い部類だが、`s7_select_axis`の重なり要件や
wt_overlap/mark3ゲートとの組み合わせで実効的な絞り込みが強くなっている
可能性がある。entropy閾値を 1.8329(現行)/1.90/1.9459(無制限)で振り、
axis_sumは無制限固定、wt_overlap/wt_mark3は現行のまま、日次capを5/10/12で
組み合わせて評価する。

honest: 月次凍結vintageモデル。配分はuniform固定。DB書き込みなし・読み取り専用。
"""
import statistics
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import S7_MARK3_OVERLAP_MAX, S7_STAKE
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

ENTROPY_LN7 = 1.9459101090932196   # 理論上限 ln(7)（7車一様分布）

ENTROPY_REGIMES = [1.8329, 1.90, 1.93, ENTROPY_LN7]
CAP_TARGETS = [5, 10, 12]


def gate_pass_counts(candidates):
    """各ゲート単独・組み合わせでの通過数を返す（axis_sumは無制限とみなす）。"""
    n_total = len(candidates)
    n_entropy = sum(1 for c in candidates if c.get("entropy", float("inf")) <= 1.8329)
    n_overlap = sum(1 for c in candidates if c.get("wt_overlap_n") in (0, 1))
    n_mark3 = sum(1 for c in candidates if c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX)
    n_all = sum(
        1 for c in candidates
        if c.get("entropy", float("inf")) <= 1.8329
        and c.get("wt_overlap_n") in (0, 1)
        and c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX
    )
    return {"total": n_total, "entropy_only": n_entropy, "overlap_only": n_overlap,
            "mark3_only": n_mark3, "all_gates": n_all}


def select_with_entropy_cap(candidates, entropy_max, daily_cap):
    """axis_sum無制限・wt_overlap/mark3は現行のまま・entropy閾値と日次capのみ可変。"""
    pool = [
        c for c in candidates
        if c.get("entropy", float("inf")) <= entropy_max
        and c.get("wt_overlap_n") in (0, 1)
        and c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX
    ]
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
    payouts_on_hit = []
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
            pay = trio_pay * S7_STAKE // 100
            ret += pay
            payouts_on_hit.append(trio_pay)
    n = len(selected)
    return {
        "n": n, "bet": bet, "ret": ret, "hit": hit,
        "hit_rate": hit / n * 100 if n else 0.0,
        "roi": ret / bet * 100 if bet else 0.0,
        "avg_payout_on_hit": statistics.mean(payouts_on_hit) if payouts_on_hit else 0.0,
    }


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}")

    gate_totals = {"total": 0, "entropy_only": 0, "overlap_only": 0,
                   "mark3_only": 0, "all_gates": 0}
    total_days = 0

    regime_keys = [(e, cap) for e in ENTROPY_REGIMES for cap in CAP_TARGETS]
    monthly_roi = {k: [] for k in regime_keys}
    totals = {k: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0.0} for k in regime_keys}

    for date_from, date_to, eval_model, win_model in windows:
        n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
        total_days += n_days
        print(f"\n[main] {date_from}〜{date_to}（{n_days}日）", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue

        gp = gate_pass_counts(candidates)
        for k, v in gp.items():
            gate_totals[k] += v
        print(f"  [ゲート通過数] 生候補={gp['total']} entropy単独={gp['entropy_only']} "
              f"overlap単独={gp['overlap_only']} mark3単独={gp['mark3_only']} "
              f"全ゲート通過={gp['all_gates']}")

        for entropy_max in ENTROPY_REGIMES:
            for cap in CAP_TARGETS:
                selected = select_with_entropy_cap(candidates, entropy_max, cap)
                s = score(selected, pm)
                key = (entropy_max, cap)
                monthly_roi[key].append(s["roi"] if s["bet"] else None)
                for kk in ("n", "bet", "ret", "hit"):
                    totals[key][kk] += s[kk]
                per_day = s["n"] / n_days
                print(f"    entropy<={entropy_max:.4f} cap={cap:<3} n={s['n']:>4} "
                      f"1日平均={per_day:>5.2f}件 的中率={s['hit_rate']:>5.1f}% "
                      f"ROI={s['roi']:>6.1f}%")

    print("\n" + "=" * 116)
    print("Part1: ゲート別通過数（全期間合計・axis_sum無制限とみなした場合の内訳）")
    print("=" * 116)
    for k, v in gate_totals.items():
        print(f"  {k:<14}: {v}")
    print(f"  実日数合計: {total_days}日")

    print("\n" + "=" * 116)
    print("Part2: entropy閾値 × 日次capターゲット 全期間合計（配分uniform固定）")
    print("=" * 116)
    print(f"{'entropy上限':<14}{'cap':>5}{'総R数':>8}{'真の1日平均':>12}"
          f"{'的中率':>8}{'ROI':>8}{'月次ROI標準偏差':>16}{'月次0%回数':>10}")
    for entropy_max in ENTROPY_REGIMES:
        for cap in CAP_TARGETS:
            key = (entropy_max, cap)
            t = totals[key]
            hit_rate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
            roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
            vals = [v for v in monthly_roi[key] if v is not None]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            n_zero = sum(1 for v in vals if v == 0.0)
            per_day_true = t["n"] / total_days if total_days else 0.0
            print(f"entropy<={entropy_max:<7.4f}{cap:>5}{t['n']:>8}{per_day_true:>12.2f}"
                  f"{hit_rate:>7.1f}%{roi:>7.1f}%{sd:>16.1f}{n_zero:>10}")


if __name__ == "__main__":
    main()
