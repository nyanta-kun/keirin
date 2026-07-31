"""【S7 axis_sumゲートの再設計】的中率を十分に確保した上で安定したROIを狙う（2026-07-30）。

## 方針転換の背景

ev_threshold_filter配分は本命除外バグが確定し実装禁止
（[[keirin_s7_ev_filter_favorite_exclusion_bug_2026_07_30]]）。ユーザー方針:
「まずはベースは十分な的中率の上での安定したROI確保をS7とします」。

配分（staking）はuniformのまま変更しない（安全確認済み）。レース選択
（axis_sumゲート）のみを、EV最大化ではなく **的中率とROI変動の安定性**
という基準で再検討する。

## 現行仕様の確認（`src/strategy_wt.py:305-321`）

`S7_AXIS_SUM_MAX = 1.3`（axis_sum**以下**を採用）。axis_sumが高い
（軸2車のtop3_probs合計が大きい＝波乱寄りでない・堅い決着）レースは
「三連複が安くなりやすい極端な人気決着」として**除外**する設計。
2024-2026(935日)の検証でaxis_sum<=1.3採用によりROI 131.3%→147.1%
（旧汚染モデル時代の数値、絶対値は参考にしない）。

本スクリプトはこの閾値・方向を honest 月次vintageモデルで**掃引**し、
「的中率」「ROI」「月次ROIの変動（標準偏差）」「平均payout」を全て並べて
比較する。entropy/wt_overlap/wt_mark3_overlapゲートは現行のまま固定
（配分同様、staking以外で唯一まだ安全性未確認だった部分）。

## 掃引するレジーム

- `<=1.0` / `<=1.3`(現行) / `<=1.5` / `<=1.7` / `<=2.0` / 上限なし（axis_sum無視）
- 参考: `>=1.3`（現行の逆＝axis_sumが高い「堅い」レースのみ）

各レジームで日次cap（S7_DAILY_CAP=12・entropy昇順トリム）は現行のまま適用する
（`s7_evening_reselect`をそのまま再利用し、axis_sum条件だけ差し替え）。

honest: 月次凍結vintageモデル（`monthly_windows()`）。配分はuniform固定。
DB書き込みなし・読み取り専用。
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import S7_DAILY_CAP, S7_ENTROPY_MAX, S7_MARK3_OVERLAP_MAX, S7_STAKE
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

REGIMES = [
    ("<=1.0", lambda a: a <= 1.0, S7_DAILY_CAP),
    ("<=1.3(現行)", lambda a: a <= 1.3, S7_DAILY_CAP),
    ("<=1.5", lambda a: a <= 1.5, S7_DAILY_CAP),
    ("<=1.7", lambda a: a <= 1.7, S7_DAILY_CAP),
    ("<=2.0", lambda a: a <= 2.0, S7_DAILY_CAP),
    ("上限なし", lambda a: True, S7_DAILY_CAP),
    (">=1.3(逆方向)", lambda a: a >= 1.3, S7_DAILY_CAP),
    # ユーザー要望: 1日あたりの採用件数を固定目標にした場合（axis_sum制限なし・
    # entropy昇順=最も自信がある順でその日の上位N件のみ採用。production の
    # s7_evening_reselect と同じ仕組みでcap値だけ変える）
    ("目標5件/日", lambda a: True, 5),
    ("目標10件/日", lambda a: True, 10),
]


def gate_and_select(candidates, axis_sum_pred, daily_cap):
    """s7_daily_select 相当（axis_sum条件・日次capを差し替え可能）＋日次capトリム。"""
    pool = [
        c for c in candidates
        if axis_sum_pred(c["axis_sum"])
        and c.get("entropy", float("inf")) <= S7_ENTROPY_MAX
        and c.get("wt_overlap_n") in (0, 1)
        and c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX
    ]
    pool.sort(key=lambda c: c["axis_sum"])

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
        combos = [frozenset({axis1, axis2, x}) for x in c_["others"] if frozenset({axis1, axis2, x}) in trio]
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

    # レジームごとの月次ROIを蓄積（変動＝標準偏差の計算用）
    monthly_roi = {name: [] for name, _, _ in REGIMES}
    totals = {name: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0.0} for name, _, _ in REGIMES}

    from datetime import date as _date
    for date_from, date_to, eval_model, win_model in windows:
        n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
        print(f"\n[main] {date_from}〜{date_to}（{n_days}日）", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue
        for name, pred, cap in REGIMES:
            selected = gate_and_select(candidates, pred, cap)
            s = score(selected, pm)
            monthly_roi[name].append(s["roi"] if s["bet"] else None)
            for k in ("n", "bet", "ret", "hit"):
                totals[name][k] += s[k]
            per_day = s["n"] / n_days
            print(f"  [{name:<14}] n={s['n']:>4} 1日平均={per_day:>5.2f}件 "
                  f"的中率={s['hit_rate']:>5.1f}% ROI={s['roi']:>6.1f}% "
                  f"平均払戻(的中時)={s['avg_payout_on_hit']:>7.0f}円")

    print("\n" + "=" * 108)
    print("全期間合計（月次vintageモデル・honest walk-forward・配分はuniform固定）")
    print("=" * 108)
    print(f"{'レジーム':<16}{'総R数':>8}{'日次R数':>9}{'的中率':>8}{'ROI':>8}{'月次ROI標準偏差':>16}{'月次0%回数':>10}")
    n_months = len(windows)
    for name, _, _ in REGIMES:
        t = totals[name]
        hit_rate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in monthly_roi[name] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        n_zero = sum(1 for v in vals if v == 0.0)
        print(f"{name:<16}{t['n']:>8}{t['n']/n_months:>9.1f}{hit_rate:>7.1f}%{roi:>7.1f}%"
              f"{sd:>16.1f}{n_zero:>10}")

if __name__ == "__main__":
    main()
