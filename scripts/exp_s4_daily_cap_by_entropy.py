"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

S4 entropy<=1.8329ゲート通過後、突出日（今日2026-07-26=26件など）の対処として
日次件数capを再導入すべきか検証する（2026-07-26・ユーザー要望「朝夕合わせて
10レースちょっとに絞りたい」への対応）。

axis_sum<=1.3 ゲート通過候補（entropy計算込み）を全期間で集め、
  1. entropy<=S4_ENTROPY_MAX ゲート通過後の日次件数分布を確認
  2. 日内でentropy昇順に順位付けし、順位別（1位・2位…）の的中/ROIが
     単調に悪化するか（＝entropyランキングが日内でも有効な選別基準か）を確認
  3. 日次capをK件（entropy昇順）で導入した場合の全期間ROI/日次平均件数を
     K=8/10/12/15/無制限で比較

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_daily_cap_by_entropy.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_s4_entropy_uncapped_wt import build_uncapped
from src.strategy_wt import S4_ENTROPY_MAX

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
    ("2026-04-13", "2026-07-25", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label}: n=0")
        return
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"  {label:<30} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%")


def main():
    all_cands = []
    for f, t, m, w in QUARTERS:
        rows = build_uncapped(m, w, f, t)
        all_cands.extend(rows)
        print(f"{f}~{t}: axis_sum<=1.3ゲート通過 n={len(rows)}", flush=True)

    # entropyゲート適用（現行ライブと同一条件）
    gated = [r for r in all_cands if r["entropy"] <= S4_ENTROPY_MAX]
    print(f"\n現行(entropy<=1.8329)ゲート通過合計: n={len(gated)}")
    summarize(gated, "現行ゲート全体")

    # 日次件数分布
    by_day = defaultdict(list)
    for r in gated:
        by_day[r["race_date"]].append(r)
    counts = sorted((len(v) for v in by_day.values()), reverse=True)
    print(f"日数={len(by_day)}  平均={len(gated)/len(by_day):.2f}件/日  "
          f"最大={counts[0]}  上位10日={counts[:10]}")

    # 日内entropy順位別の成績（1位=最もentropyが低い＝最も自信がある）
    print("\n===== 日内entropy順位別 成績（同一順位を全日でプール） =====")
    rank_buckets = defaultdict(list)
    for day, cands in by_day.items():
        ordered = sorted(cands, key=lambda r: r["entropy"])
        for i, c in enumerate(ordered, 1):
            rank_buckets[min(i, 15)].append(c)  # 15位以降はまとめる
    for rank in sorted(rank_buckets):
        label = f"日内{rank}位" if rank < 15 else "日内15位以降"
        summarize(rank_buckets[rank], label)

    # 日次capをK件で導入した場合のシミュレーション
    print("\n===== 日次cap(entropy昇順)導入シミュレーション =====")
    for K in (8, 10, 12, 15, None):
        capped = []
        for day, cands in by_day.items():
            ordered = sorted(cands, key=lambda r: r["entropy"])
            capped.extend(ordered if K is None else ordered[:K])
        avg = len(capped) / len(by_day)
        label = f"cap無し（現行）" if K is None else f"日次cap={K}件"
        summarize(capped, f"{label}（平均{avg:.2f}件/日）")


if __name__ == "__main__":
    main()
