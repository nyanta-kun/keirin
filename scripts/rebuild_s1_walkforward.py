#!/usr/bin/env python3
"""S1(SEVEN_S1)行を四半期対応のリークなしモデルで全期間再構築する（2026-07-19）。

背景: backfill_s1w_rank_wt.py の既定 --model lgbm_wt_eval --win-model lgbm_wt_win は
どちらも「今日学習した・全履歴を知っているモデル」で過去のpicks_historyを構築していた
（S3で発覚したのと同型のリーク・[[keirin_composite_ratio_gate]]参照）。
本スクリプトは四半期ウォークフォワードモデル（eval_q24xx/win_q24xx等）で
SEVEN_S1行をリークなしで全期間再構築する。あわせてS1W_TOP3_GAP_MIN=0.22への
閾値変更（strategy_wt.py）も同時に反映する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s1_walkforward.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_s1w_rank_wt import build_rows, insert_rows, wipe_rows
from scripts.rebuild_coverage_guard import assert_local_covers_pg

# get_connection() は KEIRIN_DB_URL が立っていると読み取りも VPS PG に向く。
# PG側wt_oddsは2026-06以降のみのミラーで全履歴を持たないため、読み取りは
# 常にローカル完全SQLiteを使う必要がある（rebuild_s3_walkforward.pyと同じ対策）。
_PG_URL = os.environ.pop("KEIRIN_DB_URL", None)

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
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="末尾窓の終了日（省略時は昨日）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                     help="ローカルDBカバレッジ不足の警告を無視して続行する")
    args = ap.parse_args()
    if not args.end:
        from datetime import date, timedelta
        args.end = (date.today() - timedelta(days=1)).isoformat()

    assert_local_covers_pg("2024-01-01", args.end, _PG_URL, args.force)

    quarters = list(QUARTERS)
    quarters.append(("2026-04-13", args.end, "lgbm_wt_eval", "lgbm_wt_win_eval"))

    all_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in quarters:
        print(f"\n[rebuild-s1] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (
            __import__("datetime").date.fromisoformat(date_to)
            - __import__("datetime").date.fromisoformat(date_from)
        ).days + 1
        print(f"[rebuild-s1]   S1: {len(rows)}R ({len(rows)/n_days:.1f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-s1] ===== 全期間合計 =====")
    print(f"[rebuild-s1] S1: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    if _PG_URL:
        os.environ["KEIRIN_DB_URL"] = _PG_URL
    wipe_rows("2024-01-01", args.end, args.dry_run)
    insert_rows(all_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s1] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
