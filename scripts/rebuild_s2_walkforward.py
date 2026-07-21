#!/usr/bin/env python3
"""S2(U)行を四半期対応のリークなしモデルで全期間再構築する（2026-07-21）。

背景: S2(U)のentropy/mto/穴選定は「eval」モデル（pred_prob=3着内確率）に基づく。
build_rows()のデフォルト model_name="lgbm_wt_eval" は S3(M)のgap12計算で問題に
なったのと同じ full_refit=True（ホールドアウトなし・毎回全期間で再学習）モデルで、
2024-01-01〜のS2過去分backfillもこのモデルで行われていた可能性が高い
（= 今日時点の「未来を知るモデル」で過去レースをスコアリングするリーク）。

rebuild_s3_walkforward.py で S3(gap12/win_rank) 用に用意した四半期ウォークフォワード
モデル群（lgbm_wt_eval_qXXXX）は entropy/mto の算出にも使えるため、同じ QUARTERS
定義を流用して S2(U)行のみをリークなしで再構築する。

あわせて 2026-07-21 の厳選判断（U_MTO_MIN: 4.3→4.5）も同時に反映する
（strategy_wt.py の定数を読むため、本スクリプト実行時点の値がそのまま使われる）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s2_walkforward.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_um_rank_wt import build_rows, insert_rows, wipe_rank_rows
from scripts.rebuild_coverage_guard import assert_local_covers_pg

_PG_URL = os.environ.pop("KEIRIN_DB_URL", None)

# S3と同一の四半期定義（win_modelはS2では未使用だが build_rows のシグネチャ上必要）。
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

    all_u_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in quarters:
        print(f"\n[rebuild-s2] {date_from}〜{date_to}  eval={eval_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        u_rows = [r for r in rows if r["rank"] == "7PLUS_U"]
        n_hit = sum(r["hit"] for r in u_rows)
        bet = sum(r["bet_amount"] for r in u_rows)
        pay = sum(r["payout"] for r in u_rows)
        print(f"[rebuild-s2]   S2(U): {len(u_rows)}R 的中{n_hit} "
              f"({n_hit / len(u_rows) * 100 if u_rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_u_rows.extend(u_rows)

    total_hit = sum(r["hit"] for r in all_u_rows)
    total_bet = sum(r["bet_amount"] for r in all_u_rows)
    total_pay = sum(r["payout"] for r in all_u_rows)
    print(f"\n[rebuild-s2] ===== 全期間合計 =====")
    print(f"[rebuild-s2] S2(U): {len(all_u_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_u_rows) * 100 if all_u_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    if _PG_URL:
        os.environ["KEIRIN_DB_URL"] = _PG_URL
    wipe_rank_rows("7PLUS_U", "#7U", "2024-01-01", args.end, args.dry_run)
    insert_rows(all_u_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s2] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
