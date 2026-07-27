#!/usr/bin/env python3
"""S1(SEVEN_S1) の全期間honest再構築（quarterly walk-forwardモデル使用・VPS PG専用）。

rebuild_s1_walkforward.py はローカルSQLite前提（KEIRIN_DB_URLをpopして
ローカル読み取り）だが、2026-07-22にローカルSQLiteは廃止されVPS PGへ
一本化済み（wt_odds含め2024-01-01〜のtrifectaオッズを確認済み）。
S4/S9のrebuild_s4_walkforward_pg.py/rebuild_s9_walkforward_pg.pyと同じ設計で、
環境変数をpopしないPG直読みの単発スクリプトとして実行する（get_connection()
単一経路のみ使用。backfill_s1w_rank_wt.wipe_rows/insert_rowsをそのまま使うと
「ローカル+VPSミラー」二重書き込みが同一PGへ二重に当たるため専用実装）。

2026-07-27: entropyゲート(S1W_ENTROPY_MAX)導入分の再構築にも使用。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s1_walkforward_pg.py [--dry-run]
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_s1w_rank_wt import build_rows
from src.database import get_connection


def wipe_rows_pg(date_from: str, date_to: str, dry_run: bool) -> None:
    cond = "rank='SEVEN_S1' AND race_key LIKE '%#7S1' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
                          (date_from, date_to)).fetchone()[0]
        print(f"[rebuild-s1-pg] 既存 #7S1 行（{date_from}〜{date_to}）: {n}件 → 削除"
              f"{'（dry-run）' if dry_run else ''}")
        if not dry_run and n:
            conn.execute(f"DELETE FROM picks_history WHERE {cond}", (date_from, date_to))
            conn.commit()


def insert_rows_pg(rows: list[dict], dry_run: bool) -> None:
    if dry_run or not rows:
        return
    rows_ins = [{**r, "miwokuri": False} for r in rows]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trifecta_payout,bet_amount,route,miwokuri) "
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
            " :payout,:trifecta_payout,:bet_amount,'wt',:miwokuri)",
            rows_ins)
        conn.commit()
    print(f"[rebuild-s1-pg] {len(rows)}件 書き込み完了（VPS PG）")


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


TAIL_FROM = "2026-04-13"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="末尾窓の終了日（省略時は昨日）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tail-only", action="store_true",
                     help="末尾窓（現行eval modelのholdout範囲）のみ再構築する日次軽量運用向け"
                          "オプション。四半期vintageモデル分は既に確定済みで結果が変わらない"
                          "ため毎日再計算する必要がなく、これのみ再実行すれば直近日を"
                          "honestな状態に保てる。")
    args = ap.parse_args()
    if not args.end:
        args.end = (date.today() - timedelta(days=1)).isoformat()

    if args.tail_only:
        quarters = [(TAIL_FROM, args.end, "lgbm_wt_eval", "lgbm_wt_win_eval")]
        wipe_from = TAIL_FROM
    else:
        quarters = list(QUARTERS)
        quarters.append((TAIL_FROM, args.end, "lgbm_wt_eval", "lgbm_wt_win_eval"))
        wipe_from = "2024-01-01"

    all_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in quarters:
        print(f"\n[rebuild-s1-pg] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
        print(f"[rebuild-s1-pg]   S1: {len(rows)}R ({len(rows)/n_days:.1f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-s1-pg] ===== 全期間合計 =====")
    print(f"[rebuild-s1-pg] S1: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    wipe_rows_pg(wipe_from, args.end, args.dry_run)
    insert_rows_pg(all_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s1-pg] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
