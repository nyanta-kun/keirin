#!/usr/bin/env python3
"""S9(NINE_S9) の全期間honest再構築（quarterly walk-forwardモデル使用）。

rebuild_s7_walkforward_pg.py の9車版。VPS PG一本化（2026-07-22〜）に伴い、
環境変数をpopしないPG直読みの単発スクリプトとして実行する。

【2026-07-29改定】期間定義を`src.wt_vintage_config.monthly_windows()`（月次凍結
vintageモデル・唯一の正本）に統一。詳細は`rebuild_s7_walkforward_pg.py`の
モジュールdocstring参照（同一設計）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s9_walkforward_pg.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_s9_rank_wt import build_rows
from src.database import get_connection
from src.wt_vintage_config import monthly_windows


def wipe_rows_pg(date_from: str, date_to: str, dry_run: bool) -> None:
    cond = "rank='NINE_S9' AND race_key LIKE '%#9S9' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
                          (date_from, date_to)).fetchone()[0]
        print(f"[rebuild-s9-pg] 既存 #9S9 行（{date_from}〜{date_to}）: {n}件 → 削除"
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
            " trio_payout,bet_amount,route,miwokuri,gate_label) "
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
            " :payout,:trio_payout,:bet_amount,'wt',:miwokuri,:gate_label)",
            rows_ins)
        conn.commit()
    print(f"[rebuild-s9-pg] {len(rows)}件 書き込み完了（VPS PG）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tail-only", action="store_true",
                     help="直近月（今月）の窓のみ再構築する日次軽量運用向けオプション。"
                          "確定済み過去月は結果が変わらないため毎日再計算する必要がなく、"
                          "これのみ再実行すれば直近日をhonestな状態に保てる。")
    args = ap.parse_args()

    windows = monthly_windows()
    if args.tail_only:
        windows = windows[-1:]
    wipe_from = windows[0][0]
    wipe_to = windows[-1][1]

    all_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"\n[rebuild-s9-pg] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
        print(f"[rebuild-s9-pg]   S9: {len(rows)}R ({len(rows)/n_days:.2f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-s9-pg] ===== 全期間合計 =====")
    print(f"[rebuild-s9-pg] S9: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    wipe_rows_pg(wipe_from, wipe_to, args.dry_run)
    insert_rows_pg(all_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s9-pg] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
