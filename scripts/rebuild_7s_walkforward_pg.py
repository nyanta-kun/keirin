#!/usr/bin/env python3
"""S7(RANK_7S)の全期間honest再構築（月次凍結vintageモデル体系・2026-07-29改定）。

[[keirin_s7_foundational_rethink_2026_07_29]]。従来は四半期QUARTERS+静的TAIL_FROM
という2層構造で、TAIL_FROMと実際のlgbm_wt_evalのtest_fromが週次で乖離し続ける
バグ（2週間分がリーク区間化）を抱えていた。加えてQUARTERSが6ファイルに
コピーされ将来の食い違いリスクを抱えていた。

新設計: `src.wt_vintage_config.monthly_windows()`を唯一の正本として使う。
月次凍結モデル(lgbm_wt_eval_mYYMM等)は「その月のレースは必ず前月末までの
データで学習したモデルでスコアする」契約が当月中ずっと不変なため、
「未確定tail」という別概念が構造的に不要になる。`--tail-only`は単に
「直近月の窓のみ再構築」を意味する（月次モデルの学習自体は
`scripts/train_monthly_vintage_models.py`が別途担当）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_7s_walkforward_pg.py [--dry-run]
    PYTHONPATH=. .venv/bin/python scripts/rebuild_7s_walkforward_pg.py --tail-only
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_7s_rank_wt import build_rows
from src.database import get_connection
from src.wt_vintage_config import monthly_windows


def wipe_rows_pg(date_from: str, date_to: str, dry_run: bool) -> None:
    """backfill_7s_rank_wt.wipe_rows と違い get_connection() 単一経路のみ使う
    （KEIRIN_DB_URLをpopしないためget_connection()自体が既にVPS PGを指しており、
    そのままだと元のwipe_rows/insert_rowsの「ローカル+VPSミラー」二重書き込みが
    同一PGへ二重に当たり、insert側はUNIQUE(race_key)違反で失敗するため）。"""
    cond = "rank='RANK_7S' AND race_key LIKE '%#7S' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
                          (date_from, date_to)).fetchone()[0]
        print(f"[rebuild-s4-pg] 既存 #7S 行（{date_from}〜{date_to}）: {n}件 → 削除"
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
    print(f"[rebuild-s4-pg] {len(rows)}件 書き込み完了（VPS PG）")

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
    else:
        wipe_from = windows[0][0]
    wipe_to = windows[-1][1]

    all_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"\n[rebuild-s4-pg] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
        print(f"[rebuild-s4-pg]   S7: {len(rows)}R ({len(rows)/n_days:.1f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-s4-pg] ===== 全期間合計 =====")
    print(f"[rebuild-s4-pg] S7: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    wipe_rows_pg(wipe_from, wipe_to, args.dry_run)
    insert_rows_pg(all_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s4-pg] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
