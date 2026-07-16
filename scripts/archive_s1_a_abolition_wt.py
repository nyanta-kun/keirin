#!/usr/bin/env python3
"""S1(SIX_S1)/A(7PLUS_A) 全廃（2026-07-17）に伴う picks_history 行の退避。

正規プロトコル再検証（学習〜2025-03-31／検証2025-04-01〜2026-03-31／
テスト2026-04-01〜07-15）で両ランクとも検証ROI100%超なし → 全廃。
picks_history の該当行を退避テーブルへ移動して現行集計（kiseki Web・
save_model_eval）から外す。表示系譜に合わせて退避先を分ける:

  SIX_S1（#6S1・6車三連単）  → picks_history_r_archive（S1系譜。旧7PLUS_R と同居）
  7PLUS_A（#7A・二連単）     → picks_history_a_archive（A系譜。旧Aと同居）

SQLite（Mac 正本）と VPS PG（KEIRIN_DB_URL 設定時）の両方を処理する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/archive_s1_a_abolition_wt.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

# (rank, race_key suffix LIKE, 退避先テーブル)
TARGETS = [
    ("SIX_S1", "%#6S1", "picks_history_r_archive"),
    ("7PLUS_A", "%#7A", "picks_history_a_archive"),
]


def archive_sqlite(dry_run: bool) -> None:
    with get_connection() as conn:
        for rank, like, table in TARGETS:
            cond = "rank=? AND race_key LIKE ?"
            n = conn.execute(
                f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
                (rank, like)).fetchone()[0]
            print(f"[archive] SQLite {rank}: {n}件 → {table}"
                  f"{'（dry-run）' if dry_run else ''}")
            if dry_run or not n:
                continue
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} AS "
                "SELECT * FROM picks_history WHERE 0")
            conn.execute(
                f"INSERT INTO {table} SELECT * FROM picks_history WHERE {cond}",
                (rank, like))
            conn.execute(f"DELETE FROM picks_history WHERE {cond}", (rank, like))
        conn.commit()


def archive_pg(dry_run: bool) -> None:
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        print("[archive] KEIRIN_DB_URL 未設定 → VPS PG スキップ")
        return
    import psycopg2
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            for rank, like, table in TARGETS:
                cond = "rank=%s AND race_key LIKE %s"
                cur.execute(
                    f"SELECT COUNT(*) FROM keirin.picks_history WHERE {cond}",
                    (rank, like))
                n = cur.fetchone()[0]
                print(f"[archive] VPS PG {rank}: {n}件 → keirin.{table}"
                      f"{'（dry-run）' if dry_run else ''}")
                if dry_run or not n:
                    continue
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS keirin.{table} "
                    "(LIKE keirin.picks_history INCLUDING ALL)")
                cur.execute(
                    f"INSERT INTO keirin.{table} "
                    f"SELECT * FROM keirin.picks_history WHERE {cond} "
                    "ON CONFLICT DO NOTHING",
                    (rank, like))
                cur.execute(
                    f"DELETE FROM keirin.picks_history WHERE {cond}",
                    (rank, like))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    archive_sqlite(args.dry_run)
    archive_pg(args.dry_run)
    print("[archive] 完了")


if __name__ == "__main__":
    main()
