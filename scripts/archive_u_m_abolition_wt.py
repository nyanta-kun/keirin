#!/usr/bin/env python3
"""廃止ランクの picks_history 行の退避（S2(7PLUS_U)/S3(7PLUS_M) 全廃・2026-07-21〜）。

対象レース数・的中率・期待値の観点で継続困難と判断し全廃。
honest全期間実績: S2(7PLUS_U) ROI84.8%(1155R・全期間では損失圏内)・
S3(7PLUS_M) ROI120.4%(801R・厳選により黒字化していたが全廃対象に含む)。

picks_history の該当行を退避テーブルへ移動して現行集計（kiseki Web・
save_model_eval）から外す（scripts/archive_s1_a_abolition_wt.py と同一パターン）:

  7PLUS_U（#7U・波乱ライン連れ込み）  → picks_history_u_archive
  7PLUS_M（#7M・◎不一致×軸信頼）     → picks_history_m_archive

冪等: 挿入は ON CONFLICT DO NOTHING（PG）／既存重複はそのままスキップ（SQLite）で
既に退避済みの行を再実行しても安全。

SQLite（Mac 正本）と VPS PG（KEIRIN_DB_URL 設定時）の両方を処理する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/archive_u_m_abolition_wt.py [--dry-run]
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
    ("7PLUS_U", "%#7U", "picks_history_u_archive"),
    ("7PLUS_M", "%#7M", "picks_history_m_archive"),
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


def archive_pg(dry_run: bool, db_url: str | None) -> None:
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
    # KEIRIN_DB_URL が設定されていると get_connection が PG 直結になり
    # SQLite 側の退避ができないため、env から退避して SQLite → PG の順に処理する
    db_url = os.environ.pop("KEIRIN_DB_URL", None)
    archive_sqlite(args.dry_run)
    archive_pg(args.dry_run, db_url)
    print("[archive] 完了")


if __name__ == "__main__":
    main()
