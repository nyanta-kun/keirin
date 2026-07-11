#!/usr/bin/env python3
"""picks_history を SQLite → PostgreSQL に直接 upsert するスクリプト。

migrate_sqlite_to_pg.py は id 列を含めて INSERT するため、
PostgreSQL 側の SERIAL PK と衝突してサイレントに失敗するケースがある。
このスクリプトは id を除いて ON CONFLICT (race_key) DO UPDATE する。

実行例:
  export KEIRIN_DB_URL="postgresql://hrdb_user:pass@host:5432/hrdb"
  python3 scripts/push_picks_history_to_pg.py
  python3 scripts/push_picks_history_to_pg.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "keirin.db"
BATCH_SIZE  = 2000

COLS = [
    "race_date", "race_key", "rank", "pred_combo",
    "n_combos", "hit", "payout", "bet_amount", "route", "miwokuri",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="stale ガードを無視して強制実行")
    args = parser.parse_args()

    db_url = os.environ.get("KEIRIN_DB_URL", "")
    if not db_url:
        print("ERROR: KEIRIN_DB_URL が未設定です。")
        sys.exit(1)

    if not SQLITE_PATH.exists():
        print(f"[skip] SQLite DB が見つかりません（VPS直接書き込み運用）: {SQLITE_PATH}")
        sys.exit(0)

    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.row_factory = sqlite3.Row

    # ── stale ガード（migrate_sqlite_to_pg.py と同型・2026-07-12 追加）──────
    # VPS 直接書き込み運用へ移行後は SQLite が凍結されている。誤って再実行すると
    # PG の picks_history（prerace_decisions 再採点済みの hit/payout/miwokuri）を
    # 古いスナップショットで上書きしてしまうため、SQLite の最新日が PG より
    # 2 日以上古い場合はスキップする。
    if not args.force:
        try:
            _row = sqlite_conn.execute(
                "SELECT MAX(race_date) FROM picks_history").fetchone()
            sqlite_max = _row[0] if _row else None
            _pg = psycopg2.connect(db_url, connect_timeout=10)
            with _pg.cursor() as _cur:
                _cur.execute("SELECT MAX(race_date)::text FROM keirin.picks_history")
                pg_max = _cur.fetchone()[0]
            _pg.close()
            if sqlite_max and pg_max and str(sqlite_max) < str(pg_max):
                from datetime import datetime
                _days = (datetime.strptime(str(pg_max)[:10], "%Y-%m-%d")
                         - datetime.strptime(str(sqlite_max)[:10], "%Y-%m-%d")).days
                if _days >= 2:
                    print(f"[skip] SQLite が stale です（SQLite最新 {sqlite_max} / "
                          f"PG最新 {pg_max}, {_days}日差）。PG を古い値で上書きする恐れが"
                          "あるためスキップします（--force で強制実行）。")
                    sqlite_conn.close()
                    sys.exit(0)
        except Exception as e:  # noqa: BLE001 - ガード失敗時は安全側（中止）に倒す
            print(f"ERROR: stale チェック失敗のため中止します（--force で強制実行）: {e}")
            sqlite_conn.close()
            sys.exit(1)

    # SQLite の実在カラムを確認（miwokuri/route が無い場合は除外）
    existing_cols_info = sqlite_conn.execute("PRAGMA table_info(picks_history)").fetchall()
    sqlite_col_names = {row[1] for row in existing_cols_info}
    cols = [c for c in COLS if c in sqlite_col_names]
    print(f"移行カラム: {cols}", flush=True)

    total = sqlite_conn.execute("SELECT COUNT(*) FROM picks_history").fetchone()[0]
    print(f"SQLite picks_history: {total:,} 行", flush=True)

    if args.dry_run:
        print("(dry-run: DB書き込みなし)")
        sqlite_conn.close()
        return

    pg_conn = psycopg2.connect(db_url, connect_timeout=30)
    pg_conn.autocommit = False

    non_conf = [c for c in cols if c != "race_key"]
    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_conf)
    upsert_sql = (
        f"INSERT INTO keirin.picks_history ({', '.join(cols)}) VALUES %s "
        f"ON CONFLICT (race_key) DO UPDATE SET {upd}"
    )

    cursor = sqlite_conn.execute(f"SELECT {', '.join(cols)} FROM picks_history")
    pg_cur = pg_conn.cursor()
    inserted = 0

    try:
        while True:
            raw = cursor.fetchmany(BATCH_SIZE)
            if not raw:
                break
            data = [
                tuple(bool(row[c]) if c == "miwokuri" else row[c] for c in cols)
                for row in raw
            ]
            psycopg2.extras.execute_values(pg_cur, upsert_sql, data, page_size=BATCH_SIZE)
            inserted += len(raw)
            print(f"  {inserted:,}/{total:,} 行処理中...", flush=True)

        pg_conn.commit()
        print(f"完了: {inserted:,} 行を keirin.picks_history に upsert しました。", flush=True)
    except Exception as e:
        pg_conn.rollback()
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        pg_conn.close()
        sqlite_conn.close()


if __name__ == "__main__":
    main()
