"""SQLite → PostgreSQL データ移行スクリプト（ワンショット）

使い方:
  export KEIRIN_DB_URL="postgresql://user:pass@vps-host:5432/keiba"
  python scripts/migrate_sqlite_to_pg.py [--dry-run] [--full]

オプション:
  --dry-run  実行せずに行数のみ表示
  --full     wt_odds(34M行) と wt_weather(1.3M行) も移行（デフォルトは除外）

事前条件:
  - KEIRIN_DB_URL 環境変数が設定されていること
  - kiseki の Alembic マイグレーション (c1d2e3f4a5b6) 適用済みであること
  - psycopg2-binary がインストールされていること: pip install psycopg2-binary

移行対象テーブル（keirin スキーマ）:
  デフォルト: venue_info, wt_races, wt_entries, wt_odds_snapshot, picks_history
  --full 時:  上記 + wt_odds, wt_weather
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "keirin.db"

# 必須テーブル（kiseki フロントエンドが参照するもの）
TABLES_ESSENTIAL = [
    # (table_name, conflict_columns)
    ("venue_info",       ("venue_code",)),
    ("wt_races",         ("race_key",)),
    ("wt_entries",       ("race_key", "frame_no")),
    ("wt_odds_snapshot", ("race_key", "bet_type", "combination", "snapshot_type")),
    ("picks_history",    ("race_key",)),
]

# 大容量テーブル（--full 時のみ移行）
TABLES_LARGE = [
    ("wt_odds",    ("race_key", "bet_type", "combination")),
    ("wt_weather", ("venue_id", "dt_hour")),
]

BATCH_SIZE = 2000


def get_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_pg_columns(pg_conn, table: str) -> set[str]:
    """PostgreSQL テーブルの実在カラム名を取得する。"""
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'keirin' AND table_name = %s",
        (table,),
    )
    return {r[0] for r in cur.fetchall()}


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn,
    table: str,
    conflict_cols: tuple[str, ...],
    dry_run: bool,
) -> int:
    rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        print(f"  {table}: 0 行（スキップ）")
        return 0

    sqlite_cols = list(rows[0].keys())

    if dry_run:
        print(f"  {table}: {len(rows)} 行（dry-run・スキップ）")
        return len(rows)

    # PostgreSQL に存在するカラムのみ使用（SQLite 独自カラムを除外）
    pg_cols = get_pg_columns(pg_conn, table)
    cols = [c for c in sqlite_cols if c in pg_cols]
    if len(cols) < len(sqlite_cols):
        skipped = set(sqlite_cols) - set(cols)
        print(f"  {table}: PG に存在しないカラムをスキップ: {skipped}")

    conflict_set = set(conflict_cols)
    non_conf = [c for c in cols if c not in conflict_set]

    placeholders = ", ".join(["%s"] * len(cols))
    if non_conf:
        upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_conf)
        upsert = (
            f"INSERT INTO keirin.{table} ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {upd}"
        )
    else:
        upsert = (
            f"INSERT INTO keirin.{table} ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
        )

    total = len(rows)
    cur = pg_conn.cursor()
    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        data = [tuple(r[c] for c in cols) for r in batch]
        cur.executemany(upsert, data)
        inserted += len(batch)
        if inserted % 50000 == 0:
            print(f"    {table}: {inserted}/{total} 行...")

    pg_conn.commit()
    print(f"  {table}: {inserted}/{total} 行 → keirin.{table}")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 移行")
    parser.add_argument("--dry-run", action="store_true", help="実行せずに行数のみ表示")
    parser.add_argument("--full", action="store_true", help="wt_odds(34M) と wt_weather(1.3M) も移行")
    args = parser.parse_args()

    db_url = os.environ.get("KEIRIN_DB_URL", "")
    if not db_url:
        print("ERROR: KEIRIN_DB_URL 環境変数が未設定です。")
        sys.exit(1)

    if not SQLITE_PATH.exists():
        print(f"ERROR: SQLite DB が見つかりません: {SQLITE_PATH}")
        sys.exit(1)

    import psycopg2
    import psycopg2.extras

    tables = TABLES_ESSENTIAL + (TABLES_LARGE if args.full else [])

    print(f"SQLite: {SQLITE_PATH}")
    print(f"PostgreSQL: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print(f"dry-run: {args.dry_run}")
    print(f"mode: {'--full (全テーブル)' if args.full else 'essential only (wt_odds/wt_weather 除外)'}")
    print()

    sqlite_conn = get_sqlite(SQLITE_PATH)
    pg_conn = psycopg2.connect(db_url)

    total_rows = 0
    for table, conflict_cols in tables:
        # テーブルが SQLite に存在するか確認
        exists = sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            print(f"  {table}: SQLite に存在しない（スキップ）")
            continue
        n = migrate_table(sqlite_conn, pg_conn, table, conflict_cols, args.dry_run)
        total_rows += n

    sqlite_conn.close()
    pg_conn.close()

    print()
    print(f"完了: 合計 {total_rows} 行を移行しました。")


if __name__ == "__main__":
    main()
