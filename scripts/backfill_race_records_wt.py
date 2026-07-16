#!/usr/bin/env python3
"""過去レースの S/B 取得・上がりタイムを winticket racecard の再取得で補完する。

winticket racecard ページの window.__PRELOADED_STATE__ 内、
tanStackQuery の "FETCH_KEIRIN_RACE" クエリが持つ `results`（playerId キー）に、
レース単位の以下フィールドが入っている:

    standing        (bool)  S（スタンディング/先行取得）取得の有無
    back            (bool)  B（バック/後方取得）取得の有無
    finalHalfRecord (str)   後半上がりタイム（秒、例: "11.9"）

現行の `src.scraper.winticket.fetch_race_data` はこれらを取り込んでいない
（別セッションで拡張中）。このスクリプトは `fetch_race_data` に依存せず、
`_extract_state` / `_get_query` / `VENUE_SLUGS` / `_BASE` のみを import して
自前で racecard ページの state を読み、wt_entries を補完する。

対象: wt_races から race_date が [--start, --end] 区間かつ status=3（確定）の
レースを日付昇順で列挙する。wt_entries に res_back IS NOT NULL の行が
1行でもあるレースはバックフィル済みとみなしスキップする（resume 対応）。

書き込み先:
    - ローカル SQLite（src.database.get_connection・KEIRIN_DB_URL 未設定前提）
    - 環境変数 PG_MIRROR_URL が設定されていれば、VPS PostgreSQL の
      keirin.wt_entries にも同じ UPDATE をミラーする（psycopg2 使用）

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_race_records_wt.py \
        --start 2024-01-01 --end 2026-07-16

    # 動作確認（先頭10レースのみ・書き込みなし）
    PYTHONPATH=. .venv/bin/python scripts/backfill_race_records_wt.py \
        --limit 10 --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.database import get_connection
from src.scraper.winticket import _BASE, VENUE_SLUGS, _extract_state, _get_query

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.5",
}

# レース間リクエスト間隔（秒）
_REQUEST_INTERVAL = 1.2

# 追加する列: (列名, SQLite型, PostgreSQL型)
_NEW_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("res_standing", "INTEGER", "INTEGER"),
    ("res_back", "INTEGER", "INTEGER"),
    ("final_half", "REAL", "DOUBLE PRECISION"),
)


def _ensure_sqlite_columns(conn: Any) -> None:
    """wt_entries に res_standing/res_back/final_half 列が無ければ追加する（SQLite用）。

    KEIRIN_DB_URL が設定されている場合 conn は _PgConn（PostgreSQL 互換ラッパー）に
    なるが、PRAGMA / ALTER TABLE ADD COLUMN は _pg_translate 側のスキップ規則で
    自動的に no-op になるため、ここでは分岐せずそのまま呼び出してよい。
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(wt_entries)").fetchall()}
    for col, sqlite_type, _pg_type in _NEW_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE wt_entries ADD COLUMN {col} {sqlite_type}")


def _ensure_pg_mirror_columns(pg_conn: Any) -> None:
    """PG_MIRROR_URL 先の keirin.wt_entries に不足列を追加する（PostgreSQL は
    ADD COLUMN IF NOT EXISTS が使えるため冪等に実行できる）。"""
    with pg_conn.cursor() as cur:
        for col, _sqlite_type, pg_type in _NEW_COLUMNS:
            cur.execute(f"ALTER TABLE keirin.wt_entries ADD COLUMN IF NOT EXISTS {col} {pg_type}")
    pg_conn.commit()


def _fetch_race_state(session: requests.Session, venue_id: str, cup_id: str,
                       day_index: int, race_no: int) -> dict | None:
    """racecard ページを取得し FETCH_KEIRIN_RACE クエリの data を返す。取得不可時 None。"""
    slug = VENUE_SLUGS.get(venue_id)
    if not slug:
        return None
    url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/{race_no}"
    try:
        resp = session.get(url, headers=_HEADERS, timeout=15)
    except requests.RequestException as exc:
        print(f"[backfill] GET失敗 {url}: {exc}", flush=True)
        return None
    if resp.status_code != 200:
        print(f"[backfill] status={resp.status_code} {url}", flush=True)
        return None
    state = _extract_state(resp.text)
    return _get_query(state, "FETCH_KEIRIN_RACE")


def _build_update_rows(data: dict, race_key: str) -> list[tuple[int, int, float | None, str, int]]:
    """FETCH_KEIRIN_RACE の data から (res_standing, res_back, final_half, race_key, frame_no) を作る。

    data["entries"] の playerId→number(frame_no) 対応で
    data["results"] の playerId をひも付ける。results が空のレースは空リストを返す。
    """
    entries_raw = data.get("entries") or []
    player_to_frame = {
        e["playerId"]: e["number"] for e in entries_raw
        if not e.get("absent") and "playerId" in e and "number" in e
    }

    rows: list[tuple[int, int, float | None, str, int]] = []
    for item in data.get("results") or []:
        player_id = item.get("playerId")
        frame_no = player_to_frame.get(player_id)
        if frame_no is None:
            continue
        res_standing = int(bool(item.get("standing")))
        res_back = int(bool(item.get("back")))
        raw_final_half = item.get("finalHalfRecord")
        try:
            final_half = float(raw_final_half) if raw_final_half not in (None, "") else None
        except (TypeError, ValueError):
            final_half = None
        rows.append((res_standing, res_back, final_half, race_key, int(frame_no)))
    return rows


def _target_races(start: str, end: str, limit: int | None) -> tuple[list[dict], set[str]]:
    """バックフィル対象レース一覧と、resume 判定用の完了済み race_key 集合を返す。"""
    with get_connection() as conn:
        _ensure_sqlite_columns(conn)
        rows = conn.execute(
            "SELECT race_key, venue_id, race_date, race_no, cup_id, day_index "
            "FROM wt_races WHERE race_date BETWEEN ? AND ? AND status = 3 "
            "ORDER BY race_date ASC, race_key ASC",
            (start, end),
        ).fetchall()
        done_rows = conn.execute(
            "SELECT DISTINCT race_key FROM wt_entries WHERE res_back IS NOT NULL"
        ).fetchall()

    races = [
        {
            "race_key": r["race_key"],
            "venue_id": r["venue_id"],
            "race_date": r["race_date"],
            "race_no": r["race_no"],
            "cup_id": r["cup_id"],
            "day_index": r["day_index"],
        }
        for r in rows
    ]
    if limit is not None:
        races = races[:limit]
    done_keys = {r["race_key"] for r in done_rows}
    return races, done_keys


def _write_rows(rows: list[tuple[int, int, float | None, str, int]],
                pg_conn: Any | None, dry_run: bool) -> None:
    """1レース分の更新行を SQLite（+ 設定時は PG ミラー）へ書き込む。レース単位 commit。"""
    if dry_run or not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            "UPDATE wt_entries SET res_standing=?, res_back=?, final_half=? "
            "WHERE race_key=? AND frame_no=?",
            rows,
        )
    if pg_conn is not None:
        with pg_conn.cursor() as cur:
            cur.executemany(
                "UPDATE keirin.wt_entries SET res_standing=%s, res_back=%s, final_half=%s "
                "WHERE race_key=%s AND frame_no=%s",
                rows,
            )
        pg_conn.commit()


def run(start: str, end: str, limit: int | None, dry_run: bool) -> None:
    """バックフィル本体。"""
    import os

    races, done_keys = _target_races(start, end, limit)
    total = len(races)
    print(f"[backfill] 対象 {total}件 ({start}〜{end})"
          f"{'（dry-run）' if dry_run else ''}", flush=True)

    pg_conn = None
    pg_mirror_url = os.environ.get("PG_MIRROR_URL")
    if pg_mirror_url and not dry_run:
        import psycopg2
        pg_conn = psycopg2.connect(pg_mirror_url)
        _ensure_pg_mirror_columns(pg_conn)
        print("[backfill] PG_MIRROR_URL 設定あり → keirin.wt_entries にもミラー", flush=True)
    elif pg_mirror_url and dry_run:
        print("[backfill] PG_MIRROR_URL 設定あり（dry-runのため接続はスキップ）", flush=True)

    session = requests.Session()
    session.headers.update(_HEADERS)

    n_resume_skipped = 0
    n_no_data_skipped = 0
    n_fetch_failed = 0
    n_races_updated = 0
    n_rows_updated = 0

    try:
        for i, race in enumerate(races, start=1):
            race_key = race["race_key"]
            if race_key in done_keys:
                n_resume_skipped += 1
            else:
                try:
                    data = _fetch_race_state(
                        session, race["venue_id"], race["cup_id"],
                        race["day_index"], race["race_no"],
                    )
                    time.sleep(_REQUEST_INTERVAL)
                    if not data:
                        n_fetch_failed += 1
                    else:
                        rows = _build_update_rows(data, race_key)
                        if not rows:
                            n_no_data_skipped += 1
                        else:
                            _write_rows(rows, pg_conn, dry_run)
                            n_races_updated += 1
                            n_rows_updated += len(rows)
                except Exception as exc:  # noqa: BLE001 - 1レース分の例外で全体を止めない
                    print(f"[backfill] 例外 {race_key}: {exc}", flush=True)
                    n_fetch_failed += 1
                    time.sleep(_REQUEST_INTERVAL)

            if i % 100 == 0:
                skipped = n_resume_skipped + n_no_data_skipped + n_fetch_failed
                print(f"[backfill] {i}/{total} {race['race_date']} "
                      f"更新{n_rows_updated}件 スキップ{skipped}件", flush=True)
    except KeyboardInterrupt:
        print("[backfill] 中断（Ctrl-C）。ここまでの更新は commit 済み。"
              "再実行で resume されます。", flush=True)
    finally:
        if pg_conn is not None:
            pg_conn.close()

    print(f"[backfill] 完了: 対象{total}件 "
          f"更新{n_races_updated}レース({n_rows_updated}行) "
          f"resumeスキップ{n_resume_skipped}件 "
          f"データ無しスキップ{n_no_data_skipped}件 "
          f"fetch失敗{n_fetch_failed}件"
          f"{'（dry-run・書き込みなし）' if dry_run else ''}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="winticket racecard 再取得で wt_entries の S/B取得・上がりタイムを補完する",
    )
    ap.add_argument("--start", default="2024-01-01", help="開始日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="終了日 YYYY-MM-DD（デフォルト: 昨日）")
    ap.add_argument("--dry-run", action="store_true", help="書き込みなしで動作確認する")
    ap.add_argument("--limit", type=int, default=None,
                    help="先頭Nレースのみ処理する（動作確認用）")
    args = ap.parse_args()
    end = args.end or (date.today() - timedelta(days=1)).isoformat()

    run(args.start, end, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
