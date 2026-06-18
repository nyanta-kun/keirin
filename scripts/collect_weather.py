"""気象データ収集スクリプト — Open-Meteo Historical Weather API バックフィル

使い方:
    # 全会場のバックフィル（2022-12-01 〜 今日）
    python3 scripts/collect_weather.py

    # 特定期間
    python3 scripts/collect_weather.py --from 2024-01-01 --to 2024-12-31

    # 特定会場のみ
    python3 scripts/collect_weather.py --venue 61 --venue 83

    # カバレッジレポートのみ（データ取得なし）
    python3 scripts/collect_weather.py --report-only

収集後に wt_races の (venue_id, 日付) に対するカバレッジを報告する。
INSERT OR REPLACE なので再実行は安全。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scraper.weather import (
    VENUE_COORDS,
    ensure_table,
    fetch_weather,
    upsert_rows,
)
from src.scraper.winticket import VENUE_SLUGS

DB_PATH = ROOT / "data" / "keirin.db"
DEFAULT_START = "2022-12-01"
# APIレート配慮: 会場間の待機時間（秒）
_INTER_VENUE_SLEEP = 1.0


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def collect(
    venue_ids: list[str],
    start: str,
    end: str,
    db_path: str | Path = DB_PATH,
) -> dict[str, int]:
    """指定会場・期間のデータを取得して DB に保存する。

    Returns:
        venue_id -> 取得行数 のマップ
    """
    conn = sqlite3.connect(str(db_path))
    ensure_table(conn)

    result: dict[str, int] = {}
    total = len(venue_ids)
    for idx, vid in enumerate(venue_ids, 1):
        print(f"[{idx}/{total}] venue={vid}  {start} 〜 {end} ...", flush=True)
        try:
            rows = fetch_weather(vid, start, end)
            n = upsert_rows(conn, rows)
            result[vid] = n
            print(f"  → {n} rows upserted", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True, file=sys.stderr)
            result[vid] = -1
        if idx < total:
            time.sleep(_INTER_VENUE_SLEEP)

    conn.close()
    return result


def coverage_report(
    venue_ids: list[str] | None = None,
    db_path: str | Path = DB_PATH,
) -> None:
    """wt_races の (venue_id, 日付) に対して wt_weather のカバレッジを出力する。

    カバレッジ基準:
        wt_races に存在するレース日の開催時間帯 (09:00〜22:00) において、
        wt_weather に 1 件以上のレコードがあれば「カバー済み」と判定する。
    """
    conn = sqlite3.connect(str(db_path))

    # wt_races から venue_id・レース日一覧を取得
    query = "SELECT DISTINCT venue_id, race_date FROM wt_races WHERE venue_id IN ({seq})"
    if venue_ids:
        placeholders = ",".join("?" * len(venue_ids))
        rows = conn.execute(
            query.format(seq=placeholders), venue_ids
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT venue_id, race_date FROM wt_races"
        ).fetchall()

    if not rows:
        print("wt_races にデータがありません。")
        conn.close()
        return

    # 対象 venue_id ごとにカバレッジを集計
    total_race_days = len(rows)
    covered = 0
    missing_by_venue: dict[str, int] = {}

    for venue_id, race_date in rows:
        # 開催日の 09:00〜22:00 に 1 件でも気象データがあれば OK
        cnt = conn.execute(
            """
            SELECT COUNT(*) FROM wt_weather
             WHERE venue_id = ?
               AND dt_hour BETWEEN ? AND ?
            """,
            (venue_id, f"{race_date} 09:00", f"{race_date} 22:00"),
        ).fetchone()[0]
        if cnt > 0:
            covered += 1
        else:
            missing_by_venue[venue_id] = missing_by_venue.get(venue_id, 0) + 1

    coverage_pct = covered / total_race_days * 100 if total_race_days else 0.0

    print("\n=== カバレッジレポート ===")
    print(f"総レース日数 (venue×date):   {total_race_days:,}")
    print(f"気象データあり:              {covered:,}")
    print(f"カバレッジ:                  {coverage_pct:.1f}%")

    if missing_by_venue:
        print(f"\n欠損のある会場 ({len(missing_by_venue)} 会場):")
        for vid in sorted(missing_by_venue):
            n = missing_by_venue[vid]
            print(f"  venue={vid}  欠損レース日={n}")
    else:
        print("\n全会場でカバー済み。")

    # wt_weather の総行数
    total_weather = conn.execute("SELECT COUNT(*) FROM wt_weather").fetchone()[0]
    print(f"\nwt_weather 総行数: {total_weather:,}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-Meteo から競輪場気象データを収集・バックフィルする"
    )
    parser.add_argument(
        "--from", dest="start", default=DEFAULT_START,
        metavar="YYYY-MM-DD", help=f"収集開始日 (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--to", dest="end", default=None,
        metavar="YYYY-MM-DD",
        help="収集終了日 (default: 昨日・archive API は当日未収録)",
    )
    parser.add_argument(
        "--venue", dest="venues", action="append", default=None,
        metavar="VID", help="収集する会場ID (複数指定可・省略時は全43会場)"
    )
    parser.add_argument(
        "--db", default=str(DB_PATH),
        metavar="PATH", help=f"SQLite DB パス (default: {DB_PATH})"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="データ取得せずカバレッジレポートのみ出力"
    )
    args = parser.parse_args()

    end = args.end or _yesterday()
    venue_ids = args.venues if args.venues else sorted(VENUE_COORDS.keys())

    # 入力検証
    for vid in venue_ids:
        if vid not in VENUE_COORDS:
            print(f"ERROR: 不明な venue_id={vid!r}。有効値: {sorted(VENUE_COORDS.keys())}", file=sys.stderr)
            sys.exit(1)
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: 日付フォーマットが不正: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"対象会場: {len(venue_ids)} 会場")
    print(f"期間:     {args.start} 〜 {end}")
    print(f"DB:       {args.db}")

    if not args.report_only:
        result = collect(venue_ids, args.start, end, db_path=args.db)
        errors = [vid for vid, n in result.items() if n < 0]
        if errors:
            print(f"\nWARNING: 以下の会場でエラーが発生しました: {errors}", file=sys.stderr)

    coverage_report(venue_ids if args.venues else None, db_path=args.db)


if __name__ == "__main__":
    main()
