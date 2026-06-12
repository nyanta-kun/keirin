"""日中オッズスナップショット（winticket・money-flow 素材収集）

目的:
  朝（morning）・夕（evening）の 2 時点に加え、日中複数時点のオッズを記録して
  money-flow（朝→直前のオッズ変動）検証の素材を蓄積する。

  実行時刻から snapshot_type を自動決定する: h08, h10, h12, h14, h16, h18, h20, h22 等
  （当日 UTC+9 の時刻 HH を使用。例: 10:xx → 'h10'）

  INSERT OR REPLACE: 同 (race_key, bet_type, combination, snapshot_type) を
  同一時間帯内に再実行した場合は最新値で更新する（冪等性を保証）。
  ※ morning スナップショット（INSERT OR IGNORE の初回保持）とは用途が異なる。

使い方:
  # 取得（cron用・既定は当日・snapshot_type は実行時刻から自動決定）
  PYTHONPATH=. .venv/bin/python3 scripts/snapshot_intraday_odds_wt.py [YYYY-MM-DD]

  # snapshot_type を明示指定（手動バックフィル用）
  PYTHONPATH=. .venv/bin/python3 scripts/snapshot_intraday_odds_wt.py --type h10 [YYYY-MM-DD]

  # ドリフト系列レポート（race_key ごとに時点別オッズを表示）
  PYTHONPATH=. .venv/bin/python3 scripts/snapshot_intraday_odds_wt.py --report [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--race RACE_KEY]

ヘルパー（G04 money-flow 分析から import 可能）:
  from scripts.snapshot_intraday_odds_wt import get_nearest_snapshot
"""
from __future__ import annotations

import sys
import argparse
import logging
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# プロジェクトルートをパスに追加（PYTHONPATH 未設定でも動くように）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection, init_db
from src.scraper.winticket import WinticketScraper

logger = logging.getLogger(__name__)

# 日本標準時 (UTC+9)
_JST = timezone(timedelta(hours=9))

# fetch する bet_type（wt_odds_snapshot に保存する市場）
_TARGET_BET_TYPES = {"trio", "trifecta", "quinellaPlace"}

# 順序あり（vehicle_no の順序が意味を持つ）bet_type → セパレータ "-"
# それ以外（順序なし）→ "="
# pipeline_wt.py の _ORDERED に合わせる
_ORDERED_BET_TYPES = {"exacta", "trifecta"}


def _jst_now() -> datetime:
    """現在の JST 時刻を返す。"""
    return datetime.now(_JST)


def auto_snapshot_type(dt: Optional[datetime] = None) -> str:
    """実行時刻（JST）から snapshot_type 文字列 'h{HH}' を返す。

    例: 10:35 → 'h10', 08:05 → 'h08'
    """
    if dt is None:
        dt = _jst_now()
    return f"h{dt.hour:02d}"


def snapshot(target_date: str, snapshot_type: Optional[str] = None,
             dry_run: bool = False, sleep_sec: float = 1.5) -> None:
    """target_date の未発走レースの現在オッズを winticket から取得して保存する。

    Parameters
    ----------
    target_date   : 対象日 YYYY-MM-DD
    snapshot_type : 明示指定する場合。None なら実行時刻から自動決定 (h{HH})
    dry_run       : True なら DB 書き込みをスキップして件数だけ表示
    sleep_sec     : レース間リクエスト間隔（秒）
    """
    if snapshot_type is None:
        snapshot_type = auto_snapshot_type()

    init_db()

    # 当日の未発走レース（start_at > 現在の UNIX 秒、cancel=0）を取得
    now_unix = int(_jst_now().timestamp())
    with get_connection() as conn:
        races = conn.execute(
            """
            SELECT race_key, venue_id, race_no, cup_id, day_index
            FROM wt_races
            WHERE race_date = ?
              AND start_at > ?
              AND cancel = 0
            ORDER BY start_at
            """,
            (target_date, now_unix),
        ).fetchall()

    if not races:
        print(f"[intraday] {target_date} ({snapshot_type}): "
              f"未発走レースなし（全発走済み or 開催なし）")
        return

    print(f"[intraday] {target_date} ({snapshot_type}): "
          f"未発走 {len(races)} レースのオッズ取得開始...")

    scraper = WinticketScraper(request_interval=sleep_sec)
    total_inserted = 0
    total_races_ok = 0

    for race_key, venue_id, race_no, cup_id, day_index in races:
        try:
            odds_data = scraper.fetch_odds(
                venue_id, target_date, race_no, cup_id, day_index
            )
        except Exception as e:
            logger.warning("[intraday] %s fetch_odds failed: %s", race_key, e)
            continue

        if not odds_data:
            logger.debug("[intraday] %s: odds データなし（発走直前 or スキップ）", race_key)
            continue

        rows = []
        for bet_type, items in odds_data.items():
            if bet_type not in _TARGET_BET_TYPES:
                continue
            sep = "-" if bet_type in _ORDERED_BET_TYPES else "="
            for item in items:
                combo = item.get("combination", "")
                # combination は list/tuple の場合があるので文字列に変換する
                # （pipeline_wt.py の _save_batch と同方式）
                if isinstance(combo, (list, tuple)):
                    combo = sep.join(str(x) for x in combo)
                odds_val = item.get("odds_value")
                if combo and odds_val:
                    rows.append((
                        race_key, bet_type, combo, odds_val,
                        snapshot_type,
                        datetime.now(_JST).isoformat(timespec="seconds"),
                    ))

        if not rows:
            continue

        if dry_run:
            total_inserted += len(rows)
            total_races_ok += 1
            continue

        with get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO wt_odds_snapshot
                    (race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        # executemany の rowcount は -1 になる場合があるため行数で直接カウント
        total_inserted += len(rows)
        total_races_ok += 1

    dry_label = " [DRY RUN]" if dry_run else ""
    print(f"[intraday] {target_date} ({snapshot_type}){dry_label}: "
          f"{total_races_ok} レース / {total_inserted:,} 行を保存")


def report(date_from: Optional[str], date_to: Optional[str],
           race_filter: Optional[str] = None) -> None:
    """morning → h{XX} → 確定 のドリフト系列をレース毎に表示する。

    表示する snapshot_type 順: morning, h08, h10, h12, h14, h16, h18, h20, h22, evening
    """
    where = ["1=1"]
    params: list = []
    if date_from:
        where.append("r.race_date >= ?"); params.append(date_from)
    if date_to:
        where.append("r.race_date <= ?"); params.append(date_to)
    if race_filter:
        where.append("s.race_key = ?"); params.append(race_filter)

    where_sql = " AND ".join(where)

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT s.race_key, r.race_date, s.bet_type, s.combination,
                   s.snapshot_type, s.odds_value, s.snapshot_at
            FROM wt_odds_snapshot s
            JOIN wt_races r ON s.race_key = r.race_key
            WHERE {where_sql}
            ORDER BY s.race_key, s.bet_type, s.combination, s.snapshot_at
            """,
            params,
        ).fetchall()

    if not rows:
        print("[report] 表示可能なデータがありません。")
        return

    # snapshot_type の表示順
    TYPE_ORDER = ["morning", "h06", "h07", "h08", "h09", "h10", "h11", "h12",
                  "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20",
                  "h21", "h22", "h23", "evening"]

    # race_key × bet_type × combination → {snapshot_type: odds_value}
    from collections import defaultdict
    data: dict[tuple, dict] = defaultdict(dict)
    for race_key, race_date, bet_type, combination, snap_type, odds_val, _ in rows:
        data[(race_key, race_date, bet_type, combination)][snap_type] = odds_val

    # 存在する snapshot_types を収集・ソート
    all_types: set[str] = set()
    for d in data.values():
        all_types.update(d.keys())
    sorted_types = [t for t in TYPE_ORDER if t in all_types] + \
                   sorted(all_types - set(TYPE_ORDER))

    print(f"\n{'='*80}")
    print(f"オッズ ドリフト系列レポート")
    if date_from or date_to:
        print(f"  期間: {date_from or '開始日未指定'} 〜 {date_to or '終了日未指定'}")
    if race_filter:
        print(f"  レース: {race_filter}")
    print(f"  snapshot_type 列: {', '.join(sorted_types)}")
    print(f"{'='*80}")

    header = f"  {'race_key':<22} {'type':<14} {'combo':<12} " + \
             "  ".join(f"{t:>8}" for t in sorted_types)
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_race = None
    shown = 0
    for (race_key, race_date, bet_type, combination), snap_dict in sorted(data.items()):
        if race_key != prev_race:
            if prev_race is not None:
                print()
            prev_race = race_key
        vals = "  ".join(
            f"{snap_dict[t]:>8.1f}" if t in snap_dict else f"{'---':>8}"
            for t in sorted_types
        )
        print(f"  {race_key:<22} {bet_type:<14} {combination:<12} {vals}")
        shown += 1

    print(f"\n  合計 {shown:,} 組合せ / {len({k[0] for k in data.keys()}):,} レース")
    print(f"{'='*80}")

    # ドリフト統計（morning→最後の intraday スナップショット）
    if "morning" in all_types:
        intraday_types = [t for t in sorted_types if t not in ("morning", "evening")]
        if intraday_types:
            last_type = intraday_types[-1]
            ratios = []
            for snap_dict in data.values():
                m = snap_dict.get("morning")
                l = snap_dict.get(last_type)
                if m and l and m > 0 and l > 0:
                    ratios.append(l / m)
            if ratios:
                import statistics
                print(f"\n  morning → {last_type} ドリフト統計 (n={len(ratios):,})")
                print(f"  中央値 最新/morning: {statistics.median(ratios):.3f}")
                abs_pct = [abs(r - 1.0) for r in ratios]
                print(f"  |変動| 中央値:       {statistics.median(abs_pct):.1%}")
                within20 = sum(1 for x in abs_pct if x <= 0.20) / len(abs_pct)
                print(f"  ±20%以内:          {within20:.1%}")


def get_nearest_snapshot(
    race_key: str,
    reference_unix: int,
    delta_minutes: int = 60,
    bet_type: Optional[str] = None,
) -> list[dict]:
    """race_key と任意時点 T（UNIX 秒）に対し、T ± delta_minutes に最も近い
    snapshot を返す。G04 の money-flow 分析からの import 用。

    Parameters
    ----------
    race_key        : 対象レースキー
    reference_unix  : 基準時刻（UNIX 秒。通常は発走時刻の T-30分 や T-60分）
    delta_minutes   : 基準時刻前後の許容幅（分）。既定 60 分
    bet_type        : 絞り込む bet_type（None なら全市場）

    Returns
    -------
    list of dict: [{"race_key", "bet_type", "combination", "odds_value",
                    "snapshot_type", "snapshot_at", "diff_seconds"}, ...]
    最も近い snapshot_at のものを返す。該当なしは空リスト。
    """
    where_bt = "AND s.bet_type = ?" if bet_type else ""
    params: list = [race_key]
    if bet_type:
        params.append(bet_type)

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT s.race_key, s.bet_type, s.combination,
                   s.odds_value, s.snapshot_type, s.snapshot_at
            FROM wt_odds_snapshot s
            WHERE s.race_key = ?
              {where_bt}
            ORDER BY s.snapshot_at
            """,
            params,
        ).fetchall()

    if not rows:
        return []

    delta_sec = delta_minutes * 60

    def _abs_diff(snapshot_at: str) -> int:
        """snapshot_at の ISO 文字列を UNIX 秒に変換して基準時刻との差（秒）を返す。
        タイムゾーン付き文字列（+09:00 等）も naive 文字列（JST とみなす）も処理する。
        """
        try:
            dt = datetime.fromisoformat(snapshot_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_JST)
            return abs(int(dt.timestamp()) - reference_unix)
        except ValueError:
            return 10 ** 9

    # delta_minutes 内に収まる候補だけ残す
    candidates = [r for r in rows if _abs_diff(r[5]) <= delta_sec]
    if not candidates:
        return []

    min_diff = min(_abs_diff(r[5]) for r in candidates)
    tolerance = 60  # 同一時点とみなす幅（秒）

    result = []
    for race_key_, bet_type_, combo, odds_val, snap_type, snap_at in candidates:
        diff = _abs_diff(snap_at)
        if diff <= min_diff + tolerance:
            result.append({
                "race_key": race_key_,
                "bet_type": bet_type_,
                "combination": combo,
                "odds_value": odds_val,
                "snapshot_type": snap_type,
                "snapshot_at": snap_at,
                "diff_seconds": diff,
            })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="日中オッズスナップショット（winticket・money-flow 素材収集）"
    )
    ap.add_argument(
        "target_date", nargs="?", default=None,
        help="取得対象日 YYYY-MM-DD（既定: 当日）",
    )
    ap.add_argument(
        "--type", dest="snap_type", default=None,
        help="snapshot_type を明示指定（例: h10, h14。既定: 実行時刻から自動決定）",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="DB 書き込みをせず取得件数だけ表示",
    )
    ap.add_argument(
        "--sleep", type=float, default=1.5,
        help="レース間リクエスト間隔 秒（既定: 1.5）",
    )
    ap.add_argument(
        "--report", action="store_true",
        help="ドリフト系列レポートを出力（取得は行わない）",
    )
    ap.add_argument("--from", dest="date_from", default=None, help="--report 時の開始日")
    ap.add_argument("--to", dest="date_to", default=None, help="--report 時の終了日")
    ap.add_argument(
        "--race", dest="race_key", default=None,
        help="--report 時に絞り込むレースキー",
    )
    args = ap.parse_args()

    if args.report:
        report(args.date_from, args.date_to, args.race_key)
    else:
        target = args.target_date or date.today().isoformat()
        snapshot(target, args.snap_type, dry_run=args.dry_run, sleep_sec=args.sleep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
