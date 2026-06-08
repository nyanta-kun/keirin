"""
データ収集パイプライン（並列処理対応版）

改善点:
1. 収集済みレースのスキップ（再実行時の無駄を排除）
2. 出走表+結果の並列取得（1レースあたり2リクエストを同時実行）
3. 複数開催場の並列処理（最大3会場同時）
4. 開催場単位でのバッチDB書き込み
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..database import get_connection, init_db
from .keirin_station import KeirinStationScraper


logger = logging.getLogger(__name__)

_db_lock = threading.Lock()

MAX_VENUE_WORKERS = 4   # 並列処理する最大会場数（増やしすぎるとBANリスク）
MAX_DAY_WORKERS = 3     # collect_month で並列処理する最大日数（3日×4会場×2=24同時リクエスト）


class CollectionPipeline:
    """データ収集パイプライン"""

    def collect_date(self, target_date: str, dry_run: bool = False) -> dict:
        """指定日のすべてのレースデータを収集。

        スケジュールは「イベント開始日」しか返さないため、複数日イベントの
        途中日（2日目・3日目）はスケジュールに現れない。常に全会場をスキャン
        してスケジュール結果とマージすることで取りこぼしを防ぐ。
        """
        logger.info(f"Collecting data for {target_date}")
        stats = {"venues": 0, "races": 0, "results": 0, "errors": 0}

        # 常に全会場スキャン（スケジュールは途中日を返さないため）
        logger.info(f"Scanning all venues for {target_date}...")
        day_schedules = _scan_all_venues(target_date)

        stats["venues"] = len(day_schedules)
        if not day_schedules:
            logger.info(f"No events found on {target_date}")
            return stats

        venue_stats = _collect_venues_parallel(day_schedules, dry_run)
        for k in stats:
            stats[k] += venue_stats.get(k, 0)

        logger.info(f"Collection complete for {target_date}: {stats}")
        return stats

    def collect_month(self, year: int, month: int, dry_run: bool = False) -> dict:
        """指定年月のデータを一括収集（全日付を並列処理）"""
        from calendar import monthrange
        from datetime import date, timedelta

        _, n_days = monthrange(year, month)
        start = date(year, month, 1)
        days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]
        total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

        with ThreadPoolExecutor(max_workers=MAX_DAY_WORKERS) as executor:
            futures = {executor.submit(self.collect_date, d, dry_run): d for d in days}
            for future in as_completed(futures):
                d = futures[future]
                try:
                    stats = future.result()
                    for k in total:
                        total[k] += stats.get(k, 0)
                except Exception as e:
                    logger.error(f"Day {d} failed: {e}")
                    total["errors"] += 1

        logger.info(f"collect_month {year}/{month:02d} complete: {total}")
        return total


# ---------------------------------------------------------------------------
# 内部関数
# ---------------------------------------------------------------------------

def _scan_all_venues(target_date: str) -> list[dict]:
    """全会場コードに対して指定日のレース一覧を試行し、開催中の会場だけ返す。
    スケジュールに出ない複数日イベントの途中日に使用する。"""
    from .keirin_station import VENUE_CODES

    schedules = []

    def _try_venue(vc: str, vname: str) -> dict | None:
        races = _make_scraper().scrape_race_list(vc, target_date)
        if races:
            logger.info(f"[scan] Found {len(races)} races at {vname}({vc}) on {target_date}")
            return {"venue_code": vc, "venue_name": vname, "date": target_date}
        return None

    with ThreadPoolExecutor(max_workers=MAX_VENUE_WORKERS) as ex:
        futures = {ex.submit(_try_venue, vc, vname): vc
                   for vc, vname in VENUE_CODES.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                schedules.append(result)

    logger.info(f"[scan] {len(schedules)} venues active on {target_date}")
    return schedules


def _make_scraper() -> KeirinStationScraper:
    """スレッドごとに独立したスクレイパーを生成"""
    return KeirinStationScraper()


def _collect_venues_parallel(schedules: list[dict], dry_run: bool) -> dict:
    """複数開催場を並列処理する"""
    total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

    with ThreadPoolExecutor(max_workers=MAX_VENUE_WORKERS) as executor:
        futures = {
            executor.submit(_collect_one_venue, s, dry_run): s
            for s in schedules
        }
        for future in as_completed(futures):
            schedule = futures[future]
            try:
                stats = future.result()
                for k in total:
                    total[k] += stats.get(k, 0)
            except Exception as e:
                logger.error(f"Venue {schedule['venue_code']} on {schedule['date']} failed: {e}")
                total["errors"] += 1

    return total


def _collect_one_venue(schedule: dict, dry_run: bool) -> dict:
    """1開催場のデータを収集（スレッド内で実行）"""
    venue_code = schedule["venue_code"]
    venue_name = schedule["venue_name"]
    target_date = schedule["date"]
    stats = {"venues": 1, "races": 0, "results": 0, "errors": 0}

    scraper = _make_scraper()  # スレッドごとに独立したセッション

    races = scraper.scrape_race_list(venue_code, target_date)
    logger.info(f"[{venue_name}({venue_code}) {target_date}] {len(races)} races")

    # 収集済みのrace_keyを一括確認（DBアクセスを最小化）
    race_keys = [r["race_key"] for r in races]
    already_collected = _get_collected_race_keys(race_keys)

    batch: list[tuple[dict, dict, dict | None]] = []

    for race_info in races:
        race_key = race_info["race_key"]
        if race_key in already_collected:
            logger.debug(f"Skip (already collected): {race_key}")
            stats["races"] += 1
            continue

        try:
            # 出走表と結果を並列取得
            detail, result = _fetch_race_parallel(scraper, race_key)

            if result and result.get("finish_order"):
                stats["results"] += 1

            if detail:
                batch.append((race_info, detail, result))

        except Exception as e:
            logger.error(f"Error collecting {race_key}: {e}")
            stats["errors"] += 1

    # バッチDB書き込み（開催場単位でまとめて1トランザクション）
    if batch and not dry_run:
        try:
            _save_batch(batch, venue_name)
            stats["races"] += len(batch)
        except Exception as e:
            logger.error(f"DB write failed for {venue_name}: {e}")
            stats["errors"] += len(batch)

    return stats


def _fetch_race_parallel(scraper: KeirinStationScraper, race_key: str) -> tuple:
    """出走表と結果を並列取得する"""
    scraper2 = _make_scraper()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_detail = ex.submit(scraper.scrape_race_detail, race_key)
        f_result = ex.submit(scraper2.scrape_race_result, race_key)
        return f_detail.result(), f_result.result()


def _get_collected_race_keys(race_keys: list[str]) -> set[str]:
    """出走表・結果の両方が揃っている race_key のみスキップ対象とする。
    出走表があっても結果が未取得のレースは再スクレイピングして結果を保存する。"""
    if not race_keys:
        return set()
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT re.race_key FROM race_entries re
                JOIN race_results rr ON re.race_key = rr.race_key
                WHERE re.race_key IN ({placeholders})
                AND re.quinella_rate IS NOT NULL""",
            race_keys,
        ).fetchall()
    return {row[0] for row in rows}


def _save_batch(batch: list[tuple[dict, dict, dict | None]], venue_name: str):
    """複数レースをまとめて1トランザクションでDB保存"""
    with _db_lock:
        with get_connection() as conn:
            for race_info, detail, result in batch:
                _write_race(conn, race_info, detail, result, venue_name)


def _write_race(conn, race_info: dict, detail: dict, result: dict | None, venue_name: str):
    """1レース分をDB書き込み（トランザクション内で呼ぶ）"""
    conn.execute(
        "INSERT OR IGNORE INTO venues (code, name) VALUES (?, ?)",
        (race_info["venue_code"], venue_name),
    )

    race_meta = detail.get("race_info", {})
    conn.execute("""
        INSERT INTO races
        (race_key, venue_code, race_date, race_no, grade, distance, start_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_key) DO UPDATE SET
            start_time = excluded.start_time
    """, (
        race_info["race_key"],
        race_info["venue_code"],
        race_info["date"],
        race_info["race_no"],
        race_meta.get("grade_text"),
        race_meta.get("distance"),
        race_meta.get("start_time"),
    ))

    frame_to_player: dict[int, dict] = {}
    if result and result.get("finish_order"):
        for f in result["finish_order"]:
            frame_to_player[f["frame_no"]] = {
                "player_id": f.get("player_id"),
                "player_name": f.get("player_name", ""),
            }

    for player_data in frame_to_player.values():
        pid = player_data.get("player_id")
        if pid:
            conn.execute(
                "INSERT OR IGNORE INTO players (player_id, name, prefecture) VALUES (?, ?, ?)",
                (pid, player_data.get("player_name", ""), player_data.get("prefecture")),
            )

    for i, entry in enumerate(detail.get("entries", [])):
        frame_no = i + 1
        player_info = frame_to_player.get(frame_no, {})
        # 優先順: 結果ページ → 出走表 → frame_N フォールバック
        player_id = (player_info.get("player_id")
                     or entry.get("player_id")
                     or f"frame_{frame_no}")
        conn.execute("""
            INSERT INTO race_entries
            (race_key, player_id, frame_no, line_position,
             gear_ratio, racing_score, recent_win_rate_3m, recent_top3_rate_3m,
             quinella_rate, period, prefecture, player_class)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_key, frame_no) DO UPDATE SET
                player_id            = excluded.player_id,
                line_position        = excluded.line_position,
                gear_ratio           = excluded.gear_ratio,
                racing_score         = excluded.racing_score,
                recent_win_rate_3m   = excluded.recent_win_rate_3m,
                recent_top3_rate_3m  = excluded.recent_top3_rate_3m,
                quinella_rate        = excluded.quinella_rate,
                period               = excluded.period,
                prefecture           = excluded.prefecture,
                player_class         = excluded.player_class
        """, (
            race_info["race_key"], player_id, frame_no,
            entry.get("riding_style"), entry.get("gear_ratio"),
            entry.get("racing_score"), entry.get("win_rate"), entry.get("top3_rate"),
            entry.get("quinella_rate"), entry.get("period"),
            entry.get("prefecture"), entry.get("player_class"),
        ))

    if result and result.get("finish_order"):
        for f in result["finish_order"]:
            pid = f.get("player_id") or f"frame_{f['frame_no']}"
            conn.execute("""
                INSERT OR REPLACE INTO race_results
                (race_key, player_id, frame_no, finish_position)
                VALUES (?, ?, ?, ?)
            """, (race_info["race_key"], pid, f["frame_no"], f["position"]))

        for key, payout in result.get("payouts", {}).items():
            bet_type, combo = _split_payout_key(key)
            if bet_type:
                conn.execute("""
                    INSERT OR REPLACE INTO odds
                    (race_key, bet_type, combination, payout)
                    VALUES (?, ?, ?, ?)
                """, (race_info["race_key"], bet_type, combo, payout))


_KNOWN_BET_TYPES = ("trifecta_box", "trifecta", "quinella", "exacta", "win", "place", "wide")


def _split_payout_key(key: str) -> tuple[str, str]:
    """'trifecta_box_1=2=3' → ('trifecta_box', '1=2=3') のように分割する"""
    for bt in _KNOWN_BET_TYPES:
        prefix = bt + "_"
        if key.startswith(prefix):
            return bt, key[len(prefix):]
    return "", ""


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
