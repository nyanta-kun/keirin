"""winticket データ収集パイプライン"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..database import get_connection
from .winticket import WinticketScraper, VENUE_SLUGS

logger = logging.getLogger(__name__)
_db_lock = threading.Lock()

MAX_VENUE_WORKERS = 4  # 会場絞り込み後の並列度（単一ドメインなので控えめに）


class WinticketPipeline:
    def collect_date(self, target_date: str, dry_run: bool = False,
                     full_scan: bool = False) -> dict:
        """指定日の全winticket会場のレースデータ（+オッズ）を収集

        会場リストの決定:
          - 既に wt_races(本番) に当日記録があり full_scan=False なら、その会場だけ
            再収集して探索リクエストを削減（結果再取得・バックフィル向け）。
          - 未収集の日 or full_scan=True なら 全 VENUE_SLUGS を走査して開催を検出。
        ※ 旧実装は停止済み keirin-station の races を参照していたため、ks停止後に
          始まった初日開催（例: 宇都宮/別府の F2 ミッドナイト初日）を取りこぼした。
          現行は wt_races 基準＋未収集日は全会場走査でこれを防ぐ。
        """
        stats = {"venues": 0, "races": 0, "results": 0, "errors": 0}
        known = _venues_racing_on(target_date)
        venue_ids = list(VENUE_SLUGS.keys()) if (full_scan or not known) else known

        with ThreadPoolExecutor(max_workers=MAX_VENUE_WORKERS) as ex:
            futures = {
                ex.submit(_collect_venue, vid, target_date, dry_run): vid
                for vid in venue_ids
            }
            for future in as_completed(futures):
                vid = futures[future]
                try:
                    vstats = future.result()
                    for k in stats:
                        stats[k] += vstats.get(k, 0)
                except Exception as e:
                    logger.error(f"[wt] venue {vid} failed: {e}", exc_info=True)
                    stats["errors"] += 1

        logger.info(f"[wt] {target_date}: {stats}")
        return stats

    def collect_month(self, year: int, month: int, dry_run: bool = False) -> dict:
        from calendar import monthrange
        from datetime import date, timedelta

        _, n_days = monthrange(year, month)
        start = date(year, month, 1)
        days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]
        total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

        for d in days:
            stats = self.collect_date(d, dry_run=dry_run)
            for k in total:
                total[k] += stats.get(k, 0)

        return total


def _collect_venue(venue_id: str, target_date: str, dry_run: bool) -> dict:
    """1会場分のデータを収集（スレッド内で実行）"""
    scraper = WinticketScraper(request_interval=2.0)
    stats = {"venues": 0, "races": 0, "results": 0, "errors": 0}

    # DBキャッシュを先に確認（再収集時の高速化）
    cup_info = _get_cup_info_from_db(venue_id, target_date)
    if cup_info is None:
        cup_info = scraper.find_cup_info(venue_id, target_date)
    if cup_info is None:
        return stats

    cup_id, day_index = cup_info
    n_races = scraper.get_race_count(venue_id, cup_id, day_index)
    if n_races == 0:
        return stats

    stats["venues"] = 1
    logger.info(f"[wt] venue={venue_id} {target_date}: {n_races}R (cup={cup_id} day={day_index})")

    date_str = target_date.replace("-", "")
    race_keys = [f"{date_str}_{venue_id}_{i:02d}" for i in range(1, n_races + 1)]
    already = _get_collected_keys(race_keys)

    batch: list[dict] = []
    for race_no in range(1, n_races + 1):
        race_key = f"{date_str}_{venue_id}_{race_no:02d}"
        if race_key in already:
            logger.debug(f"[wt] skip {race_key}")
            stats["races"] += 1
            continue

        try:
            data = scraper.fetch_race_data(
                venue_id, target_date, race_no,
                cup_id=cup_id, day_index=day_index,
            )
            if data:
                # オッズも取得（失敗しても継続）
                try:
                    odds = scraper.fetch_odds(venue_id, target_date, race_no, cup_id, day_index)
                    data["odds"] = odds or {}
                except Exception as e:
                    logger.debug(f"[wt] odds fetch skipped {race_key}: {e}")
                    data["odds"] = {}

                batch.append(data)
                has_result = any(
                    e.get("finish_order") is not None for e in data["entries"]
                )
                if has_result:
                    stats["results"] += 1
        except Exception as e:
            logger.error(f"[wt] {race_key} fetch failed: {e}")
            stats["errors"] += 1

    if batch and not dry_run:
        try:
            _save_batch(batch)
            stats["races"] += len(batch)
        except Exception as e:
            logger.error(f"[wt] DB write failed venue={venue_id}: {e}")
            stats["errors"] += len(batch)
    elif dry_run:
        stats["races"] += len(batch)

    return stats


# ---------------------------------------------------------------------------
# DB ヘルパー
# ---------------------------------------------------------------------------

def _venues_racing_on(target_date: str) -> list[str]:
    """既に wt_races(本番) に当日記録のある会場コードを返す（再収集の絞り込み用）。

    停止済み keirin-station の races ではなく現行 wt_races を参照する。
    未収集の日は空リストを返し、呼び出し側で全 VENUE_SLUGS を走査して開催を検出する
    （初日開催の取りこぼし防止）。
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT venue_id FROM wt_races WHERE race_date = ?",
                (target_date,),
            ).fetchall()
    except Exception:
        return []
    return [r[0] for r in rows if r[0] in VENUE_SLUGS]


def _get_cup_info_from_db(venue_id: str, target_date: str) -> tuple[str, int] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT cup_id, day_index FROM wt_races "
            "WHERE venue_id = ? AND race_date = ? LIMIT 1",
            (venue_id, target_date),
        ).fetchone()
    return (row[0], row[1]) if row else None


def _get_collected_keys(race_keys: list[str]) -> set[str]:
    """「結果確定済み(finish_order あり)」のレースのみスキップ対象とする。

    出走表のみ(前日朝に予想用収集した未確定レース)は翌日再収集で結果を取得
    したいのでスキップしない（ks ルートの _get_collected_race_keys と同方針）。
    """
    if not race_keys:
        return set()
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT race_key FROM wt_entries "
            f"WHERE race_key IN ({placeholders}) AND finish_order >= 1",
            race_keys,
        ).fetchall()
    return {row[0] for row in rows}


def _save_batch(batch: list[dict]):
    with _db_lock:
        with get_connection() as conn:
            for data in batch:
                _write_race(conn, data)


def _write_race(conn, data: dict):
    ri = data["race_info"]
    conn.execute("""
        INSERT OR REPLACE INTO wt_races
        (race_key, venue_id, race_date, race_no, cup_id, day_index,
         grade, race_type, distance, n_entries, start_at, status, cancel)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["race_key"], data["venue_id"], data["race_date"], data["race_no"],
        data["cup_id"], data["day_index"],
        ri.get("grade"), ri.get("race_type"), ri.get("distance"),
        ri.get("n_entries"), ri.get("start_at"),
        int(ri.get("status", 0)), int(bool(ri.get("cancel", False))),
    ))

    for e in data["entries"]:
        conn.execute("""
            INSERT OR REPLACE INTO wt_entries
            (race_key, frame_no, player_id, name, prefecture, player_class, term,
             gear_ratio, style, race_point, comment, prediction_mark,
             s_count, h_count, b_count,
             front_runner, stalker, deep_closer, marker,
             first_rate, second_rate, third_rate,
             ex_spurt_pct, ex_thrust_pct, ex_left_behind_pct,
             ex_split_line_pct, ex_snatch_pct,
             line_group, line_size, line_pos, is_line_leader, n_lines,
             finish_order, factor, res_standing, res_back, final_half)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["race_key"], e["frame_no"], e["player_id"],
            e["name"], e["prefecture"], e["player_class"], e["term"],
            e["gear_ratio"], e["style"], e["race_point"], e["comment"],
            e["prediction_mark"],
            e["s_count"], e["h_count"], e["b_count"],
            e["front_runner"], e["stalker"], e["deep_closer"], e["marker"],
            e["first_rate"], e["second_rate"], e["third_rate"],
            e["ex_spurt_pct"], e["ex_thrust_pct"], e["ex_left_behind_pct"],
            e["ex_split_line_pct"], e["ex_snatch_pct"],
            e["line_group"], e["line_size"], e["line_pos"],
            e["is_line_leader"], e["n_lines"],
            e["finish_order"], e["factor"],
            e.get("res_standing"), e.get("res_back"), e.get("final_half"),
        ))

    # 欠車ガード: スクレイパーは absent(欠車)選手を除外する（winticket.py で continue）。
    # INSERT OR REPLACE は欠車になった選手の古い行を消さないため、再収集後も
    # 「出走表から消えた=欠車」の行が wt_entries に残り、wave-picks-wt が拾って
    # 購入不可な軸/相手を含む買い目を生成してしまう。これを防ぐため、現在の出走表に
    # 無いframe_noの行を削除して wt_entries を最新の出走表と一致させる。
    # entries が空（API失敗等）のときは誤削除を避けるため何もしない。
    cur_frames = [e["frame_no"] for e in data["entries"]]
    if cur_frames:
        ph = ",".join("?" * len(cur_frames))
        conn.execute(
            f"DELETE FROM wt_entries WHERE race_key=? AND frame_no NOT IN ({ph})",
            [data["race_key"], *cur_frames],
        )

    # 順序あり(車単/三連単)は "-"、順序なし(車複/三連複/ワイド)は "=" で結合
    _ORDERED = {"exacta", "trifecta"}
    for bet_type, items in data.get("odds", {}).items():
        sep = "-" if bet_type in _ORDERED else "="
        for item in items:
            combo = item["combination"]
            if isinstance(combo, (list, tuple)):
                combo = sep.join(str(x) for x in combo)
            conn.execute("""
                INSERT OR REPLACE INTO wt_odds
                (race_key, bet_type, combination, odds_value)
                VALUES (?, ?, ?, ?)
            """, (
                data["race_key"], bet_type,
                combo, item["odds_value"],
            ))
