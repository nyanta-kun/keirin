"""気象データ収集モジュール — Open-Meteo Historical Weather API 使用

全43競輪場の緯度経度テーブルと、会場×時間帯別気象データの取得・格納を担当する。

DB テーブル:
    wt_weather(venue_id TEXT, dt_hour TEXT,  -- 'YYYY-MM-DD HH:00' JST
               wind_speed REAL, wind_dir REAL, wind_gust REAL,
               temp REAL, precip REAL,
               PRIMARY KEY(venue_id, dt_hour))

API: https://archive-api.open-meteo.com/v1/archive
  hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,precipitation
  timezone=Asia/Tokyo  (Open-Meteo が JST に変換して返す)
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Generator

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全43会場の緯度経度テーブル（市レベル精度 ±数km）
# venue_id は VENUE_SLUGS のキー（2桁 JKA コード）と対応
# ---------------------------------------------------------------------------
VENUE_COORDS: dict[str, tuple[float, float]] = {
    # --- 北海道・東北 ---
    "11": (41.769, 140.729),  # 函館競輪場（函館市）
    "12": (40.824, 140.740),  # 青森競輪場（青森市）
    "13": (37.049, 140.886),  # いわき平競輪場（いわき市）
    # --- 関東 ---
    "21": (37.604, 138.885),  # 弥彦競輪場（弥彦村・燕市近郊）
    "22": (36.391, 139.060),  # 前橋競輪場（前橋市）
    "23": (35.905, 140.095),  # 取手競輪場（取手市）
    "24": (36.566, 139.883),  # 宇都宮競輪場（宇都宮市）
    "25": (35.906, 139.624),  # 大宮競輪場（さいたま市大宮区）
    "26": (35.813, 139.394),  # 西武園競輪場（東村山市）
    "27": (35.627, 139.449),  # 京王閣競輪場（調布市）
    "28": (35.698, 139.413),  # 立川競輪場（立川市）
    "31": (35.781, 139.903),  # 松戸競輪場（松戸市）
    "32": (35.605, 140.123),  # 千葉競輪場（千葉市）
    "34": (35.530, 139.703),  # 川崎競輪場（川崎市）
    "35": (35.329, 139.347),  # 平塚競輪場（平塚市）
    "36": (35.254, 139.154),  # 小田原競輪場（小田原市）
    "37": (34.965, 139.101),  # 伊東競輪場（伊東市）
    "38": (34.975, 138.383),  # 静岡競輪場（静岡市）
    # --- 中部 ---
    "42": (35.183, 136.906),  # 名古屋競輪場（名古屋市中村区）
    "43": (35.423, 136.762),  # 岐阜競輪場（岐阜市）
    "44": (35.358, 136.612),  # 大垣競輪場（大垣市）
    "45": (34.769, 137.391),  # 豊橋競輪場（豊橋市）
    "46": (36.695, 137.213),  # 富山競輪場（富山市）
    "47": (34.578, 136.527),  # 松阪競輪場（松阪市）
    "48": (34.973, 136.624),  # 四日市競輪場（四日市市）
    "51": (36.065, 136.222),  # 福井競輪場（福井市）
    # --- 近畿 ---
    "53": (34.685, 135.832),  # 奈良競輪場（奈良市）
    "54": (34.931, 135.699),  # 向日町競輪場（向日市）
    "55": (34.226, 135.167),  # 和歌山競輪場（和歌山市）
    "56": (34.461, 135.371),  # 岸和田競輪場（岸和田市）
    # --- 中国 ---
    "61": (34.489, 133.953),  # 玉野競輪場（玉野市）
    "62": (34.385, 132.455),  # 広島競輪場（広島市南区）
    "63": (34.051, 131.562),  # 防府競輪場（防府市）
    # --- 四国 ---
    "71": (34.340, 134.043),  # 高松競輪場（高松市）
    "73": (34.000, 134.589),  # 小松島競輪場（小松島市）
    "74": (33.559, 133.531),  # 高知競輪場（高知市）
    "75": (33.839, 132.766),  # 松山競輪場（松山市）
    # --- 九州 ---
    "81": (33.883, 130.878),  # 小倉競輪場（北九州市小倉北区）
    "83": (33.321, 130.508),  # 久留米競輪場（久留米市）
    "84": (33.190, 130.010),  # 武雄競輪場（武雄市）
    "85": (33.160, 129.716),  # 佐世保競輪場（佐世保市）
    "86": (33.284, 131.500),  # 別府競輪場（別府市）
    "87": (32.803, 130.706),  # 熊本競輪場（熊本市）
}

# Open-Meteo archive エンドポイント
_API_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,precipitation"
_TIMEZONE = "Asia/Tokyo"
# APIレート配慮: 会場ごとに最大365日を1リクエストで取得
_MAX_DAYS_PER_REQUEST = 365


def _iter_date_chunks(
    start: str, end: str, max_days: int = _MAX_DAYS_PER_REQUEST
) -> Generator[tuple[str, str], None, None]:
    """start〜end を max_days 単位でチャンクに分割する。"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        t = min(s + timedelta(days=max_days - 1), e)
        yield s.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%d")
        s = t + timedelta(days=1)


def _archive_max_date() -> str:
    """Open-Meteo archive API の利用可能な最大日付（today-1）を返す。"""
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_weather(
    venue_id: str,
    start_date: str,
    end_date: str,
    retry: int = 3,
    backoff: float = 5.0,
) -> list[dict]:
    """Open-Meteo から会場の気象データを取得する。

    Open-Meteo archive API のデータは当日(today)を含まない。
    end_date が today 以降の場合は自動的に today-1 に丸める。

    Args:
        venue_id: 2桁 JKA コード（VENUE_COORDS のキー）
        start_date: "YYYY-MM-DD" (inclusive)
        end_date:   "YYYY-MM-DD" (inclusive)
        retry:      失敗時のリトライ回数
        backoff:    リトライ待機秒数（指数バックオフの初期値）

    Returns:
        list of dict:
            {"venue_id": ..., "dt_hour": "YYYY-MM-DD HH:00",
             "wind_speed": ..., "wind_dir": ..., "wind_gust": ...,
             "temp": ..., "precip": ...}
    """
    if venue_id not in VENUE_COORDS:
        raise ValueError(f"Unknown venue_id: {venue_id!r}")

    # end_date を API の上限に丸める（today は archive 対象外）
    max_dt = _archive_max_date()
    if end_date > max_dt:
        logger.info(
            "fetch_weather %s: end_date=%s を API 上限 %s に丸めます",
            venue_id, end_date, max_dt,
        )
        end_date = max_dt
    if start_date > end_date:
        logger.info(
            "fetch_weather %s: start_date=%s > end_date=%s のためスキップ",
            venue_id, start_date, end_date,
        )
        return []

    lat, lon = VENUE_COORDS[venue_id]
    rows: list[dict] = []

    for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": _HOURLY_VARS,
            "timezone": _TIMEZONE,
        }
        last_err: Exception | None = None
        for attempt in range(retry):
            try:
                resp = requests.get(_API_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "fetch_weather %s %s〜%s attempt %d/%d failed: %s — retry in %.0fs",
                    venue_id, chunk_start, chunk_end, attempt + 1, retry, e, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"fetch_weather {venue_id} {chunk_start}〜{chunk_end} failed after {retry} retries"
            ) from last_err

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        wind_dirs = hourly.get("wind_direction_10m", [])
        wind_gusts = hourly.get("wind_gusts_10m", [])
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation", [])

        for i, t in enumerate(times):
            # API は "YYYY-MM-DDTHH:MM" 形式。"YYYY-MM-DD HH:00" に変換。
            dt_hour = t.replace("T", " ").replace(":00", ":00")
            # 分部分を必ず :00 に正規化
            if len(dt_hour) == 16:  # "YYYY-MM-DD HH:MM"
                dt_hour = dt_hour[:14] + "00"
            rows.append(
                {
                    "venue_id": venue_id,
                    "dt_hour": dt_hour,
                    "wind_speed": _safe_float(wind_speeds, i),
                    "wind_dir": _safe_float(wind_dirs, i),
                    "wind_gust": _safe_float(wind_gusts, i),
                    "temp": _safe_float(temps, i),
                    "precip": _safe_float(precips, i),
                }
            )

    return rows


def _safe_float(lst: list, idx: int) -> float | None:
    """リストから安全に float を取得する。"""
    try:
        v = lst[idx]
        return float(v) if v is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def ensure_table(conn: sqlite3.Connection) -> None:
    """wt_weather テーブルが存在しなければ作成する。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wt_weather (
            venue_id  TEXT NOT NULL,
            dt_hour   TEXT NOT NULL,   -- 'YYYY-MM-DD HH:00' JST
            wind_speed REAL,
            wind_dir   REAL,
            wind_gust  REAL,
            temp       REAL,
            precip     REAL,
            PRIMARY KEY (venue_id, dt_hour)
        )
        """
    )
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """rows を wt_weather に INSERT OR REPLACE で書き込む。

    Returns:
        挿入/更新された行数
    """
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO wt_weather
            (venue_id, dt_hour, wind_speed, wind_dir, wind_gust, temp, precip)
        VALUES
            (:venue_id, :dt_hour, :wind_speed, :wind_dir, :wind_gust, :temp, :precip)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def weather_for_race(
    race_key: str,
    conn: sqlite3.Connection,
) -> dict | None:
    """レースキーに対応する最近傍時刻の気象データを返す（G06 から import される）。

    Args:
        race_key: wt_races.race_key
        conn: SQLite 接続

    Returns:
        dict {"venue_id", "dt_hour", "wind_speed", "wind_dir", "wind_gust", "temp", "precip"}
        or None if not found
    """
    row = conn.execute(
        "SELECT venue_id, start_at FROM wt_races WHERE race_key = ?", (race_key,)
    ).fetchone()
    if row is None:
        return None

    venue_id, start_at = row
    if not start_at:
        return None

    # start_at を "YYYY-MM-DD HH:00" に丸める（下方向・00分区切り）
    try:
        dt = datetime.strptime(start_at[:16], "%Y-%m-%d %H:%M")
        dt_hour = dt.strftime("%Y-%m-%d %H:00")
    except ValueError:
        logger.warning("weather_for_race: invalid start_at=%r for race_key=%s", start_at, race_key)
        return None

    result = conn.execute(
        """
        SELECT venue_id, dt_hour, wind_speed, wind_dir, wind_gust, temp, precip
          FROM wt_weather
         WHERE venue_id = ? AND dt_hour = ?
        """,
        (venue_id, dt_hour),
    ).fetchone()

    if result is None:
        # 前後±1時間で最近傍を探す
        dt_prev = (dt - timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
        dt_next = (dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
        result = conn.execute(
            """
            SELECT venue_id, dt_hour, wind_speed, wind_dir, wind_gust, temp, precip
              FROM wt_weather
             WHERE venue_id = ? AND dt_hour IN (?, ?)
             ORDER BY ABS(
                 (strftime('%s', dt_hour) - strftime('%s', ?))
             )
             LIMIT 1
            """,
            (venue_id, dt_prev, dt_next, dt_hour),
        ).fetchone()

    if result is None:
        return None

    cols = ["venue_id", "dt_hour", "wind_speed", "wind_dir", "wind_gust", "temp", "precip"]
    return dict(zip(cols, result))
