"""ユニットテスト: src/scraper/weather.py

テスト対象:
- VENUE_COORDS: 43会場全件 × 日本国内緯度経度
- ensure_table / upsert_rows: DB 操作
- weather_for_race: 最近傍時刻のデータ取得
- fetch_weather のモック（API 呼び出しなし）
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.scraper.weather import (
    VENUE_COORDS,
    ensure_table,
    fetch_weather,
    upsert_rows,
    weather_for_race,
)
from src.scraper.winticket import VENUE_SLUGS


# ---------------------------------------------------------------------------
# 座標テーブルのユニットテスト
# ---------------------------------------------------------------------------

class TestVenueCoords:
    """VENUE_COORDS の整合性チェック"""

    def test_all_43_venues_present(self) -> None:
        """VENUE_SLUGS の全43会場に座標が登録されている"""
        missing = set(VENUE_SLUGS.keys()) - set(VENUE_COORDS.keys())
        assert not missing, f"座標未登録の venue_id: {sorted(missing)}"

    def test_no_extra_venues(self) -> None:
        """VENUE_COORDS に VENUE_SLUGS 以外のキーがない"""
        extra = set(VENUE_COORDS.keys()) - set(VENUE_SLUGS.keys())
        assert not extra, f"VENUE_SLUGS に存在しない venue_id: {sorted(extra)}"

    def test_all_latitudes_in_japan(self) -> None:
        """全会場の緯度が日本国内範囲（24°〜46°N）"""
        for vid, (lat, lon) in VENUE_COORDS.items():
            assert 24.0 <= lat <= 46.0, (
                f"venue_id={vid}: 緯度 {lat} が日本国内範囲外"
            )

    def test_all_longitudes_in_japan(self) -> None:
        """全会場の経度が日本国内範囲（122°〜154°E）"""
        for vid, (lat, lon) in VENUE_COORDS.items():
            assert 122.0 <= lon <= 154.0, (
                f"venue_id={vid}: 経度 {lon} が日本国内範囲外"
            )

    def test_coordinate_count(self) -> None:
        """座標テーブルが正確に43件"""
        assert len(VENUE_COORDS) == 43

    @pytest.mark.parametrize("venue_id", sorted(VENUE_COORDS.keys()))
    def test_each_venue_has_valid_tuple(self, venue_id: str) -> None:
        """各会場の座標が (lat, lon) のタプル"""
        coords = VENUE_COORDS[venue_id]
        assert isinstance(coords, tuple), f"venue_id={venue_id}: tuple でない"
        assert len(coords) == 2, f"venue_id={venue_id}: 要素数が2でない"
        lat, lon = coords
        assert isinstance(lat, float), f"venue_id={venue_id}: lat が float でない"
        assert isinstance(lon, float), f"venue_id={venue_id}: lon が float でない"


# ---------------------------------------------------------------------------
# DB 操作のユニットテスト（インメモリ DB 使用）
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """インメモリ SQLite 接続（wt_weather テーブル付き）"""
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    yield conn
    conn.close()


@pytest.fixture
def mem_conn_with_races(mem_conn):
    """wt_races テーブルも持つインメモリ DB"""
    mem_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wt_races (
            race_key TEXT PRIMARY KEY,
            venue_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            race_no INTEGER NOT NULL,
            cup_id TEXT NOT NULL,
            day_index INTEGER NOT NULL,
            grade TEXT,
            race_type TEXT,
            distance INTEGER,
            n_entries INTEGER,
            start_at TEXT,
            status INTEGER DEFAULT 0,
            cancel INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    mem_conn.commit()
    return mem_conn


class TestEnsureTable:
    def test_table_created(self, mem_conn) -> None:
        tables = mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='wt_weather'"
        ).fetchall()
        assert len(tables) == 1

    def test_idempotent(self, mem_conn) -> None:
        """2回呼び出しても例外が出ない"""
        ensure_table(mem_conn)  # 2回目
        ensure_table(mem_conn)  # 3回目

    def test_schema_columns(self, mem_conn) -> None:
        cols = {row[1] for row in mem_conn.execute("PRAGMA table_info(wt_weather)")}
        assert cols >= {"venue_id", "dt_hour", "wind_speed", "wind_dir", "wind_gust", "temp", "precip"}


class TestUpsertRows:
    def _sample_rows(self) -> list[dict]:
        return [
            {
                "venue_id": "61",
                "dt_hour": "2025-01-01 10:00",
                "wind_speed": 5.0,
                "wind_dir": 180.0,
                "wind_gust": 8.0,
                "temp": 15.0,
                "precip": 0.0,
            },
            {
                "venue_id": "61",
                "dt_hour": "2025-01-01 11:00",
                "wind_speed": 6.0,
                "wind_dir": 200.0,
                "wind_gust": 9.0,
                "temp": 16.0,
                "precip": 0.2,
            },
        ]

    def test_insert(self, mem_conn) -> None:
        n = upsert_rows(mem_conn, self._sample_rows())
        assert n == 2
        cnt = mem_conn.execute("SELECT COUNT(*) FROM wt_weather").fetchone()[0]
        assert cnt == 2

    def test_replace(self, mem_conn) -> None:
        """INSERT OR REPLACE: 同一 PK で上書き"""
        upsert_rows(mem_conn, self._sample_rows())
        updated = [
            {**self._sample_rows()[0], "wind_speed": 99.9}
        ]
        upsert_rows(mem_conn, updated)
        row = mem_conn.execute(
            "SELECT wind_speed FROM wt_weather WHERE venue_id='61' AND dt_hour='2025-01-01 10:00'"
        ).fetchone()
        assert row[0] == pytest.approx(99.9)
        cnt = mem_conn.execute("SELECT COUNT(*) FROM wt_weather").fetchone()[0]
        assert cnt == 2  # 行数は増えない

    def test_empty(self, mem_conn) -> None:
        n = upsert_rows(mem_conn, [])
        assert n == 0

    def test_none_values_stored(self, mem_conn) -> None:
        """NULL 値は格納できる（センサー欠損対応）"""
        rows = [
            {
                "venue_id": "61",
                "dt_hour": "2025-01-02 10:00",
                "wind_speed": None,
                "wind_dir": None,
                "wind_gust": None,
                "temp": None,
                "precip": None,
            }
        ]
        upsert_rows(mem_conn, rows)
        row = mem_conn.execute(
            "SELECT wind_speed FROM wt_weather WHERE venue_id='61' AND dt_hour='2025-01-02 10:00'"
        ).fetchone()
        assert row[0] is None


# ---------------------------------------------------------------------------
# weather_for_race のユニットテスト
# ---------------------------------------------------------------------------

class TestWeatherForRace:
    def _insert_race(self, conn, race_key, venue_id, race_date, start_at):
        conn.execute(
            """
            INSERT INTO wt_races
                (race_key, venue_id, race_date, race_no, cup_id, day_index, start_at)
            VALUES (?, ?, ?, 1, 'DUMMY', 1, ?)
            """,
            (race_key, venue_id, race_date, start_at),
        )
        conn.commit()

    def _insert_weather(self, conn, venue_id, dt_hour, wind_speed=5.0):
        upsert_rows(
            conn,
            [
                {
                    "venue_id": venue_id,
                    "dt_hour": dt_hour,
                    "wind_speed": wind_speed,
                    "wind_dir": 180.0,
                    "wind_gust": 8.0,
                    "temp": 20.0,
                    "precip": 0.0,
                }
            ],
        )

    def test_exact_match(self, mem_conn_with_races) -> None:
        conn = mem_conn_with_races
        self._insert_race(conn, "RK001", "61", "2025-06-01", "2025-06-01 14:30")
        self._insert_weather(conn, "61", "2025-06-01 14:00", wind_speed=7.7)
        result = weather_for_race("RK001", conn)
        assert result is not None
        assert result["venue_id"] == "61"
        assert result["dt_hour"] == "2025-06-01 14:00"
        assert result["wind_speed"] == pytest.approx(7.7)

    def test_race_not_found(self, mem_conn_with_races) -> None:
        result = weather_for_race("NONEXISTENT", mem_conn_with_races)
        assert result is None

    def test_no_weather_data(self, mem_conn_with_races) -> None:
        conn = mem_conn_with_races
        self._insert_race(conn, "RK002", "61", "2025-06-02", "2025-06-02 14:30")
        result = weather_for_race("RK002", conn)
        assert result is None

    def test_fallback_to_adjacent_hour(self, mem_conn_with_races) -> None:
        """当該時刻データなし → 前後1時間から最近傍を返す"""
        conn = mem_conn_with_races
        self._insert_race(conn, "RK003", "61", "2025-06-03", "2025-06-03 14:45")
        # 14:00 のデータのみ（14:45 の切り捨て時刻 14:00 にデータなし→15:00を探す）
        # 実際には 14:00 がないので 13:00 と 15:00 を試す
        self._insert_weather(conn, "61", "2025-06-03 15:00", wind_speed=3.3)
        result = weather_for_race("RK003", conn)
        # 15:00 のデータが返るか None（前後1時間に見つかれば OK）
        # race start_at 14:45 → dt_hour=14:00 (切捨) → fallback=13:00,15:00
        assert result is not None
        assert result["dt_hour"] == "2025-06-03 15:00"

    def test_missing_start_at(self, mem_conn_with_races) -> None:
        conn = mem_conn_with_races
        self._insert_race(conn, "RK004", "61", "2025-06-04", None)
        result = weather_for_race("RK004", conn)
        assert result is None

    def test_return_keys(self, mem_conn_with_races) -> None:
        """返り値の dict が必要なキーを持つ"""
        conn = mem_conn_with_races
        self._insert_race(conn, "RK005", "61", "2025-06-05", "2025-06-05 10:00")
        self._insert_weather(conn, "61", "2025-06-05 10:00")
        result = weather_for_race("RK005", conn)
        assert result is not None
        expected_keys = {"venue_id", "dt_hour", "wind_speed", "wind_dir", "wind_gust", "temp", "precip"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# fetch_weather のモックテスト（API 呼び出しなし）
# ---------------------------------------------------------------------------

class TestFetchWeather:
    def _mock_response(self) -> dict:
        return {
            "hourly": {
                "time": [
                    "2025-01-01T00:00",
                    "2025-01-01T01:00",
                ],
                "wind_speed_10m": [5.0, 6.0],
                "wind_direction_10m": [180.0, 200.0],
                "wind_gusts_10m": [8.0, 9.0],
                "temperature_2m": [15.0, 14.5],
                "precipitation": [0.0, 0.2],
            }
        }

    def test_fetch_returns_correct_rows(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.scraper.weather.requests.get", return_value=mock_resp):
            rows = fetch_weather("61", "2025-01-01", "2025-01-01")

        assert len(rows) == 2
        assert rows[0]["venue_id"] == "61"
        assert rows[0]["dt_hour"] == "2025-01-01 00:00"
        assert rows[0]["wind_speed"] == pytest.approx(5.0)
        assert rows[1]["dt_hour"] == "2025-01-01 01:00"
        assert rows[1]["precip"] == pytest.approx(0.2)

    def test_fetch_unknown_venue_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown venue_id"):
            fetch_weather("99", "2025-01-01", "2025-01-01")

    def test_fetch_retry_on_error(self) -> None:
        """接続エラー時にリトライし、最終的に例外を投げる"""
        with patch(
            "src.scraper.weather.requests.get",
            side_effect=ConnectionError("timeout"),
        ):
            with patch("src.scraper.weather.time.sleep"):  # sleep をスキップ
                with pytest.raises(RuntimeError, match="failed after"):
                    fetch_weather("61", "2025-01-01", "2025-01-01", retry=2, backoff=0.1)
