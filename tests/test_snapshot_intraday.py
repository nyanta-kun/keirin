"""snapshot_intraday_odds_wt.py のテスト。

主要テスト:
1. auto_snapshot_type: 時刻から 'h{HH}' を正しく生成する
2. snapshot: wt_odds_snapshot への INSERT OR REPLACE が冪等であることを確認
3. get_nearest_snapshot: 基準時刻に最も近い snapshot を返すことを確認
4. UNIQUE 制約: 同一 (race_key, bet_type, combination, snapshot_type) を
   INSERT OR REPLACE で再実行しても行数が増えないことを確認
"""
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

# conftest.py でプロジェクトルートが sys.path に追加されている
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snapshot_intraday_odds_wt import (
    auto_snapshot_type,
    snapshot,
    get_nearest_snapshot,
)


# ---------------------------------------------------------------------------
# テスト用インメモリ DB セットアップ
# ---------------------------------------------------------------------------

_JST = timezone(timedelta(hours=9))


def _make_in_memory_db():
    """テスト用のインメモリ SQLite DB を初期化して返す。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE wt_races (
            race_key  TEXT PRIMARY KEY,
            venue_id  TEXT NOT NULL,
            race_date TEXT NOT NULL,
            race_no   INTEGER NOT NULL,
            cup_id    TEXT NOT NULL,
            day_index INTEGER NOT NULL,
            grade     TEXT,
            race_type TEXT,
            distance  INTEGER,
            n_entries INTEGER,
            start_at  TEXT,
            status    INTEGER DEFAULT 0,
            cancel    INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE wt_odds_snapshot (
            id            INTEGER PRIMARY KEY,
            race_key      TEXT NOT NULL,
            bet_type      TEXT NOT NULL,
            combination   TEXT NOT NULL,
            odds_value    REAL,
            snapshot_type TEXT NOT NULL DEFAULT 'morning',
            snapshot_at   TEXT DEFAULT (datetime('now')),
            UNIQUE(race_key, bet_type, combination, snapshot_type)
        );
        CREATE INDEX idx_wt_odds_snap_race ON wt_odds_snapshot(race_key);
    """)
    return conn


# ---------------------------------------------------------------------------
# auto_snapshot_type のテスト
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour, expected", [
    (0,  "h00"),
    (8,  "h08"),
    (10, "h10"),
    (12, "h12"),
    (18, "h18"),
    (20, "h20"),
    (23, "h23"),
])
def test_auto_snapshot_type(hour, expected):
    dt = datetime(2026, 6, 13, hour, 30, 0, tzinfo=_JST)
    assert auto_snapshot_type(dt) == expected


# ---------------------------------------------------------------------------
# snapshot: INSERT OR REPLACE の冪等性テスト
# ---------------------------------------------------------------------------

def _fake_get_connection(conn):
    """get_connection をインメモリ DB にパッチする contextmanager。"""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield conn
    return _ctx


def _make_fake_scraper(odds_data: dict) -> MagicMock:
    """winticket から指定の odds_data を返すモックスクレイパーを作成する。"""
    scraper = MagicMock()
    scraper.fetch_odds.return_value = odds_data
    return scraper


def test_snapshot_inserts_rows(tmp_path):
    """snapshot() が未発走レースのオッズを正しく保存することを確認。"""
    conn = _make_in_memory_db()

    # 未来のレースを 2 件挿入（start_at = unix epoch 2099年）
    future_ts = 4102444800  # 2099-12-31 00:00:00 UTC
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_01", "23", "2026-06-13", 1, "2026061323", 1,
         None, None, None, 9, future_ts, 1, 0, "2026-06-13T08:00:00"),
    )
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_02", "23", "2026-06-13", 2, "2026061323", 1,
         None, None, None, 9, future_ts + 1200, 1, 0, "2026-06-13T08:00:00"),
    )
    conn.commit()

    fake_odds = {
        "trio": [
            {"combination": "1-2-3", "odds_value": 5.5, "bet_type": "trio"},
            {"combination": "1-2-4", "odds_value": 8.0, "bet_type": "trio"},
        ],
        "trifecta": [
            {"combination": "1-2-3", "odds_value": 12.0, "bet_type": "trifecta"},
        ],
        "quinellaPlace": [
            {"combination": "1-2", "odds_value": 2.5, "bet_type": "quinellaPlace"},
        ],
        "exacta": [
            {"combination": "1-2", "odds_value": 3.0, "bet_type": "exacta"},
        ],
    }

    fake_scraper = _make_fake_scraper(fake_odds)

    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper", return_value=fake_scraper),
    ):
        snapshot("2026-06-13", snapshot_type="h10")

    # trio + trifecta + quinellaPlace = 2+1+1 = 4 行 × 2 レース = 8 行
    count = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot WHERE snapshot_type = 'h10'"
    ).fetchone()[0]
    assert count == 8, f"期待 8 行, 実際 {count} 行"

    # exacta は保存しない
    bad = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot WHERE bet_type = 'exacta'"
    ).fetchone()[0]
    assert bad == 0


def test_snapshot_replace_is_idempotent(tmp_path):
    """同一 snapshot_type の再実行で行数が増えないことを確認（INSERT OR REPLACE）。"""
    conn = _make_in_memory_db()
    future_ts = 4102444800
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_01", "23", "2026-06-13", 1, "2026061323", 1,
         None, None, None, 9, future_ts, 1, 0, "2026-06-13T08:00:00"),
    )
    conn.commit()

    fake_odds_1 = {
        "trio": [{"combination": "1-2-3", "odds_value": 5.5, "bet_type": "trio"}],
        "trifecta": [],
        "quinellaPlace": [],
    }
    fake_odds_2 = {
        "trio": [{"combination": "1-2-3", "odds_value": 6.0, "bet_type": "trio"}],
        "trifecta": [],
        "quinellaPlace": [],
    }

    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper",
              return_value=_make_fake_scraper(fake_odds_1)),
    ):
        snapshot("2026-06-13", snapshot_type="h10")

    count_first = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot WHERE snapshot_type = 'h10'"
    ).fetchone()[0]

    # 2 回目（オッズ値が変わっても行数は変わらないはず）
    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper",
              return_value=_make_fake_scraper(fake_odds_2)),
    ):
        snapshot("2026-06-13", snapshot_type="h10")

    count_second = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot WHERE snapshot_type = 'h10'"
    ).fetchone()[0]

    assert count_first == count_second == 1, (
        f"INSERT OR REPLACE の冪等性違反: 1回目={count_first}, 2回目={count_second}"
    )

    # 値は最新に更新されていること
    odds_val = conn.execute(
        "SELECT odds_value FROM wt_odds_snapshot WHERE combination = '1-2-3' "
        "AND snapshot_type = 'h10'"
    ).fetchone()[0]
    assert odds_val == 6.0, f"最新値 6.0 に更新されるべき, 実際: {odds_val}"


def test_snapshot_skips_started_races():
    """発走済みレース（start_at < now）はスキップされることを確認。"""
    conn = _make_in_memory_db()
    # 過去のレース（start_at = 1000 = UNIX epoch 初期）
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_99", "23", "2026-06-13", 99, "2026061323", 1,
         None, None, None, 9, 1000, 1, 0, "2026-06-13T08:00:00"),
    )
    conn.commit()

    fake_scraper = _make_fake_scraper({"trio": [], "trifecta": [], "quinellaPlace": []})

    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper", return_value=fake_scraper),
    ):
        snapshot("2026-06-13", snapshot_type="h10")

    count = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot"
    ).fetchone()[0]
    assert count == 0, "発走済みレースは保存されないはず"
    fake_scraper.fetch_odds.assert_not_called()


def test_snapshot_skips_cancelled_races():
    """中止レース（cancel=1）はスキップされることを確認。"""
    conn = _make_in_memory_db()
    future_ts = 4102444800
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_01", "23", "2026-06-13", 1, "2026061323", 1,
         None, None, None, 9, future_ts, 1, 1, "2026-06-13T08:00:00"),  # cancel=1
    )
    conn.commit()

    fake_scraper = _make_fake_scraper({"trio": [], "trifecta": [], "quinellaPlace": []})

    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper", return_value=fake_scraper),
    ):
        snapshot("2026-06-13", snapshot_type="h10")

    count = conn.execute("SELECT COUNT(*) FROM wt_odds_snapshot").fetchone()[0]
    assert count == 0, "中止レースは保存されないはず"
    fake_scraper.fetch_odds.assert_not_called()


def test_snapshot_dry_run_no_write():
    """--dry-run は DB に書き込まないことを確認。"""
    conn = _make_in_memory_db()
    future_ts = 4102444800
    conn.execute(
        "INSERT INTO wt_races VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("20260613_23_01", "23", "2026-06-13", 1, "2026061323", 1,
         None, None, None, 9, future_ts, 1, 0, "2026-06-13T08:00:00"),
    )
    conn.commit()

    fake_odds = {"trio": [{"combination": "1-2-3", "odds_value": 5.5, "bet_type": "trio"}],
                 "trifecta": [], "quinellaPlace": []}

    with (
        patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)),
        patch("snapshot_intraday_odds_wt.init_db"),
        patch("snapshot_intraday_odds_wt.WinticketScraper",
              return_value=_make_fake_scraper(fake_odds)),
    ):
        snapshot("2026-06-13", snapshot_type="h10", dry_run=True)

    count = conn.execute("SELECT COUNT(*) FROM wt_odds_snapshot").fetchone()[0]
    assert count == 0, "dry_run=True なら DB に書かないはず"


# ---------------------------------------------------------------------------
# UNIQUE 制約の直接確認
# ---------------------------------------------------------------------------

def test_unique_constraint_replace():
    """wt_odds_snapshot の UNIQUE 制約で INSERT OR REPLACE が正しく動作する。"""
    conn = _make_in_memory_db()

    # 1 行目
    conn.execute(
        "INSERT OR REPLACE INTO wt_odds_snapshot "
        "(race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at) "
        "VALUES (?,?,?,?,?,?)",
        ("TEST_RACE", "trio", "1-2-3", 5.0, "h10", "2026-06-13T10:00:00+09:00"),
    )
    conn.commit()

    count_before = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot"
    ).fetchone()[0]
    assert count_before == 1

    # 同じキーで値を変えて再挿入
    conn.execute(
        "INSERT OR REPLACE INTO wt_odds_snapshot "
        "(race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at) "
        "VALUES (?,?,?,?,?,?)",
        ("TEST_RACE", "trio", "1-2-3", 7.5, "h10", "2026-06-13T10:30:00+09:00"),
    )
    conn.commit()

    count_after = conn.execute(
        "SELECT COUNT(*) FROM wt_odds_snapshot"
    ).fetchone()[0]
    assert count_after == 1, "REPLACE のため行数は変わらないはず"

    val = conn.execute(
        "SELECT odds_value FROM wt_odds_snapshot"
    ).fetchone()[0]
    assert val == 7.5, f"最新値 7.5 に更新されるべき, 実際: {val}"


# ---------------------------------------------------------------------------
# get_nearest_snapshot のテスト
# ---------------------------------------------------------------------------

def test_get_nearest_snapshot_returns_closest():
    """get_nearest_snapshot が基準時刻に最も近い snapshot を返すことを確認。"""
    conn = _make_in_memory_db()

    # reference: 2026-06-13 10:00:00 JST = 1781312400
    _JST_OFFSET = timedelta(hours=9)
    ref_unix = int(datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone(_JST_OFFSET)).timestamp())

    # h08 snapshot (08:50 JST) = ref - 70min → delta60 外
    # h10 snapshot (09:55 JST) = ref - 5min  → 最も近い（delta60 内）
    # h12 snapshot (12:05 JST) = ref + 125min → delta60 外
    snap_09 = datetime(2026, 6, 13, 8, 50, 0, tzinfo=timezone(_JST_OFFSET)).isoformat(
        timespec="seconds"
    )
    snap_10 = datetime(2026, 6, 13, 9, 55, 0, tzinfo=timezone(_JST_OFFSET)).isoformat(
        timespec="seconds"
    )
    snap_12 = datetime(2026, 6, 13, 12, 5, 0, tzinfo=timezone(_JST_OFFSET)).isoformat(
        timespec="seconds"
    )

    conn.executemany(
        "INSERT INTO wt_odds_snapshot "
        "(race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("RACE_X", "trio", "1-2-3", 5.0, "h09", snap_09),
            ("RACE_X", "trio", "1-2-3", 6.0, "h10", snap_10),
            ("RACE_X", "trio", "1-2-3", 7.0, "h12", snap_12),
        ],
    )
    conn.commit()

    with patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)):
        results = get_nearest_snapshot("RACE_X", ref_unix, delta_minutes=60)

    assert len(results) == 1, f"最も近い1件だけ返すべき, 実際: {len(results)}"
    assert results[0]["odds_value"] == 6.0
    assert results[0]["snapshot_type"] == "h10"


def test_get_nearest_snapshot_no_data():
    """対象データがない場合は空リストを返す。"""
    conn = _make_in_memory_db()

    with patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)):
        results = get_nearest_snapshot("NONEXIST", 1781298000, delta_minutes=60)

    assert results == []


def test_get_nearest_snapshot_filter_bet_type():
    """bet_type フィルタが機能することを確認。"""
    conn = _make_in_memory_db()

    snap_at = datetime(2026, 6, 13, 10, 0, 0, tzinfo=_JST).isoformat(timespec="seconds")
    ref_unix = int(datetime(2026, 6, 13, 10, 0, 0, tzinfo=_JST).timestamp())

    conn.executemany(
        "INSERT INTO wt_odds_snapshot "
        "(race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("RACE_Y", "trio", "1-2-3", 5.0, "h10", snap_at),
            ("RACE_Y", "trifecta", "1-2-3", 12.0, "h10", snap_at),
        ],
    )
    conn.commit()

    with patch("snapshot_intraday_odds_wt.get_connection", _fake_get_connection(conn)):
        results = get_nearest_snapshot("RACE_Y", ref_unix, delta_minutes=60,
                                       bet_type="trio")

    assert all(r["bet_type"] == "trio" for r in results)
    assert len(results) == 1
