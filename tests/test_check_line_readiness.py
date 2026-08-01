"""scripts/check_line_readiness.py のテスト（2026-08-01 新設）。

背景: 朝(daily_picks_wt.sh)を7:00・夕方(evening_picks_wt.sh)を8:00へ前倒しする
にあたり、ライン情報(winticket linePrediction)が想定より遅れて公開された場合に
備えたリトライ判定用ヘルパー。判定ロジックをシェルへ埋め込むとテストできないため
Python側に切り出した（タスク仕様の推奨(b)方式）。

DB アクセスは一切実DB（本番VPS PostgreSQL）に触れないよう、
get_connection をフェイクへ差し替える
（tests/test_7s_evening_reselect_netkeirin_notify.py と同じ安全策）。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

import check_line_readiness as clr  # scripts/ は conftest で path 追加済み

JST = timezone(timedelta(hours=9))


def _ts(hh: int, mm: int = 0, day: str = "2026-08-01") -> str:
    """JST の日時から winticket 形式(unix秒文字列)の start_at を作る。"""
    dt = datetime.strptime(f"{day} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M").replace(tzinfo=JST)
    return str(int(dt.timestamp()))


class _FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict]:
        return self._rows


class _FakeConnection:
    """get_connection() の戻り値（with 文で使う想定）を模したフェイク。

    check() が発行する唯一のSELECTに対して、コンストラクタで渡した行を返す。
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        self.executed.append((sql, tuple(params)))
        return _FakeCursor(self.rows)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """全テスト共通の安全策: 実DB（本番VPS PostgreSQL）へは絶対に接続しない。"""
    monkeypatch.delenv("KEIRIN_DB_URL", raising=False)


def _install_fake_connection(monkeypatch, rows: list[dict]) -> _FakeConnection:
    fake = _FakeConnection(rows)
    monkeypatch.setattr(clr, "get_connection", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# _hour_of: winticket start_at (JST unix秒文字列) の変換
# ---------------------------------------------------------------------------

def test_hour_of_parses_unix_seconds_as_jst():
    assert clr._hour_of(_ts(7, 30)) == 7
    assert clr._hour_of(_ts(19, 45)) == 19
    assert clr._hour_of(_ts(0, 5)) == 0


def test_hour_of_none_for_missing_or_invalid():
    assert clr._hour_of(None) is None
    assert clr._hour_of("not-a-number") is None


# ---------------------------------------------------------------------------
# _in_target_window: src/cli/main.py の _hour_skip() を反転した意味論
# ---------------------------------------------------------------------------

def test_target_window_morning_excludes_night_and_includes_unknown():
    # start_to_hour=19 (朝の部): hh<19 は対象、hh>=19 は対象外、hh不明は対象に含む
    assert clr._in_target_window(10, None, 19) is True
    assert clr._in_target_window(18, None, 19) is True
    assert clr._in_target_window(19, None, 19) is False
    assert clr._in_target_window(None, None, 19) is True


def test_target_window_evening_excludes_day_and_unknown():
    # start_from_hour=19 (夜の部): hh>=19 は対象、hh<19・hh不明は対象外
    assert clr._in_target_window(19, 19, None) is True
    assert clr._in_target_window(23, 19, None) is True
    assert clr._in_target_window(18, 19, None) is False
    assert clr._in_target_window(None, 19, None) is False


# ---------------------------------------------------------------------------
# check(): 充足/不足の判定
# ---------------------------------------------------------------------------

def test_check_ok_when_all_races_have_lines(monkeypatch):
    rows = [
        {"race_key": f"20260801_01_{i:02d}", "start_at": _ts(10 + i), "n_lines": 3}
        for i in range(5)
    ]
    _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=None, start_to_hour=19)

    assert is_ok is True
    assert "充足" in message


def test_check_insufficient_when_ratio_exceeds_threshold(monkeypatch):
    # 5レース中2レースがn_lines=0 → 40% > 30%閾値 → 不足
    rows = [
        {"race_key": "20260801_01_01", "start_at": _ts(10), "n_lines": 0},
        {"race_key": "20260801_01_02", "start_at": _ts(11), "n_lines": 0},
        {"race_key": "20260801_01_03", "start_at": _ts(12), "n_lines": 3},
        {"race_key": "20260801_01_04", "start_at": _ts(13), "n_lines": 3},
        {"race_key": "20260801_01_05", "start_at": _ts(14), "n_lines": 3},
    ]
    _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=None, start_to_hour=19)

    assert is_ok is False
    assert "不足" in message


def test_check_ok_when_ratio_within_threshold(monkeypatch):
    # 10レース中1レースのみ0件 → 10% <= 30%閾値 → 充足扱い（数レース程度のばらつきは許容）
    rows = [
        {"race_key": f"20260801_01_{i:02d}", "start_at": _ts(10 + i), "n_lines": 0 if i == 0 else 3}
        for i in range(10)
    ]
    _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=None, start_to_hour=19)

    assert is_ok is True


def test_check_skips_judgement_when_too_few_target_races(monkeypatch):
    # 対象2レースのみ（MIN_TARGET_RACES=3未満）→ 全滅でも判定スキップ扱い
    rows = [
        {"race_key": "20260801_01_01", "start_at": _ts(10), "n_lines": 0},
        {"race_key": "20260801_01_02", "start_at": _ts(11), "n_lines": 0},
    ]
    _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=None, start_to_hour=19)

    assert is_ok is True
    assert "判定スキップ" in message


def test_check_null_n_lines_treated_as_no_line(monkeypatch):
    """wt_entriesが1件も無い（LEFT JOINでn_lines=NULL）レースも未公開扱いにすること。"""
    rows = [
        {"race_key": "20260801_01_01", "start_at": _ts(10), "n_lines": None},
        {"race_key": "20260801_01_02", "start_at": _ts(11), "n_lines": None},
        {"race_key": "20260801_01_03", "start_at": _ts(12), "n_lines": None},
        {"race_key": "20260801_01_04", "start_at": _ts(13), "n_lines": 3},
    ]
    _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=None, start_to_hour=19)

    # 4レース中3レースが未公開(NULL) = 75% > 30% → 不足
    assert is_ok is False


def test_check_evening_window_only_counts_night_races(monkeypatch):
    rows = [
        {"race_key": "20260801_01_01", "start_at": _ts(10), "n_lines": 0},  # 昼・対象外
        {"race_key": "20260801_01_02", "start_at": _ts(19), "n_lines": 3},  # 夜・対象
        {"race_key": "20260801_01_03", "start_at": _ts(20), "n_lines": 3},  # 夜・対象
        {"race_key": "20260801_01_04", "start_at": _ts(21), "n_lines": 3},  # 夜・対象
    ]
    fake = _install_fake_connection(monkeypatch, rows)

    is_ok, message = clr.check("2026-08-01", start_from_hour=19, start_to_hour=None)

    assert is_ok is True  # 夜3レースは全て充足（昼の0件は対象外なので無関係）
    assert fake.executed  # SELECTが発行されたこと


# ---------------------------------------------------------------------------
# main(): 終了コード
# ---------------------------------------------------------------------------

def test_main_exit_code_0_when_ok(monkeypatch, capsys):
    rows = [
        {"race_key": f"20260801_01_{i:02d}", "start_at": _ts(10 + i), "n_lines": 3}
        for i in range(5)
    ]
    _install_fake_connection(monkeypatch, rows)
    monkeypatch.setattr(sys, "argv", ["check_line_readiness.py", "--date", "2026-08-01", "--start-to-hour", "19"])

    with pytest.raises(SystemExit) as exc_info:
        clr.main()

    assert exc_info.value.code == 0
    assert "充足" in capsys.readouterr().out


def test_main_exit_code_1_when_insufficient(monkeypatch, capsys):
    rows = [
        {"race_key": f"20260801_01_{i:02d}", "start_at": _ts(10 + i), "n_lines": 0}
        for i in range(5)
    ]
    _install_fake_connection(monkeypatch, rows)
    monkeypatch.setattr(sys, "argv", ["check_line_readiness.py", "--date", "2026-08-01", "--start-to-hour", "19"])

    with pytest.raises(SystemExit) as exc_info:
        clr.main()

    assert exc_info.value.code == 1
    assert "不足" in capsys.readouterr().out
