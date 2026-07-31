"""reselect_7s_evening.py の netkeirin 取り下げ漏れ検知・通知のテスト（D-7）。

背景: reselect_7s_evening.py はentropyゲート通過後の日次件数トリムで未購入
（bet_amount=0）プレースホルダを picks_history から削除する
（_delete_dropped_placeholders）。この際、対象レースが既に netkeirin へ
入稿済み（netkeirin_submissions に存在）だと、netkeirin側の下書きが
「幽霊ピック」として取り下げられずに残ってしまう。本テストは以下を検証する:

1. 入稿済みレースがドロップされたとき Discord 通知が呼ばれること
2. 未入稿レースがドロップされたときは通知されないこと
3. Discord 通知が例外を投げてもトリム処理（DELETE）が継続すること
4. bet_amount>0（ロック済み）のレースはそもそもドロップ対象にならないこと

DB アクセスは一切実 DB に触れないよう、_find_netkeirin_submitted /
_delete_dropped_placeholders が使う get_connection をフェイクへ差し替える。
本リポジトリの実行環境では KEIRIN_DB_URL が本番 VPS PostgreSQL を指しているため、
_delete_dropped_placeholders 内の psycopg2 分岐が発火しないよう
KEIRIN_DB_URL も明示的に unset する（get_connection モックだけに頼らない
二重の安全策）。
"""
from __future__ import annotations

import sys

import pytest

import reselect_7s_evening as ser  # scripts/ は conftest で path 追加済


# ---------------------------------------------------------------------------
# フェイク DB 接続（netkeirin_submissions の SELECT のみ中身を模す。
# picks_history への DELETE は記録するだけで何もしない）。
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConnection:
    """get_connection() の戻り値（with 文で使う想定）を模したフェイク。

    netkeirin_submissions に対する SELECT だけ、コンストラクタで渡した
    submissions（[(race_key, rank_key), ...]）から実際にフィルタして返す。
    それ以外（picks_history への DELETE 等）は呼び出しを記録するのみ。
    """

    def __init__(self, submissions: list[tuple[str, str]]) -> None:
        self.submissions = submissions
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        self.executed.append((sql, tuple(params)))
        normalized = sql.strip().upper()
        if normalized.startswith("SELECT") and "NETKEIRIN_SUBMISSIONS" in sql.upper():
            n_rank_keys = len(ser._RANK_7S_NETKEIRIN_RANK_KEYS)
            keys = set(params[:-n_rank_keys])
            rank_keys = set(params[-n_rank_keys:])
            rows = [
                (rk, rank) for rk, rank in self.submissions
                if rk in keys and rank in rank_keys
            ]
            return _FakeCursor(rows)
        return _FakeCursor([])

    def commit(self) -> None:
        pass

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """全テスト共通の安全策: 実DB（本番VPS PostgreSQL）へは絶対に接続しない。

    KEIRIN_DB_URL がこの実行環境で本番VPSを指していても、
    _delete_dropped_placeholders の psycopg2 分岐へ入らないよう unset する。
    """
    monkeypatch.delenv("KEIRIN_DB_URL", raising=False)


def _install_fake_connection(monkeypatch, submissions: list[tuple[str, str]]) -> _FakeConnection:
    fake = _FakeConnection(submissions)
    monkeypatch.setattr(ser, "get_connection", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# 1. 入稿済みレースがドロップされたとき Discord 通知が呼ばれること
# ---------------------------------------------------------------------------

def test_notify_called_when_dropped_race_already_submitted(monkeypatch):
    fake = _install_fake_connection(
        monkeypatch, submissions=[("20260731_04_07", "7S")],
    )
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(ser, "send", lambda content, channel: sent.append((content, channel)))

    ser._notify_netkeirin_orphans("2026-07-31", {"20260731_04_07"})

    assert len(sent) == 1
    content, channel = sent[0]
    assert channel == "netkeirin"
    assert "20260731_04_07" in content
    assert "7S" in content
    assert "netkeirin取り下げ漏れ" in content
    # SELECT が netkeirin_submissions に対して実際に発行されたこと
    assert any("NETKEIRIN_SUBMISSIONS" in sql.upper() for sql, _ in fake.executed)


# ---------------------------------------------------------------------------
# 2. 未入稿レースがドロップされたときは通知されないこと
# ---------------------------------------------------------------------------

def test_notify_not_called_when_dropped_race_not_submitted(monkeypatch):
    _install_fake_connection(monkeypatch, submissions=[])
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(ser, "send", lambda content, channel: sent.append((content, channel)))

    ser._notify_netkeirin_orphans("2026-07-31", {"20260731_04_07"})

    assert sent == []


def test_notify_not_called_for_unrelated_rank_key(monkeypatch):
    """同一race_keyでも7SS/7S以外のランク（例: 7A）への入稿は無関係のため無視する。"""
    _install_fake_connection(
        monkeypatch, submissions=[("20260731_04_07", "7A")],
    )
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(ser, "send", lambda content, channel: sent.append((content, channel)))

    ser._notify_netkeirin_orphans("2026-07-31", {"20260731_04_07"})

    assert sent == []


# ---------------------------------------------------------------------------
# 3. Discord 通知が例外を投げてもトリム処理（DELETE）が継続すること
# ---------------------------------------------------------------------------

def test_trim_continues_when_discord_notify_raises(monkeypatch):
    fake = _install_fake_connection(
        monkeypatch, submissions=[("20260731_04_07", "7S")],
    )

    def _boom(content, channel):
        raise RuntimeError("discord down")

    monkeypatch.setattr(ser, "send", _boom)

    # 例外が外へ伝播しないこと
    ser._delete_dropped_placeholders("2026-07-31", {"20260731_04_07"})

    # DELETE FROM picks_history が実際に発行されたこと（通知失敗でスキップされていない）
    delete_calls = [
        (sql, params) for sql, params in fake.executed
        if sql.strip().upper().startswith("DELETE")
    ]
    assert len(delete_calls) == 1
    assert delete_calls[0][1] == ("20260731_04_07#7S",)


def test_trim_continues_when_netkeirin_select_raises(monkeypatch):
    """netkeirin_submissions への問い合わせ自体が失敗しても削除処理は継続する。"""

    class _BoomConnection:
        def __enter__(self):
            raise RuntimeError("db down")

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = {"n": 0}
    real_fake = _FakeConnection([])

    def _get_connection():
        calls["n"] += 1
        if calls["n"] == 1:
            # 1回目（netkeirin_submissions確認）は失敗させる
            return _BoomConnection()
        # 2回目（picks_history削除）は成功させる
        return real_fake

    monkeypatch.setattr(ser, "get_connection", _get_connection)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(ser, "send", lambda content, channel: sent.append((content, channel)))

    ser._delete_dropped_placeholders("2026-07-31", {"20260731_04_07"})

    assert sent == []  # 確認できなかったので通知はしない
    delete_calls = [
        (sql, params) for sql, params in real_fake.executed
        if sql.strip().upper().startswith("DELETE")
    ]
    assert len(delete_calls) == 1


# ---------------------------------------------------------------------------
# 4. bet_amount>0（ロック済み）のレースはドロップ対象にならないこと
# ---------------------------------------------------------------------------

def test_locked_race_excluded_from_dropped_set(monkeypatch):
    """main() のトリム集合計算 (day_selected - final) - locked を検証する。

    day_selected には rk_a（未購入・トリムで外れる）・rk_b（入稿済み・未購入・
    トリムで外れる）・rk_locked（購入済みロック・rank_7s_evening_reselect の結果
    final から漏れる想定）の3件を用意する。rk_locked は _locked_keys() が
    ロック対象として返すため、たとえ final から漏れても dropped 集合には
    含まれてはならない（実購入を誤って「取り下げてください」と警告しては
    ならないため）。
    """
    day_raw = [
        {"race_key": "rk_a", "axis_sum": 1.0},
        {"race_key": "rk_b", "axis_sum": 2.0},
        {"race_key": "rk_locked", "axis_sum": 3.0},
    ]
    night_raw: list[dict] = []
    day_selected = [
        {"race_key": "rk_a"}, {"race_key": "rk_b"}, {"race_key": "rk_locked"},
    ]

    def _fake_load_raw(path):
        name = path.name
        if name.endswith("_night_s7_raw_candidates.json"):
            return night_raw
        if name.endswith("_s7_raw_candidates.json"):
            return day_raw
        if name.endswith("_s7_candidates.json"):
            return day_selected
        raise AssertionError(f"想定外のパス: {path}")

    monkeypatch.setattr(ser, "_load_raw", _fake_load_raw)
    monkeypatch.setattr(ser, "_locked_keys", lambda target_date: {"rk_locked"})

    # rank_7s_evening_reselect: 全員トリムで落ちる（rk_locked含む）という極端な
    # ケースを想定し、ロック除外が dropped 計算側で効くことを確認する。
    monkeypatch.setattr(ser, "rank_7s_evening_reselect", lambda day, night, locked: [])

    recorded: list[tuple[str, set]] = []
    monkeypatch.setattr(
        ser, "_delete_dropped_placeholders",
        lambda target_date, dropped_keys: recorded.append((target_date, dropped_keys)),
    )

    # main() 末尾のファイル書き込みは実ファイルに触れないようフェイク化する。
    written: list[tuple[str, object]] = []

    class _FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data):
            written.append(data)

    monkeypatch.setattr(ser, "open", lambda *a, **k: _FakeFile(), raising=False)
    monkeypatch.setattr(sys, "argv", ["reselect_7s_evening.py", "2026-07-31"])

    ser.main()

    assert len(recorded) == 1
    target_date, dropped_keys = recorded[0]
    assert target_date == "2026-07-31"
    assert dropped_keys == {"rk_a", "rk_b"}
    assert "rk_locked" not in dropped_keys
