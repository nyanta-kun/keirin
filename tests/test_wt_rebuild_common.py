"""src/wt_rebuild_common.py の単体テスト（2026-08-01・F-4対応）。

背景: 月初にその月のvintageモデルが未生成だと、rebuild_{7s,7a,9s,9a}_walkforward_pg.py
が全期間の計算(約40分規模)を終えた後にFileNotFoundErrorで失敗し、結果が丸ごと
失われる事故が実際に発生した(2026-08-01)。加えてwipe(DELETE)とinsertが別々の
接続・トランザクションだったため、wipe成功後にinsertが失敗するとpicks_historyが
空のまま残るリスクもあった。

本テストは以下を実DB・実モデルファイルに一切触れずに検証する:
  1. split_by_model_availability(): モデル存在チェックが正しく窓を仕分けること
  2. rebuild_pg_atomic():
     a. 挿入対象0件なら get_connection() 自体を呼ばず、DB に一切触れないこと
     b. 全窓の wipe+insert が単一の `with get_connection()` ブロック内
        （＝単一トランザクション）で行われること
     c. 個別の窓が0行ならその窓だけ wipe をスキップすること
     d. dry_run 時は SELECT COUNT のみで DELETE/INSERT を発行しないこと
  3. notify_discord_warning(): 送信失敗・例外時も呼び出し元に伝播しないこと

DBアクセスはFakeConnで完全に差し替え、実DB（VPS PG/ローカルSQLite）へは
一切アクセスしない。モデルファイルの存在確認もtmp_pathでMODEL_DIRを差し替える。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import wt_rebuild_common as wrc

#: 本物の状態ファイル（import 時点＝差し替え前に控える）。下の隔離テストで使う。
_REAL_ZERO_ROW_STATE = wrc._ZERO_ROW_STATE


# 隔離そのものは `tests/conftest.py::_isolate_zero_row_state` が repo 全体へ効かせる
# （書き込み元は本ファイルではなく test_three_head_rebuild_guard.py だったため）。


# ---------------------------------------------------------------------------
# split_by_model_availability
# ---------------------------------------------------------------------------


def test_split_by_model_availability_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(wrc, "MODEL_DIR", tmp_path)
    for name in ("lgbm_wt_eval_m2607", "lgbm_wt_win_m2607"):
        (tmp_path / f"{name}.pkl").write_bytes(b"x")

    windows = [("2026-07-01", "2026-07-31", "lgbm_wt_eval_m2607", "lgbm_wt_win_m2607")]
    available, missing = wrc.split_by_model_availability(windows)

    assert available == windows
    assert missing == []


def test_split_by_model_availability_missing_tail(tmp_path, monkeypatch):
    """月初にありがちなケース: 直近月(m2608)のモデルだけが未生成。"""
    monkeypatch.setattr(wrc, "MODEL_DIR", tmp_path)
    for name in ("lgbm_wt_eval_m2607", "lgbm_wt_win_m2607"):
        (tmp_path / f"{name}.pkl").write_bytes(b"x")
    # m2608 は作成しない（未生成を模す）

    w_ok = ("2026-07-01", "2026-07-31", "lgbm_wt_eval_m2607", "lgbm_wt_win_m2607")
    w_missing = ("2026-08-01", "2026-08-01", "lgbm_wt_eval_m2608", "lgbm_wt_win_m2608")
    available, missing = wrc.split_by_model_availability([w_ok, w_missing])

    assert available == [w_ok]
    assert len(missing) == 1
    window, missing_names = missing[0]
    assert window == w_missing
    assert set(missing_names) == {"lgbm_wt_eval_m2608", "lgbm_wt_win_m2608"}


def test_split_by_model_availability_partial_missing(tmp_path, monkeypatch):
    """eval/winの片方だけが存在しないケースも不足として検出されること。"""
    monkeypatch.setattr(wrc, "MODEL_DIR", tmp_path)
    (tmp_path / "lgbm_wt_eval_m2608.pkl").write_bytes(b"x")
    # win モデルは作らない

    w = ("2026-08-01", "2026-08-01", "lgbm_wt_eval_m2608", "lgbm_wt_win_m2608")
    available, missing = wrc.split_by_model_availability([w])

    assert available == []
    assert missing == [(w, ["lgbm_wt_win_m2608"])]


def test_format_missing_report_contains_rank_and_window_info():
    w = ("2026-08-01", "2026-08-01", "lgbm_wt_eval_m2608", "lgbm_wt_win_m2608")
    report = wrc.format_missing_report("RANK_7S", [(w, ["lgbm_wt_eval_m2608", "lgbm_wt_win_m2608"])])
    assert "RANK_7S" in report
    assert "2026-08-01" in report
    assert "lgbm_wt_eval_m2608" in report


# ---------------------------------------------------------------------------
# notify_discord_warning
# ---------------------------------------------------------------------------


def test_notify_discord_warning_swallow_false_return(monkeypatch, capsys):
    monkeypatch.setattr(wrc, "_discord_send", lambda content, channel: False)
    wrc.notify_discord_warning("test message")  # 例外を送出しないこと
    assert "Discord通知に失敗しました" in capsys.readouterr().out


def test_notify_discord_warning_swallow_exception(monkeypatch, capsys):
    def _raise(content, channel):
        raise RuntimeError("network down")

    monkeypatch.setattr(wrc, "_discord_send", _raise)
    wrc.notify_discord_warning("test message")  # 例外を送出しないこと
    assert "例外が発生しました" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# rebuild_pg_atomic: Fake DB接続
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    """get_connection() の代替。実接続を一切開かずSQL呼び出しを記録する。"""

    def __init__(self, count_row: int = 0):
        self.calls: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list]] = []
        self._count_row = count_row
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, tuple(params)))
        return _FakeCursor((self._count_row,))

    def executemany(self, sql, rows_list):
        self.executemany_calls.append((sql, list(rows_list)))

    def delete_calls(self):
        return [c for c in self.calls if c[0].strip().upper().startswith("DELETE")]

    def select_calls(self):
        return [c for c in self.calls if c[0].strip().upper().startswith("SELECT")]


def _row(race_date="2026-08-01", race_key="k1", hit=1, bet=100, payout=500):
    return {
        "race_date": race_date,
        "race_key": race_key,
        "rank": "RANK_TEST",
        "pred_combo": "1-2-3",
        "n_combos": 5,
        "hit": hit,
        "payout": payout,
        "trio_payout": payout,
        "bet_amount": bet,
        "gate_label": "S",
    }


_COND = "rank='RANK_TEST' AND race_date BETWEEN ? AND ?"


def test_rebuild_pg_atomic_zero_total_rows_never_touches_db(monkeypatch):
    """挿入対象が1件も無い場合、get_connection()自体を呼ばずDBに一切触れない。"""

    def _boom():
        raise AssertionError("get_connection should not be called when total rows is 0")

    monkeypatch.setattr(wrc, "get_connection", _boom)
    notified = []
    monkeypatch.setattr(wrc, "notify_discord_warning", lambda msg: notified.append(msg))

    per_window_rows = [("2026-08-01", "2026-08-31", [])]
    wrc.rebuild_pg_atomic("RANK_TEST", _COND, per_window_rows, dry_run=False)

    assert notified, "0件時はDiscordへ警告するはず"


def test_rebuild_pg_atomic_single_transaction_across_windows(monkeypatch):
    """複数窓のwipe+insertが単一の`with get_connection()`ブロック内で行われる
    （＝単一トランザクション）こと。窓ごとに個別接続していないことを確認する。"""
    conn = FakeConn(count_row=3)
    monkeypatch.setattr(wrc, "get_connection", lambda: conn)

    per_window_rows = [
        ("2026-08-01", "2026-08-01", [_row(race_date="2026-08-01", race_key="k1")]),
        ("2026-08-02", "2026-08-02", [_row(race_date="2026-08-02", race_key="k2")]),
    ]
    wrc.rebuild_pg_atomic("RANK_TEST", _COND, per_window_rows, dry_run=False)

    assert conn.entered == 1, "get_connection()は1回しか呼ばれない（単一トランザクション）"
    assert len(conn.delete_calls()) == 2
    assert len(conn.executemany_calls) == 2


def test_rebuild_pg_atomic_skips_window_with_zero_rows_but_processes_others(monkeypatch):
    """個別の窓が0行なら、その窓のwipeはスキップし、他窓は通常通り処理する。"""
    conn = FakeConn(count_row=5)
    monkeypatch.setattr(wrc, "get_connection", lambda: conn)

    per_window_rows = [
        ("2026-08-01", "2026-08-01", []),  # 0件 → wipeスキップ
        ("2026-08-02", "2026-08-02", [_row(race_date="2026-08-02", race_key="k2")]),
    ]
    wrc.rebuild_pg_atomic("RANK_TEST", _COND, per_window_rows, dry_run=False)

    assert conn.entered == 1
    assert len(conn.delete_calls()) == 1, "0件窓のDELETEは発行されないこと"
    assert len(conn.executemany_calls) == 1


def test_rebuild_pg_atomic_dry_run_no_writes(monkeypatch):
    """dry_run時はSELECT COUNTのみ発行し、DELETE/INSERTは一切発行しない。"""
    conn = FakeConn(count_row=7)
    monkeypatch.setattr(wrc, "get_connection", lambda: conn)

    per_window_rows = [("2026-08-01", "2026-08-01", [_row()])]
    wrc.rebuild_pg_atomic("RANK_TEST", _COND, per_window_rows, dry_run=True)

    assert conn.executemany_calls == []
    assert conn.delete_calls() == []
    assert len(conn.select_calls()) == 1


def test_rebuild_pg_atomic_exception_mid_loop_propagates_for_rollback(monkeypatch):
    """途中で例外が発生した場合、呼び出し元(get_connection()のcontextmanager)が
    rollbackできるよう例外がそのまま伝播すること（本関数内でexceptを握りつぶさない）。
    """

    class _RaisingConn(FakeConn):
        def executemany(self, sql, rows_list):
            raise RuntimeError("simulated insert failure")

    conn = _RaisingConn(count_row=1)
    monkeypatch.setattr(wrc, "get_connection", lambda: conn)

    per_window_rows = [("2026-08-01", "2026-08-01", [_row()])]
    with pytest.raises(RuntimeError, match="simulated insert failure"):
        wrc.rebuild_pg_atomic("RANK_TEST", _COND, per_window_rows, dry_run=False)


# ---------------------------------------------------------------------------
# 0件wipe見送りの通知抑制（2026-08-09）
#   挙動（wipeしない）は変えず、Discord通知だけを絞る。
#   9S のように候補がほぼ出ないランクが毎朝同じ警告を流すと、
#   警告そのものが読まれなくなるため。
# ---------------------------------------------------------------------------

def test_zero_row_state_is_isolated_from_the_repo_file():
    """状態ファイルの差し替えが効いていること（効いていないと自己汚染する）。"""
    assert wrc._ZERO_ROW_STATE != _REAL_ZERO_ROW_STATE


def test_zero_row_初回は通知し2回目は抑制する(tmp_path, monkeypatch):
    import src.wt_rebuild_common as m
    monkeypatch.setattr(m, "_ZERO_ROW_STATE", tmp_path / "s.json")
    win = [("2026-08-01", "2026-08-08", [])]
    assert m._zero_row_should_notify("RANK_9S", win) is True
    assert m._zero_row_should_notify("RANK_9S", win) is False


def test_zero_row_窓の終端が進んでも抑制が効く(tmp_path, monkeypatch):
    """tail の窓は終端が毎日進む。終端を鍵にすると毎日通知が出てしまう。"""
    import src.wt_rebuild_common as m
    monkeypatch.setattr(m, "_ZERO_ROW_STATE", tmp_path / "s.json")
    assert m._zero_row_should_notify("RANK_9S", [("2026-08-01", "2026-08-08", [])]) is True
    assert m._zero_row_should_notify("RANK_9S", [("2026-08-01", "2026-08-09", [])]) is False


def test_zero_row_月が変われば再通知する(tmp_path, monkeypatch):
    import src.wt_rebuild_common as m
    monkeypatch.setattr(m, "_ZERO_ROW_STATE", tmp_path / "s.json")
    assert m._zero_row_should_notify("RANK_9S", [("2026-08-01", "2026-08-08", [])]) is True
    assert m._zero_row_should_notify("RANK_9S", [("2026-09-01", "2026-09-08", [])]) is True


def test_zero_row_ランクが違えば独立して通知する(tmp_path, monkeypatch):
    import src.wt_rebuild_common as m
    monkeypatch.setattr(m, "_ZERO_ROW_STATE", tmp_path / "s.json")
    win = [("2026-08-01", "2026-08-08", [])]
    assert m._zero_row_should_notify("RANK_9S", win) is True
    assert m._zero_row_should_notify("RANK_9A", win) is True


def test_zero_row_状態ファイルが壊れていても通知する側に倒す(tmp_path, monkeypatch):
    """fail-open。黙らせる方に倒すと本当の異常まで気づけなくなる。"""
    import src.wt_rebuild_common as m
    bad = tmp_path / "s.json"
    bad.write_text("{ これはJSONではない", encoding="utf-8")
    monkeypatch.setattr(m, "_ZERO_ROW_STATE", bad)
    assert m._zero_row_should_notify("RANK_9S", [("2026-08-01", "2026-08-08", [])]) is True
