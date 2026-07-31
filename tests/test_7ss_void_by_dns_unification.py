"""backfill_7ss_rank_wt.py の欠車判定を void_by_dns へ統一した変更（2026-07-31
是正・PMタスク C-2b-7ss）の回帰テスト。

対象: scripts/backfill_7ss_rank_wt.py

7SS は S7/S9/S7A/S9A とは異なりモデル予測（LightGBM）に依存せず、
wt_entries の公表値（race_point/prediction_mark/first_rate/third_rate）のみで
判定する（load_model/build_features_wt/prepare_X を一切呼ばない）ため、
tests/test_void_by_dns_backfill_unification.py の FakeDB とは異なるクエリ形状
（wt_races の1クエリで n_entries=7・cancel=0 を直接絞り込み、wt_entries から
race_point 等の生値を取得）専用の FakeDB を用意する。

検証する性質:
  1. `_load_board_frames_wt()` が本番 notify_results_wt._board_frames と
     同一の構築方法（bet_type='trio' の combination の車番和集合。
     odds_value によるフィルタなし）で構築されること
     （odds_value が無効値/NULL/不正文字列でも board に含めること）。
  2. 盤面が完全一致するレースは従来と同じ買い目・点数(5点)になること（回帰）。
  3. 相手候補の1台だけが盤面から欠けたレースで、レースが除外されず
     4点の買い目になること（本タスクの核心）。
  4. 軸（軸1 or 軸2）が盤面から欠けたレースは除外されること。
  5. 有効な目が0（相手も全員欠車）になるケースは除外されること。
  6. DNF（finish_order=0 だが board には残る）が返還されず外れ計上されること。

DB アクセス・モデルロードは monkeypatch で差し替え、実DBへは一切アクセスしない
（7SS はそもそもモデルをロードしないため patch 対象は get_connection のみ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backfill_7ss_rank_wt as mod  # noqa: E402


# ---------------------------------------------------------------------------
# FakeDB: backfill_7ss_rank_wt.py 専用のクエリ形状に合わせた最小限のインメモリDB
# ---------------------------------------------------------------------------

class _Rows(list):
    """sqlite3.Cursor 相当（イテレート可能 かつ .fetchall() を持つ）。"""

    def fetchall(self):
        return list(self)


class FakeConn:
    """get_connection() の代替。with 文コンテキストマネージャとして振る舞う。"""

    def __init__(self, db: "FakeDB"):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        return _Rows(self.db.dispatch(sql, params))

    def executemany(self, *a, **k):
        raise AssertionError("build_rows は executemany を呼ばない想定（insert_rowsのみ使用）")

    def commit(self):
        pass


class FakeDB:
    """wt_races / wt_entries / wt_odds を模擬する最小限のインメモリDB。

    board（欠車判定・odds_value フィルタなし）と trio（購入可否判定・odds_value
    フィルタあり）を明確に分離して管理する。add_trio() で odds_value を渡すと
    自動的に両方（+payout）へ登録し、odds_value=None または無効値を渡すと
    board のみに登録される（「盤面には掲載されたが有効オッズが無い」ケースを
    模擬）。
    """

    def __init__(self):
        self.races: dict[str, str] = {}
        self.entries: dict[str, list[dict]] = {}
        self.board_rows: dict[str, list[str]] = {}
        self.trio_rows: dict[str, list[tuple[str, float]]] = {}
        self.payout_rows: dict[str, list[tuple[str, str, float | None]]] = {}

    def add_race(self, race_key: str, race_date: str) -> None:
        self.races[race_key] = race_date

    def add_entry(self, race_key: str, frame_no: int, *, race_point: float,
                  first_rate: float, third_rate: float, finish_order: int | None,
                  prediction_mark: int | None) -> None:
        self.entries.setdefault(race_key, []).append({
            "frame_no": frame_no, "race_point": race_point, "line_group": None,
            "line_size": None, "n_lines": None, "first_rate": first_rate,
            "third_rate": third_rate, "finish_order": finish_order,
            "prediction_mark": prediction_mark,
        })

    def add_trio(self, race_key: str, combo_str: str, odds_value: float | None) -> None:
        """combo_str: 例 "1-2-3"。odds_value を渡すと board + payout に登録、
        さらに odds_value が有効値（>0）なら trio(_load_trio_boards 用) にも
        登録する。odds_value=None なら「盤面には車番として存在するが有効オッズ
        が無い」状態。
        """
        self.board_rows.setdefault(race_key, []).append(combo_str)
        self.payout_rows.setdefault(race_key, []).append(("trio", combo_str, odds_value))
        if odds_value is not None and odds_value > 0:
            self.trio_rows.setdefault(race_key, []).append((combo_str, odds_value))

    # -- SQL dispatch -------------------------------------------------
    def dispatch(self, sql: str, params):
        if sql.startswith("SELECT race_key, race_date FROM wt_races"):
            return [{"race_key": rk, "race_date": d} for rk, d in self.races.items()]
        if sql.startswith("SELECT race_key, frame_no, race_point"):
            out = []
            for rk in params:
                for e in self.entries.get(rk, []):
                    out.append({"race_key": rk, **e})
            return out
        if sql.startswith("SELECT race_key, combination, odds_value FROM wt_odds"):
            out = []
            for rk in params:
                for (comb, odds) in self.trio_rows.get(rk, []):
                    out.append((rk, comb, odds))
            return out
        if sql.startswith("SELECT race_key, combination FROM wt_odds"):
            out = []
            for rk in params:
                for comb in self.board_rows.get(rk, []):
                    out.append((rk, comb))
            return out
        if sql.startswith("SELECT race_key, bet_type, combination, odds_value FROM wt_odds"):
            out = []
            for rk in params:
                for (bt, comb, odds) in self.payout_rows.get(rk, []):
                    out.append((rk, bt, comb, odds))
            return out
        raise AssertionError(f"FakeDB.dispatch: 未対応のSQL: {sql!r}")


def _patch_common(monkeypatch, db: FakeDB) -> None:
    monkeypatch.setattr(mod, "get_connection", lambda: FakeConn(db))
    import src.evaluation.backtest_wt as backtest_wt
    monkeypatch.setattr(backtest_wt, "get_connection", lambda: FakeConn(db))


# ---------------------------------------------------------------------------
# 7車立て・rank_7ss_score が RANK_7SS_SCORE_THRESHOLD(=4.796886) を確実に上回る
# フィールド（実際の rank_7ss_field_features/rank_7ss_score で確認済み・score≈13.2）。
# axis1=frame1(race_point最大)・axis2=frame2(◯・third_rate=12.0が◯△✕中で最大)。
# ---------------------------------------------------------------------------

_RP = [89.0, 88.9, 88.8, 88.7, 88.6, 88.5, 88.4]
_FR = [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4]
_TR = [10.0, 12.0, 11.0, 10.5, 9.0, 8.0, 7.0]
_MARKS = {2: 2, 3: 3, 4: 4}  # frame2=◯, frame3=△, frame4=✕（frame2がthird_rate最大→axis2）
AXIS1, AXIS2 = 1, 2


def _score_precondition_holds() -> float:
    ents = [
        {"frame_no": i + 1, "race_point": _RP[i], "first_rate": _FR[i],
         "third_rate": _TR[i], "prediction_mark": _MARKS.get(i + 1), "line_group": None,
         "n_lines": None}
        for i in range(7)
    ]
    feat = mod.rank_7ss_field_features(ents)
    return mod.rank_7ss_score(feat)


_SCORE = _score_precondition_holds()
assert _SCORE >= mod.RANK_7SS_SCORE_THRESHOLD, f"テスト用フィールドが閾値未満: {_SCORE}"
assert mod.rank_7ss_select_axis([
    {"frame_no": i + 1, "race_point": _RP[i], "third_rate": _TR[i],
     "prediction_mark": _MARKS.get(i + 1)} for i in range(7)
]) == (AXIS1, AXIS2)


# ===========================================================================
# 1) _load_board_frames_wt: 本番 _board_frames と同一構築方法であることの確認
# ===========================================================================

def test_load_board_frames_wt_matches_board_frames_semantics(monkeypatch):
    """odds_value を一切問わず、combination に現れる車番の和集合を返すこと。

    odds_value が無効値(0以下)/NULL(None)/不正文字列でも board には含める。
    """
    db = FakeDB()
    db.add_trio("R1", "1-2-3", odds_value=None)          # NULL でも board には載る
    db.add_trio("R1", "1-2-4", odds_value=5.5)
    db.add_trio("R1", "2-5-6", odds_value=0.0)            # 0以下の異常値でも載る
    db.board_rows.setdefault("R1", []).append("x-y-z")    # 不正な組合せ文字列は無視

    monkeypatch.setattr(mod, "get_connection", lambda: FakeConn(db))
    board_map = mod._load_board_frames_wt(["R1"])
    assert board_map["R1"] == {1, 2, 3, 4, 5, 6}

    # race_keys=[] は空dictを返す（クエリを発行しない）
    assert mod._load_board_frames_wt([]) == {}


# ===========================================================================
# 2)〜6) build_rows の欠車統一シナリオ
# ===========================================================================

class TestRank7SSBuildRowsVoidUnification:
    RACE_A = "RA20240101"   # 盤面7車ちょうど（回帰: 従来と同じ5点になること）
    RACE_B = "RB20240102"   # 相手1台だけ盤面欠け（4点に減るがレースは除外されない）
    RACE_C = "RC20240103"   # 軸1(frame1)が盤面欠け（レース無効）
    RACE_D = "RD20240104"   # 相手が全員盤面欠け（レース無効）
    RACE_E = "RE20240105"   # DNF(finish_order=0)は盤面に残る（返還されず外れ計上）

    def _add_field(self, db: FakeDB, rk: str, finish_orders: dict[int, int | None]) -> None:
        for i in range(7):
            fno = i + 1
            db.add_entry(rk, fno, race_point=_RP[i], first_rate=_FR[i],
                         third_rate=_TR[i], finish_order=finish_orders.get(fno),
                         prediction_mark=_MARKS.get(fno))

    def test_full_board_match_is_regression_stable(self, monkeypatch):
        """盤面7車ちょうど: 従来と同じ5点の買い目になること。"""
        db = FakeDB()
        rk = self.RACE_A
        db.add_race(rk, "2024-01-01")
        # 実際の着順: 1着=frame1, 2着=frame2, 3着=frame6
        self._add_field(db, rk, {1: 1, 2: 2, 6: 3, 7: 4})
        for x in (3, 4, 5, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        assert r["n_combos"] == 5
        assert r["bet_amount"] == 500
        assert r["hit"] == 1  # 実際の3着=frame6 → combo{1,2,6}が的中
        assert r["payout"] == 1000  # trio_pay(1000=10.0*100) * STAKE(100) // 100
        assert "3,4,5,6,7" in r["pred_combo"]

    def test_one_third_missing_from_board_reduces_points_without_voiding_race(self, monkeypatch):
        """相手1台(frame5)だけ盤面から欠けている: レースは除外されず4点になる。

        旧実装は `if len(others) != 5: continue` で本レース全体を候補プールから
        除外していた（本タスクで是正した核心バグ）。
        """
        db = FakeDB()
        rk = self.RACE_B
        db.add_race(rk, "2024-01-02")
        self._add_field(db, rk, {1: 1, 2: 2, 6: 3, 7: 4})
        # frame5 を含む trio コンボは一切登録しない → board に5が現れない
        for x in (3, 4, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        assert r["n_combos"] == 4
        assert r["bet_amount"] == 400
        assert r["hit"] == 1
        thirds_str = r["pred_combo"].split("-", 2)[-1].split(" ")[0]
        assert "5" not in thirds_str.split(",")
        for x in ("3", "4", "6", "7"):
            assert x in thirds_str.split(",")

    def test_axis_missing_from_board_voids_whole_race(self, monkeypatch):
        """軸1(frame1)が盤面に無い: void_by_dns の「軸欠車=レース無効」と同一に、
        本レースからは何も生成されない（従来通り）。"""
        db = FakeDB()
        rk = self.RACE_C
        db.add_race(rk, "2024-01-03")
        self._add_field(db, rk, {2: 1, 6: 2, 7: 3})
        # frame1（軸1）を一切含まない盤面にする（軸1が欠車の状況を再現）
        for x in (3, 4, 5, 7):
            db.add_trio(rk, f"2-6-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert [r for r in rows if rk in r["race_key"]] == []

    def test_all_thirds_missing_from_board_voids_race(self, monkeypatch):
        """軸2車は盤面に有るが、相手候補5車が全員盤面に無い: 買える目が無く無効。"""
        db = FakeDB()
        rk = self.RACE_D
        db.add_race(rk, "2024-01-04")
        self._add_field(db, rk, {1: 1, 2: 2, 6: 3, 7: 4})
        # 軸2車(1,2)のみが盤面に登場する架空のcomboにして、相手5車を一切含めない
        db.board_rows[rk] = ["1-2-1"]  # パース結果は {1,2}（重複は無視される）
        db.trio_rows[rk] = []
        db.payout_rows[rk] = []

        rows = self._run(db, monkeypatch)
        assert [r for r in rows if rk in r["race_key"]] == []

    def test_dnf_remains_on_board_and_counts_as_loss_not_void(self, monkeypatch):
        """frame7がDNF(finish_order=0)でも盤面には残るため、欠車扱いされず
        購入対象に含まれる（的中しなければ普通に外れ計上・返還されない）。
        """
        db = FakeDB()
        rk = self.RACE_E
        db.add_race(rk, "2024-01-05")
        self._add_field(db, rk, {1: 1, 2: 2, 3: 3, 7: 0})  # frame7はDNF
        for x in (3, 4, 5, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        # frame7 は欠車ではないので5点のまま（DNFによる自動除外は発生しない）
        assert r["n_combos"] == 5
        assert r["bet_amount"] == 500
        thirds_str = r["pred_combo"].split("-", 2)[-1].split(" ")[0]
        for x in ("3", "4", "5", "6", "7"):
            assert x in thirds_str.split(",")
        # 的中は frame3 が3着のため combo{1,2,3}。frame7を含むcomboは外れ計上のみ。
        assert r["hit"] == 1

    # -- 実行ヘルパー ----------------------------------------------------
    def _run(self, db: FakeDB, monkeypatch) -> list[dict]:
        _patch_common(monkeypatch, db)
        return mod.build_rows("2024-01-01", "2024-01-31")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
