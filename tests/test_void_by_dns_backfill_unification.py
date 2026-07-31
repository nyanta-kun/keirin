"""backfill_*_rank_wt.py の欠車判定を void_by_dns へ統一した変更（2026-07-31
是正・PMタスク C-2b）の回帰テスト。

対象: scripts/backfill_7s_rank_wt.py / backfill_9s_rank_wt.py /
      backfill_7a_rank_wt.py / backfill_9a_rank_wt.py / backfill_um_rank_wt.py

検証する性質:
  1. board（欠車判定用の盤面掲載車集合）が本番 notify_results_wt._board_frames
     と同一の構築方法（bet_type='trio' の combination の車番和集合。
     odds_value によるフィルタなし）で構築されること。
  2. 盤面が完全一致するレースは従来と同じ買い目・点数になること（回帰）。
  3. 相手候補の1台だけが盤面から欠けたレースで、レースが除外されず
     点数が1つ減った買い目になること（本タスクの核心）。
  4. 軸が盤面から欠けたレースは従来通り除外されること。
  5. 有効な目が0（相手も全員欠車）になるケースは除外されること。
  6. DNF（finish_order=0 だが board には残る）が返還されず外れ計上されること。

DB アクセスは全て monkeypatch で差し替え、実DBへは一切アクセスしない。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# 共通テストダブル: DB / モデル
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

    - board 用（欠車判定・odds_value フィルタなし）と trio 用（購入可否判定・
      odds_value フィルタあり）を明確に分離して管理する。add_trio() で
      odds_value を渡すと自動的に両方へ登録し、odds_value=None または
      無効値を渡すと board のみに登録される（「盤面には掲載されたが
      有効オッズが無い」ケースを模擬）。
    """

    def __init__(self):
        self.races: dict[str, tuple[int, str]] = {}
        self.entries: dict[str, list[tuple[int, int, int | None]]] = {}
        self.board_rows: dict[str, list[str]] = {}
        self.trio_rows: dict[str, list[tuple[str, float]]] = {}
        self.payout_rows: dict[str, list[tuple[str, str, float]]] = {}

    def add_race(self, race_key: str, n_entries: int, race_date: str) -> None:
        self.races[race_key] = (n_entries, race_date)

    def add_entry(self, race_key: str, frame_no: int, finish_order: int,
                  prediction_mark: int | None) -> None:
        self.entries.setdefault(race_key, []).append((frame_no, finish_order, prediction_mark))

    def add_trio(self, race_key: str, combo_str: str, odds_value: float | None,
                 leg_min_ok: bool = True) -> None:
        """combo_str: 例 "1-2-3"。odds_value を渡すと board + payout に登録、
        さらに odds_value が有効値（>0）なら trio(_load_trio_boards 用) にも登録する。
        odds_value=None なら「盤面には車番として存在するが有効オッズが無い」状態。
        """
        self.board_rows.setdefault(race_key, []).append(combo_str)
        self.payout_rows.setdefault(race_key, []).append(("trio", combo_str, odds_value))
        if odds_value is not None and odds_value > 0:
            self.trio_rows.setdefault(race_key, []).append((combo_str, odds_value))

    # -- SQL dispatch -------------------------------------------------
    def dispatch(self, sql: str, params):
        if sql.startswith("SELECT race_key, n_entries FROM wt_races"):
            return [(rk, ne) for rk, (ne, _d) in self.races.items()]
        if sql.startswith("SELECT race_key, race_date FROM wt_races"):
            return [(rk, d) for rk, (_ne, d) in self.races.items()]
        if sql.startswith("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries"):
            out = []
            for rk in params:
                for (fno, fo, pmv) in self.entries.get(rk, []):
                    out.append((rk, fno, fo, pmv))
            return out
        if sql.startswith("SELECT race_key, frame_no, prediction_mark, finish_order FROM wt_entries"):
            out = []
            for rk in params:
                for (fno, fo, pmv) in self.entries.get(rk, []):
                    out.append((rk, fno, pmv, fo))
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


class StubModel:
    """model.predict_proba(X)[:, 1] を、あらかじめ df に仕込んだ列から返す。"""

    def __init__(self, col: str):
        self.col = col

    def predict_proba(self, X: pd.DataFrame):
        p = X[self.col].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def _identity_prepare_x(df: pd.DataFrame) -> pd.DataFrame:
    return df


# ---------------------------------------------------------------------------
# 7車立て・軸1/軸2 = frame1/frame2、entropy/axis_sum ゲート通過を保証する
# 標準フィールド（S7/S9/S7A/S9A 共通で使う）。
# ---------------------------------------------------------------------------

# top3_probs=win_probs（同一）: frame1=0.30, frame2=0.25, frame3=0.15,
# frame4=0.10, frame5=0.08, frame6=0.07, frame7=0.05
_TOP3 = {1: 0.30, 2: 0.25, 3: 0.15, 4: 0.10, 5: 0.08, 6: 0.07, 7: 0.05}


def _field_entropy(probs: dict[int, float]) -> float:
    total = sum(probs.values())
    ent = 0.0
    for v in probs.values():
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def _make_field_df(race_key: str, n_car: int = 7,
                    probs: dict[int, float] | None = None) -> pd.DataFrame:
    """axis1=1, axis2=2 で選定されるフィールドの DataFrame を返す。

    probs を渡さない場合は標準の _TOP3（7車専用・低entropy）を使う。
    """
    if probs is None:
        if n_car != 7:
            raise ValueError("n_car!=7 では probs を明示的に渡すこと（_TOP3は7車専用）")
        probs = dict(_TOP3)
    rows = []
    for fno, p in probs.items():
        rows.append({
            "race_key": race_key, "frame_no": fno,
            "_stub_top3": p, "_stub_win": p,
            "player_class": "A1",
        })
    return pd.DataFrame(rows)


# axis1=1, axis2=2 は上記フィールドで rank_7s_select_axis から一意に選定される
# （win_top3==place_top3=={1,2,3}, overlap>=2 → top3_probs降順上位2 = 1,2）。
AXIS1, AXIS2 = 1, 2
# axis_sum=0.55<=RANK_7S_AXIS_SUM_MAX(1.5)。entropy(7車)≈1.7607<=RANK_7S_ENTROPY_MAX(1.8329)。
assert 0.30 + 0.25 == pytest.approx(0.55)
assert _field_entropy(_TOP3) < 1.8329

# 9車版・低entropy分布（S9: entropy<=RANK_9S_ENTROPY_MAX(1.9938)を満たす。axis1=1,axis2=2）。
_TOP3_9CAR = {1: 0.30, 2: 0.25, 3: 0.15, 4: 0.10, 5: 0.08, 6: 0.05, 7: 0.04, 8: 0.02, 9: 0.01}
assert _field_entropy(_TOP3_9CAR) < 1.9938  # RANK_9S_ENTROPY_MAX

# 7A用・axis_sum(=1.99)がRANK_7S_AXIS_SUM_MAX(1.5)を超えるがentropyは低い分布
# （2ゲートのうちちょうど1つ（axis_sum）だけ不合格＝7A対象）。
_TOP3_7A_AXISFAIL = {1: 1.0, 2: 0.99, 3: 0.05, 4: 0.03, 5: 0.02, 6: 0.01, 7: 0.01}
assert _TOP3_7A_AXISFAIL[1] + _TOP3_7A_AXISFAIL[2] > 1.5  # axis_sum gate: FAIL
assert _field_entropy(_TOP3_7A_AXISFAIL) < 1.8329          # entropy gate: PASS


def _patch_common(monkeypatch, module, db: FakeDB, *, win_model_col="_stub_win",
                   top3_model_col="_stub_top3"):
    """load_model/prepare_X/build_features_wt/load_raw_data_wt/get_connection を
    まとめて monkeypatch する（S7/S9/S7A/S9A の4スクリプト共通）。
    """
    monkeypatch.setattr(module, "get_connection", lambda: FakeConn(db))
    monkeypatch.setattr(module, "prepare_X", _identity_prepare_x)
    monkeypatch.setattr(module, "load_raw_data_wt", lambda **kw: object())
    # backtest_wt._load_payouts_wt が使う get_connection も同じ FakeDB に向ける。
    import src.evaluation.backtest_wt as backtest_wt
    monkeypatch.setattr(backtest_wt, "get_connection", lambda: FakeConn(db))

    def _load_model(name):
        return StubModel(top3_model_col if "win" not in name else win_model_col)

    monkeypatch.setattr(module, "load_model", _load_model)


# ===========================================================================
# 1) _load_board_frames_wt: 本番 _board_frames と同一構築方法であることの確認
# ===========================================================================

_MODULES_WITH_BOARD_LOADER = [
    "backfill_7s_rank_wt",
    "backfill_9s_rank_wt",
    "backfill_7a_rank_wt",
    "backfill_9a_rank_wt",
    "backfill_um_rank_wt",
]


@pytest.mark.parametrize("modname", _MODULES_WITH_BOARD_LOADER)
def test_load_board_frames_wt_matches_board_frames_semantics(monkeypatch, modname):
    """odds_value を一切問わず、combination に現れる車番の和集合を返すこと。

    本番 notify_results_wt._board_frames は odds_value を SELECT しない
    （フィルタ不可能）。本テストは同一のクエリ形状（SELECT race_key, combination
    FROM wt_odds WHERE bet_type='trio' ...）で応答することを保証する。
    """
    _ensure_strategy_wt_um_stub(monkeypatch)
    module = __import__(modname)

    db = FakeDB()
    # 正常な3車combo
    db.add_trio("R1", "1-2-3", odds_value=None)  # 有効オッズなしでも board には載る
    db.add_trio("R1", "1-2-4", odds_value=5.5)
    db.add_trio("R1", "2-5-6", odds_value=0.0)   # 0以下の異常値でも board には載る
    # 不正な組み合わせ文字列（数値化できない要素）は無視される
    db.board_rows.setdefault("R1", []).append("x-y-z")

    monkeypatch.setattr(module, "get_connection", lambda: FakeConn(db))
    board_map = module._load_board_frames_wt(["R1"])
    assert board_map["R1"] == {1, 2, 3, 4, 5, 6}

    # race_keys=[] は空dictを返す（クエリを発行しない）
    assert module._load_board_frames_wt([]) == {}


def _ensure_strategy_wt_um_stub(monkeypatch):
    """backfill_um_rank_wt.py が import する src.strategy_wt の定数/関数
    （M_LEG_MIN_ODDS 等）が、S2/S3全廃に伴い strategy_wt.py から既に削除されて
    おり、モジュールとして import できない状態であることが判明した
    （2026-07-31 調査・本タスクとは無関係の既存バグ。詳細はテストの docstring
    下部コメント・最終報告を参照）。テストのためだけに欠落属性を注入する
    （strategy_wt.py 自体は変更しない・モジュールオブジェクトへの一時的な
    monkeypatch のみ）。
    """
    import src.strategy_wt as strategy_wt
    stub_attrs = {
        "U_ENTROPY_MIN": 4.5,
        "U_MTO_MIN": 4.5,
        "U_LEG_MIN_ODDS": 1.0,
        "U_STAKE": 100,
        "M_LEG_MIN_ODDS": 20.0,
        "M_STAKE": 100,
        "m_axis_gate": lambda gap12, win_rank, ratio: (True, "TEST"),
    }
    for name, val in stub_attrs.items():
        if not hasattr(strategy_wt, name):
            monkeypatch.setattr(strategy_wt, name, val, raising=False)


# ===========================================================================
# 2)〜6) build_rows の欠車統一シナリオ（S7 系: backfill_7s_rank_wt.py）
# ===========================================================================

class TestS7BuildRowsVoidUnification:
    """backfill_7s_rank_wt.build_rows() の欠車判定シナリオ。

    5レース(race_key)をそれぞれ別日に配置し、RANK_7S_DAILY_CAP(=12)によるトリムの
    影響を受けないようにする（本テストの主目的は日次トリムではなく個々の
    レースの欠車判定なので、トリムが発火しない設計にしている）。
    """

    RACE_A = "RA20240101"   # 盤面7車ちょうど（回帰: 従来と同じ5点になること）
    RACE_B = "RB20240102"   # 相手1台だけ盤面欠け（4点に減るがレースは除外されない）
    RACE_C = "RC20240103"   # 軸(frame1)が盤面欠け（レース無効）
    RACE_D = "RD20240104"   # 相手が全員盤面欠け（レース無効）
    RACE_E = "RE20240105"   # DNF(finish_order=0)は盤面に残る（返還されず外れ計上）

    def _base_db(self) -> FakeDB:
        db = FakeDB()
        for rk, date in (
            (self.RACE_A, "2024-01-01"), (self.RACE_B, "2024-01-02"),
            (self.RACE_C, "2024-01-03"), (self.RACE_D, "2024-01-04"),
            (self.RACE_E, "2024-01-05"),
        ):
            db.add_race(rk, 7, date)
            # WT公式印: honmei=frame3, taikou=frame4（axis{1,2}と重ならない→wt_overlap_n=0）
            db.add_entry(rk, 3, 0, 1)
            db.add_entry(rk, 4, 0, 2)
            db.add_entry(rk, 5, 0, 3)
        return db

    def test_full_board_match_is_regression_stable(self, monkeypatch):
        """盤面7車ちょうど: 従来と同じ5点の買い目になること。"""
        db = self._base_db()
        rk = self.RACE_A
        # 出走7車の finish_order（1,2,3着 + DNS無し）
        db.add_entry(rk, 1, 1, None)
        db.add_entry(rk, 2, 2, None)
        db.add_entry(rk, 6, 3, None)
        db.add_entry(rk, 7, 4, None)
        # 盤面(board)・購入可能コンボ(trio)ともに全5通り
        for x in (3, 4, 5, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        assert r["n_combos"] == 5
        assert r["bet_amount"] == 500
        assert r["hit"] == 1  # 実際の3着=frame6, combo{1,2,6}が的中
        assert r["payout"] == 1000  # trio_pay(1000) * STAKE(100) // 100
        # pred_combo は実際に購入した5目のみを列挙
        assert "3,4,5,6,7" in r["pred_combo"]

    def test_one_third_missing_from_board_reduces_points_without_voiding_race(self, monkeypatch):
        """相手1台(frame5)だけ盤面から欠けている: レースは除外されず4点になる。

        旧実装は `if len(others) != 5: continue` で本レース全体を候補プールから
        除外していた（本タスクで是正した核心バグ）。
        """
        db = self._base_db()
        rk = self.RACE_B
        db.add_entry(rk, 1, 1, None)
        db.add_entry(rk, 2, 2, None)
        db.add_entry(rk, 6, 3, None)
        db.add_entry(rk, 7, 4, None)
        # frame5 を含む trio コンボは一切登録しない → board に5が現れない
        for x in (3, 4, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        assert r["n_combos"] == 4
        assert r["bet_amount"] == 400
        assert r["hit"] == 1
        assert "5" not in r["pred_combo"].split("-")[-1].split(" ")[0].split(",")
        for x in ("3", "4", "6", "7"):
            assert x in r["pred_combo"]

    def test_axis_missing_from_board_voids_whole_race(self, monkeypatch):
        """軸(frame1)が盤面に無い: void_by_dns の「軸欠車=レース無効」と同一に、
        本レースからは何も生成されない（従来通り）。"""
        db = self._base_db()
        rk = self.RACE_C
        db.add_entry(rk, 2, 1, None)
        db.add_entry(rk, 6, 2, None)
        db.add_entry(rk, 7, 3, None)
        # frame1（軸1）を一切含まない盤面にする（軸1が欠車の状況を再現）
        for x in (3, 4, 5, 7):
            db.add_trio(rk, f"2-6-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert [r for r in rows if rk in r["race_key"]] == []

    def test_all_thirds_missing_from_board_voids_race(self, monkeypatch):
        """軸2車は盤面に有るが、相手候補5車が全員盤面に無い: 買える目が無く無効。"""
        db = self._base_db()
        rk = self.RACE_D
        db.add_entry(rk, 1, 1, None)
        db.add_entry(rk, 2, 2, None)
        db.add_entry(rk, 6, 3, None)
        db.add_entry(rk, 7, 4, None)
        # 軸2車のみが載る（相手を一切含まない）架空のcombo。3車必要なのでダミーの
        # 3人目として軸自身を重複させず、「1」「2」しか登場しない状況を再現する
        # ため combination の車番集合が {1,2} のみになるよう調整する。
        db.board_rows[rk] = ["1-2-1"]  # パース結果は {1,2}（重複は無視される）
        db.trio_rows[rk] = []          # 有効オッズの購入可能コンボは無し
        db.payout_rows[rk] = []

        rows = self._run(db, monkeypatch)
        assert [r for r in rows if rk in r["race_key"]] == []

    def test_dnf_remains_on_board_and_counts_as_loss_not_void(self, monkeypatch):
        """frame7がDNF(finish_order=0)でも盤面には残るため、欠車扱いされず
        購入対象に含まれる（的中しなければ普通に外れ計上・返還されない）。
        """
        db = self._base_db()
        rk = self.RACE_E
        db.add_entry(rk, 1, 1, None)
        db.add_entry(rk, 2, 2, None)
        db.add_entry(rk, 3, 3, None)
        db.add_entry(rk, 7, 0, None)  # DNF: finish_orderは0だが盤面には残る
        # 盤面・購入可能コンボは通常通り全5通り（frame7を含むコンボもある）
        for x in (3, 4, 5, 6, 7):
            db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

        rows = self._run(db, monkeypatch)
        assert len(rows) == 1
        r = rows[0]
        # frame7 は欠車ではないので5点のまま（DNFによる自動除外は発生しない）
        assert r["n_combos"] == 5
        assert r["bet_amount"] == 500
        for x in ("3", "4", "5", "6", "7"):
            assert x in r["pred_combo"]
        # 的中は frame3 が3着のため combo{1,2,3}。frame7を含むcomboは外れ計上のみ。
        assert r["hit"] == 1

    # -- 実行ヘルパー ----------------------------------------------------
    def _run(self, db: FakeDB, monkeypatch) -> list[dict]:
        import backfill_7s_rank_wt as mod

        _patch_common(monkeypatch, mod, db)

        def _fake_build_features_wt(_raw):
            frames = []
            for rk in db.races:
                frames.append(_make_field_df(rk))
            return pd.concat(frames, ignore_index=True)

        monkeypatch.setattr(mod, "build_features_wt", _fake_build_features_wt)
        return mod.build_rows("lgbm_wt_eval", "2024-01-01", "2024-01-31", "lgbm_wt_win")


# ===========================================================================
# S9 / S7A / S9A: 同型構造のため「相手1台欠け→点数減で継続」の核心ケースのみ
# 個別に確認する（board loaderの共通性は上のパラメトライズテストで確認済み）。
# 各ランク固有のゲート条件（S9=entropy単独・7A/9A=2ゲート中ちょうど1個不合格）
# を満たす専用のフィールド分布を使うため、個別テスト関数に分けている
# （単一の分布を使い回すと各ランク固有のゲートを満たせない）。
# ===========================================================================

def _run_gate_variant(monkeypatch, module, build_fn_name: str, db: FakeDB,
                       rk: str, probs: dict[int, float]) -> list[dict]:
    monkeypatch.setattr(module, "get_connection", lambda: FakeConn(db))
    monkeypatch.setattr(module, "prepare_X", _identity_prepare_x)
    monkeypatch.setattr(module, "load_raw_data_wt", lambda **kw: object())
    import src.evaluation.backtest_wt as backtest_wt
    monkeypatch.setattr(backtest_wt, "get_connection", lambda: FakeConn(db))
    monkeypatch.setattr(module, "load_model",
                         lambda name: StubModel("_stub_win" if "win" in name else "_stub_top3"))
    monkeypatch.setattr(module, "build_features_wt",
                         lambda _raw: _make_field_df(rk, n_car=len(probs), probs=probs))
    build_rows = getattr(module, build_fn_name)
    return build_rows("lgbm_wt_eval", "2024-01-01", "2024-01-31", "lgbm_wt_win")


def test_rank_9s_partial_third_exclusion_does_not_void_race(monkeypatch):
    """S9(9車): 相手候補の1台(frame9)が盤面から欠けても除外されず6点になる。"""
    module = __import__("backfill_9s_rank_wt")
    rk = "R_S9_PARTIAL"
    db = FakeDB()
    db.add_race(rk, 9, "2024-01-01")
    db.add_entry(rk, 3, 0, 1)   # honmei（axis外・overlap_n=0を保証）
    db.add_entry(rk, 4, 0, 2)   # taikou（axis外）
    db.add_entry(rk, 8, 0, 3)   # ana（axis外・mark3=0を保証）
    db.add_entry(rk, 1, 1, None)
    db.add_entry(rk, 2, 2, None)
    db.add_entry(rk, 6, 3, None)
    present = [3, 4, 5, 6, 7, 8]  # frame9 のみ盤面から欠落させる
    for x in present:
        db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

    rows = _run_gate_variant(monkeypatch, module, "build_rows", db, rk, _TOP3_9CAR)
    matching = [r for r in rows if r["race_key"].endswith("#9S")]
    assert len(matching) == 1, f"S9: 相手1台欠けのレースが除外されている（{rows}）"
    r = matching[0]
    assert r["n_combos"] == 6
    assert r["bet_amount"] == 600
    assert "9" not in r["pred_combo"]


def test_rank_7a_partial_third_exclusion_does_not_void_race(monkeypatch):
    """7A(7車): 相手候補の1台(frame7)が盤面から欠けても除外されず4点になる。

    7Aは axis_sum/entropy の2ゲートのうちちょうど1個不合格の候補が対象
    （_TOP3_7A_AXISFAIL は axis_sum のみ不合格・entropyは合格）。
    """
    module = __import__("backfill_7a_rank_wt")
    rk = "R_S7A_PARTIAL"
    db = FakeDB()
    db.add_race(rk, 7, "2024-01-01")
    db.add_entry(rk, 3, 0, 1)   # honmei（axis外・overlap_n=0を保証）
    db.add_entry(rk, 4, 0, 2)   # taikou（axis外）
    db.add_entry(rk, 1, 1, None)
    db.add_entry(rk, 2, 2, None)
    db.add_entry(rk, 6, 3, None)
    present = [3, 4, 5, 6]  # frame7 のみ盤面から欠落させる
    for x in present:
        db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

    rows = _run_gate_variant(monkeypatch, module, "build_rows", db, rk, _TOP3_7A_AXISFAIL)
    matching = [r for r in rows if r["race_key"].endswith("#7A")]
    assert len(matching) == 1, f"7A: 相手1台欠けのレースが除外されている（{rows}）"
    r = matching[0]
    assert r["n_combos"] == 4
    assert r["bet_amount"] == 400
    assert "7" not in r["pred_combo"]


def test_rank_9a_partial_third_exclusion_does_not_void_race(monkeypatch):
    """9A(9車): 相手候補の1台(frame9)が盤面から欠けても除外されず5点になる。

    9Aは entropy/mark3 の2ゲートのうちちょうど1個不合格の候補が対象。
    ここでは entropy 合格・mark3(=2)不合格の組み合わせを使う
    （honmei=frame1(=axis1と一致)・ana=frame2(=axis2と一致)・
    taikou=frame5(axis外) → wt_overlap_n=len({1,2}&{1,5})=1(合格)、
    wt_mark3_overlap_n=len({1,2}&{1,5,2})=2(不合格)）。
    """
    module = __import__("backfill_9a_rank_wt")
    rk = "R_S9A_PARTIAL"
    db = FakeDB()
    db.add_race(rk, 9, "2024-01-01")
    db.add_entry(rk, 1, 0, 1)   # honmei = axis1
    db.add_entry(rk, 5, 0, 2)   # taikou（axis外）
    db.add_entry(rk, 2, 0, 3)   # ana = axis2
    db.add_entry(rk, 1, 1, None)
    db.add_entry(rk, 2, 2, None)
    db.add_entry(rk, 6, 3, None)
    present = [3, 4, 5, 6, 7, 8]  # frame9 のみ盤面から欠落させる
    for x in present:
        db.add_trio(rk, f"1-2-{x}", odds_value=10.0)

    rows = _run_gate_variant(monkeypatch, module, "build_rows", db, rk, _TOP3_9CAR)
    matching = [r for r in rows if r["race_key"].endswith("#9A")]
    assert len(matching) == 1, f"9A: 相手1台欠けのレースが除外されている（{rows}）"
    r = matching[0]
    assert r["n_combos"] == 6
    assert r["bet_amount"] == 600
    assert "9" not in r["pred_combo"]


# ===========================================================================
# UM (S2/S3): board loader + 明示的 void_by_dns 呼び出しの確認
#
# 注意（重要な既存バグの発見）: scripts/backfill_um_rank_wt.py は
# `from src.strategy_wt import (M_LEG_MIN_ODDS, M_STAKE, U_ENTROPY_MIN,
# U_LEG_MIN_ODDS, U_MTO_MIN, U_STAKE, m_axis_gate)` としているが、これらの
# 名前は現在の src/strategy_wt.py（コミット済みHEAD時点も含む）に一切存在
# しない。S2/S3全廃コミット(5d8b258)でこれらの定義が削除された後、本スクリプト
# の import 文が追随しておらず、素の `import backfill_um_rank_wt` は
# ImportError になる（本タスクの変更に起因せず、着手前から存在した状態。
# strategy_wt.py の編集は本タスクで禁止されているため修正しない）。
# 以下のテストは、この既存バグの影響を受けずに「本タスクで実装した
# board構築＋void_by_dns統合」を検証するため、テストプロセス内でのみ
# src.strategy_wt モジュールへ欠落属性を一時注入する
# （_ensure_strategy_wt_um_stub。ファイルは変更しない）。
# ===========================================================================

def _make_um_field_df(race_key: str) -> pd.DataFrame:
    """backfill_um_rank_wt 用の7車フィールド（line_group/line_size/line_pos/
    style 列を含む）。

    frame1 は pred_prob 最上位だが line_size!=1 かつ line_pos が(1,2)以外
    なので U の「穴(dark)」候補から除外される（judge_u相当ロジックの
    ls==1 or lp∈(1,2) 条件を満たさない）。
    frame2（pred_prob2位・lg=5・lp=1）が dark 候補。frame3（同じlg=5・
    style=逃）が「同ラインの逃」として mate に選ばれる。frame3自身は
    lp=3 のため dark 候補からは除外され、frame2 とのペアが一意に定まる。
    """
    rows = []
    specs = [
        # fno, pred_prob(top3), pred_win, lg, ls, lp, style
        (1, 0.30, 0.30, 9, 2, 3, "先行"),   # dark候補から除外（ls!=1・lp not in(1,2)）
        (2, 0.25, 0.10, 5, 2, 1, "先行"),   # dark候補: lg=5, lp=1
        (3, 0.20, 0.05, 5, 2, 3, "逃"),     # frame2と同じlg=5・style=逃 → mate
        (4, 0.10, 0.20, 3, 1, 1, "先行"),
        (5, 0.08, 0.15, 4, 1, 1, "先行"),
        (6, 0.04, 0.12, 6, 1, 1, "先行"),
        (7, 0.03, 0.08, 7, 1, 1, "先行"),
    ]
    for fno, p3, pw, lg, ls, lp, style in specs:
        rows.append({
            "race_key": race_key, "frame_no": fno,
            "_stub_top3": p3, "_stub_win": pw,
            "line_group": lg, "line_size": ls, "line_pos": lp, "style": style,
        })
    return pd.DataFrame(rows)


class TestUmBuildRowsVoidUnification:
    """backfill_um_rank_wt.build_rows() の欠車判定シナリオ（S2/U 経路）。"""

    def _patch(self, monkeypatch, db: FakeDB):
        _ensure_strategy_wt_um_stub(monkeypatch)
        import backfill_um_rank_wt as mod
        monkeypatch.setattr(mod, "get_connection", lambda: FakeConn(db))
        monkeypatch.setattr(mod, "prepare_X", _identity_prepare_x)
        monkeypatch.setattr(mod, "load_raw_data_wt", lambda **kw: object())
        import src.evaluation.backtest_wt as backtest_wt
        monkeypatch.setattr(backtest_wt, "get_connection", lambda: FakeConn(db))

        def _load_model(name):
            if "win" in name:
                raise FileNotFoundError(name)  # win_model無し→gap12単独ゲートにフォールバック
            return StubModel("_stub_top3")

        monkeypatch.setattr(mod, "load_model", _load_model)
        return mod

    def _base_db(self, rk: str) -> FakeDB:
        db = FakeDB()
        db.add_race(rk, 7, "2024-01-01")
        # WT◎=frame4（m1候補のWT不一致判定用。frame1がpred_prob最上位=m1想定なので
        # frame1と不一致にするためWT◎はframe4にする）。
        # finish_order は実際の着順（1着=frame1, 2着=frame4, 3着=frame5）を与える
        # （fins に3件以上の finish_order>=1 が無いと build_rows が早期 continue する）。
        db.add_entry(rk, 1, 1, None)
        db.add_entry(rk, 4, 2, 1)
        db.add_entry(rk, 5, 3, None)
        for fno in (2, 3, 6, 7):
            db.add_entry(rk, fno, 0, None)
        return db

    def test_board_loader_matches_board_frames(self, monkeypatch):
        db = FakeDB()
        db.add_trio("RX", "1-2-3", odds_value=None)
        db.add_trio("RX", "1-2-4", odds_value=999999)  # 異常値でも board には載る
        import backfill_um_rank_wt as mod
        monkeypatch.setattr(mod, "get_connection", lambda: FakeConn(db))
        assert mod._load_board_frames_wt(["RX"]) == {"RX": {1, 2, 3, 4}}

    def test_one_third_missing_from_board_reduces_points_without_voiding_race(self, monkeypatch):
        """穴(dark)=frame2, 相方(mate)=frame3（同ラインlg=5・style=逃）が
        軸ペア。相手候補{1,4,5,6,7}のうちframe7を盤面から欠落させる。

        dark(2)/mate(3)を含むコンボにはあえて高いオッズ(=弱い市場評価)を
        与え、2/3を含まないコンボには低いオッズ(=強い市場評価)を与える
        ことで、市場評価順位ゲート(4<=mkt_rank(dark)<=7)を満たすよう
        frame2の市場順位を意図的に下位へ追いやっている。
        """
        rk = "UM_B"
        db = self._base_db(rk)
        mod = self._patch(monkeypatch, db)
        # entropy/mto ゲートを事実上無効化するため、テスト側の閾値を大きく緩和。
        # `from src.strategy_wt import U_ENTROPY_MIN` により backfill_um_rank_wt
        # モジュール自身の名前空間に束縛済みのため、strategy_wt 側ではなく
        # モジュール自身の属性を monkeypatch する必要がある。
        monkeypatch.setattr(mod, "U_ENTROPY_MIN", -1.0, raising=False)
        monkeypatch.setattr(mod, "U_MTO_MIN", -1.0, raising=False)

        for x in (1, 4, 5, 6):  # frame7 を除外（盤面から欠落）
            db.add_trio(rk, f"2-3-{x}", odds_value=50.0)
        for combo in ("1-4-5", "1-4-6", "1-5-6", "4-5-6"):
            db.add_trio(rk, combo, odds_value=2.0)

        monkeypatch.setattr(mod, "build_features_wt", lambda _raw: _make_um_field_df(rk))
        rows = mod.build_rows("lgbm_wt_eval", "2024-01-01", "2024-01-31", win_model_name="lgbm_wt_win")

        u_rows = [r for r in rows if r["rank"] == "7PLUS_U"]
        assert len(u_rows) == 1, f"U判定行が想定通り1件生成されていない: {rows}"
        r = u_rows[0]
        assert r["n_combos"] == 4
        assert r["bet_amount"] == 400
        thirds_str = r["pred_combo"].split("-", 2)[-1]
        assert "7" not in thirds_str.split(",")
        for x in ("1", "4", "5", "6"):
            assert x in thirds_str.split(",")

    def test_axis_missing_from_board_yields_no_u_row(self, monkeypatch):
        """軸(dark=frame2)が盤面から完全に欠落していれば、U行は生成されない。"""
        rk = "UM_C"
        db = self._base_db(rk)
        mod = self._patch(monkeypatch, db)
        monkeypatch.setattr(mod, "U_ENTROPY_MIN", -1.0, raising=False)
        monkeypatch.setattr(mod, "U_MTO_MIN", -1.0, raising=False)

        # frame2(dark)・frame3(mate) を一切含まないコンボのみを登録する
        for combo in ("1-4-5", "1-4-6", "1-4-7", "1-5-6", "1-5-7", "1-6-7", "4-5-6"):
            db.add_trio(rk, combo, odds_value=10.0)

        monkeypatch.setattr(mod, "build_features_wt", lambda _raw: _make_um_field_df(rk))
        rows = mod.build_rows("lgbm_wt_eval", "2024-01-01", "2024-01-31", win_model_name="lgbm_wt_win")
        assert [r for r in rows if r["rank"] == "7PLUS_U"] == []
