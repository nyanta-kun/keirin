"""backfill_*_rank_wt.py の欠車判定を void_by_dns へ統一した変更（2026-07-31
是正・PMタスク C-2b）の回帰テスト。

対象: scripts/backfill_7s_rank_wt.py / backfill_9s_rank_wt.py /
      backfill_7a_rank_wt.py / backfill_9a_rank_wt.py

（backfill_um_rank_wt.py は 2026-08-01 に削除済み。S2(7PLUS_U)/S3(7PLUS_M) 全廃
コミット 5d8b258 で `M_LEG_MIN_ODDS` 等の定数が src/strategy_wt.py から消えて
以降 import 不能のまま放置されており、対象ランク自体が既に無いため復旧せず
スクリプトごと削除した。本ファイルの UM 用テスト・stub もあわせて撤去済み。）

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
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.strategy_wt as sw

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
# axis_sum=0.55<=RANK_7S_AXIS_SUM_MAX(1.40・2026-08-05に1.5から引き下げ)。entropy(7車)≈1.7607<=RANK_7S_ENTROPY_MAX(1.8329)。
assert 0.30 + 0.25 == pytest.approx(0.55)
assert _field_entropy(_TOP3) < 1.8329

# 9車版・低entropy分布（S9: entropy<=RANK_9S_ENTROPY_MAX(1.9938)を満たす。axis1=1,axis2=2）。
_TOP3_9CAR = {1: 0.30, 2: 0.25, 3: 0.15, 4: 0.10, 5: 0.08, 6: 0.05, 7: 0.04, 8: 0.02, 9: 0.01}
assert _field_entropy(_TOP3_9CAR) < 1.9938  # RANK_9S_ENTROPY_MAX

# 7A用・axis_sum(=1.99)がRANK_7S_AXIS_SUM_MAX(1.40)を超えるがentropyは低い分布
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
]


@pytest.mark.parametrize("modname", _MODULES_WITH_BOARD_LOADER)
def test_load_board_frames_wt_matches_board_frames_semantics(monkeypatch, modname):
    """odds_value を一切問わず、combination に現れる車番の和集合を返すこと。

    本番 notify_results_wt._board_frames は odds_value を SELECT しない
    （フィルタ不可能）。本テストは同一のクエリ形状（SELECT race_key, combination
    FROM wt_odds WHERE bet_type='trio' ...）で応答することを保証する。
    """
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
        # 2026-08-07: 賭け金は1レース RACE_BUDGET(10,000円)。同日中に均等割りから
        # **入稿と同じ傾斜配分**へ変えたので、1点あたりの額は目によって変わる。
        # ここで見たいのは「部分的な除外でレースが無効にならないこと」なので、
        # 配分方式に依存しない形（総額と、的中目に乗った額×オッズ）で確認する。
        assert r["bet_amount"] == 10000
        assert r["hit"] == 1  # 実際の3着=frame6, combo{1,2,6}が的中
        assert 0 < r["payout"] == r["payout"] // 10 * 10
        # 払戻 = 的中目の賭け金 × オッズ。賭け金は 100円単位で総額の一部。
        assert r["payout"] % (1000 // 100) == 0
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
        # 欠車で1点減っても投資は予算枠に揃う（2,500円×4点）
        assert r["bet_amount"] == 4 * sw.unit_stake(4) == 10000
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
        assert r["bet_amount"] == 5 * sw.unit_stake(5) == 10000
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
    # 🔴 本テストの関心事は**欠車時の無効判定**であって選抜ではない。
    #    7A の低配当見送りゲート(2026-08-09)は合成データを閾値で落としてしまうので
    #    明示的に無効化する（ゲート自体は test_dutch_and_7a_gate.py で検査済み）。
    kwargs = {}
    if "apply_top2_gate" in inspect.signature(build_rows).parameters:
        kwargs["apply_top2_gate"] = False
    return build_rows("lgbm_wt_eval", "2024-01-01", "2024-01-31", "lgbm_wt_win", **kwargs)


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
    # 2026-08-07: 傾斜配分は端数（100円未満に割り切れない分）まで配り切るので
    # 投資額はちょうど予算枠になる。均等割りは切り捨てで 6点なら 1,600×6=9,600 円と
    # 400円余らせていた（予算枠1万円という建て付けと食い違っていた）。
    assert r["bet_amount"] == 10000
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
    assert r["bet_amount"] == 4 * sw.unit_stake(4) == 10000
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
    # 2026-08-07: 傾斜配分は端数（100円未満に割り切れない分）まで配り切るので
    # 投資額はちょうど予算枠になる。均等割りは切り捨てで 6点なら 1,600×6=9,600 円と
    # 400円余らせていた（予算枠1万円という建て付けと食い違っていた）。
    assert r["bet_amount"] == 10000
    assert "9" not in r["pred_combo"]
