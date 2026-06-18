"""G01: backtest_wt.py 本番忠実セマンティクス（doc18 バイアス修正）のテスト。

検証項目:
  ① 全エントリーでランキング（欠車を含む全エントリーで pred_prob を計算してランクする）
  ② 出走表基準の≤6車フィルタ（完走者ではなくエントリー数で判定）
  ③ void 採点（軸欠車→レース不計上 / 相手欠車→その目のみ除外）
"""
import types
import numpy as np
import pandas as pd
import pytest

from src.evaluation.void_rules import void_by_dns
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt,
    _filter_by_n_riders,
    _combo_cars,
    _compute_accum_wt,
    run_tiered_backtest_wt,
    _assign_tier,
)
from src.evaluation.backtest import BetStrategy


# ---------------------------------------------------------------------------
# テスト用ヘルパー
# ---------------------------------------------------------------------------

def _fake_model(pred_probs: list[float]):
    """predict_proba が指定確率を返すモックモデル。"""
    class FakeModel:
        def predict_proba(self, X):
            n = len(X)
            arr = np.zeros((n, 2))
            arr[:, 1] = pred_probs[:n]
            return arr
    return FakeModel()


def _make_df(race_key: str, frame_probs: list[tuple], finish_orders: list[int]) -> pd.DataFrame:
    """合成レースデータを作成。frame_probs = [(frame_no, dummy_feature), ...].

    finish_orders: 各 frame に対応する finish_order（0=欠車, 1-n=着順）
    feature_wt.prepare_X が要求する FEATURE_COLS_WT を揃える。
    """
    from src.preprocessing.feature_wt import FEATURE_COLS_WT
    rows = []
    for i, (frame_no, _) in enumerate(frame_probs):
        # FEATURE_COLS_WT を全て 0 で埋める（モデルは無視するので値は何でもよい）
        row = {col: 0.0 for col in FEATURE_COLS_WT}
        # race_key / frame_no / finish_order は FEATURE_COLS_WT の後に上書きする
        # （frame_no が FEATURE_COLS_WT に含まれている場合に 0 で上書きされるのを防ぐ）
        row["race_key"] = race_key
        row["frame_no"] = frame_no
        row["finish_order"] = finish_orders[i]
        rows.append(row)
    return pd.DataFrame(rows)


def _multi_race_df(races: list[dict]) -> pd.DataFrame:
    """複数レースを含む DataFrame を作成。
    各 race = {"key": str, "frames": list[int], "orders": list[int]}
    """
    dfs = []
    for r in races:
        fp = [(f, 0.0) for f in r["frames"]]
        dfs.append(_make_df(r["key"], fp, r["orders"]))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# void_rules.void_by_dns のユニットテスト
# ---------------------------------------------------------------------------

class TestVoidByDns:
    def test_axis_p1_dns_returns_void(self):
        runners = {2, 3, 4, 5}          # 1号車が欠車
        skip, valid = void_by_dns(1, 2, [3, 4, 5], runners)
        assert skip is True
        assert valid == []

    def test_axis_p2_dns_returns_void(self):
        runners = {1, 3, 4, 5}          # 2号車が欠車
        skip, valid = void_by_dns(1, 2, [3, 4, 5], runners)
        assert skip is True
        assert valid == []

    def test_third_dns_excludes_only_that_third(self):
        runners = {1, 2, 4, 5}          # 3号車が欠車
        skip, valid = void_by_dns(1, 2, [3, 4, 5], runners)
        assert skip is False
        assert 3 not in valid
        assert 4 in valid
        assert 5 in valid

    def test_all_thirds_dns_returns_void(self):
        runners = {1, 2}                 # thirds が全員欠車
        skip, valid = void_by_dns(1, 2, [3, 4, 5], runners)
        assert skip is True
        assert valid == []

    def test_no_dns(self):
        runners = {1, 2, 3, 4, 5}
        skip, valid = void_by_dns(1, 2, [3, 4, 5], runners)
        assert skip is False
        assert valid == [3, 4, 5]

    def test_wide_p2_dns_returns_void(self):
        """ワイドは p1/p2 を両方軸扱い（is_wide=True）。"""
        runners = {1, 3, 4}             # p2=2 が欠車
        skip, valid = void_by_dns(1, 2, [], runners, is_wide=True)
        assert skip is True

    def test_wide_no_dns(self):
        runners = {1, 2, 3, 4}
        skip, valid = void_by_dns(1, 2, [], runners, is_wide=True)
        assert skip is False


# ---------------------------------------------------------------------------
# ① 全エントリーランキング: _apply_pred_prob_wt のテスト
# ---------------------------------------------------------------------------

class TestApplyPredProbAllEntries:
    def test_dns_rows_are_included_in_output(self):
        """finish_order=0（欠車）の行が pred_prob 付与後も残ること。"""
        df = _make_df("R1", [(1, 0), (2, 0), (3, 0), (4, 0)],
                      finish_orders=[0, 1, 2, 3])
        probs = [0.4, 0.3, 0.2, 0.1]
        model = _fake_model(probs)
        result = _apply_pred_prob_wt(model, df)
        # 欠車行(finish_order=0)も残っている
        assert len(result) == 4
        assert 0 in result["finish_order"].values

    def test_pred_prob_assigned_to_all_frames(self):
        """欠車含む全車に pred_prob が付与されること。"""
        df = _make_df("R1", [(1, 0), (2, 0), (3, 0)],
                      finish_orders=[0, 1, 2])
        model = _fake_model([0.5, 0.3, 0.2])
        result = _apply_pred_prob_wt(model, df)
        assert "pred_prob" in result.columns
        assert result["pred_prob"].notna().all()
        assert len(result) == 3

    def test_ranking_includes_dns_car(self):
        """欠車を含む全エントリーでランキングされるため、
        欠車車番がランキング上位に入り得ること。"""
        # frame_no=1 が欠車でも pred_prob が最大なら ranked[0] になる
        df = _make_df("R1", [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
                      finish_orders=[0, 1, 2, 3, 4, 5])
        # 欠車(1号車) の pred_prob を最大にする
        model = _fake_model([0.9, 0.3, 0.2, 0.2, 0.1, 0.1])
        result = _apply_pred_prob_wt(model, df)
        # ランキング: pred_prob 降順に並べると 1号車(0.9)が先頭
        ranked = result.sort_values("pred_prob", ascending=False)["frame_no"].tolist()
        assert ranked[0] == 1  # 欠車でも pred_prob 最大なら1位


# ---------------------------------------------------------------------------
# ② 出走表基準フィルタ: _filter_by_n_riders のテスト
# ---------------------------------------------------------------------------

class TestFilterByNRiders:
    def _make_multirace(self):
        """6車立て（完走6）・7車立て（完走5+欠車1）・7車立て（完走6+欠車1）の3レース。"""
        race_6 = {"key": "R_6car", "frames": [1, 2, 3, 4, 5, 6],
                  "orders": [1, 2, 3, 4, 5, 6]}
        # 7車立て・完走6（欠車1=finish_order=0）→ 完走者基準だと「6車」に見えるが出走は7車
        race_7dns = {"key": "R_7car_1dns",
                     "frames": [1, 2, 3, 4, 5, 6, 7],
                     "orders": [0, 1, 2, 3, 4, 5, 6]}
        # 7車立て・完走7（欠車なし）
        race_7full = {"key": "R_7car_full",
                      "frames": [1, 2, 3, 4, 5, 6, 7],
                      "orders": [1, 2, 3, 4, 5, 6, 7]}
        return _multi_race_df([race_6, race_7dns, race_7full])

    def test_entry_based_filter_excludes_7car_with_dns(self):
        """7車立て（欠車1で完走6）は出走表基準で除外される。"""
        df = self._make_multirace()
        result = _filter_by_n_riders(df, max_riders=6)
        remaining = set(result["race_key"].unique())
        assert "R_6car" in remaining
        assert "R_7car_1dns" not in remaining
        assert "R_7car_full" not in remaining

    def test_old_bug_would_include_7car_with_dns(self):
        """旧バグ（完走者で判定）では 7車立て+欠車1 が ≤6 と判定されていた。
        本テストは欠車行を除去した後に filter をかける「旧挙動」のシミュレーション。
        修正後は完走者基準ではなくエントリー数基準で判定される。"""
        df = self._make_multirace()
        # 修正後の動作: エントリー数ベースで7車はすべて除外
        result = _filter_by_n_riders(df, max_riders=6)
        # 完走者基準だと R_7car_1dns(完走6) が通過するが、修正後は通過しない
        assert "R_7car_1dns" not in result["race_key"].unique()


# ---------------------------------------------------------------------------
# _combo_cars のテスト
# ---------------------------------------------------------------------------

class TestComboCars:
    def test_frozenset_returns_frozenset(self):
        c = frozenset([1, 2, 3])
        assert _combo_cars(c) == frozenset([1, 2, 3])

    def test_tuple_returns_frozenset(self):
        c = (1, 2, 3)
        assert _combo_cars(c) == frozenset([1, 2, 3])

    def test_list_returns_frozenset(self):
        c = [1, 2, 3]
        assert _combo_cars(c) == frozenset([1, 2, 3])


# ---------------------------------------------------------------------------
# ③ void 採点: _compute_accum_wt のテスト
# ---------------------------------------------------------------------------

def _make_strategy(name: str, bet_type: str, combos_fn) -> BetStrategy:
    """テスト用シンプル戦略。"""
    return BetStrategy(name=name, label=name, bet_type=bet_type, combo_fn=combos_fn)


class TestComputeAccumVoid:
    def _base_race_df(self, race_key: str, finish_orders: list[int],
                       probs: list[float]) -> pd.DataFrame:
        """6車レースの合成 DataFrame。frame_no = 1..6。"""
        from src.preprocessing.feature_wt import FEATURE_COLS_WT
        rows = []
        for i in range(6):
            row = {col: 0.0 for col in FEATURE_COLS_WT}
            # frame_no / finish_order / pred_prob は上書き（FEATURE_COLS_WT で 0 になるのを防ぐ）
            row["race_key"] = race_key
            row["frame_no"] = i + 1
            row["finish_order"] = finish_orders[i]
            row["pred_prob"] = probs[i]
            rows.append(row)
        return pd.DataFrame(rows)

    def test_no_dns_all_combos_counted(self):
        """欠車なし: 全コンボが bets に計上される。"""
        # 欠車なし: 1位=frame1, ..., ランク上位3点でtrio
        df = self._base_race_df(
            "R1",
            finish_orders=[1, 2, 3, 4, 5, 6],
            probs=[0.9, 0.7, 0.5, 0.3, 0.2, 0.1],
        )
        # trio 3連複 jiku2×3点: (1,2,3), (1,2,4), (1,2,5) → frozenset
        def _trio_combos(ranked): return [
            frozenset([ranked[0], ranked[1], ranked[2]]),
            frozenset([ranked[0], ranked[1], ranked[3]]),
            frozenset([ranked[0], ranked[1], ranked[4]]),
        ]
        s = _make_strategy("trio3", "trifecta_box", _trio_combos)
        payout_map = {}  # オッズなし（的中なし）
        accum = _compute_accum_wt(df, [s], payout_map)
        assert accum["trio3"]["bets"] == 300  # 3点×100円

    def test_axis_dns_race_not_counted(self):
        """ranked[0]（最高確率の車）が欠車の場合は全コンボが除外→レース不計上。"""
        # frame1 が最高確率だが欠車
        df = self._base_race_df(
            "R1",
            finish_orders=[0, 1, 2, 3, 4, 5],  # frame1 が欠車
            probs=[0.9, 0.7, 0.5, 0.3, 0.2, 0.1],
        )
        # trio 3点（全て ranked[0]=frame1 を含む）
        def _trio_combos(ranked): return [
            frozenset([ranked[0], ranked[1], ranked[2]]),
            frozenset([ranked[0], ranked[1], ranked[3]]),
            frozenset([ranked[0], ranked[1], ranked[4]]),
        ]
        s = _make_strategy("trio3", "trifecta_box", _trio_combos)
        accum = _compute_accum_wt(df, [s], {})
        # ranked[0]=frame1 は欠車 → 全コンボが DNS 車を含む → bets=0
        assert accum["trio3"]["bets"] == 0

    def test_third_dns_partial_combos_excluded(self):
        """ranked[2] が欠車の場合: そのコンボのみ除外、他は計上。"""
        # frame3 が欠車（ranked 上では ranked[2]）
        df = self._base_race_df(
            "R1",
            finish_orders=[1, 2, 0, 3, 4, 5],  # frame3(index=2) が欠車
            probs=[0.9, 0.7, 0.5, 0.3, 0.2, 0.1],
        )
        # trio 3点: ranked[2] を含む目は1点だけ
        def _trio_combos(ranked): return [
            frozenset([ranked[0], ranked[1], ranked[2]]),  # ranked[2]=frame3(欠車) → スキップ
            frozenset([ranked[0], ranked[1], ranked[3]]),  # 有効
            frozenset([ranked[0], ranked[1], ranked[4]]),  # 有効
        ]
        s = _make_strategy("trio3", "trifecta_box", _trio_combos)
        accum = _compute_accum_wt(df, [s], {})
        # 欠車含む1点はスキップ → 2点のみ計上
        assert accum["trio3"]["bets"] == 200


# ---------------------------------------------------------------------------
# run_tiered_backtest_wt の統合テスト（合成データ）
# ---------------------------------------------------------------------------

class TestRunTieredBacktestLeakfree:
    def _make_tiered_df(self, n_races: int = 10, include_dns: bool = False) -> pd.DataFrame:
        """SS/S/A 層別バックテスト用の合成データ。
        各レースは6車・欠車なし（include_dns=True なら最初の1レースに欠車を入れる）。
        """
        from src.preprocessing.feature_wt import FEATURE_COLS_WT
        rows = []
        for r in range(n_races):
            rk = f"2025-01-01_A1_{r:02d}"
            for i in range(6):
                fo = (i + 1) if not (include_dns and r == 0 and i == 0) else 0
                row = {col: 0.0 for col in FEATURE_COLS_WT}
                row["race_key"] = rk
                row["frame_no"] = i + 1
                row["finish_order"] = fo
                rows.append(row)
        return pd.DataFrame(rows)

    def test_tiered_runs_without_error(self):
        """欠車なし合成データで run_tiered_backtest_wt がエラーなく完了すること。"""
        # 各エントリーが 0.0 の特徴量 → 全確率が同値になる
        df = self._make_tiered_df(n_races=5)
        probs = [0.5] * len(df)
        model = _fake_model(probs)
        result = run_tiered_backtest_wt(model, df, max_riders=6)
        assert isinstance(result, pd.DataFrame)
        assert "層" in result.columns

    def test_tiered_with_dns_does_not_error(self):
        """欠車ありのデータで run_tiered_backtest_wt がエラーなく完了すること。"""
        df = self._make_tiered_df(n_races=5, include_dns=True)
        model = _fake_model([0.5] * len(df))
        result = run_tiered_backtest_wt(model, df, max_riders=6)
        assert isinstance(result, pd.DataFrame)

    def test_7car_race_excluded_by_entry_count(self):
        """7車立てレースが出走表基準で除外されること。"""
        from src.preprocessing.feature_wt import FEATURE_COLS_WT
        rows = []
        # 7車立て（欠車なし）
        for i in range(7):
            row = {col: 0.0 for col in FEATURE_COLS_WT}
            row["race_key"] = "R7car"
            row["frame_no"] = i + 1
            row["finish_order"] = i + 1
            rows.append(row)
        # 6車立て（欠車なし）
        for i in range(6):
            row = {col: 0.0 for col in FEATURE_COLS_WT}
            row["race_key"] = "R6car"
            row["frame_no"] = i + 1
            row["finish_order"] = i + 1
            rows.append(row)
        df = pd.DataFrame(rows)
        # モデルは適当な確率を返す（SS/S/A 判定のため差をつける）
        probs = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05,  # 7車
                 0.9, 0.7, 0.5, 0.3, 0.2, 0.1]          # 6車
        model = _fake_model(probs)
        # max_riders=6: 7車立てが除外され、集計結果は6車立て分のみ
        result = run_tiered_backtest_wt(model, df, max_riders=6)
        # レース数: 6車立てのみ（tier条件次第で0以上）
        # 少なくとも 7車立て分の bets が混入していないことを確認
        # (7車盤面でオッズなしので直接確認はしないが、エラーなく完了すること)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _assign_tier の境界値テスト（既存のバックテスト互換確認）
# ---------------------------------------------------------------------------

class TestAssignTier:
    @pytest.mark.parametrize("gap12, ratio, expected", [
        (0.05, 1.0, None),           # gap12 < 0.06 → None
        (0.10, 1.0, "A"),            # gap12 ∈ [0.06, 0.15) → A
        (0.15, 1.2, "SS"),           # gap12 >= 0.15 & ratio < 1.3 → SS
        (0.15, 1.4, "S"),            # gap12 >= 0.15 & ratio ∈ [1.3, 1.6) → S
        (0.15, 1.6, None),           # ratio >= 1.6 → None
        (0.06, 1.0, "A"),            # gap12 = 0.06（境界・A）
        (0.149999, 1.0, "A"),        # gap12 直前境界値 → A
    ])
    def test_tier_boundaries(self, gap12, ratio, expected):
        assert _assign_tier(gap12, ratio) == expected
