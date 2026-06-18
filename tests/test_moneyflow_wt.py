"""exp_moneyflow_wt.py のユニットテスト（G04）。

テスト方針:
  - DB/モデルに依存するデータ収集部分 (collect) は対象外（統合テストは手動実行）。
  - 純粋関数（_gami_band, _hit_vs_nonhit, estimate_min_n, cell_a, cell_b, cell_c）
    および記述統計集計の補助関数をテストする。
  - doc18 セマンティクス（欠車void・全エントリーランキング・≤6車出走表基準）が
    セル関数に正しく反映されているかを確認する。
"""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest

from exp_moneyflow_wt import (
    _gami_band,
    _hit_vs_nonhit,
    _market_fav_from_board,
    estimate_min_n,
    cell_a,
    cell_b,
    cell_c,
)


# ── _gami_band ────────────────────────────────────────────────────────────────

class TestGamiBand:
    def test_lt3(self):
        assert _gami_band(1.0) == "<3"
        assert _gami_band(2.9) == "<3"

    def test_3to5(self):
        assert _gami_band(3.0) == "3-5"
        assert _gami_band(4.9) == "3-5"

    def test_ge5(self):
        assert _gami_band(5.0) == ">=5"
        assert _gami_band(100.0) == ">=5"

    def test_boundary_3(self):
        """3.0 は '<3' でなく '3-5' に入る。"""
        assert _gami_band(3.0) == "3-5"

    def test_boundary_5(self):
        """5.0 は '3-5' でなく '>=5' に入る。"""
        assert _gami_band(5.0) == ">=5"


# ── _market_fav_from_board ────────────────────────────────────────────────────

class TestMarketFav:
    def test_simple_board(self):
        """最も高頻度で低オッズに現れる frame が市場本命として返る。"""
        # frame 1 が全組み合わせに含まれ、低オッズ → 1/odds が大きい
        board = {
            frozenset({1, 2, 3}): 2.0,
            frozenset({1, 2, 4}): 3.0,
            frozenset({1, 3, 4}): 4.0,
            frozenset({2, 3, 4}): 50.0,
        }
        assert _market_fav_from_board(board) == 1

    def test_none_on_insufficient_combos(self):
        """4 組み合わせ未満ではNoneを返す。"""
        board = {frozenset({1, 2, 3}): 5.0}
        assert _market_fav_from_board(board) is None

    def test_none_on_empty(self):
        assert _market_fav_from_board({}) is None

    def test_skips_high_odds(self):
        """9000倍以上の組み合わせはスキップされる。"""
        board = {
            frozenset({1, 2, 3}): 9000.0,
            frozenset({1, 2, 4}): 9001.0,
            frozenset({1, 3, 4}): 9100.0,
            frozenset({2, 3, 4}): 9200.0,
        }
        # 全てスキップ → None
        assert _market_fav_from_board(board) is None


# ── _hit_vs_nonhit ────────────────────────────────────────────────────────────

class TestHitVsNonhit:
    def test_normal_case(self):
        """的中目と非的中目の短縮率差が正しく計算される。"""
        hit_rows = [
            {"shortened": True, "drift": 0.8},
            {"shortened": True, "drift": 0.9},
        ]
        nonhit_rows = [
            {"shortened": False, "drift": 1.2},
            {"shortened": False, "drift": 1.1},
        ]
        result = _hit_vs_nonhit(hit_rows, nonhit_rows)
        assert result["n_hit"] == 2
        assert result["n_nonhit"] == 2
        assert result["hit_pct_shortened"] == pytest.approx(1.0)
        assert result["nonhit_pct_shortened"] == pytest.approx(0.0)
        assert result["diff_pct_shortened"] == pytest.approx(1.0)
        assert result["hit_drift_median"] == pytest.approx(0.85)

    def test_empty_hit(self):
        """的中目が空の場合はNoneを返す。"""
        result = _hit_vs_nonhit([], [{"shortened": False, "drift": 1.1}])
        assert result["hit_pct_shortened"] is None
        assert "標本不足" in result["note"]

    def test_empty_nonhit(self):
        """非的中目が空の場合はNoneを返す。"""
        result = _hit_vs_nonhit([{"shortened": True, "drift": 0.9}], [])
        assert result["hit_pct_shortened"] is None


# ── estimate_min_n ────────────────────────────────────────────────────────────

class TestEstimateMinN:
    def test_returns_positive(self):
        n = estimate_min_n()
        assert n > 0

    def test_larger_effect_smaller_n(self):
        """効果量が大きいほど必要標本数は小さい。"""
        n_small = estimate_min_n(effect_roi=0.10)
        n_large = estimate_min_n(effect_roi=0.40)
        assert n_small > n_large

    def test_floor_100(self):
        """最小値は100を下回らない。"""
        n = estimate_min_n(effect_roi=10.0)  # 非常に大きな効果量
        assert n >= 100

    def test_default_reasonable_range(self):
        """デフォルト設定で現実的な範囲（100〜10000）に収まる。"""
        n = estimate_min_n()
        assert 100 <= n <= 10000, f"n={n} が想定外の範囲"


# ── cell_a ────────────────────────────────────────────────────────────────────

def _make_race(
    tier="A",
    t3s=0.9,
    fav_mismatch=True,
    fav_swapped=False,
    axis_void=False,
    trio3_final=None,
    trio3_morning=None,
    min3_final=None,
    n=6,
):
    """テスト用レース構造体を生成するファクトリ関数。"""
    if trio3_final is None:
        trio3_final = []
    if trio3_morning is None:
        trio3_morning = []
    if min3_final is None and trio3_final:
        min3_final = min(o for o, _ in trio3_final)
    return {
        "race_key": "TEST_RACE",
        "date": "2026-06-10",
        "n": n,
        "p1": 1, "p2": 2,
        "top3": frozenset({1, 2, 3}),
        "dns": set(),
        "tier": tier,
        "t3s": t3s,
        "mf_morning": 5,
        "mf_final": 1 if not fav_mismatch else 5,
        "fav_mismatch": fav_mismatch,
        "fav_swapped": fav_swapped,
        "axis_void": axis_void,
        "trio3_final": trio3_final,
        "trio3_morning": trio3_morning,
        "min3_final": min3_final,
    }


class TestCellA:
    def test_shortened_vs_elongated(self):
        """短縮目のROIが伸長目より高い（スマートマネー方向）とき roi_diff > 0。"""
        # 短縮目（朝10.0→確定8.0）的中1回、伸長目（朝5.0→確定7.0）不的中
        race = _make_race(
            tier="A",
            trio3_final=[(8.0, True), (7.0, False)],
            trio3_morning=[(10.0, True), (5.0, False)],
            min3_final=7.0,
        )
        result = cell_a([race])
        # 短縮: 1目(8.0倍的中) → pays=[800], bets=[100]
        assert result["shortened"]["n_races"] == 1
        assert result["elongated"]["n_races"] == 1
        # 短縮ROI = 800/100 = 8.0; 伸長ROI = 0/100 = 0.0
        assert result["shortened"]["roi"]["roi"] == pytest.approx(8.0)
        assert result["elongated"]["roi"]["roi"] == pytest.approx(0.0)
        assert result["roi_diff"] == pytest.approx(8.0)

    def test_axis_void_skipped(self):
        """axis_void=True のレースはスキップされる（欠車void・doc18）。"""
        race = _make_race(
            axis_void=True,
            trio3_final=[(8.0, True), (7.0, False)],
            trio3_morning=[(10.0, True), (5.0, False)],
        )
        result = cell_a([race])
        assert result["shortened"]["n_races"] == 0
        assert result["elongated"]["n_races"] == 0

    def test_empty_races(self):
        """レースなしのとき ROI=0, n_races=0。"""
        result = cell_a([])
        assert result["shortened"]["n_races"] == 0
        assert result["elongated"]["n_races"] == 0

    def test_no_morning_data_skipped(self):
        """trio3_final が空のレースはスキップ。"""
        race = _make_race(trio3_final=[], trio3_morning=[])
        result = cell_a([race])
        assert result["shortened"]["n_races"] == 0

    def test_multiple_races_aggregated(self):
        """複数レースの集計が正しく行われる。"""
        # 両方とも短縮目のみ
        r1 = _make_race(
            tier="A",
            trio3_final=[(6.0, True)],
            trio3_morning=[(8.0, True)],
            min3_final=6.0,
        )
        r2 = _make_race(
            tier="A",
            trio3_final=[(10.0, False)],
            trio3_morning=[(12.0, False)],
            min3_final=10.0,
        )
        result = cell_a([r1, r2])
        # r1: 短縮目1本 的中 → pays=600, bets=100
        # r2: 短縮目1本 不的中 → pays=0, bets=100
        # 合計: pays=[600, 0], bets=[100, 100]
        assert result["shortened"]["n_races"] == 2
        assert result["elongated"]["n_races"] == 0
        assert result["shortened"]["roi"]["roi"] == pytest.approx(3.0)  # 600/200


# ── cell_b ────────────────────────────────────────────────────────────────────

class TestCellB:
    def test_c0_condition_requires_tier_and_min5(self):
        """tier=None のレースや min3_final<5 のレースは C0 に含まれない。"""
        race_no_tier = _make_race(
            tier=None,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(5.0, True), (6.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result_no_tier = cell_b([race_no_tier])
        assert result_no_tier["c0_all"]["n_races"] == 0

        race_low_odds = _make_race(
            tier="A",
            trio3_final=[(2.0, False), (3.0, False), (4.0, False)],
            trio3_morning=[(1.5, False), (2.5, False), (3.5, False)],
            min3_final=2.0,
        )
        result_low = cell_b([race_low_odds])
        assert result_low["c0_all"]["n_races"] == 0

    def test_gate_majority_shortened(self):
        """全推奨目の過半数が短縮されていればゲート通過。"""
        # 3点中2点が短縮（過半数）→ ゲート通過
        race = _make_race(
            tier="A",
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],  # 1,2本目が短縮
            min3_final=6.0,
        )
        result = cell_b([race])
        assert result["c0_all"]["n_races"] == 1
        assert result["c0_gate"]["n_races"] == 1  # 2/3点短縮 → ゲート通過
        assert result["gate_pct"] == pytest.approx(1.0)

    def test_gate_not_met(self):
        """過半数が伸長→ゲート不通過。"""
        # 1点のみ短縮 / 2点が伸長
        race = _make_race(
            tier="A",
            trio3_final=[(6.0, False), (7.0, False), (8.0, True)],
            trio3_morning=[(5.0, False), (6.0, False), (9.0, True)],  # 3本目のみ短縮
            min3_final=6.0,
        )
        result = cell_b([race])
        assert result["c0_all"]["n_races"] == 1
        # 1/3点のみ短縮（過半数未満）→ ゲート不通過
        assert result["c0_gate"]["n_races"] == 0
        assert result["gate_pct"] == pytest.approx(0.0)

    def test_axis_void_skipped(self):
        race = _make_race(
            tier="A", axis_void=True,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_b([race])
        assert result["c0_all"]["n_races"] == 0


# ── cell_c ────────────────────────────────────────────────────────────────────

class TestCellC:
    def test_fav_mismatch_only(self):
        """fav_mismatch=False のレースは含まれない（doc18：市場本命≠モデル1位のみ）。"""
        race_match = _make_race(
            tier="A", fav_mismatch=False,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(5.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_c([race_match])
        assert result["fav_swap"]["n_races"] == 0
        assert result["fav_noswap"]["n_races"] == 0

    def test_swap_vs_noswap_split(self):
        """fav_swapped の有無で2グループに分割される。"""
        race_swap = _make_race(
            tier="A", fav_mismatch=True, fav_swapped=True,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        race_noswap = _make_race(
            tier="A", fav_mismatch=True, fav_swapped=False,
            trio3_final=[(6.0, False), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, False), (5.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_c([race_swap, race_noswap])
        assert result["fav_swap"]["n_races"] == 1
        assert result["fav_noswap"]["n_races"] == 1

    def test_roi_diff_is_none_when_one_group_empty(self):
        """片方のグループが空なら roi_diff は None。"""
        race = _make_race(
            tier="A", fav_mismatch=True, fav_swapped=True,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_c([race])
        assert result["roi_diff"] is None  # noswap グループが空

    def test_axis_void_skipped(self):
        race = _make_race(
            tier="A", fav_mismatch=True, fav_swapped=True, axis_void=True,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_c([race])
        assert result["fav_swap"]["n_races"] == 0

    def test_min3_lt5_skipped(self):
        """min3_final < 5 のレースは C0 条件不成立→スキップ。"""
        race = _make_race(
            tier="A", fav_mismatch=True, fav_swapped=True,
            trio3_final=[(4.0, True), (3.0, False)],
            trio3_morning=[(5.0, True), (4.0, False)],
            min3_final=3.0,
        )
        result = cell_c([race])
        assert result["fav_swap"]["n_races"] == 0

    def test_none_tier_skipped(self):
        """tier=None は C0 条件不成立→スキップ。"""
        race = _make_race(
            tier=None, fav_mismatch=True, fav_swapped=True,
            trio3_final=[(6.0, True), (7.0, False), (8.0, False)],
            trio3_morning=[(8.0, True), (9.0, False), (7.0, False)],
            min3_final=6.0,
        )
        result = cell_c([race])
        assert result["fav_swap"]["n_races"] == 0


# ── doc18 セマンティクス確認 ──────────────────────────────────────────────────

class TestDoc18Semantics:
    """doc18 セマンティクスが cell 関数に正しく反映されていることを確認する統合的テスト。"""

    def test_dns_axis_void_in_cell_a(self):
        """欠車で軸が void になったレースは全セルでスキップされる。"""
        race = _make_race(
            axis_void=True,
            trio3_final=[(8.0, True), (6.0, False)],
            trio3_morning=[(10.0, True), (5.0, False)],
        )
        assert cell_a([race])["shortened"]["n_races"] == 0
        assert cell_b([race])["c0_all"]["n_races"] == 0
        assert cell_c([race])["fav_swap"]["n_races"] == 0

    def test_no_snapshot_no_result(self):
        """trio3_final が空（スナップショットなし）のレースはスキップ。"""
        race = _make_race(trio3_final=[], trio3_morning=[])
        assert cell_a([race])["shortened"]["n_races"] == 0
        assert cell_b([race])["c0_all"]["n_races"] == 0
        assert cell_c([race])["fav_swap"]["n_races"] == 0
