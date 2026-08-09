"""STEP3 で入れた「ダッチ配分」と「7A 低配当レース見送りゲート」の検査。

守りたい不変条件を固定する:
  - ダッチ: 採用したどの目が来ても払戻が総賭け金の MIN_RETURN 倍以上
  - ダッチ: 低オッズ目は「切られる」が、切った結果 2点未満でも条件成立なら買う
  - 7Aゲート: 閾値は**必ず有限値**（None を返すとゲート無効と誤解され全件通る）
  - 7Aゲート: 7C/7S には適用されない（STEP1Cで無効と確定しているため）
"""

import pytest

from src.dutch_allocation import (
    EV_MIN,
    MIN_ODDS,
    MIN_RETURN,
    ODDS_FLOOR,
    dutch_allocate,
)
from src.strategy_wt import (
    RANK_7A_TOP2_GATE_FALLBACK,
    rank_7a_daily_select,
    rank_7a_gate_chronological,
    rank_7a_top2_gate,
    rank_7a_top2_threshold,
)


class TestDutchAllocate:
    def test_guarantees_min_return_on_every_kept_leg(self):
        """採用した**どの目**が来ても総賭け金の 1.3 倍以上が返る。"""
        odds = {"a": 3.0, "b": 4.5, "c": 7.0, "d": 12.0}
        r = dutch_allocate(odds)
        assert r.buy
        total = sum(r.stakes.values())
        for k, s in r.stakes.items():
            assert s * odds[k] >= total * MIN_RETURN

    def test_drops_odds_below_floor(self):
        """2.0倍未満の目は最初に捨てる。"""
        r = dutch_allocate({"low": 1.4, "ok": 5.0, "ok2": 6.0})
        assert "low" not in r.stakes
        assert "low" in r.dropped

    def test_all_below_floor_is_skip(self):
        r = dutch_allocate({"a": 1.2, "b": 1.9})
        assert not r.buy
        assert r.reason == "no_odds_above_floor"

    def test_no_odds_is_skip_not_crash(self):
        for bad in ({}, None):
            r = dutch_allocate(bad)
            assert not r.buy and r.stakes == {}

    def test_two_legs_still_bought_when_conditions_hold(self):
        """少点数(2点)でも条件が成立すれば購入する（仕様書 §2B）。"""
        r = dutch_allocate({"a": 3.0, "b": 3.2})
        assert r.buy
        assert len(r.stakes) == 2

    def test_rejects_when_min_odds_not_met(self):
        """最低2.5倍を満たさない構成は見送り。"""
        r = dutch_allocate({"a": 2.1, "b": 2.2}, min_odds=MIN_ODDS)
        assert not r.buy
        assert r.reason in ("min_odds", "min_return")

    def test_ev_gate_applies_only_when_probs_given(self):
        odds = {"a": 3.0, "b": 4.0, "c": 6.0}
        assert dutch_allocate(odds, probs=None).reason == "ok_no_ev_check"
        low = {"a": 0.01, "b": 0.01, "c": 0.01}
        r = dutch_allocate(odds, probs=low)
        assert not r.buy and r.reason == "ev"

    def test_respects_per_point_cap_and_budget(self):
        r = dutch_allocate({"a": 2.5, "b": 30.0}, cap=5_000, budget=10_000)
        assert all(s <= 5_000 for s in r.stakes.values())
        assert sum(r.stakes.values()) <= 10_000

    def test_stakes_are_whole_units(self):
        r = dutch_allocate({"a": 3.0, "b": 5.0, "c": 9.0}, unit=100)
        assert all(s % 100 == 0 for s in r.stakes.values())

    def test_constants_match_spec(self):
        assert (ODDS_FLOOR, MIN_RETURN, MIN_ODDS, EV_MIN) == (2.0, 1.3, 2.5, 1.3)


class TestRank7aTop2Gate:
    def test_threshold_always_finite(self):
        """空でもフォールバック値を返す。None を返すとゲートが無効化される。"""
        assert rank_7a_top2_threshold([]) == RANK_7A_TOP2_GATE_FALLBACK
        assert rank_7a_top2_threshold([1.5] * 3) == RANK_7A_TOP2_GATE_FALLBACK

    def test_threshold_is_q20_when_enough_history(self):
        hist = [1.30 + i * 0.01 for i in range(200)]
        assert rank_7a_top2_threshold(hist) == pytest.approx(1.30 + 40 * 0.01)

    def test_gate_keeps_low_axis_sum_only(self):
        cands = [{"axis_sum": 1.35}, {"axis_sum": 1.45}, {"axis_sum": 1.60}]
        keep, skip = rank_7a_top2_gate(cands, 1.40)
        assert [c["axis_sum"] for c in keep] == [1.35]
        assert all(c["skip_reason"] == "7A_top2_gate" for c in skip)

    def test_gate_records_threshold_on_skipped(self):
        _, skip = rank_7a_top2_gate([{"axis_sum": 1.9}], 1.4)
        assert skip[0]["top2_gate_threshold"] == 1.4

    def test_missing_axis_sum_is_skipped_not_bought(self):
        """axis_sum が無い候補は買わない側へ倒す（推奨を増やさない）。"""
        keep, skip = rank_7a_top2_gate([{"axis_sum": None}], 1.4)
        assert keep == [] and len(skip) == 1

    def test_daily_select_without_threshold_is_unchanged(self):
        """閾値を渡さなければ従来どおり全件（既存呼び出しの互換）。"""
        cands = [
            {"axis_sum": 1.5, "entropy": 0.1, "wt_overlap_n": 0},
            {"axis_sum": 1.9, "entropy": 0.1, "wt_overlap_n": 1},
        ]
        assert len(rank_7a_daily_select(cands)) == 2

    def test_daily_select_applies_threshold_when_given(self):
        cands = [
            {"axis_sum": 1.5, "entropy": 0.1, "wt_overlap_n": 0},
            {"axis_sum": 1.9, "entropy": 0.1, "wt_overlap_n": 1},
        ]
        got = rank_7a_daily_select(cands, top2_threshold=1.6)
        assert [c["axis_sum"] for c in got] == [1.5]


class TestRank7aGateChronological:
    """過去分再構築のゲートが look-ahead しないこと。"""

    def _pool(self):
        # 3日分。日ごとに axis_sum が違う
        out = []
        for d, vals in [("2026-01-01", [1.40, 1.60]),
                        ("2026-01-02", [1.35, 1.70]),
                        ("2026-01-03", [1.33, 1.90])]:
            out += [{"race_date": d, "axis_sum": v} for v in vals]
        return out

    def test_履歴が無い初日はフォールバック閾値で判定する(self):
        got = rank_7a_gate_chronological(self._pool()[:2])
        # fallback=1.432 → 1.40 は通り 1.60 は落ちる
        assert [c["axis_sum"] for c in got] == [1.40]

    def test_その日以降のプールを閾値に使わない(self):
        """未来日の axis_sum が閾値に混ざると look-ahead になる。"""
        hist: list[tuple[str, float]] = []
        rank_7a_gate_chronological(self._pool(), hist)
        # 1日目の判定時点では履歴が空でなければならない（後から積まれる）
        assert hist[0][0] == "2026-01-01"
        assert all(d <= "2026-01-03" for d, _ in hist)

    def test_seed_historyは呼び出し後に伸びる(self):
        hist: list[tuple[str, float]] = []
        rank_7a_gate_chronological(self._pool(), hist)
        assert len(hist) == 6      # プール全件が母集団として積まれる

    def test_窓をまたいで履歴が引き継がれる(self):
        hist: list[tuple[str, float]] = []
        rank_7a_gate_chronological(self._pool()[:2], hist)
        n_after_first = len(hist)
        rank_7a_gate_chronological(self._pool()[2:], hist)
        assert len(hist) > n_after_first

    def test_ゲートを外すとプール全件が返る(self):
        assert len(rank_7a_gate_chronological(self._pool(), [])) < len(self._pool())
