"""波乱ゲート純粋関数のテスト（境界値・ゲート・カット読込フォールバック）。"""
import json
import importlib

import pytest

import src.strategy_wt as sw


# ── upset_tier 境界値（既定カット 1.70 / 1.90 / 2.08）──
@pytest.mark.parametrize("top3_sum, expected", [
    (1.0, "Q1_loose"),
    (1.6999, "Q1_loose"),
    (1.70, "Q2"),       # カット境界は下側の帯に含めない（< 判定）
    (1.80, "Q2"),
    (1.90, "Q3"),
    (2.00, "Q3"),
    (2.08, "Q4_chalk"),
    (2.50, "Q4_chalk"),
])
def test_upset_tier_boundaries(monkeypatch, top3_sum, expected):
    # 既定カットで判定（JSON の影響を排除）
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.upset_tier(top3_sum) == expected


# ── passes_upset_gate（loose側のみ通す）──
@pytest.mark.parametrize("top3_sum, max_tier, expected", [
    (1.5, "Q1_loose", True),    # Q1_loose は通る
    (1.8, "Q1_loose", False),   # Q2 は Q1ゲートでは通さない
    (1.8, "Q2", True),          # Q2 までなら通る
    (2.0, "Q2", False),         # Q3 は通さない
    (2.0, "Q3", True),
    (2.2, "Q3", False),         # Q4_chalk は常に通さない
])
def test_passes_upset_gate(monkeypatch, top3_sum, max_tier, expected):
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.passes_upset_gate(top3_sum, max_tier) is expected


# ── _load_cuts のフォールバック ──
def _write(tmp_path, content):
    p = tmp_path / "upset_cuts_wt.json"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_cuts_valid(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.6, 1.8, 2.0]}))
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == (1.6, 1.8, 2.0)


def test_load_cuts_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sw, "_CUTS_PATH", tmp_path / "nope.json")
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_corrupt_json(monkeypatch, tmp_path):
    p = _write(tmp_path, "not a json {")
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_non_monotonic(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [2.0, 1.9, 2.1]}))   # 単調でない
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_wrong_length(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.7, 1.9]}))
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_equal_values_rejected(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.7, 1.7, 2.0]}))   # 等値は単調NG
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


# ── stake_units（波乱ステーク傾斜） ──
@pytest.mark.parametrize("top3_sum, expected_mult", [
    (1.5, 2),    # Q1_loose → 2倍
    (1.8, 1),    # Q2 → 1倍
    (2.0, 0),    # Q3 → 見送り
    (2.3, 0),    # Q4_chalk → 見送り
])
def test_stake_units(monkeypatch, top3_sum, expected_mult):
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.stake_units(top3_sum) == expected_mult


# ── doc53 統合ポリシー（2026-07-12） ─────────────────────────────────────────

from src.strategy_wt import (  # noqa: E402
    SS_STAKE, is_senbatsu, line_score_features,
    ss_policy,
)


class TestLineScoreFeatures:
    def test_basic_two_lines(self):
        # ライン1: 平均90 / ライン2: 平均88 → avg_gap=2.0
        pairs = [(1, 92.0), (1, 88.0), (2, 88.0), (2, 88.0), (3, 85.0), (3, 85.0), (3, 85.0)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert gap == 2.0
        assert n_lines == 3
        assert all_solo is False

    def test_all_solo(self):
        pairs = [(i, 80.0 + i) for i in range(1, 8)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert n_lines == 7
        assert all_solo is True
        assert gap == 1.0  # 86-85（単騎も1本のラインとして格差計算）

    def test_missing_line_group(self):
        pairs = [(1, 90.0), (None, 88.0), (2, 85.0)]
        assert line_score_features(pairs) == (None, None, None)

    def test_single_line(self):
        pairs = [(1, 90.0), (1, 88.0)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert gap is None
        assert n_lines == 1

    def test_empty(self):
        assert line_score_features([]) == (None, None, None)


class TestSsPolicy:
    def test_normal(self):
        assert ss_policy("Ａ級一般", 0.5, 3, False) == (None, SS_STAKE)

    def test_senbatsu_skip(self):
        reason, _ = ss_policy("Ａ級選抜", 0.5, 3, False)
        assert reason == "選抜"

    def test_four_lines_not_skipped(self):
        # 4分戦カットは2026-07-16廃止
        assert ss_policy("Ａ級一般", 3.0, 4, False) == (None, SS_STAKE)

    def test_no_boost(self):
        # 格差増額は2026-07-16廃止（常に100円/点）
        assert ss_policy("Ａ級一般", 2.0, 3, False) == (None, SS_STAKE)

    def test_none_context_fallback(self):
        assert ss_policy(None, None, None, None) == (None, SS_STAKE)



def test_is_senbatsu():
    assert is_senbatsu("Ａ級選抜")
    assert is_senbatsu("Ａ級チャレンジ選抜")
    assert is_senbatsu("Ｌ級ガールズ選抜")
    assert not is_senbatsu("Ａ級特選")  # 特選は選抜ではない
    assert not is_senbatsu("Ａ級一般")
    assert not is_senbatsu(None)
