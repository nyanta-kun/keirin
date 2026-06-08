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
