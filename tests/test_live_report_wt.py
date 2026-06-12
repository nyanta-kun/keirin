"""ユニットテスト: live_report_wt.py — 集計・void除外・WIDE分離。

合成 picks_history を使って純粋関数をテストする。
DBアクセス・ファイルI/O は monkeypatch で差し替え。
"""
import importlib
import sys
from pathlib import Path
import pytest
import numpy as np

# conftest で sys.path に scripts/ が追加済み
import live_report_wt as lr


# ── ヘルパー ──────────────────────────────────────────────────────────

def _make_picks(*rows):
    """picks を辞書リストで生成。fields: race_date, race_key, rank, n_combos, hit, payout, bet_amount"""
    result = []
    for r in rows:
        result.append({
            "race_date": r.get("race_date", "2026-06-01"),
            "race_key": r.get("race_key", "20260601_11_01"),
            "rank": r.get("rank", "A"),
            "n_combos": r.get("n_combos", 3),
            "hit": r.get("hit", False),
            "payout": r.get("payout", 0),
            "bet_amount": r.get("bet_amount", 300),
        })
    return result


# ── 1. _rank_section ──────────────────────────────────────────────────

def test_rank_section_basic():
    """SS/S/A が別々に集計される。"""
    picks = _make_picks(
        {"rank": "SS", "hit": True, "payout": 900, "bet_amount": 300},
        {"rank": "S",  "hit": False, "payout": 0,   "bet_amount": 300},
        {"rank": "A",  "hit": True, "payout": 600, "bet_amount": 300},
    )
    result = lr._rank_section(picks)
    assert result["SS"]["n"] == 1
    assert abs(result["SS"]["roi"] - 3.0) < 1e-9   # 900/300
    assert result["S"]["n"] == 1
    assert result["S"]["roi"] == 0.0
    assert result["A"]["n"] == 1
    assert abs(result["A"]["roi"] - 2.0) < 1e-9   # 600/300


def test_rank_section_wide_separate():
    """WIDE は独立して集計される（SS/S/A と混ざらない）。"""
    picks = _make_picks(
        {"rank": "A",    "hit": True,  "payout": 300, "bet_amount": 300},
        {"rank": "WIDE", "hit": False, "payout": 0,   "bet_amount": 100},
    )
    result = lr._rank_section(picks)
    assert "A" in result and "WIDE" in result
    assert result["A"]["n"] == 1
    assert result["WIDE"]["n"] == 1
    assert result["WIDE"]["roi"] == 0.0


def test_rank_section_empty():
    """空のとき各ランクが存在しない（KeyError なし）。"""
    result = lr._rank_section([])
    assert len(result) == 0


# ── 2. WIDE 分離（build_report レベル） ──────────────────────────────

def test_build_report_wide_excluded_from_main(monkeypatch):
    """build_report の main_total に WIDE が含まれないことを確認。"""
    picks = _make_picks(
        {"rank": "SS",   "hit": True,  "payout": 900, "bet_amount": 300},
        {"rank": "WIDE", "hit": True,  "payout": 200, "bet_amount": 100},
        {"rank": "A",    "hit": False, "payout": 0,   "bet_amount": 300},
    )
    # DB・ファイルアクセスを差し替え
    monkeypatch.setattr(lr, "_load_picks", lambda *a, **k: picks)
    monkeypatch.setattr(lr, "_load_tags", lambda *a, **k: {})
    monkeypatch.setattr(lr, "_drift_section", lambda: {"morning": {}, "evening": {}})

    result = lr.build_report()
    # main_total は SS+S+A のみ（WIDE除外・2件）
    assert result["main_total"]["n"] == 2
    # WIDE は rank_raw["WIDE"] に記録
    assert result["rank_raw"]["WIDE"]["total_bet"] == 100
    assert result["rank_raw"]["WIDE"]["total_pay"] == 200


# ── 3. void 除外（notifyで計上しない行はDBに存在しない） ──────────────

def test_void_races_not_in_picks(monkeypatch):
    """欠車無効化されたレースは picks_history に存在しない（採点から除外済み）。
    build_report がその前提で動作することを確認（void行=空）。"""
    picks = _make_picks(
        {"rank": "A", "hit": True,  "payout": 500, "bet_amount": 300},
        {"rank": "A", "hit": False, "payout": 0,   "bet_amount": 300},
    )
    monkeypatch.setattr(lr, "_load_picks", lambda *a, **k: picks)
    monkeypatch.setattr(lr, "_load_tags", lambda *a, **k: {})
    monkeypatch.setattr(lr, "_drift_section", lambda: {})

    result = lr.build_report()
    # 2件のみ集計（void行なし）
    assert result["rank"]["A"]["n"] == 2


# ── 4. タグ突合 ─────────────────────────────────────────────────────

def test_tag_section_fav_mismatch(monkeypatch):
    """fav_mismatch=True のレースのみ別集計される。"""
    picks = _make_picks(
        {"race_key": "rk1", "rank": "SS", "hit": True,  "payout": 900, "bet_amount": 300},
        {"race_key": "rk2", "rank": "A",  "hit": False, "payout": 0,   "bet_amount": 300},
        {"race_key": "rk3", "rank": "A",  "hit": False, "payout": 0,   "bet_amount": 300},
    )
    tags = {
        "rk1": {"fav_mismatch": True,  "upset_tier": "Q1_loose(<1.70)", "top3_sum": 1.5, "top3_sum_band": "Q1_loose(<1.70)"},
        "rk2": {"fav_mismatch": False, "upset_tier": "Q3(1.90-2.08)",  "top3_sum": 2.0, "top3_sum_band": "Q3(1.90-2.08)"},
        # rk3 はタグなし → 未記録扱い
    }
    result = lr._tag_section(picks, tags)
    # fav_mismatch=True: rk1 のみ
    assert result["fav_mismatch=True"]["n"] == 1
    assert abs(result["fav_mismatch=True"]["roi"] - 3.0) < 1e-9
    # fav_mismatch=False: rk2
    assert result["fav_mismatch=False"]["n"] == 1
    assert result["fav_mismatch=False"]["roi"] == 0.0
    # 未記録: rk3
    assert result["fav_mismatch=未記録"]["n"] == 1


def test_tag_section_wide_excluded():
    """タグ別集計では WIDE が除外される。"""
    picks = _make_picks(
        {"race_key": "rk1", "rank": "A",    "hit": True, "payout": 500, "bet_amount": 300},
        {"race_key": "rk1#W", "rank": "WIDE", "hit": True, "payout": 200, "bet_amount": 100},
    )
    tags = {"rk1": {"fav_mismatch": True, "top3_sum": 1.5, "top3_sum_band": "Q1_loose(<1.70)", "upset_tier": None}}
    result = lr._tag_section(picks, tags)
    # rk1#W は WIDE なのでタグ集計には含まれない
    assert result["fav_mismatch=True"]["n"] == 1  # rk1(A)のみ


# ── 5. top3_sum バンド割り当て ─────────────────────────────────────

@pytest.mark.parametrize("v, expected", [
    (None,  None),
    (1.0,   "Q1_loose(<1.70)"),
    (1.6999, "Q1_loose(<1.70)"),
    (1.70,  "Q2(1.70-1.90)"),
    (1.8,   "Q2(1.70-1.90)"),
    (1.90,  "Q3(1.90-2.08)"),
    (2.0,   "Q3(1.90-2.08)"),
    (2.08,  "Q4_chalk(>=2.08)"),
    (3.0,   "Q4_chalk(>=2.08)"),
])
def test_top3_band(v, expected):
    assert lr._top3_band(v) == expected


# ── 6. 必要標本数推定 ─────────────────────────────────────────────

def test_required_n_empty():
    r = lr._required_n_section([], [])
    assert r["current_n"] == 0
    assert r["needed_additional"] is None


def test_required_n_already_above_100(monkeypatch):
    """CI下限がすでに100%超 → needed_additional=0。"""
    # roi_summary を差し替えて ci_lo > 1.0 を返す
    monkeypatch.setattr(lr, "roi_summary", lambda *a, **k: {
        "n": 200, "hits": 150, "hit_rate": 0.75, "roi": 2.0,
        "ci_lo": 1.5, "ci_hi": 2.5,
        "roi_ex_max": 1.8, "roi_ex_top2": 1.7, "median_hit": 400.0
    })
    r = lr._required_n_section([300] * 200, [100] * 200)
    assert r["needed_additional"] == 0
    assert "すでに" in r["note"]


def test_required_n_roi_below_target():
    """ROI が 100% 未満の場合 → '>5000' を返す（現在分布から追加しても届かない）。"""
    # 全ハズレのケース
    pays = [0] * 10
    bets = [300] * 10
    r = lr._required_n_section(pays, bets, n_sim=100)
    assert r["needed_additional"] == ">5000"


# ── 7. _render_text / _render_md の煙テスト ────────────────────────

def _make_result():
    from scripts.roi_robustness_wt import roi_summary
    main_s = roi_summary([900, 0, 300], [300, 300, 300])
    return {
        "rank": {"SS": main_s, "A": roi_summary([300], [300])},
        "rank_raw": {
            "SS": {"total_bet": 300, "total_pay": 900},
            "A":  {"total_bet": 300, "total_pay": 300},
            "S":  {"total_bet": 0, "total_pay": 0},
            "B":  {"total_bet": 0, "total_pay": 0},
            "WIDE": {"total_bet": 100, "total_pay": 0},
            "_main_inv": 600, "_main_pay": 1200,
        },
        "main_total": main_s,
        "tag": {"fav_mismatch=True": roi_summary([900], [300])},
        "drift": {},
        "required_n": {"SS+S+A合算": {"current_n": 3, "ci_lo_now": 0.5,
                                       "needed_additional": 500, "note": "test"}},
    }


def test_render_text_runs():
    result = _make_result()
    text = lr._render_text(result, None, None)
    assert "live実測レポート" in text
    assert "ランク別成績" in text
    assert "ドリフト" in text
    assert "必要標本数" in text


def test_render_md_runs():
    result = _make_result()
    md = lr._render_md(result, "2026-06-01", "2026-06-13")
    assert "# live実測レポート" in md
    assert "## 1." in md
    assert "## 4." in md
    assert "2026-06-01" in md
