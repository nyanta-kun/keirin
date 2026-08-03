"""ユニットテスト: live_report_wt.py — 集計・見送り/候補除外・ランク分離。

合成 picks_history を使って純粋関数をテストする。
DBアクセス・ファイルI/O は monkeypatch で差し替え。

ランク体系（2026-07-31〜）: 購入対象は RANK_7S(7S) / RANK_7A(7A) /
RANK_9S(9S) / RANK_9A(9A) の4ランク。
- 旧 7PLUS_R(表示SS) は 2026-07-16 全廃
- 旧 7PLUS_ST/STP(S/S+) は 2026-07-15 全廃
- SEVEN_S1 は 2026-07-31 全廃（picks_history からも削除済み）
本テストは 2026-07-31 のレビューで live_report_wt.py の RANKS が全廃済みの
7PLUS_R のまま放置されていた（＝常に n=0 で空振りしていた）ことが判明し、
本体の修正に合わせて現行4ランクへ更新したもの。
"""
import importlib
import sys
from pathlib import Path
import pytest
import numpy as np

# conftest で sys.path に scripts/ が追加済み
import live_report_wt as lr
import src.strategy_wt as sw


# ── ヘルパー ──────────────────────────────────────────────────────────

def _make_picks(*rows):
    """picks を辞書リストで生成。fields: race_date, race_key, rank, n_combos, hit, payout, bet_amount"""
    result = []
    for r in rows:
        result.append({
            "race_date": r.get("race_date", "2026-06-01"),
            "race_key": r.get("race_key", "20260601_11_01"),
            "rank": r.get("rank", "RANK_7S"),
            "n_combos": r.get("n_combos", 5),
            "hit": r.get("hit", False),
            "payout": r.get("payout", 0),
            "bet_amount": r.get("bet_amount", 500),
        })
    return result


# ── 0. ランク定義そのもの ─────────────────────────────────────────────

def test_ranks_are_current_and_exclude_abolished():
    """RANKS が単一正本の現行ランク（in_live_report=True）と一致し、
    全廃済みランクを含まないこと。

    2026-07-31 のレビューで検出した「全廃ランクのまま放置」の再発防止。
    期待値は src.strategy_wt.CURRENT_PAPER_RANKS から導出する（ランクの
    増減のたびにこのテストのリテラルを書き換える運用にすると、
    「数を合わせるだけ」の修正で本来の検出力が失われるため）。
    """
    expected = [spec.rank for spec in sw.CURRENT_PAPER_RANKS if spec.in_live_report]
    assert lr.RANKS == expected
    for abolished in ("7PLUS_R", "7PLUS_ST", "7PLUS_STP", "7PLUS_U",
                      "7PLUS_M", "SEVEN_S1", "SIX_S1"):
        assert abolished not in lr.RANKS
    # 全ランクに表示ラベルが定義されていること
    for rank in lr.RANKS:
        assert rank in lr.RANK_LABELS


# ── 1. _rank_section ──────────────────────────────────────────────────

def test_rank_section_basic():
    """RANK_7S が集計され、全廃済みランクはバケットに含まれない。"""
    picks = _make_picks(
        {"rank": "RANK_7S", "hit": True, "payout": 1500, "bet_amount": 500},
        {"rank": "RANK_7S", "hit": False, "payout": 0, "bet_amount": 500,
         "race_key": "20260601_11_02"},
    )
    result = lr._rank_section(picks)
    assert result["RANK_7S"]["n"] == 2
    assert abs(result["RANK_7S"]["roi"] - 1.5) < 1e-9   # 1500/1000
    assert "7PLUS_R" not in result
    assert "7PLUS_ST" not in result


def test_rank_section_separates_all_four_ranks():
    """4ランクが混在しても互いに混ざらず分離集計されること。"""
    picks = _make_picks(
        {"rank": "RANK_7S", "hit": True,  "payout": 1500, "bet_amount": 500,
         "race_key": "rk1"},
        {"rank": "RANK_7A", "hit": False, "payout": 0,    "bet_amount": 500,
         "race_key": "rk2"},
        {"rank": "RANK_9S",  "hit": True,  "payout": 800,  "bet_amount": 500,
         "race_key": "rk3"},
        {"rank": "RANK_9A",  "hit": False, "payout": 0,    "bet_amount": 500,
         "race_key": "rk4"},
    )
    result = lr._rank_section(picks)
    for rank in ("RANK_7S", "RANK_7A", "RANK_9S", "RANK_9A"):
        assert result[rank]["n"] == 1
    assert abs(result["RANK_7S"]["roi"] - 3.0) < 1e-9
    assert result["RANK_7A"]["roi"] == 0.0


def test_rank_section_empty():
    """空のとき各ランクが存在しない（KeyError なし）。"""
    result = lr._rank_section([])
    assert len(result) == 0


# ── 2. 全ランク合算（build_report レベル） ────────────────────────────

def test_build_report_all_ranks_in_main(monkeypatch):
    """build_report の main_total に全4ランクが合算されることを確認。"""
    picks = _make_picks(
        {"rank": "RANK_7S", "hit": True,  "payout": 1500, "bet_amount": 500,
         "race_key": "rk1"},
        {"rank": "RANK_7A", "hit": True,  "payout": 400,  "bet_amount": 500,
         "race_key": "rk2"},
        {"rank": "RANK_9S",  "hit": False, "payout": 0,    "bet_amount": 500,
         "race_key": "rk3"},
        {"rank": "RANK_9A",  "hit": False, "payout": 0,    "bet_amount": 500,
         "race_key": "rk4"},
    )
    # DB・ファイルアクセスを差し替え
    monkeypatch.setattr(lr, "_load_picks", lambda *a, **k: picks)
    monkeypatch.setattr(lr, "_load_tags", lambda *a, **k: {})
    monkeypatch.setattr(lr, "_drift_section", lambda: {"morning": {}, "evening": {}})

    result = lr.build_report()
    assert result["main_total"]["n"] == 4
    assert result["rank_raw"]["RANK_7S"]["total_bet"] == 500
    assert result["rank_raw"]["RANK_7S"]["total_pay"] == 1500
    assert result["rank_raw"]["RANK_7A"]["total_pay"] == 400
    # 合算は4ランク分（500*4=2000 / 1500+400=1900）
    assert result["rank_raw"]["_main_inv"] == 2000
    assert result["rank_raw"]["_main_pay"] == 1900


# ── 3. 見送り・候補・void 除外は _load_picks（SQL側）で行う ────────────

def test_picks_already_filtered_are_all_counted(monkeypatch):
    """_load_picks が返す行（見送り/候補/void除外済み）は build_report でそのまま計上される。"""
    picks = _make_picks(
        {"rank": "RANK_7S", "hit": True,  "payout": 900, "bet_amount": 500},
        {"rank": "RANK_7S", "hit": False, "payout": 0,   "bet_amount": 500},
    )
    monkeypatch.setattr(lr, "_load_picks", lambda *a, **k: picks)
    monkeypatch.setattr(lr, "_load_tags", lambda *a, **k: {})
    monkeypatch.setattr(lr, "_drift_section", lambda: {})

    result = lr.build_report()
    # 2件のみ計上
    assert result["rank"]["RANK_7S"]["n"] == 2


def test_load_picks_sql_excludes_miwokuri_and_candidate_and_zero_bet():
    """_load_picks の SQL が見送り(miwokuri)・候補・bet_amount=0 を除外する条件を持つ。"""
    assert "7PLUS_CAND" not in lr.RANKS  # IN句には含めない（除外対象）
    import inspect
    src = inspect.getsource(lr._load_picks)
    # SQL 本文（docstring より後）のみを対象に、除外条件が実装されていることを確認
    sql_body = src.split('"""', 2)[-1]
    assert "miwokuri" in sql_body
    assert "bet_amount > 0" in sql_body
    assert "rank IN" in sql_body


# ── 4. タグ突合 ─────────────────────────────────────────────────────

def test_tag_section_fav_mismatch(monkeypatch):
    """fav_mismatch=True のレースのみ別集計される。"""
    picks = _make_picks(
        {"race_key": "rk1", "rank": "RANK_7S", "hit": True,  "payout": 1500, "bet_amount": 500},
        {"race_key": "rk2", "rank": "RANK_7S", "hit": False, "payout": 0,    "bet_amount": 500},
        {"race_key": "rk3", "rank": "RANK_7S", "hit": False, "payout": 0,    "bet_amount": 500},
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


def test_tag_section_all_ranks_included():
    """タグ別集計は購入対象ランクの全件が対象になる（独立除外は不要）。"""
    picks = _make_picks(
        {"race_key": "rk1", "rank": "RANK_7S", "hit": True, "payout": 500, "bet_amount": 500},
        {"race_key": "rk1", "rank": "RANK_9S",  "hit": True, "payout": 200, "bet_amount": 500},
    )
    tags = {"rk1": {"fav_mismatch": True, "top3_sum": 1.5, "top3_sum_band": "Q1_loose(<1.70)", "upset_tier": None}}
    result = lr._tag_section(picks, tags)
    # 両方とも fav_mismatch=True 集計に含まれる（2件）
    assert result["fav_mismatch=True"]["n"] == 2


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
    r = lr._required_n_section([500] * 200, [100] * 200)
    assert r["needed_additional"] == 0
    assert "すでに" in r["note"]


def test_required_n_roi_below_target():
    """ROI が 100% 未満の場合 → '>5000' を返す（現在分布から追加しても届かない）。"""
    # 全ハズレのケース
    pays = [0] * 10
    bets = [500] * 10
    r = lr._required_n_section(pays, bets, n_sim=100)
    assert r["needed_additional"] == ">5000"


# ── 7. _render_text / _render_md の煙テスト ────────────────────────

def _make_result():
    from scripts.roi_robustness_wt import roi_summary
    main_s = roi_summary([1500, 0, 500], [500, 500, 500])
    return {
        "rank": {
            "RANK_7S": main_s,
            "RANK_7A": roi_summary([500], [500]),
        },
        "rank_raw": {
            "RANK_7S": {"total_bet": 500, "total_pay": 1500},
            "RANK_7A": {"total_bet": 500, "total_pay": 500},
            "RANK_9S":  {"total_bet": 0, "total_pay": 0},
            "RANK_9A":  {"total_bet": 0, "total_pay": 0},
            "_main_inv": 1000, "_main_pay": 2000,
        },
        "main_total": main_s,
        "tag": {"fav_mismatch=True": roi_summary([1500], [500])},
        "drift": {},
        "required_n": {"全ランク合算": {"current_n": 3, "ci_lo_now": 0.5,
                                        "needed_additional": 500, "note": "test"}},
    }


def test_render_text_runs():
    result = _make_result()
    text = lr._render_text(result, None, None)
    assert "live実測レポート" in text
    assert "ランク別成績" in text
    assert "ドリフト" in text
    assert "必要標本数" in text
    # 表示ラベル（内部rankの生文字列 SEVEN_* は表に出さない）
    assert "7S" in text
    assert "7A" in text
    assert "RANK_7S" not in text


def test_render_md_runs():
    result = _make_result()
    md = lr._render_md(result, "2026-06-01", "2026-06-13")
    assert "# live実測レポート" in md
    assert "## 1." in md
    assert "## 4." in md
    assert "2026-06-01" in md
