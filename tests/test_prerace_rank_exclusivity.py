"""7S/7A・9S/9A の排他性ガードの回帰テスト（2026-08-01・commit 参照）。

背景:
    `rank_7s_daily_select()`（2ゲート不合格0個）と `rank_7a_daily_select()`
    （ちょうど1個）は定義上排他だが、候補JSONが昼/夜の2ファイルに分かれる構成
    では同一レースが両方に載りえた（朝は1ゲート不合格で7A → 夕方の再収集で
    ライン情報が更新され0個不合格＝7S に転ぶ）。
    `notify_prerace_wt._load_rank_*_candidates()` は昼+夜を無条件に連結する
    ため、両方が判定・記録され `#7A` と `#7S` の2行が picks_history に
    書かれてしまう（1レースに1,000円投資として二重計上）。
    実測: 2026-07-28〜31 に6レース（うち3件は買い目も完全一致）。

    本テストは、ローダ段で 7S/9S を優先して A 系から重複レースを除外する
    ガードが効いていることを保証する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import notify_prerace_wt as mod  # noqa: E402


# ---------------------------------------------------------------------------
# _exclude_overlapping_races（純関数）
# ---------------------------------------------------------------------------

def test_exclude_overlapping_races_drops_only_duplicates():
    loser = [{"race_key": "A"}, {"race_key": "B"}, {"race_key": "C"}]
    winner = [{"race_key": "B"}]
    kept = mod._exclude_overlapping_races(loser, winner, loser="7A", winner="7S")
    assert [c["race_key"] for c in kept] == ["A", "C"]


def test_exclude_overlapping_races_noop_when_winner_empty():
    """勝者側が0件なら敗者側は一切削らない（元のリストをそのまま返す）。"""
    loser = [{"race_key": "A"}, {"race_key": "B"}]
    kept = mod._exclude_overlapping_races(loser, [], loser="7A", winner="7S")
    assert [c["race_key"] for c in kept] == ["A", "B"]


def test_exclude_overlapping_races_ignores_entries_without_race_key():
    """race_key を持たない不正エントリは勝者集合に混入させない（None で全滅しない）。"""
    loser = [{"race_key": "A"}, {"foo": 1}]
    winner = [{"foo": 2}]
    kept = mod._exclude_overlapping_races(loser, winner, loser="9A", winner="9S")
    assert kept == loser


# ---------------------------------------------------------------------------
# ローダ結合テスト（昼/夜ファイルをまたいだ実際の再現ケース）
# ---------------------------------------------------------------------------

def _write(picks_dir: Path, name: str, payload: list[dict]) -> None:
    (picks_dir / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def picks_dir(tmp_path, monkeypatch):
    """notify_prerace_wt が読む data/picks を tmp へ差し替える。

    ローダは `Path(__file__).parent.parent / "data" / "picks"` を組み立てるため、
    モジュールの __file__ を tmp 配下の scripts/ に見せかける。
    """
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    d = fake_repo / "data" / "picks"
    d.mkdir(parents=True)
    monkeypatch.setattr(mod, "__file__", str(fake_repo / "scripts" / "notify_prerace_wt.py"))
    return d


def test_7a_excludes_race_that_flipped_to_7s_in_night_file(picks_dir):
    """朝7A・夜7Sに転んだレースが7A側から消え、7S側には残ること（本バグの再現）。"""
    today = "2026-07-31"
    _write(picks_dir, f"wave_picks_wt_{today}_s7a_candidates.json",
           [{"race_key": "20260731_13_12"}, {"race_key": "20260731_13_05"}])
    _write(picks_dir, f"wave_picks_wt_{today}_night_s7_candidates.json",
           [{"race_key": "20260731_13_12"}])

    assert [c["race_key"] for c in mod._load_rank_7a_candidates(today)] == ["20260731_13_05"]
    assert [c["race_key"] for c in mod._load_rank_7s_candidates(today)] == ["20260731_13_12"]


def test_7a_unaffected_when_no_overlap(picks_dir):
    today = "2026-07-31"
    _write(picks_dir, f"wave_picks_wt_{today}_s7a_candidates.json",
           [{"race_key": "R_A"}])
    _write(picks_dir, f"wave_picks_wt_{today}_s7_candidates.json",
           [{"race_key": "R_S"}])

    assert [c["race_key"] for c in mod._load_rank_7a_candidates(today)] == ["R_A"]


def test_9a_excludes_race_present_in_9s(picks_dir):
    """9S/9Aにも同じガードが掛かっていること（実測0件だが構造は同型）。"""
    today = "2026-07-31"
    _write(picks_dir, f"wave_picks_wt_{today}_s9a_candidates.json",
           [{"race_key": "R1"}, {"race_key": "R2"}])
    _write(picks_dir, f"wave_picks_wt_{today}_night_s9s_unused.json", [])
    _write(picks_dir, f"wave_picks_wt_{today}_s9_candidates.json",
           [{"race_key": "R2"}])

    assert [c["race_key"] for c in mod._load_rank_9a_candidates(today)] == ["R1"]


def test_loaders_return_empty_when_no_files(picks_dir):
    today = "2026-07-31"
    assert mod._load_rank_7a_candidates(today) == []
    assert mod._load_rank_9a_candidates(today) == []
