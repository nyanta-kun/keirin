"""新S1（6車三連単）judge_s1 の純関数テスト。"""
import sys
from itertools import combinations, permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from notify_prerace_wt import judge_s1  # noqa: E402


def _trio6(frames):
    return {frozenset(c): 10.0 for c in combinations(frames, 3)}


def _trifecta6(frames):
    return {p: 50.0 for p in permutations(frames, 3)}


def test_buy_normal():
    frames = [1, 2, 3, 4, 5, 6]
    d, det = judge_s1([3, 1, 5, 2], _trifecta6(frames), _trio6(frames))
    assert d == "buy"
    assert det["combos"] == ["3>1>5", "3>1>2"]
    assert det["leg_odds"]["3>1>5"] == 50.0


def test_skip_board5():
    frames5 = [1, 2, 3, 4, 5]
    d, det = judge_s1([1, 2, 3, 4], _trifecta6(frames5), _trio6(frames5))
    assert d == "skip"
    assert "盤面5車" in det["skip_reason"]


def test_skip_axis_out_of_board():
    frames = [1, 2, 3, 4, 5, 7]  # 6 が欠車で 7 繰り上がり等
    d, det = judge_s1([6, 1, 2, 3], _trifecta6(frames), _trio6(frames))
    assert d == "skip"
    assert "盤面外" in det["skip_reason"]


def test_unknown_when_no_board():
    d, det = judge_s1([1, 2, 3, 4], {}, {})
    assert d == "不明"


def test_buy_even_if_odds_missing():
    frames = [1, 2, 3, 4, 5, 6]
    d, det = judge_s1([1, 2, 3, 4], {}, _trio6(frames))  # trifectaオッズ取得不可
    assert d == "buy"
    assert det["leg_odds"]["1>2>3"] is None
