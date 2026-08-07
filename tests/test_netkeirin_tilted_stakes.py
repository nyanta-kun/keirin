"""netkeirin 入稿の傾斜配分まわりのテスト（2026-08-07）。

ここで守るのは3点:
  1. **買い目の集合が変わらない**（配分は金額の話で、買う目を減らしてはいけない）
  2. **印が均等経路と同じ**（軸=◎○ / 買った相手=△ / 買っていない車は印なし）
  3. **総額が予算どおり**
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import netkeirin_submit_wt as sub  # noqa: E402
from src.netkeirin_client import (  # noqa: E402
    BET_KIND_TRIO_AXIS2,
    expand_bet,
)
from src.strategy_wt import RACE_BUDGET  # noqa: E402


@pytest.fixture
def board(monkeypatch):
    """DB を触らずに盤面とモデル確率を差し込む。"""
    state = {"board": {}, "p3": {}}

    monkeypatch.setattr(sub, "_load_trio_board", lambda rk: state["board"])
    monkeypatch.setattr(sub, "_load_top3_probs", lambda rk: state["p3"])
    return state


CFG = {"stake_budget": RACE_BUDGET, "n_cars": 7}


def _points(legs):
    """買い目行を展開して目の集合に戻す。"""
    out = set()
    for leg in legs:
        assert leg.bet_kind == BET_KIND_TRIO_AXIS2
        out |= {frozenset(p) for p in expand_bet(leg.bet_kind, leg.groups)}
    return out


def test_買い目の集合は総流しと一致する(board):
    a1, a2, partners = 1, 2, [3, 4, 5, 6, 7]
    board["board"] = {frozenset({a1, a2, t}): 3.0 + t for t in partners}
    legs, source, stakes = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert _points(legs) == {frozenset({a1, a2, t}) for t in partners}
    assert set(stakes) == set(partners)
    assert source == "odds"


def test_総額は予算どおり(board):
    a1, a2, partners = 1, 2, [3, 4, 5, 6, 7]
    board["board"] = {frozenset({a1, a2, t}): 3.0 + t for t in partners}
    legs, _, stakes = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert sum(stakes.values()) == RACE_BUDGET
    assert sum(leg.stake_per_line * len(leg.groups[2]) for leg in legs) == RACE_BUDGET


def test_低オッズの相手ほど厚く積まれる(board):
    a1, a2 = 1, 2
    partners = [3, 4, 5]
    board["board"] = {frozenset({1, 2, 3}): 2.0,
                      frozenset({1, 2, 4}): 10.0,
                      frozenset({1, 2, 5}): 60.0}
    _, _, stakes = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert stakes[3] > stakes[4] > stakes[5]


def test_盤面が空ならモデル確率へ落ちる(board):
    a1, a2, partners = 1, 2, [3, 4, 5]
    board["board"] = {}
    board["p3"] = {3: 0.6, 4: 0.4, 5: 0.2}
    _, source, stakes = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert source == "model"
    assert stakes[3] > stakes[4] > stakes[5]


def test_盤面もモデルも無ければ均等になる(board):
    a1, a2, partners = 1, 2, [3, 4, 5, 6, 7]
    _, source, stakes = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert source == "equal"
    assert set(stakes.values()) == {2000}


def test_盤面が一部しか無ければモデルへ落ちる(board):
    """一部だけオッズを使うと点どうしの比率が壊れる。"""
    a1, a2, partners = 1, 2, [3, 4, 5]
    board["board"] = {frozenset({1, 2, 3}): 2.0}      # 4,5 が無い
    board["p3"] = {3: 0.6, 4: 0.4, 5: 0.2}
    _, source, _ = sub._build_tilted_legs("rk", CFG, a1, a2, partners)
    assert source == "model"


def test_三連複軸2車のランクはすべて傾斜配分の対象():
    """当初 7B だけ除外していたが、3点買いでも 3.0倍未満はガミになる。
    実測 +1.05pt [+0.66, +1.43] P=100% だったので全ランク一律にした。"""
    assert sub.RANK_CONFIGS["7B"].get("tilt_stakes")
    # 7H1 は三連単+三連複の併せ買いで予算配分の考え方が別（対象外のまま）。
    assert not sub.RANK_CONFIGS["7H1"].get("tilt_stakes")


def test_新ランクへの付け忘れを検出する():
    """RANK_ORDER で実際に起きた「一覧の手書き二重管理」型の事故を防ぐ。"""
    for key, cfg in sub.RANK_CONFIGS.items():
        if cfg.get("bet_kind") != BET_KIND_TRIO_AXIS2:
            continue
        assert cfg.get("tilt_stakes"), f"{key} に tilt_stakes が付いていません"


def test_自信ありランクの正本は一つ():
    assert sub.CONFIDENT_RANKS == {"7SS"}
