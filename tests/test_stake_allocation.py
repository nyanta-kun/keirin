"""賭け金の傾斜配分（src/stake_allocation.py）のテスト。"""
from __future__ import annotations

import pytest

from src.stake_allocation import (
    BUDGET_DEFAULT,
    SOURCE_BLEND,
    SOURCE_EQUAL,
    SOURCE_MODEL,
    SOURCE_ODDS,
    UNIT_DEFAULT,
    allocate_budget,
    group_by_stake,
    landing_weights,
    tilted_stakes,
)


# ── landing_weights: どの材料で重みを作ったか ──────────────────────────

def test_両方あればブレンドになる():
    w, src = landing_weights([3, 4, 5], {3: 5.0, 4: 20.0, 5: 50.0},
                             {3: 0.5, 4: 0.3, 5: 0.1})
    assert src == SOURCE_BLEND
    # 低オッズ・高確率の点ほど重い
    assert w[3] > w[4] > w[5]


def test_オッズだけならオッズを使う():
    w, src = landing_weights([3, 4], {3: 5.0, 4: 20.0}, None)
    assert src == SOURCE_ODDS
    assert w[3] == pytest.approx(1 / 5.0)


def test_モデルだけならモデルを使う():
    w, src = landing_weights([3, 4], None, {3: 0.5, 4: 0.2})
    assert src == SOURCE_MODEL
    assert w[3] == pytest.approx(0.5)


def test_どちらも無ければ均等へ落ちる():
    w, src = landing_weights([3, 4], None, None)
    assert src == SOURCE_EQUAL
    assert w[3] == w[4]


def test_買う点の一部しかオッズが無いなら使わない():
    """一部だけ使うと欠けた点の重みを別尺度で決めることになり比率が壊れる。"""
    w, src = landing_weights([3, 4, 5], {3: 5.0, 4: 20.0}, {3: 0.5, 4: 0.3, 5: 0.1})
    assert src == SOURCE_MODEL


def test_ゼロや負のオッズは使わない():
    _, src = landing_weights([3, 4], {3: 0.0, 4: 20.0}, {3: 0.5, 4: 0.3})
    assert src == SOURCE_MODEL


def test_legsが空なら落とす():
    with pytest.raises(ValueError):
        landing_weights([], {}, {})


# ── allocate_budget: 予算の割り方 ─────────────────────────────────────

def test_合計は必ず予算に一致する():
    for weights in ({3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
                    {3: 0.5, 4: 0.2, 5: 0.11, 6: 0.03},
                    {3: 1.0, 4: 0.001}):
        stakes = allocate_budget(weights)
        assert sum(stakes.values()) == BUDGET_DEFAULT


def test_すべて100円単位になる():
    stakes = allocate_budget({3: 0.5, 4: 0.2, 5: 0.11, 6: 0.03})
    assert all(s % UNIT_DEFAULT == 0 for s in stakes.values())


def test_均等な重みなら均等に割れる():
    stakes = allocate_budget({3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0})
    assert set(stakes.values()) == {2000}


def test_極端に薄い点でも0円にはならない():
    """0円になると買い目の集合が黙って変わる（点数が減る）。配分の役目ではない。"""
    stakes = allocate_budget({3: 1.0, 4: 1e-9, 5: 1e-9})
    assert all(s >= UNIT_DEFAULT for s in stakes.values())
    assert sum(stakes.values()) == BUDGET_DEFAULT


def test_重みが大きい点ほど賭け金も大きい():
    stakes = allocate_budget({3: 0.5, 4: 0.25, 5: 0.125})
    assert stakes[3] > stakes[4] > stakes[5]


def test_予算が点数に足りなければ落とす():
    with pytest.raises(ValueError):
        allocate_budget({i: 1.0 for i in range(20)}, budget=1000, unit=100)


def test_重みが全部0なら落とす():
    with pytest.raises(ValueError):
        allocate_budget({3: 0.0, 4: 0.0})


# ── tilted_stakes: 目的（ガミの解消）を満たしているか ─────────────────

def test_想定オッズどおりに決まれば全点が元返し以上になる():
    """dutch 配分の狙いそのもの。買う点の合成ブックが1未満なら成立する。"""
    odds = {3: 3.0, 4: 8.0, 5: 15.0, 6: 40.0}   # Σ(1/o)=0.5 なので余裕がある
    stakes, src = tilted_stakes(list(odds), odds, None)
    assert src == SOURCE_ODDS
    for car, o in odds.items():
        assert stakes[car] * o >= BUDGET_DEFAULT


def test_均等割りでは低オッズ点がガミになることの確認():
    """現行方式が壊れていることをテストとして固定しておく（比較の基準）。"""
    odds = {3: 3.0, 4: 8.0, 5: 15.0, 6: 40.0}
    stakes, _ = tilted_stakes(list(odds), None, None)   # 均等
    assert stakes[3] * odds[3] < BUDGET_DEFAULT         # 2,500円 × 3.0 = 7,500円


def test_7C想定の可変点数でも成立する():
    for n in (4, 5):
        legs = list(range(3, 3 + n))
        stakes, _ = tilted_stakes(legs, None, {c: 0.3 for c in legs})
        assert sum(stakes.values()) == BUDGET_DEFAULT
        assert len(stakes) == n


def test_9車想定の7点でも成立する():
    legs = list(range(3, 10))
    stakes, _ = tilted_stakes(legs, {c: 5.0 + c for c in legs}, None)
    assert sum(stakes.values()) == BUDGET_DEFAULT
    assert len(stakes) == 7


# ── group_by_stake: 入稿行へのまとめ方 ────────────────────────────────

def test_同額はまとめられ金額の降順で返る():
    groups = group_by_stake({3: 3000, 4: 3000, 5: 2000, 6: 2000})
    assert groups == [(3000, [3, 4]), (2000, [5, 6])]


def test_まとめても車の集合は失われない():
    stakes = {3: 3000, 4: 2500, 5: 2500, 6: 2000}
    groups = group_by_stake(stakes)
    assert sorted(c for _, cars in groups for c in cars) == [3, 4, 5, 6]
    assert sum(s * len(cars) for s, cars in groups) == sum(stakes.values())
