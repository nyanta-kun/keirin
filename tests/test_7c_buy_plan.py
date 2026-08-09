"""7C の買い方（券種と買う相手）の単一正本を検査する（2026-08-09）。

## 設計

    pw1 >= RANK_7C_TRIFECTA_PW_MIN → 三連単 順序固定・**相手は全部**
    それ以外 ∧ p3_sum >= RANK_7C_TRIO_P3_SUM_MIN → 三連複・**相手は上位2点**
    それ以外 → 買わない

実測（13,960R・掃引/確認）: 網羅 100%→53.3% / 実質的中 31.7→33.6・32.0→33.0 /
ROI 75.8→78.5・76.8→79.7。

## 守る不変条件

1. **三連単は絞らない。** 点数を変えると効果が消える
   （[[keirin_7c_trifecta_switch_2026_08_09]]）
2. **相手2点は「絞る」とセットでしか効かない。** 単独だと ROI −1.35pt
3. 買い方を決めるのは `rank_7c_buy_plan` **だけ**。候補生成・発走前・入稿・Web が
   同じ結論になること（表示と入稿の食い違いはこのリポジトリの定番事故）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_TRIFECTA_PW_MIN,
    RANK_7C_TRIO_LEGS,
    RANK_7C_TRIO_P3_SUM_MIN,
    rank_7c_buy_plan,
)

LEGS = [3, 4, 5, 6]


def _p3(sum_top2: float) -> dict[int, float]:
    """上位2車の合計が sum_top2 になる 7車分の3着内率。"""
    a = sum_top2 / 2
    return {1: a, 2: a, 3: 0.40, 4: 0.35, 5: 0.30, 6: 0.20, 7: 0.05}


def test_trifecta_keeps_all_partners() -> None:
    """三連単は相手を絞らない（点数を変えると効果が消える）。"""
    pw = {1: RANK_7C_TRIFECTA_PW_MIN, 2: 0.10}
    kind, legs = rank_7c_buy_plan(_p3(1.60), pw, 1, LEGS)
    assert kind == "trifecta"
    assert legs == LEGS


def test_trifecta_ignores_the_trio_gate() -> None:
    """三連単側には p3_sum ゲートを掛けない（単独で最良のため）。"""
    pw = {1: RANK_7C_TRIFECTA_PW_MIN, 2: 0.10}
    plan = rank_7c_buy_plan(_p3(RANK_7C_TRIO_P3_SUM_MIN - 0.10), pw, 1, LEGS)
    assert plan is not None and plan[0] == "trifecta"


def test_trio_narrows_to_two_partners() -> None:
    """三連複は上位2点だけ買う。"""
    pw = {1: 0.30, 2: 0.20}
    kind, legs = rank_7c_buy_plan(_p3(RANK_7C_TRIO_P3_SUM_MIN), pw, 1, LEGS)
    assert kind == "trio"
    assert legs == LEGS[:RANK_7C_TRIO_LEGS] == [3, 4]


def test_trio_below_gate_is_not_bought() -> None:
    """三連複側でゲートを下回るレースは買わない（見送り）。"""
    pw = {1: 0.30, 2: 0.20}
    assert rank_7c_buy_plan(_p3(RANK_7C_TRIO_P3_SUM_MIN - 0.01), pw, 1, LEGS) is None


def test_missing_win_probs_falls_back_to_trio() -> None:
    """単勝率が無いときは三連単へ切り替えない（検証済みの既定へ倒す）。"""
    plan = rank_7c_buy_plan(_p3(1.60), None, 1, LEGS)
    assert plan is not None and plan[0] == "trio"


def test_no_partners_means_no_bet() -> None:
    assert rank_7c_buy_plan(_p3(1.60), {1: 0.9}, 1, []) is None


def test_submit_reads_the_bought_partners_key() -> None:
    """入稿が `legs_7c_buy` を読んでいること。

    `legs_7c`（選別用の全リスト）を読むと絞り込みが効かず、
    **表示と入稿が食い違う**。
    """
    from scripts.netkeirin_submit_wt import RANK_CONFIGS
    assert RANK_CONFIGS["7C"]["partners_key"] == "legs_7c_buy"
