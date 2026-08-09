"""7C の三連単切替（1着=軸1 / 2着=軸2 / 3着=相手流し）の不変条件を検査する。

## 守る不変条件

1. **点数が三連複と同じ**。この予算方式では「ガミ ⟺ 的中オッズ < 点数」なので、
   点数を増やすとガミ境界も上がり切替の意味が消える。検証では 2k点/4k点の案が
   いずれも実質的中を落とした（`RANK_7C_TRIFECTA_PW_MIN` の定義部）。
2. **切替の判定は軸1の単勝率だけ**。「2着との差」を足しても該当がほぼ変わらない。
3. **切替時に文面が差し替わる**。既定文は「三連複・軸2車流し」と書いてあり、
   そのまま出すと買っていない券種を説明することになる（7B と同型の事故）。
4. **legs を組んだら submit_pick には流さない**。`submit_pick` は `cfg["bet_kind"]`
   （= 三連複）で券種を決めるため、三連単を組んだのに三連複が入稿される。
   9H1 で実際に起きた「記録経路のフラグ列挙漏れ」と同じ型なので、
   **個別フラグではなく legs の有無**で分岐していることを構造的に検査する。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.netkeirin_submit_wt import (  # noqa: E402
    RANK_CONFIGS,
    _build_trifecta_head_legs,
    _stake_per_line,
)
from src.netkeirin_client import (  # noqa: E402
    BET_KIND_TRIFECTA_FORMATION,
    BET_KIND_TRIO_AXIS2,
    expand_bet,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7C_TRIFECTA_PW_MIN,
    rank_7c_use_trifecta,
)

CFG_7C = RANK_CONFIGS["7C"]


@pytest.mark.parametrize("partners", [[2, 3, 4, 5], [2, 3, 4, 5, 6]])
def test_trifecta_point_count_equals_trio(partners: list[int]) -> None:
    """三連単の点数が三連複（軸2車流し）と一致する。"""
    a1, a2 = 1, 7
    legs, _ = _build_trifecta_head_legs(CFG_7C, a1, a2, partners)
    assert len(legs) == 1
    tf = expand_bet(BET_KIND_TRIFECTA_FORMATION, legs[0].groups)
    trio = expand_bet(BET_KIND_TRIO_AXIS2, [[a1], [a2], partners])
    assert len(tf) == len(trio) == len(partners), (
        f"点数が三連複と食い違う: 三連単{len(tf)}点 / 三連複{len(trio)}点"
    )


def test_trifecta_order_is_axis1_then_axis2() -> None:
    """全ての目が「1着=軸1・2着=軸2」で、3着だけが相手。"""
    a1, a2, partners = 3, 5, [1, 2, 4, 6]
    legs, marks = _build_trifecta_head_legs(CFG_7C, a1, a2, partners)
    points = expand_bet(BET_KIND_TRIFECTA_FORMATION, legs[0].groups)
    assert {p[0] for p in points} == {a1}
    assert {p[1] for p in points} == {a2}
    assert {p[2] for p in points} == set(partners)
    assert marks[a1] == "◎" and marks[a2] == "○"
    assert all(marks[p] == "△" for p in partners)


def test_trifecta_stake_uses_budget_split() -> None:
    """賭け金は予算枠を点数で割った額（固定額ではない）。"""
    partners = [2, 3, 4, 5]
    legs, _ = _build_trifecta_head_legs(CFG_7C, 1, 7, partners)
    assert legs[0].stake_per_line == _stake_per_line(CFG_7C, len(partners))


def test_switch_depends_only_on_axis1_win_prob() -> None:
    """判定は軸1の単勝率のみ。軸2や相手の値を変えても結果が動かない。"""
    below = {1: RANK_7C_TRIFECTA_PW_MIN - 0.01, 2: 0.10, 3: 0.05}
    at = {1: RANK_7C_TRIFECTA_PW_MIN, 2: 0.10, 3: 0.05}
    assert rank_7c_use_trifecta(below, 1) is False
    assert rank_7c_use_trifecta(at, 1) is True
    # 軸2以下をどう動かしても閾値ちょうどなら True のまま
    assert rank_7c_use_trifecta({**at, 2: 0.0, 3: 0.0}, 1) is True
    # 情報が無いときは切り替えない（検証済みの既定＝三連複へ倒す）
    assert rank_7c_use_trifecta(None, 1) is False
    assert rank_7c_use_trifecta({}, 1) is False


def test_switching_ranks_have_no_bet_kind_claim_in_comment() -> None:
    """切替キーを持つランクの文面が、券種を断定していないこと。

    同じランクが三連複と三連単を出し分けるので、**どちらかを名指しした瞬間に
    片方で嘘になる**。2026-08-09 に【この買い目について】を全ランクから削除した
    結果、既定文（DBテンプレート）は券種に言及しなくなり専用文面が不要になった。

    ⚠️ ランク固有の `default_comment` を後から足すときはここに引っかかる。
       券種を書きたくなったら、切替を前提に**両方成り立つ書き方**にすること。
    """
    for rank_key, cfg in RANK_CONFIGS.items():
        if not cfg.get("trifecta_switch_key"):
            continue
        assert "trifecta_comment" not in cfg, (
            f"{rank_key}: 専用文面は廃止済み（既定文が券種に言及しないため不要）"
        )
        tpl = cfg.get("default_comment") or ""
        for bad in ("買い目は三連複", "三連複・軸2車流し", "買い目は三連単"):
            assert bad not in tpl, (
                f"{rank_key}: 券種を出し分けるランクの文面が『{bad}』と断定している"
            )


def _process_rank_source() -> ast.FunctionDef:
    src = (Path(__file__).parent.parent / "scripts" / "netkeirin_submit_wt.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_process_rank":
            return node
    raise AssertionError("_process_rank が見つからない")


def test_send_and_record_branch_on_legs_not_flags() -> None:
    """送信・記録の分岐が「legs の有無」であること（フラグ列挙に戻さない）。

    `submit_pick` は cfg["bet_kind"] で券種を決めるので、legs を組んだ買い目を
    そこへ流すと**別の券種が入稿される**。個別フラグの列挙は増えるたびに
    漏れる（9H1 で実際に事故）。構造をパースして固定する。
    """
    fn = _process_rank_source()
    tests = [n.test for n in ast.walk(fn) if isinstance(n, ast.If)]
    tests += [n.test for n in ast.walk(fn) if isinstance(n, ast.IfExp)]
    plain_legs = [t for t in tests if isinstance(t, ast.Name) and t.id == "legs"]
    assert len(plain_legs) >= 2, (
        "送信分岐と記録分岐の両方が `if legs` / `legs if legs else ...` に"
        "なっていること（フラグの列挙に戻すと新ランクで漏れる）"
    )
