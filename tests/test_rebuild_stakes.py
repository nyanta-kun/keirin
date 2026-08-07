"""再構築の傾斜配分（src/rebuild_stakes.py）のテスト。

## なぜこのテストが要るのか

2026-08-07、7ランク中**5つ**で候補dictへの `top3_probs` 登録が漏れていた。
`stakes_for_combos` は p3 が空だと本番と同じく**均等へフォールバック**するため、
**エラーも警告も出ないまま11時間の全期間再構築が丸ごと均等配分で走った**。
気づいたのは実質的中率が想定 +7pt に対し **+0.22pt** しか動かなかったからで、
数字を突き合わせなければ最後まで分からなかった。

→ 再構築では p3 は**必ず手元にある**ので、空なら呼び出し側のバグ。
   本番（入稿時に予測が読めないことがある）とは事情が違うので、
   **再構築側は落とす**。この検査はその契約を固定する。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rebuild_stakes import stakes_for_combos  # noqa: E402
from src.strategy_wt import RACE_BUDGET  # noqa: E402

# 傾斜配分の対象ランク（`netkeirin_submit_wt.RANK_CONFIGS` の tilt_stakes と対応）
TILTED_BACKFILLS = ("7ss", "7s", "7a", "7b", "7c", "9s", "9a")


def _combos(axis1, axis2, thirds):
    return [frozenset({axis1, axis2, t}) for t in thirds]


def test_p3があれば低確率の相手ほど薄くなる():
    combos = _combos(1, 2, [3, 4, 5])
    st = stakes_for_combos(1, 2, combos, {1: .8, 2: .7, 3: .6, 4: .3, 5: .1})
    got = {next(iter(c - {1, 2})): v for c, v in st.items()}
    assert got[3] > got[4] > got[5]
    assert sum(st.values()) == RACE_BUDGET


def test_朝オッズがあれば併用する():
    combos = _combos(1, 2, [3, 4])
    board = {frozenset({1, 2, 3}): 3.0, frozenset({1, 2, 4}): 30.0}
    st = stakes_for_combos(1, 2, combos, {1: .8, 2: .7, 3: .5, 4: .5}, board)
    got = {next(iter(c - {1, 2})): v for c, v in st.items()}
    assert got[3] > got[4]          # 低オッズ側が厚い
    assert sum(st.values()) == RACE_BUDGET


def test_朝オッズが一部しか無ければモデルだけで決める():
    """一部だけ使うと点どうしの比率が壊れる。"""
    combos = _combos(1, 2, [3, 4])
    board = {frozenset({1, 2, 3}): 3.0}        # 4 が無い
    st = stakes_for_combos(1, 2, combos, {1: .8, 2: .7, 3: .5, 4: .5}, board)
    got = {next(iter(c - {1, 2})): v for c, v in st.items()}
    assert got[3] == got[4]        # p3 が同じなら同額（オッズは使われていない）


# ── 🔴 無言のフォールバックを禁じる ──────────────────────────────────

def test_p3が空なら落とす():
    """**均等へ黙って落ちてはいけない。** 落ちないと再構築が丸ごと均等で走る。"""
    with pytest.raises(ValueError, match="top3_probs"):
        stakes_for_combos(1, 2, _combos(1, 2, [3, 4]), {})


def test_買う相手のp3が欠けていても落とす():
    with pytest.raises(ValueError, match="top3_probs"):
        stakes_for_combos(1, 2, _combos(1, 2, [3, 4]), {1: .8, 2: .7, 3: .5})


@pytest.mark.parametrize("rank", TILTED_BACKFILLS)
def test_候補dictにtop3_probsを載せている(rank):
    """`stakes_for_combos` を使う backfill は候補dictへ `top3_probs` を載せること。

    載せ忘れると（2026-08-07 に5ランクで実際に起きた）実行時まで気づけない。
    ここは**静的に**見るので、重い再構築を回さなくても検出できる。
    """
    src = (ROOT / "scripts" / f"backfill_{rank}_rank_wt.py").read_text(encoding="utf-8")
    assert "stakes_for_combos" in src, f"{rank} が傾斜配分を使っていません"
    assert '"top3_probs": top3_probs' in src, (
        f"backfill_{rank}_rank_wt.py の candidates.append に "
        f'"top3_probs": top3_probs がありません（均等配分で走ってしまう）')


@pytest.mark.parametrize("rank", TILTED_BACKFILLS)
def test_candidates_appendの中に入っている(rank):
    """辞書リテラルの**中**にあること（コメント等に書いてあるだけでは意味がない）。"""
    tree = ast.parse((ROOT / "scripts" / f"backfill_{rank}_rank_wt.py")
                     .read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "top3_probs" for k in node.keys
        ):
            found = True
            break
    assert found, f"{rank}: top3_probs が辞書リテラルのキーになっていません"
