"""`wave_picks_wt` のローカル変数が「代入より前に読まれていない」ことを検査する。

## なぜ要るのか

2026-08-07、`src/cli/main.py` の 7C 低配当パターン判定が、4行あとで代入される
`_lg`（車番→line_group）を先に読んでいた（c713d92）。

    lowpay_7c = rank_7c_is_lowpay_pattern(top3_probs, _lg)   # ← ここで読む
    ...
    _lg = {int(r.frame_no): ... }                            # ← ここで代入

レースのループの中なので、**初回は `UnboundLocalError` で必ず落ち**、
仮に落ちなくても2周目以降は**前のレースの line_group** を読む。
当日16:00の第2パスが丸ごと失敗し、翌朝の候補生成も落ちる状態だった。

`wave_picks_wt` は DataFrame と学習済みモデルが要るためユニットテストが無く、
**この関数は本番でしか実行されない**。だから壊れても CI では分からなかった。
実行せずに済む静的検査で、同じ型だけでも塞いでおく。

## 何を見るか

`wave_picks_wt` の中で、その関数のローカル変数（＝関数内で代入されている名前）が
**最初に代入される行より前で読まれていないか**を見る。
`_` で始まる作業用ローカルに限定する（引数・import・グローバルは対象外で、
「後で代入して次の周回で使う」意図的な蓄積変数も普通この命名にはしない）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN = ROOT / "src" / "cli" / "main.py"
TARGET_FUNCS = ("wave_picks_wt",)


def _find_func(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} が {MAIN} に見つかりません")


# 内包表記と入れ子関数は**別スコープ**なので中へ入らない。
# 内包表記はソース上「要素の式」が `for` 句より前に現れるため、素朴に walk すると
# `[f(_r) for _r in xs]` を「代入より前に読んでいる」と誤検出する（実際に踏んだ）。
_OWN_SCOPE = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
              ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _first_store_and_load(func: ast.FunctionDef) -> dict[str, tuple[int | None, int | None]]:
    """関数**自身のスコープ**のローカル名について (最初の代入行, 最初の読み取り行)。"""
    stores: dict[str, int] = {}
    loads: dict[str, int] = {}

    def visit(node: ast.AST, top: bool = False) -> None:
        if not top and isinstance(node, _OWN_SCOPE):
            return
        if isinstance(node, ast.Name):
            tgt = stores if isinstance(node.ctx, ast.Store) else loads
            if isinstance(node.ctx, (ast.Store, ast.Load)):
                tgt[node.id] = min(tgt.get(node.id, node.lineno), node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(func, top=True)
    return {n: (stores.get(n), loads.get(n)) for n in set(stores) | set(loads)}


@pytest.mark.parametrize("func_name", TARGET_FUNCS)
def test_作業用ローカルが代入より前に読まれていない(func_name):
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    func = _find_func(tree, func_name)
    offenders = []
    for name, (store, load) in _first_store_and_load(func).items():
        if not name.startswith("_") or name.startswith("__"):
            continue
        if store is None or load is None:
            continue
        if load < store:
            offenders.append(f"{name}: {load}行で読んでいるが代入は{store}行")
    assert not offenders, (
        f"{func_name} で代入より前に読まれているローカルがあります "
        f"（ループ内なら初回 UnboundLocalError／2周目以降は前レースの値）:\n  "
        + "\n  ".join(offenders)
    )


def test_検査そのものが機能する():
    """わざと壊した関数を弾けることを確かめる（検査が空振りしていないこと）。"""
    src = (
        "def f(rows):\n"
        "    for r in rows:\n"
        "        y = g(_lg)\n"
        "        _lg = {r: 1}\n"
    )
    func = _find_func(ast.parse(src), "f")
    store, load = _first_store_and_load(func)["_lg"]
    assert load < store
