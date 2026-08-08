"""trio_payout / trifecta_payout の意味が全ランクで揃っていることの検査（2026-08-08）。

## 契約

`picks_history.trio_payout` / `trifecta_payout` は
**「そのレースの100円あたり確定払戻」で賭け金に依存しない生の値**。
`scripts/migrate_picks_history_stake.py` の docstring が正本。
実際に受け取った金額は `payout` に入る。

## 背景（実際に起きた不整合）

7H1 だけがこの列へ**券種別の実払戻額**を書いていた（`h1_trio_odds * u_trio // 100`）。
同じ列が他ランクと違う意味になるため、kiseki の Web は両者を同じ
`PayoutInfo` で描いて **「✓¥19,780 複¥19,780」と同じ額を2度**出していた。
さらに片方（三連単）は DB が0だったため API のフォールバックが
オッズ×100 で埋めており、**1行の中に単位の違う2つの数字が同居**していた。

2026-08-08 に追加した 9H1 の採点も、雛形にした 7H1 からこの誤りを引き継いでいた。

## 何を守るか

`history.append` が trio_payout / trifecta_payout の位置へ渡す値は、
**必ず払戻マップ `pm` から取った変数（または単一券種の 0）**であること。
`*_pay`（賭け金を掛けた実額）を渡してはいけない。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "scripts" / "notify_results_wt.py"

# history タプルの並び:
#   (date, race_key, rank, pred, n_combos, hit, payout, trio_payout, trifecta_payout, bet, ...)
_IDX_PAYOUT = 6
_IDX_TRIO = 7
_IDX_TRIFECTA = 8


def _tree() -> ast.Module:
    return ast.parse(RESULTS.read_text(encoding="utf-8"))


def _odds_variable_names(tree: ast.Module) -> set[str]:
    """`X = pm.get(...)` で払戻マップから取り出している変数名。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        src = ast.unparse(node.value)
        if src.startswith("pm.get(") or ".get((\"trio\"" in src or ".get((\"trifecta\"" in src:
            names.add(target.id)
    return names


def _history_rows(tree: ast.Module) -> list[tuple[str, ast.Tuple]]:
    """history.append((...)) のタプルを (ランク名, ノード) で返す。"""
    rows: list[tuple[str, ast.Tuple]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "append"):
            continue
        if not (isinstance(fn.value, ast.Name) and fn.value.id == "history"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Tuple):
            continue
        tup = node.args[0]
        if len(tup.elts) <= _IDX_TRIFECTA:
            continue
        rank = next(
            (e.value for e in tup.elts[:4]
             if isinstance(e, ast.Constant) and str(e.value).startswith("RANK_")),
            "?",
        )
        rows.append((str(rank), tup))
    return rows


def test_payout_columns_are_odds_not_amounts():
    """全ランクで trio/trifecta 列に「100円あたり配当」を入れていること。"""
    tree = _tree()
    odds_names = _odds_variable_names(tree)
    assert odds_names, "pm.get(...) の代入を1件も拾えていない（パース失敗）"

    bad: list[str] = []
    for rank, tup in _history_rows(tree):
        for idx, label in ((_IDX_TRIO, "trio_payout"), (_IDX_TRIFECTA, "trifecta_payout")):
            arg = tup.elts[idx]
            if isinstance(arg, ast.Constant) and arg.value == 0:
                continue  # 単一券種で常に0
            if isinstance(arg, ast.Name) and arg.id in odds_names:
                continue
            bad.append(f"{rank}.{label} = {ast.unparse(arg)}")

    assert not bad, (
        "trio_payout / trifecta_payout は「100円あたりの確定配当」でなければならない。\n"
        "賭け金を掛けた実額を入れると、同じ列がランクごとに違う意味になり\n"
        "Web が実額と配当を混ぜて表示する:\n  " + "\n  ".join(bad)
    )


def test_payout_column_differs_from_the_actual_amount():
    """payout（実額）と同じ変数を配当列へ渡していないこと。

    9H1 で実際に踏んだ形（`h9_pay, 0, h9_pay`）を直接禁じる。
    """
    offenders: list[str] = []
    for rank, tup in _history_rows(_tree()):
        amount = tup.elts[_IDX_PAYOUT]
        if not isinstance(amount, ast.Name):
            continue
        for idx, label in ((_IDX_TRIO, "trio_payout"), (_IDX_TRIFECTA, "trifecta_payout")):
            arg = tup.elts[idx]
            if isinstance(arg, ast.Name) and arg.id == amount.id:
                offenders.append(f"{rank}.{label} が payout と同一 ({arg.id})")
    assert not offenders, "\n".join(offenders)


def test_guard_catches_the_original_7h1_shape():
    """検査が空振りしていないこと（旧 7H1 の形を与えると弾かれる）。"""
    tree = _tree()
    odds_names = _odds_variable_names(tree)
    # 旧実装は h1_pay_trio / h1_pay_tf（実額）を渡していた
    assert "h1_pay_trio" not in odds_names
    assert "h1_pay_tf" not in odds_names
    assert "h1_trio_odds" in odds_names
    assert "h1_tf_odds" in odds_names


# ---------------------------------------------------------------------------
# 過去分再構築（backfill_*_rank_wt.py）も同じ契約に従うこと
#
# ⚠️ ここが抜けていると、あとで実行する再構築が本番の是正を**巻き戻す**。
#    実際 backfill_7h1_rank_wt.py は notify_results_wt と同じ誤り（実額を書く）を
#    持っており、7H1 の過去分再構築が予定されていた（2026-08-08 是正）。
# ---------------------------------------------------------------------------

_BACKFILLS = sorted((ROOT / "scripts").glob("backfill_*_rank_wt.py"))


def _dict_payout_args(path: Path) -> list[tuple[str, str, set[str]]]:
    """dict リテラル中の trio_payout / trifecta_payout の (キー, 式, 参照名)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value in ("trio_payout", "trifecta_payout"):
                names = {n.id for n in ast.walk(v) if isinstance(n, ast.Name)}
                out.append((str(k.value), ast.unparse(v), names))
    return out


def test_backfills_write_odds_not_amounts():
    """再構築スクリプトも配当（pm 由来）を書くこと。"""
    assert _BACKFILLS, "backfill_*_rank_wt.py を1本も見つけられていない"

    bad: list[str] = []
    for path in _BACKFILLS:
        odds_names = _odds_variable_names(ast.parse(path.read_text(encoding="utf-8")))
        for key, expr, names in _dict_payout_args(path):
            if expr == "0" or (names & odds_names):
                continue
            bad.append(f"{path.name}: {key} = {expr}")

    assert not bad, (
        "再構築が配当列へ実額を書いている。これを直さないと過去分の再実行が\n"
        "本番の是正を巻き戻す:\n  " + "\n  ".join(bad)
    )
