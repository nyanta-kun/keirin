"""`scripts/reconcile_walkforward_tail.sh` の登録内容を読むための共通ヘルパ。

毎朝 08:30 の tail reconcile は「当月分を月次凍結 vintage モデルで再構築する」
処理で、ここに登録し忘れたランクだけ live 行が残り、過去期間と条件が食い違う
（2026-08-06 に 7A/7B で発生。2026-08-08 に 7H1 で再発を検出）。

⚠️ **全文の文字列一致で判定してはいけない。**
   同ファイルのコメントには「実装したら "7h1:7H1" を足すこと」という TODO が
   書かれており、`'"7h1:7H1"' in sh` は**未登録のまま真になる**。
   実際それで安全網が丸ごと無効化されていた。必ず for 行をパースすること。
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_RECONCILE = _REPO / "scripts" / "reconcile_walkforward_tail.sh"
_SCRIPTS = _REPO / "scripts"

# 実体はあるが意図的に tail reconcile から外しているもの。
# 外す判断をしたときは必ず理由を書くこと（書かないと「漏れ」と区別できない）。
_INTENTIONALLY_UNREGISTERED = {
    # SEVEN_S1 は 2026-07-31 にユーザー判断で全廃し picks_history からも削除済み。
    # 登録すると tail 再構築のたびに削除済みの行が DELETE→INSERT で復活する。
    # スクリプト本体は過去日の再採点・分析用に手動実行専用として残置している。
    "s1",
}

# `for spec in "7ss:7SS" "7s:7S" ... ; do` の行を拾う
_FOR_LINE = re.compile(r"^\s*for\s+spec\s+in\s+(.+?);\s*do\s*$", re.MULTILINE)
_SPEC = re.compile(r'"([0-9a-z]+):([0-9A-Z]+)"')


def reconcile_specs() -> dict[str, str]:
    """for 行に登録された {スクリプト接尾辞: ランク表示名} を返す。

    例: {"7ss": "7SS", "7s": "7S", ..., "7h1": "7H1"}
    """
    text = _RECONCILE.read_text(encoding="utf-8")
    m = _FOR_LINE.search(text)
    if m is None:
        raise AssertionError(
            "reconcile_walkforward_tail.sh に `for spec in ...; do` が見つからない。"
            " ループの書き方を変えたなら本ヘルパも追随させること"
            "（黙って空dictを返すと登録漏れ検査が全部素通りする）")
    specs = dict(_SPEC.findall(m.group(1)))
    if not specs:
        raise AssertionError(f"for 行から spec を1件も抽出できなかった: {m.group(1)!r}")
    return specs


def rebuild_scripts() -> set[str]:
    """`scripts/rebuild_<x>_walkforward_pg.py` として実在する <x> の集合。"""
    return {
        p.name[len("rebuild_"):-len("_walkforward_pg.py")]
        for p in _SCRIPTS.glob("rebuild_*_walkforward_pg.py")
    }
