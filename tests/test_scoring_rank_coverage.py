"""採点経路の抜けを検出する（2026-08-08）。

## 背景（実際に起きた抜け）

RANK_9H1 は 2026-08-08 に新設され、候補生成・発走前判定・入稿・Web 表示まで
配線されたのに、**`notify_results_wt.py` の採点ブロックだけが無かった**。
`notify_prerace_wt._process_rank_9h1_candidates` は
「採点が legs / stake / bet_amount をここから読むため」とコメントまで書いて
decision を保存していたが、その読み手が存在しなかった。

⚠️ **これは「動かない」より悪い。** 採点されないと picks_history の行は
`hit=0 / payout=0` のまま残り、`bet_amount>0` なのでサマリーには
**純粋な損失として計上される**。実際に 2026-08-08 の 9H1 2件（各9,600円）が
その状態だった。

## 何を守るか

`notify_prerace_wt` が picks_history へ書く `RANK_*` は、必ず
`notify_results_wt` にも現れること（＝採点されて hit/payout が埋まること）。
ランク名の手書き二重管理は本リポジトリで繰り返し事故を起こしている
（netkeirin の RANK_ORDER・_THREE_HEAD_RANKS・モデル配布リスト）。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRERACE = ROOT / "scripts" / "notify_prerace_wt.py"
RESULTS = ROOT / "scripts" / "notify_results_wt.py"

_RANK_RE = re.compile(r'"(RANK_[0-9A-Z]+)"')

# 全廃済みで新規判定を行わないもの（picks_history に残骸はある）。
_ABOLISHED = {"RANK_7SS"}


def _ranks_in(path: Path) -> set[str]:
    return set(_RANK_RE.findall(path.read_text(encoding="utf-8")))


def test_every_judged_rank_is_scored():
    """発走前判定が書くランクは必ず採点側にも存在すること。"""
    judged = _ranks_in(PRERACE) - _ABOLISHED
    scored = _ranks_in(RESULTS)

    missing = sorted(judged - scored)
    assert not missing, (
        "以下のランクは発走前判定で picks_history へ書かれるのに採点されない。\n"
        "このままだと hit=0/payout=0 のまま残り、bet_amount>0 なので\n"
        "サマリーへ純損失として計上される（無言で数字が悪くなる）:\n"
        f"  {missing}\n"
        f"（判定={sorted(judged)} / 採点={sorted(scored)}）"
    )


def test_9h1_is_scored():
    """実際に踏んだ抜け。"""
    assert "RANK_9H1" in _ranks_in(RESULTS)


def test_guard_would_catch_a_removed_rank():
    """検査が空振りしていないこと（採点集合から外すと検出される）。"""
    judged = _ranks_in(PRERACE) - _ABOLISHED
    scored = _ranks_in(RESULTS) - {"RANK_9H1"}
    assert "RANK_9H1" in judged
    assert sorted(judged - scored) == ["RANK_9H1"]
