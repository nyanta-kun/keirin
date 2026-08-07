"""入稿した買い目・金額配分の記録（`build_bet_detail`）のテスト。

Web は**この記録を読むだけ**で買い目を表示する。傾斜配分の金額は入稿時点の
想定オッズから決まるため**あとから再現できない**ので、ここが唯一の正本になる。
守るのは3点:

  1. 買い目が**展開済み**であること（表示側が展開ロジックを再実装しないため）
  2. 金額の合計が実際の入稿額と一致すること
  3. 均等配分ランク（`submit_pick` 経路）でも同じ形になること
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.netkeirin_submit_wt import (  # noqa: E402
    RANK_CONFIGS,
    _legs_for_record,
    build_bet_detail,
)
from src.netkeirin_client import (  # noqa: E402
    BET_KIND_TRIFECTA_FORMATION,
    BET_KIND_TRIO_AXIS2,
    BET_KIND_TRIO_BOX,
    BetLeg,
)


def _detail(legs, source=None, odds=None):
    return json.loads(build_bet_detail(legs, source, odds))


def test_三連複軸2車が展開されて金額とともに並ぶ():
    legs = [BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3, 4]], 2500)]
    d = _detail(legs, "blend")
    assert d["source"] == "blend"
    assert d["total"] == 5000
    assert sorted(x["combo"] for x in d["lines"]) == ["1=2=3", "1=2=4"]
    assert all(x["stake"] == 2500 for x in d["lines"])
    assert all(x["bet_type"] == "3連複" for x in d["lines"])


def test_点ごとに金額が違う傾斜配分を表現できる():
    """同額どうしをまとめた複数行から、点ごとの金額へ戻せること。"""
    legs = [
        BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [5]], 4100),
        BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3, 4]], 2000),
        BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [6]], 1900),
    ]
    d = _detail(legs, "blend")
    got = {x["combo"]: x["stake"] for x in d["lines"]}
    assert got == {"1=2=5": 4100, "1=2=3": 2000, "1=2=4": 2000, "1=2=6": 1900}
    assert d["total"] == 10000


def test_三連単は着順つきで表記が変わる():
    legs = [BetLeg(BET_KIND_TRIFECTA_FORMATION, [[3], [4], [1, 2]], 900)]
    d = _detail(legs)
    assert sorted(x["combo"] for x in d["lines"]) == ["3-4-1", "3-4-2"]
    assert all(x["bet_type"] == "3連単" for x in d["lines"])
    # 3連複は "=", 3連単は "-"（着順の有無が読み取れること）
    assert all("-" in x["combo"] for x in d["lines"])


def test_2券種併買は両方が1つの記録に入る():
    legs = [
        BetLeg(BET_KIND_TRIFECTA_FORMATION, [[3], [4], [1, 2]], 900),
        BetLeg(BET_KIND_TRIO_BOX, [[1, 2, 3, 4]], 200),
    ]
    d = _detail(legs)
    assert {x["bet_type"] for x in d["lines"]} == {"3連単", "3連複"}
    assert d["total"] == 900 * 2 + 200 * 4     # 三連単2点 + BOX4点


def test_均等配分ランクも同じ形になる():
    """`submit_pick` 経路（7B 等）でも表示側の扱いが変わらないこと。"""
    cfg = RANK_CONFIGS["7B"]
    legs = _legs_for_record(cfg, 1, 3, [4, 5, 6], 3300)
    d = _detail(legs)
    assert d["source"] is None
    assert sorted(x["combo"] for x in d["lines"]) == ["1=3=4", "1=3=5", "1=3=6"]
    assert d["total"] == 9900


def test_合成した買い目は本番の入稿と同じ点数になる():
    """`_legs_for_record` が組む groups は `build_bet_id` と同じでなければならない。"""
    cfg = RANK_CONFIGS["7A"]
    partners = [3, 4, 5, 6, 7]
    legs = _legs_for_record(cfg, 1, 2, partners, 2000)
    d = _detail(legs)
    assert len(d["lines"]) == len(partners)


@pytest.mark.parametrize("source", ["blend", "odds", "model", "equal", None])
def test_配分の出どころをそのまま持つ(source):
    legs = [BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3]], 10000)]
    assert _detail(legs, source)["source"] == source


def test_JSONは日本語をエスケープしない():
    """DB を直接読んだときに券種が読めること。"""
    raw = build_bet_detail([BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3]], 100)])
    assert "3連複" in raw


# ── 入稿時点のオッズ（2026-08-07 追加）────────────────────────────────

def test_オッズを渡すと買い目に添えられる():
    """**配分の根拠そのものなので一緒に保存する。** あとから引くと発走時の値に
    なってしまい「なぜこの金額なのか」が読めなくなる。"""
    legs = [BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3, 4]], 2500)]
    odds = {frozenset({1, 2, 3}): 8.34, frozenset({1, 2, 4}): 21.0}
    d = _detail(legs, "blend", odds)
    got = {x["combo"]: x["odds"] for x in d["lines"]}
    assert got == {"1=2=3": 8.3, "1=2=4": 21.0}      # 小数第1位へ丸める


def test_三連単のオッズはtupleキーで引く():
    legs = [BetLeg(BET_KIND_TRIFECTA_FORMATION, [[3], [4], [1]], 500)]
    d = _detail(legs, None, {(3, 4, 1): 128.5})
    assert d["lines"][0] == {"bet_type": "3連単", "combo": "3-4-1",
                             "stake": 500, "odds": 128.5}


def test_オッズが取れなければNoneで残す():
    """欠損を 0 や省略にすると表示側で「オッズ0倍」と読めてしまう。"""
    legs = [BetLeg(BET_KIND_TRIO_AXIS2, [[1], [2], [3]], 10000)]
    assert _detail(legs)["lines"][0]["odds"] is None
    assert _detail(legs, None, {frozenset({1, 2, 3}): 0})["lines"][0]["odds"] is None
