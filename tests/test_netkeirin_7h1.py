"""7H1（穴推奨・本命バスト型）の netkeirin 入稿変換のテスト（2026-08-06）。

7H1 は **三連単フォーメーション + 三連複BOX の2券種**を1商品として入稿する
唯一のランク。入稿側は候補JSONの `legs_tf` / `legs_trio`（strategy_wt の
`rank_7h1_build_legs()` が生成した実際の買い目）を**正**とし、そこから
車番グループを復元する。

ここで守るのは1点だけ:
  **復元したグループを展開し直した目集合が、元の買い目と完全一致すること。**
一致しないまま入稿すると、意図と違う買い目が外部（＝有料商品）へ出る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import src.strategy_wt as sw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.netkeirin_submit_wt import (  # noqa: E402
    RANK_CONFIGS,
    _normalize_multi_candidate,
    _trifecta_formation_groups,
    _trio_box_group,
)
from src.netkeirin_client import (  # noqa: E402
    ACT_TYPE_LONGSHOT,
    BET_KIND_TRIFECTA_FORMATION,
    BET_KIND_TRIO_BOX,
    expand_bet,
)
from src.preprocessing.favbust_features import (  # noqa: E402
    ROLE_FAV_MATE,
    ROLE_LEAD_TOP,
    ROLE_OTHER_MATE,
)
from src.strategy_wt import rank_7h1_build_legs  # noqa: E402


def _legs_from_strategy(others: list[int], roles: dict[int, str]):
    """本番の買い目生成をそのまま通し、候補JSONと同じ文字列表現にする。"""
    trio, tf = rank_7h1_build_legs(others, roles)
    return ["=".join(str(x) for x in sorted(t)) for t in trio], tf


# 7車・本命=7（本命ラインは 7と6）。others はモデル3着内率の降順。
# roles は favbust_features.roles_of() の戻り値と同じ語彙。
_ROLE_LEAD_TOP = ROLE_LEAD_TOP
_FAV_LINE = ROLE_FAV_MATE
_OTHER = ROLE_OTHER_MATE


def test_formation_groups_roundtrip_on_real_shape():
    """本番の rank_7h1_build_legs() が作る8点が、フォーメーション復元で再現されること。"""
    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    _, tf = _legs_from_strategy(others, roles)
    assert len(tf) == 8

    groups = _trifecta_formation_groups(tf)
    assert groups[0] == [3]                      # 1着＝別ライン先頭で固定
    assert len(groups[1]) == 2                   # 2着＝プール上位2車
    assert groups[2] == sorted(set(others) - {3})  # 3着＝本命以外の総流し


def test_trio_box_roundtrip_on_real_shape():
    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, _ = _legs_from_strategy(others, roles)
    cars = _trio_box_group(trio)
    # 本命ライン（6）を落としたプール上位5車のBOX＝10点
    assert cars == [1, 2, 3, 4, 5]
    assert len(trio) == 10


def test_trio_box_four_cars_when_fav_line_has_three():
    """本命ラインが3車のレースはプールが4車になり、BOXは4点になる。"""
    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER,
             2: _FAV_LINE, 6: _FAV_LINE}
    trio, _ = _legs_from_strategy(others, roles)
    assert len(trio) == 4
    assert _trio_box_group(trio) == [1, 3, 4, 5]


def test_formation_groups_rejects_non_expandable_legs():
    """フォーメーションで表現できない目の集合は復元させない（黙って通さない）。"""
    # 1着[3]×2着[4,5]×3着[1,2] を展開すると4点になるが、ここでは3点しか無い
    with pytest.raises(ValueError, match="一致しません"):
        _trifecta_formation_groups(["3-4-1", "3-4-2", "3-5-1"])


def test_trio_box_rejects_non_box_legs():
    """BOXで表現できない目の集合は復元させない。"""
    with pytest.raises(ValueError, match="一致しません"):
        _trio_box_group(["1=2=3", "1=2=4"])   # {1,2,3,4} のBOXなら4点必要


def test_formation_groups_rejects_bad_format():
    with pytest.raises(ValueError):
        _trifecta_formation_groups(["3-4"])
    with pytest.raises(ValueError):
        _trifecta_formation_groups([])


def test_normalize_multi_candidate_builds_legs_and_marks():
    """候補JSON1件から (三連単F + 三連複を目ごとに1行) と印が組み上がること。

    2026-08-07: 三連複は**目ごとに金額が違う**ので 1目=1行（3車のBOX＝1点）で出す。
    同額でも束ねられない（BOXは車群でしか表現できず任意の部分集合を作れない）。

    印はユーザー確定の規則（2026-08-06）:
      ◎=三連単の1着固定車 / ○=2着列1番手 / ▲=2着列2番手 /
      △=3着だけで買っている車 / 除外した本命は印なし
    """
    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, tf = _legs_from_strategy(others, roles)
    cand = {
        "race_key": "20260807_85_07", "venue_name": "佐世保", "race_no": 7,
        "fav": 7, "others": others,
        "legs_tf": tf, "legs_trio": trio,
        "stake_tf": 900, "stake_trio": 200,
    }
    legs, marks, axis1, axis2, source = _normalize_multi_candidate(
        cand, RANK_CONFIGS["7H1"])

    # 先頭が三連単フォーメーション、以降は三連複を1目ずつ
    assert legs[0].bet_kind == BET_KIND_TRIFECTA_FORMATION
    assert all(leg.bet_kind == BET_KIND_TRIO_BOX for leg in legs[1:])
    assert len(legs) - 1 == len(trio)
    assert all(len(leg.groups[0]) == 3 for leg in legs[1:])   # 3車＝1点
    # 三連単は固定額の均等。三連複は残りを使い切る。
    assert legs[0].stake_per_line == sw.RANK_7H1_TF_UNIT
    assert source == "equal"        # オッズを渡していないので均等へ落ちる
    total = (legs[0].stake_per_line * len(tf)
             + sum(leg.stake_per_line for leg in legs[1:]))
    assert total == sw.RANK_7H1_BUDGET_CAP

    assert axis1 == 3 and marks[3] == "◎"
    assert marks[axis2] == "○"
    assert sorted(marks) == [1, 2, 3, 4, 5, 6]   # 本命(7)には印を付けない
    assert 7 not in marks
    assert marks[4] == "○" and marks[5] == "▲"   # others 順で ○/▲ を割り当てる
    assert marks[1] == marks[2] == marks[6] == "△"


def test_normalize_multi_candidate_marks_follow_others_order_not_car_number():
    """○/▲ は車番順ではなく others（モデル3着内率の降順）順で決まること。"""
    others = [3, 5, 4, 1, 2, 6]     # プール上位は 5 → 4 の順
    roles = {3: _ROLE_LEAD_TOP, 5: _OTHER, 4: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, tf = _legs_from_strategy(others, roles)
    cand = {"race_key": "x", "others": others, "legs_tf": tf, "legs_trio": trio,
            "stake_tf": 900, "stake_trio": 200}
    _, marks, _, _, _ = _normalize_multi_candidate(cand, RANK_CONFIGS["7H1"])
    assert marks[5] == "○" and marks[4] == "▲"


def _leg_odds(leg, trio_board, tf_board):
    """1点=1行の BetLeg からその点のオッズを引く（三連複 / 三連単）。"""
    from src.netkeirin_client import BET_KIND_TRIO_BOX

    if leg.bet_kind == BET_KIND_TRIO_BOX:
        return trio_board[frozenset(leg.groups[0])]
    return tf_board[tuple(g[0] for g in leg.groups)]


def test_normalize_multi_candidate_falls_back_when_odds_incomplete(monkeypatch):
    """オッズが一部でも欠けたらダッチにせず従来配分へ戻す。

    🔴 欠けたままダッチに入れると「安いから切った」のか「板が無いから消えた」のか
       区別できず、**三連単オッズだけ無いときに 7H1 が三連複単券種になる**。
    """
    import scripts.netkeirin_submit_wt as sub

    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, tf = _legs_from_strategy(others, roles)
    keys = [frozenset(int(x) for x in c.split("=")) for c in trio]
    board = {k: 20.0 + 10 * i for i, k in enumerate(keys)}
    monkeypatch.setattr(sub, "_load_trio_board", lambda rk: board)
    monkeypatch.setattr(sub, "_load_trifecta_board", lambda rk: {})   # 三連単板が無い

    cand = {"race_key": "x", "others": others, "legs_tf": tf, "legs_trio": trio,
            "stake_tf": 0, "stake_trio": 0}
    legs, _, _, _, source = _normalize_multi_candidate(
        cand, RANK_CONFIGS["7H1"], "20260807_85_07")
    assert not source.startswith("dutch:")
    # 2券種のまま（三連単の行が残っている）
    kinds = {leg.bet_kind for leg in legs}
    assert BET_KIND_TRIFECTA_FORMATION in kinds


def test_normalize_multi_candidate_uses_odds_when_available(monkeypatch):
    """オッズが買う目**すべて**に揃えば払戻が等しくなるよう配分すること。

    ⚠️ 候補JSONの `stake_tf`/`stake_trio` は**もう使わない**（朝の候補生成時点の値で、
       板が育ってから入稿する3波の設計と時点が食い違うため）。
    """
    import scripts.netkeirin_submit_wt as sub

    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, tf = _legs_from_strategy(others, roles)
    keys = [frozenset(int(x) for x in c.split("=")) for c in trio]
    board = {k: 20.0 + 10 * i for i, k in enumerate(keys)}
    monkeypatch.setattr(sub, "_load_trio_board", lambda rk: board)
    # 🔴 三連単板も**明示的に**与える。ここを実DBに任せると、板を持つ環境では
    #    ダッチ経路・持たない環境（CI）ではフォールバック経路と、同じテストが
    #    環境によって別の物を検査してしまう。
    tf_points = expand_bet(BET_KIND_TRIFECTA_FORMATION, _trifecta_formation_groups(tf))
    tf_board = {p: 60.0 + 5 * i for i, p in enumerate(sorted(tf_points))}
    monkeypatch.setattr(sub, "_load_trifecta_board", lambda rk: tf_board)

    cand = {"race_key": "x", "others": others, "legs_tf": tf, "legs_trio": trio,
            "stake_tf": 0, "stake_trio": 0}     # 候補JSONの値は無視される
    legs, _, _, _, source = _normalize_multi_candidate(
        cand, RANK_CONFIGS["7H1"], "20260807_85_07")

    # 【2026-08-09・STEP3 §2B】オッズが全点に揃うとダッチ配分になる。
    # ダッチは低オッズ目を切ったうえで**採用した目の払戻を予算の1.3倍以上**に
    # そろえるので、旧「傾斜配分」より強い条件になる。守る中身は同じ
    # （オッズの逆数へ寄せる）だが、1点=1行になり券種混在で並ぶ点が違う。
    assert source.startswith("dutch:")
    total = sum(leg.stake_per_line for leg in legs)
    assert total <= sw.RANK_7H1_BUDGET_CAP
    assert all(leg.stake_per_line <= 5_000 for leg in legs)
    # 採用したどの目が来ても予算の1.3倍以上が返る（ダッチの不変条件）
    pays = [leg.stake_per_line * _leg_odds(leg, board, tf_board) for leg in legs]
    assert min(pays) >= total * 1.3


def test_7h1_config_shape():
    """7H1 は2券種ランクなので stake_per_line/bet_kind を持たず multi_bet で分岐する。"""
    cfg = RANK_CONFIGS["7H1"]
    assert cfg["multi_bet"] is True
    assert cfg["n_cars"] == 7
    assert cfg["file_key"] == "s7h1"
    assert "bet_kind" not in cfg and "stake_per_line" not in cfg
    # 勝負アイコンは「穴狙い」（ユーザー確定・2026-08-06）
    assert cfg["act_type"] == ACT_TYPE_LONGSHOT
