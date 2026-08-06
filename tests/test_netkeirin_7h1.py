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


def test_normalize_multi_candidate_builds_two_legs_and_marks():
    """候補JSON1件から (三連単F, 三連複BOX) の2行と印が組み上がること。

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
    legs, marks, axis1, axis2 = _normalize_multi_candidate(cand, RANK_CONFIGS["7H1"])

    assert [leg.bet_kind for leg in legs] == [
        BET_KIND_TRIFECTA_FORMATION, BET_KIND_TRIO_BOX]
    assert legs[0].stake_per_line == 900
    assert legs[1].stake_per_line == 200

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
    _, marks, _, _ = _normalize_multi_candidate(cand, RANK_CONFIGS["7H1"])
    assert marks[5] == "○" and marks[4] == "▲"


def test_normalize_multi_candidate_rejects_zero_stake():
    others = [3, 4, 5, 1, 2, 6]
    roles = {3: _ROLE_LEAD_TOP, 4: _OTHER, 5: _OTHER, 1: _OTHER, 2: _OTHER,
             6: _FAV_LINE}
    trio, tf = _legs_from_strategy(others, roles)
    cand = {"race_key": "x", "others": others, "legs_tf": tf, "legs_trio": trio,
            "stake_tf": 0, "stake_trio": 200}
    with pytest.raises(ValueError):
        _normalize_multi_candidate(cand, RANK_CONFIGS["7H1"])


def test_7h1_config_shape():
    """7H1 は2券種ランクなので stake_per_line/bet_kind を持たず multi_bet で分岐する。"""
    cfg = RANK_CONFIGS["7H1"]
    assert cfg["multi_bet"] is True
    assert cfg["n_cars"] == 7
    assert cfg["file_key"] == "s7h1"
    assert "bet_kind" not in cfg and "stake_per_line" not in cfg
    # 勝負アイコンは「穴狙い」（ユーザー確定・2026-08-06）
    assert cfg["act_type"] == ACT_TYPE_LONGSHOT
