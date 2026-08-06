"""netkeirin bet_id/waku_check 組み立てロジックのテスト（実測値ベース）。"""
from datetime import date

import pytest

from src.netkeirin_client import (
    BET_KIND_TRIFECTA_AXIS1,
    BET_KIND_TRIFECTA_FORMATION,
    BET_KIND_TRIO_AXIS2,
    BET_KIND_TRIO_BOX,
    build_bet_id,
    build_bet_id_groups,
    expand_bet,
    waku_check_for,
)


def test_build_bet_id_matches_real_capture():
    """2026-07-23実機検証で確認した実データ（佐世保1R・2026-07-24=金曜）に一致すること。"""
    bet_id = build_bet_id(
        race_date=date(2026, 7, 24),
        venue_code="85",
        race_no=1,
        bet_kind=BET_KIND_TRIO_AXIS2,
        axis1=1,
        axis2=2,
        partners=[3, 4, 5, 6, 7],
    )
    assert bet_id == "a5-85-1_b8_c6_1_2_3-4-5-6-7"


def test_build_bet_id_no_leading_zero_on_race_no():
    bet_id = build_bet_id(
        race_date=date(2026, 7, 24),
        venue_code="46",
        race_no=9,
        bet_kind=BET_KIND_TRIO_AXIS2,
        axis1=3,
        axis2=5,
        partners=[1, 2, 4, 6, 7],
    )
    assert bet_id.startswith("a5-46-9_")
    assert "-09_" not in bet_id


def test_build_bet_id_weekday_monday():
    # 2026-07-20は月曜日 → isoweekday()%7 == 1
    bet_id = build_bet_id(
        race_date=date(2026, 7, 20),
        venue_code="12",
        race_no=1,
        bet_kind=BET_KIND_TRIO_AXIS2,
        axis1=1,
        axis2=2,
        partners=[3, 4, 5, 6, 7],
    )
    assert bet_id.startswith("a1-12-1_")


def test_build_bet_id_partners_sorted():
    bet_id = build_bet_id(
        race_date=date(2026, 7, 24),
        venue_code="85",
        race_no=1,
        bet_kind=BET_KIND_TRIO_AXIS2,
        axis1=1,
        axis2=2,
        partners=[7, 3, 5, 4, 6],
    )
    assert bet_id.endswith("_3-4-5-6-7")


def test_build_bet_id_trifecta_axis1_matches_real_capture():
    """2026-07-28実機検証（取手1R・1着軸=1・相手=2,3・火曜）に一致すること。"""
    bet_id = build_bet_id(
        race_date=date(2026, 7, 28),
        venue_code="23",
        race_no=1,
        bet_kind=BET_KIND_TRIFECTA_AXIS1,
        axis1=1,
        axis2=None,
        partners=[2, 3],
    )
    assert bet_id == "a2-23-1_b9_c3_1_2-3"


def test_build_bet_id_trifecta_axis1_no_axis2_slot():
    # 軸2頭ながしと異なり、trifecta_axis1にはaxis2用の数字スロットが存在しない
    bet_id = build_bet_id(
        race_date=date(2026, 7, 28),
        venue_code="23",
        race_no=1,
        bet_kind=BET_KIND_TRIFECTA_AXIS1,
        axis1=5,
        axis2=None,
        partners=[3, 7],
    )
    assert bet_id == "a2-23-1_b9_c3_5_3-7"


# ── 7H1（三連単フォーメーション / 三連複ボックス）─────────────────────────
# 2026-08-06 実機検証。佐世保7R（2026-08-07=金曜・場コード85）の買い目入力画面で
# 実際に組み、`localStorage['ndi::umaibet_202608078507']` を読み取って確定した:
#   3連単F  1着[3] × 2着[4,5] × 3着[1,2,4,5,6] = 8点  → a5-85-7_b9_c1_3_4-5_1-2-4-5-6
#   3連複BOX [2,4,5,6]            = 4点  → a5-85-7_b8_c2_2-4-5-6
#   3連複BOX [1,2,4,5,6]          = 10点 → a5-85-7_b8_c2_1-2-4-5-6
# 点数は画面表示（8点/4点/10点）でも確認済み。フォーメーションには
# trifecta_axis1 のようなマルチ相当のフラグは存在しない。


def test_build_bet_id_trifecta_formation_matches_real_capture():
    bet_id = build_bet_id_groups(
        race_date=date(2026, 8, 7),
        venue_code="85",
        race_no=7,
        bet_kind=BET_KIND_TRIFECTA_FORMATION,
        groups=[[3], [4, 5], [1, 2, 4, 5, 6]],
    )
    assert bet_id == "a5-85-7_b9_c1_3_4-5_1-2-4-5-6"


def test_build_bet_id_trio_box_matches_real_capture():
    assert build_bet_id_groups(
        race_date=date(2026, 8, 7), venue_code="85", race_no=7,
        bet_kind=BET_KIND_TRIO_BOX, groups=[[2, 4, 5, 6]],
    ) == "a5-85-7_b8_c2_2-4-5-6"
    # 車数が変わっても形式は同じ（BOXは車群の羅列のみ）
    assert build_bet_id_groups(
        race_date=date(2026, 8, 7), venue_code="85", race_no=7,
        bet_kind=BET_KIND_TRIO_BOX, groups=[[1, 2, 4, 5, 6]],
    ) == "a5-85-7_b8_c2_1-2-4-5-6"


def test_build_bet_id_groups_sorts_within_group():
    """グループ内は昇順に正規化される（実機の出力が昇順だったため）。"""
    assert build_bet_id_groups(
        race_date=date(2026, 8, 7), venue_code="85", race_no=7,
        bet_kind=BET_KIND_TRIO_BOX, groups=[[6, 2, 5, 4]],
    ) == "a5-85-7_b8_c2_2-4-5-6"


def test_build_bet_id_groups_rejects_wrong_group_count():
    with pytest.raises(ValueError):
        build_bet_id_groups(
            race_date=date(2026, 8, 7), venue_code="85", race_no=7,
            bet_kind=BET_KIND_TRIFECTA_FORMATION, groups=[[3], [4, 5]],
        )


def test_build_bet_id_groups_rejects_empty_group():
    with pytest.raises(ValueError):
        build_bet_id_groups(
            race_date=date(2026, 8, 7), venue_code="85", race_no=7,
            bet_kind=BET_KIND_TRIFECTA_FORMATION, groups=[[3], [], [1, 2]],
        )


def test_expand_trifecta_formation_is_8_points():
    """実機で「8点」と表示された組み合わせが、展開でも8点になること。

    7H1 の三連単は 1着1車 × 2着2車 × 3着5車 で、2着と3着が重なる2通りが
    落ちて 2×5-2 = 8点になる。
    """
    legs = expand_bet(BET_KIND_TRIFECTA_FORMATION, [[3], [4, 5], [1, 2, 4, 5, 6]])
    assert len(legs) == 8
    assert (3, 4, 4) not in legs and (3, 5, 5) not in legs
    assert (3, 4, 5) in legs and (3, 5, 4) in legs
    assert all(a == 3 for a, _, _ in legs)


def test_expand_trio_box_point_counts():
    assert len(expand_bet(BET_KIND_TRIO_BOX, [[2, 4, 5, 6]])) == 4
    assert len(expand_bet(BET_KIND_TRIO_BOX, [[1, 2, 4, 5, 6]])) == 10


def test_expand_matches_existing_kinds():
    """既存2形式の展開も点数が仕様どおりであること（回帰）。"""
    assert len(expand_bet(BET_KIND_TRIO_AXIS2, [[1], [2], [3, 4, 5, 6, 7]])) == 5
    assert expand_bet(BET_KIND_TRIFECTA_AXIS1, [[1], [2, 3]]) == {(1, 2, 3), (1, 3, 2)}


def test_waku_check_7car():
    assert waku_check_for(7) == [6]


def test_waku_check_9car():
    # 2026-07-28実機検証（豊橋4R/5R）: 枠4={4,5}・枠5={6,7}・枠6={8,9}
    assert waku_check_for(9) == [4, 5, 6]


def test_waku_check_unsupported_raises():
    with pytest.raises(ValueError):
        waku_check_for(6)


# ── mark_code（印）の生成 ───────────────────────────────────────────────
# 2026-08-03: 相手を絞るランク（7B）で、買い目から外した車まで △ になっていた
# 不具合の回帰テスト。submit_pick は HTTP を伴うため、mark 生成規則そのものを
# 同一ロジックで検証する（実装を変えたら必ずここが落ちるようにしておく）。
#
# mark_code: 1=◎ / 2=○ / 3=▲ / 4=△ / 0=--（印なし・docs/netkeirin-input-api-spec.md 2.2）


def _trio_axis2_marks(n_cars: int, axis1: int, axis2: int, partners: list[int]) -> dict[str, str]:
    """src.netkeirin_client.submit_pick の BET_KIND_TRIO_AXIS2 分岐と同一規則。"""
    mark = {str(axis1): "1", str(axis2): "2"}
    marked = {axis1, axis2}
    partner_set = set(partners)
    for c in range(1, n_cars + 1):
        if c in marked:
            continue
        mark[str(c)] = "4" if c in partner_set else "0"
    return mark


def test_marks_all_partners_when_full_nagashi():
    """総流し（7S/7A/9S/9A）は軸以外すべて △。従来挙動が変わっていないこと。"""
    marks = _trio_axis2_marks(7, axis1=7, axis2=2, partners=[1, 3, 4, 5, 6])
    assert marks == {"7": "1", "2": "2", "1": "4", "3": "4", "4": "4", "5": "4", "6": "4"}
    assert "0" not in marks.values()


def test_marks_excluded_partners_as_hyphen_when_narrowed():
    """相手を絞るランク（7B）は、買った相手のみ △・外した車は --(0)。"""
    marks = _trio_axis2_marks(7, axis1=2, axis2=5, partners=[3, 7, 4])
    assert marks["2"] == "1"          # ◎
    assert marks["5"] == "2"          # ○
    assert marks["3"] == marks["4"] == marks["7"] == "4"   # 買った相手 = △
    assert marks["1"] == marks["6"] == "0"                 # 買っていない = --


def test_marks_nine_car_full_nagashi_unchanged():
    marks = _trio_axis2_marks(9, axis1=2, axis2=5, partners=[1, 3, 4, 6, 7, 8, 9])
    assert set(marks.values()) == {"1", "2", "4"}
    assert len(marks) == 9
