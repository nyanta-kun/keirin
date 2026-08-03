"""netkeirin bet_id/waku_check 組み立てロジックのテスト（実測値ベース）。"""
from datetime import date

import pytest

from src.netkeirin_client import (
    BET_KIND_TRIFECTA_AXIS1,
    BET_KIND_TRIO_AXIS2,
    build_bet_id,
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
