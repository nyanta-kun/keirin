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
