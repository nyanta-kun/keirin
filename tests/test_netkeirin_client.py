"""netkeirin bet_id 組み立てロジックのテスト（実測値ベース）。"""
from datetime import date

from src.netkeirin_client import build_bet_id


def test_build_bet_id_matches_real_capture():
    """2026-07-23実機検証で確認した実データ（佐世保1R・2026-07-24=金曜）に一致すること。"""
    bet_id = build_bet_id(
        race_date=date(2026, 7, 24),
        venue_code="85",
        race_no=1,
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
        axis1=1,
        axis2=2,
        partners=[7, 3, 5, 4, 6],
    )
    assert bet_id.endswith("_3-4-5-6-7")
