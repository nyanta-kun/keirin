"""看板レース検出の不変条件（2026-08-09 新設）。

## 背景

2026-08-08 は当日売上の 84% が「外れたレース」＝看板レース（決勝・特選クラス）に
集中し、当たった準決勝・予選は買い手0だった。ユーザー決定で
**看板レースとその前後には必ず推奨を出す**方針になった。

2026-08-09 時点では検出が無く、当日の看板11件を手作業で入稿していた。

## 守る不変条件

1. 判定は **race_type**。レース番号（最終R＝決勝）で判定しない
   — ガールズ決勝が 6R と 12R の両方に置かれる開催が実在する（08-09 佐世保）
2. 「前後」は看板の ±1R。**存在しないレース番号は返さない**
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marquee import is_marquee_type, marquee_race_nos  # noqa: E402


def test_marquee_keywords() -> None:
    for t in ("決勝", "ガールズ決勝", "チャレンジ決勝", "特選", "初特選",
              "選抜", "ガールズ選抜", "特秀", "男子新人アドバンス決勝"):
        assert is_marquee_type(t), t


def test_non_marquee_types() -> None:
    for t in ("予選", "チャレンジ予選", "一般", "ガールズ一般", "準決勝",
              "特予選", "Wガル", "", None):
        assert not is_marquee_type(t), t


def test_semifinal_is_not_marquee() -> None:
    """準決勝は看板ではない（「決勝」を含むが別物）。

    ⚠️ 部分一致で拾うと準決勝まで対象になり件数が跳ねる。
    """
    assert not is_marquee_type("準決勝")
    assert not is_marquee_type("チャレンジ準決勝")


def test_neighbours_are_included() -> None:
    races = [{"race_no": n, "race_type": "予選"} for n in range(1, 13)]
    races[6]["race_type"] = "決勝"          # 7R
    assert marquee_race_nos(races) == {6, 7, 8}


def test_multiple_marquee_races_in_one_meeting() -> None:
    """ガールズ決勝が6Rと12Rの両方にある開催（2026-08-09 佐世保）。"""
    races = [{"race_no": n, "race_type": "ガールズ一般"} for n in range(1, 13)]
    races[5]["race_type"] = "ガールズ決勝"    # 6R
    races[11]["race_type"] = "ガールズ決勝"   # 12R
    assert marquee_race_nos(races) == {5, 6, 7, 11, 12}


def test_does_not_return_missing_race_numbers() -> None:
    """存在しないレース番号（欠番・最終Rの次）を返さない。"""
    races = [{"race_no": n, "race_type": "予選"} for n in (1, 2, 3)]
    races[2]["race_type"] = "決勝"           # 3R が最終
    assert marquee_race_nos(races) == {2, 3}


def test_race_no_alone_does_not_qualify() -> None:
    """🔴 最終Rでも race_type が一般なら看板ではない。"""
    races = [{"race_no": n, "race_type": "一般"} for n in range(1, 13)]
    assert marquee_race_nos(races) == set()
