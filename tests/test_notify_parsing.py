"""notify_results_wt のパース純粋関数テスト（買い目分解・Bランク採点除外）。"""
from pathlib import Path

import pytest

import notify_results_wt as nr  # scripts/ は conftest で path 追加済


# ── _parse_combo: 区切り・接頭辞・3連単/3連複 ──
@pytest.mark.parametrize("combo_str, expected", [
    ("3連複: 4-1-5,2,3", (4, 1, [5, 2, 3])),
    ("(元A) 3連複: 4-1-5,2,3", (4, 1, [5, 2, 3])),   # Bランクの (元X) 接頭辞
    ("3連単: 4→1→5", (4, 1, [5])),                    # 順序付き（→区切り）
    ("1-2-3,4,5", (1, 2, [3, 4, 5])),                 # コロン無し
    ("3連複: 1-2-3,4,5,6", (1, 2, [3, 4, 5])),        # thirds は3つに切詰め
])
def test_parse_combo(combo_str, expected):
    assert nr._parse_combo(combo_str) == expected


# ── _parse_picks_full: 【Bランク】は採点対象から除外される ──
_FIXTURE_DATE = "2099-12-31"
_FIXTURE = """\
======================================================================
 競輪AI予想PICK [wt]  2099-12-31
======================================================================

【SSランク】 0件
  (該当なし)

【Sランク】 0件
  (該当なし)

【Aランク】 1件
  10:00  京王閣 3R  [6車]  3連複: 1-2-3,4,5  (3点/300円)  [6.0倍]

【Bランク】 1件  ※各自判断
  17:43  いわき平 6R  [6車]  (元A) 3連複: 4-1-5,2,3  (3点/300円)  [4.7倍]
"""


@pytest.fixture()
def fixture_picks_file():
    path = Path(nr.__file__).resolve().parent.parent / "data" / "picks" / f"wave_picks_wt_{_FIXTURE_DATE}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_FIXTURE, encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def test_parse_picks_full_excludes_b_rank(fixture_picks_file):
    picks = nr._parse_picks_full(_FIXTURE_DATE)
    # Aランクのみ採点対象。Bランク(いわき平6R)は含めない。
    assert ("京王閣", 3) in picks
    assert picks[("京王閣", 3)][0] == "A"
    assert ("いわき平", 6) not in picks, "Bランクは採点対象から除外されるべき"
    assert len(picks) == 1


# ── notify_results_wt.main: Bランクのみ(推奨0件)を「ファイル無し」と誤通知しない ──
_B_ONLY_DATE = "2099-12-30"
_B_ONLY = """\
【SSランク】 0件
  (該当なし)
【Sランク】 0件
  (該当なし)
【Aランク】 0件
  (該当なし)
【Bランク】 1件  ※各自判断
  17:48  いわき平 6R  [6車]  (元A) 3連複: 2-3-1,5,4  (3点/300円)  [9999.9倍]
"""


def test_results_b_only_not_filemissing(monkeypatch):
    import sys as _sys
    path = Path(nr.__file__).resolve().parent.parent / "data" / "picks" / f"wave_picks_wt_{_B_ONLY_DATE}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_B_ONLY, encoding="utf-8")
    msgs = []
    monkeypatch.setattr(nr, "send", lambda m: msgs.append(m))
    monkeypatch.setattr(_sys, "argv", ["notify_results_wt.py", _B_ONLY_DATE])
    try:
        nr.main()
    finally:
        path.unlink(missing_ok=True)
    assert msgs, "通知が送られていない"
    assert "見つかりません" not in msgs[0], "Bランクのみを『ファイル無し』と誤通知している"
    assert "採点対象なし" in msgs[0] or "推奨買い目" in msgs[0]
