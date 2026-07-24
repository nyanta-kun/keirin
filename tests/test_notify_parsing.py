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
    ("3連複: 1-2-3,4,5,6", (1, 2, [3, 4, 5, 6])),      # thirds は全て返す（切り詰めなし）
    ("3連単BOX: 4⇄5→1,2,3", (4, 5, [1, 2, 3])),       # SS 1-2着BOX（⇄=両順・→区切り）
    ("ワイド: 4-5", (4, 5, [])),                        # ワイド1点（2車・thirds空）
])
def test_parse_combo(combo_str, expected):
    assert nr._parse_combo(combo_str) == expected


# ── SS 1-2着BOX(opt-in): combo_str に "BOX" を含み both-order で採点される ──
def test_box_marker_detection():
    """採点側は combo_str の 'BOX' で box を識別し、pred1,pred2 を両順で照合する。"""
    combo = "3連単BOX: 4⇄5→1,2,3"
    assert "BOX" in combo                                  # box識別フラグ
    p1, p2, thirds = nr._parse_combo(combo)
    # box は (p1,p2) と (p2,p1) の両順 × thirds = 2×len(thirds) 点
    box_orders = [(a, b, t) for t in thirds for (a, b) in ((p1, p2), (p2, p1))]
    assert len(box_orders) == 2 * len(thirds) == 6
    assert (5, 4, 1) in box_orders and (4, 5, 1) in box_orders   # 両順を含む


# ── ワイド1点の的中判定: 2車が共に top3 なら的中（順不同） ──
@pytest.mark.parametrize("combo, top3, expect_hit", [
    ("ワイド: 1-2", {1, 2, 3}, True),    # 両者top3
    ("ワイド: 1-3", {1, 2, 3}, True),    # 順不同で当たり
    ("ワイド: 1-4", {1, 2, 3}, False),   # 4が圏外
    ("ワイド: 4-5", {1, 2, 3}, False),   # 両者圏外
])
def test_wide_hit_rule(combo, top3, expect_hit):
    p1, p2, thirds = nr._parse_combo(combo)
    assert thirds == []                                   # ワイドは2車のみ
    assert frozenset((p1, p2)).issubset(frozenset(top3)) is expect_hit


# ── 欠車の無効化ルール（board=最終オッズ盤面掲載車。欠車のみ盤面から消える）:
#    軸欠車=レース無効(返還) / 相手欠車=その目除外 / 落車失格は盤面に残る=外れ計上 ──
def test_void_by_dns_axis_scratched():
    """軸(p1 or p2)が欠車ならレース無効（返還・不計上）。"""
    # p2=2 が盤面に居ない → 無効
    skip, thirds = nr._void_by_dns(5, 2, [3, 4, 1], board={3, 4, 5}, is_wide=False)
    assert skip is True and thirds == []


def test_void_by_dns_third_scratched():
    """相手(thirds)の欠車はその目のみ除外、残りで採点。"""
    # 相手 1 が欠車 → 3,4 のみ有効
    skip, thirds = nr._void_by_dns(5, 2, [3, 4, 1], board={2, 3, 4, 5}, is_wide=False)
    assert skip is False and thirds == [3, 4]


def test_void_by_dns_all_thirds_scratched():
    """相手が全員欠車なら買える目なし→無効。"""
    skip, thirds = nr._void_by_dns(5, 2, [3, 4, 1], board={2, 5}, is_wide=False)
    assert skip is True and thirds == []


def test_void_by_dns_all_runners_ok():
    """全員出走なら無効化なし・thirdsそのまま。"""
    skip, thirds = nr._void_by_dns(5, 2, [3, 4, 1], board={1, 2, 3, 4, 5}, is_wide=False)
    assert skip is False and thirds == [3, 4, 1]


def test_void_by_dns_wide_leg_scratched():
    """ワイドは2車とも軸扱い→どちらか欠車で無効。"""
    assert nr._void_by_dns(2, 4, [], board={2, 3, 5}, is_wide=True)[0] is True   # 4欠車
    assert nr._void_by_dns(2, 4, [], board={2, 3, 4, 5}, is_wide=True)[0] is False


# ── _parse_picks_full: 7+車フォーマット SS/S の採点対象確認 ──
_FIXTURE_DATE = "2099-12-31"
_FIXTURE = """\
======================================================================
 競輪AI予想PICK [wt]  2099-12-31  (7+車 三連複・SSランク/Sランク)
======================================================================

【7+車 SSランク】 0件
  (該当なし)

【7+車 Sランク】 1件
  10:00  京王閣 3R  [7車]  3連複: 1-2-3,4,5  (3点/300円)  [6.0倍]
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
    # S/S+（三連単F）は 2026-07-15 全廃 → 旧txtのSセクションは採点対象外
    assert not any(slot == "7plus_st" for (_, _, slot) in picks)
    assert len(picks) == 0


# ── _parse_picks_full: SS と S が別 slot で並立し、Aランクは無視される ──
# 7PLUS_R 時代（2026-07-10〜07-15）の日付。2026-07-16 以降は旧S1全廃で SS セクションを無視する
_WIDE_DATE = "2026-07-12"
_WIDE_FIXTURE = """\
======================================================================
 競輪AI予想PICK [wt]  2099-12-29  (7+車 三連複・SSランク/Sランク)
======================================================================
【7+車 SSランク】 1件
  10:00  京王閣 3R  [7車]  3連複: 1-2-3,4,5  (3点/300円)  [6.0倍]
【7+車 Sランク】 1件
  11:00  京王閣 5R  [7車]  3連複: 2-3-4,5,6  (3点/300円)  [8.0倍]
【7+車 Aランク】 1件
  12:00  京王閣 7R  [7車]  3連複: 3-4-5,6,7  (3点/300円)  [9.0倍]
"""


def test_parse_picks_full_wide_coexists_with_main():
    """7PLUS_R時代の日付(2026-07-10〜07-15)のSSランクは 7PLUS_R。廃止済みS/Aランクは無視。"""
    path = Path(nr.__file__).resolve().parent.parent / "data" / "picks" / f"wave_picks_wt_{_WIDE_DATE}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_WIDE_FIXTURE, encoding="utf-8")
    try:
        picks = nr._parse_picks_full(_WIDE_DATE)
    finally:
        path.unlink(missing_ok=True)
    assert ("京王閣", 3, "7plus_r") in picks
    assert ("京王閣", 5, "7plus_st") not in picks  # S/S+全廃 → 無視
    assert ("京王閣", 7, "7plus_a") not in picks   # Aランク廃止 → 無視
    assert picks[("京王閣", 3, "7plus_r")][0] == "7PLUS_R"
    assert len(picks) == 1, "7plus_r のみ1エントリ"


def test_parse_picks_full_abolished_after_20260716():
    """旧S1全廃日(2026-07-16)以降の日付では SS セクションを無視する。"""
    date = "2099-12-29"
    path = Path(nr.__file__).resolve().parent.parent / "data" / "picks" / f"wave_picks_wt_{date}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_WIDE_FIXTURE.replace(_WIDE_DATE, date), encoding="utf-8")
    try:
        picks = nr._parse_picks_full(date)
    finally:
        path.unlink(missing_ok=True)
    assert picks == {}, "全廃日以降は旧S1をパースしない"


def test_parse_picks_full_old_date_ss_is_legacy():
    """旧日付(2026-07-10 より前)のSSランクは旧カット方式 7PLUS_SS として互換維持。"""
    old_date = "2026-07-01"
    path = Path(nr.__file__).resolve().parent.parent / "data" / "picks" / f"wave_picks_wt_{old_date}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_WIDE_FIXTURE.replace(_WIDE_DATE, old_date), encoding="utf-8")
    try:
        picks = nr._parse_picks_full(old_date)
    finally:
        path.unlink(missing_ok=True)
    assert ("京王閣", 3, "7plus_ss") in picks
    assert picks[("京王閣", 3, "7plus_ss")][0] == "7PLUS_SS"


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
    monkeypatch.setattr(nr, "send", lambda m, channel=None: msgs.append(m))
    monkeypatch.setattr(_sys, "argv", ["notify_results_wt.py", _B_ONLY_DATE])
    try:
        nr.main()
    finally:
        path.unlink(missing_ok=True)
    assert msgs, "通知が送られていない"
    assert "見つかりません" not in msgs[0], "Bランクのみを『ファイル無し』と誤通知している"
    assert "採点対象なし" in msgs[0] or "推奨買い目" in msgs[0]
