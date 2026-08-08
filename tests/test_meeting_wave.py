"""開催（会場×日）→ 入稿の波 の振り分けテスト（2026-08-07）。

netkeirin は**公開後に差し替えられない**ので、板が育ってから入稿するしかない。
ここで守るのは「1つの開催が必ずちょうど1つの波に入ること」。
どこにも入らなければ商品が消え、二重に入れば先の商品が上書きされる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import netkeirin_submit_wt as sub  # noqa: E402
from src.meeting_wave import (  # noqa: E402
    WAVE_MORNING,
    WAVE_NIGHT,
    WAVE_NOON,
    WAVES,
    parse_wave,
    wave_of_first_hour,
)


# 実測（2026-07-16以降）で観測された第1R発走時刻はこの6通りだけ。
# 競輪の モーニング / デイ / ナイター / ミッドナイト に対応する。
@pytest.mark.parametrize("hour,expected", [
    (8, WAVE_MORNING),    # モーニング（最終10時）
    (10, WAVE_MORNING),   # デイ（最終15-16時）
    (11, WAVE_MORNING),   # デイ（遅め）
    (15, WAVE_NOON),      # ナイター（最終20時）
    (16, WAVE_NOON),      # ナイター（遅め）
    (20, WAVE_NIGHT),     # ミッドナイト（最終23時）
])
def test_実測される第1R発走時刻が正しい波になる(hour, expected):
    assert wave_of_first_hour(hour) == expected


def test_境界():
    assert wave_of_first_hour(11.99) == WAVE_MORNING
    assert wave_of_first_hour(12) == WAVE_NOON
    assert wave_of_first_hour(17.99) == WAVE_NOON
    assert wave_of_first_hour(18) == WAVE_NIGHT


def test_発走時刻不明は朝へ倒す():
    """分からないことを理由に入稿を落とすと商品が黙って消える。"""
    assert wave_of_first_hour(None) == WAVE_MORNING


def test_どの時刻も必ずいずれかの波に入る():
    for h in range(0, 24):
        assert wave_of_first_hour(h) in WAVES


def test_parse_wave():
    assert parse_wave("noon") == WAVE_NOON
    assert parse_wave(" NIGHT ") == WAVE_NIGHT
    assert parse_wave("yuugata") is None
    assert parse_wave(None) is None


# ── session と波の対応 ────────────────────────────────────────────────

def test_sessionと波が1対1で対応する():
    assert sub.SESSION_WAVE == {
        "morning": WAVE_MORNING, "noon": WAVE_NOON, "evening": WAVE_NIGHT}
    # 取りこぼしも重複も無いこと
    assert sorted(sub.SESSION_WAVE.values()) == sorted(WAVES)


def test_全sessionに日本語ラベルがある():
    for s in sub.SESSION_WAVE:
        assert s in sub.SESSION_LABEL_JP


# ── 開催単位でまとまること ────────────────────────────────────────────

def test_同じ開催のレースは第1Rの時刻で一括して振り分けられる(monkeypatch):
    """レース個別の発走時刻では分けない（同じ開催の商品が別々に出ると分かりにくい）。

    ナイター開催: 第1R 15時 → 最終 20時。20時発走の最終レースも「昼」の波に入る。
    """
    base = 1780000000 - (1780000000 + 9 * 3600) % 86400   # その日の JST 0:00

    rows = []
    for i, hh in enumerate([15, 17, 20], start=1):        # 会場A: ナイター
        rows.append({"race_key": f"D_A_{i}", "venue_id": "A",
                     "start_at": str(base + hh * 3600)})
    for i, hh in enumerate([8, 9, 10], start=1):          # 会場B: モーニング
        rows.append({"race_key": f"D_B_{i}", "venue_id": "B",
                     "start_at": str(base + hh * 3600)})

    class _Conn:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sub, "get_connection", lambda: _Conn())
    waves = sub._load_meeting_waves("2026-06-08")
    assert {waves[f"D_A_{i}"] for i in (1, 2, 3)} == {WAVE_NOON}
    assert {waves[f"D_B_{i}"] for i in (1, 2, 3)} == {WAVE_MORNING}


# ── 発走済みレースを出さないこと ──────────────────────────────────────

def _patch_rows(monkeypatch, rows):
    class _Conn:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sub, "get_connection", lambda: _Conn())


def test_発走済みのレースだけを拾う(monkeypatch):
    """入稿は「まだ売れるレース」にしか意味がない。

    従来は入稿が朝の1回だけで第1レースより前に必ず終わっていたので誰も見ていなかったが、
    開催単位の波と手動再実行が入ったことで終わったレースへ出しうるようになった。
    """
    import time
    now = time.time()
    _patch_rows(monkeypatch, [
        {"race_key": "past", "start_at": str(int(now - 3600))},
        {"race_key": "future", "start_at": str(int(now + 3600))},
    ])
    assert sub._load_started_races("2026-08-07") == {"past"}


def test_発走時刻不明は未発走扱い(monkeypatch):
    """安全側は「出す」。情報が無いことを理由に商品を落とすと黙って商品が消える。"""
    _patch_rows(monkeypatch, [
        {"race_key": "unknown", "start_at": None},
        {"race_key": "broken", "start_at": "not-a-number"},
    ])
    assert sub._load_started_races("2026-08-07") == set()


def test_発走時刻が全部欠けている開催は朝へ倒れる(monkeypatch):
    rows = [{"race_key": "D_C_1", "venue_id": "C", "start_at": None}]

    class _Conn:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sub, "get_connection", lambda: _Conn())
    assert sub._load_meeting_waves("2026-06-08")["D_C_1"] == WAVE_MORNING


# ── 前倒しに訂正された開催の救済（2026-08-08 レビュー指摘 M-6）─────────────

def test_waves_due_by_includes_earlier_waves():
    """自分の波 + それより前の波を対象にすること。

    波は毎回 `wt_races.start_at` から都度再計算されるため、発走時刻が
    **前倒しに訂正**されると開催が通過済みの波へ移り、「自分の波と一致するもの」
    だけを見る実装では**その日どの回からも入稿されずに終わる**。
    """
    from src.meeting_wave import (
        WAVE_MORNING, WAVE_NIGHT, WAVE_NOON, waves_due_by,
    )
    assert waves_due_by(WAVE_MORNING) == (WAVE_MORNING,)
    assert waves_due_by(WAVE_NOON) == (WAVE_MORNING, WAVE_NOON)
    assert waves_due_by(WAVE_NIGHT) == (WAVE_MORNING, WAVE_NOON, WAVE_NIGHT)


def test_waves_due_by_never_includes_later_waves():
    """🔴 後の波を含めてはいけない。

    含めるとミッドナイトを朝に入稿してしまい、「板が育ってから出す」という
    波の存在理由そのものが壊れる（推奨で買う点すべてにオッズがある割合は
    ミッドナイトだと朝の時点で 17.5% しかない）。
    """
    from src.meeting_wave import WAVES, waves_due_by
    for i, w in enumerate(WAVES):
        due = waves_due_by(w)
        assert set(due).isdisjoint(WAVES[i + 1:]), f"{w} が後の波を含んでいる: {due}"


def test_waves_due_by_tolerates_unknown_wave():
    """未知の波名でも落ちない（自分だけを返す）。"""
    from src.meeting_wave import waves_due_by
    assert waves_due_by("__unknown__") == ("__unknown__",)


def test_submit_uses_due_waves_not_exact_match():
    """入稿側が `== want_wave` ではなく `in due_waves` で絞っていること。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "netkeirin_submit_wt.py").read_text(encoding="utf-8")
    # 候補を絞り込んでいる行（waves.get(...) を評価している行）だけを見る。
    # ログ用の件数カウントは `== want_wave` のままでよいので全文検索にしない。
    filters = [ln for ln in src.splitlines()
               if "waves.get(" in ln and not ln.lstrip().startswith("#")]
    assert filters, "候補を波で絞り込んでいる行が見つからない"
    for ln in filters:
        assert "in due_waves" in ln, (
            f"波の完全一致で絞り込んでいる（前倒し訂正で取りこぼす）: {ln.strip()!r}")
