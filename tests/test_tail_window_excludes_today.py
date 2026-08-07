"""tail 再構築の窓が当日を含まないこと（2026-08-07）。

rebuild 系は対象期間の picks_history を**一旦全削除してから**再計算した行を入れ直す。
再構築できるのは結果が確定したレースだけなので、当日を窓に含めると
「当日の推奨行が削除されたまま戻ってこない」。
2026-08-07 に実際に発生し、08:40〜10:00 の約75分間 Web から推奨が消えた。

⚠️ 6本の rebuild スクリプトが**同じ3行**を持つ構造だったため、1本でも
   `monthly_windows()` に戻すと同じ事故が再発する。全本を機械的に検査する。
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from src.wt_vintage_config import monthly_windows, tail_windows

REPO = Path(__file__).resolve().parent.parent
REBUILD_SCRIPTS = [f"scripts/rebuild_{r}_walkforward_pg.py"
                   for r in ("7ss", "7s", "7a", "7b", "9s", "9a")]


@pytest.mark.parametrize("today,expected_to", [
    (date(2026, 8, 7), "2026-08-06"),    # 月の途中 → 前日で打ち切る
    (date(2026, 8, 2), "2026-08-01"),    # 2日 → 1日まで
    (date(2026, 9, 1), "2026-08-31"),    # 月初 → 前月の窓がそのまま
    (date(2026, 8, 31), "2026-08-30"),   # 月末
])
def test_tail窓は当日を含まない(today, expected_to):
    w = tail_windows(today)
    assert w, "窓が空になってはいけない"
    assert w[-1][1] == expected_to
    assert w[-1][1] < today.strftime("%Y-%m-%d")


def test_tail窓は1つだけ返す():
    assert len(tail_windows(date(2026, 8, 7))) == 1


def test_全期間側の既定は変えていない():
    """monthly_windows は当日を含んだままでよい（結果のある行だけ入り直る）。"""
    assert monthly_windows(date(2026, 8, 7))[-1][1] == "2026-08-07"


@pytest.mark.parametrize("path", REBUILD_SCRIPTS)
def test_全rebuildスクリプトが_tail_windows_を使う(path):
    src = (REPO / path).read_text(encoding="utf-8")
    assert "tail_windows" in src, f"{path} が tail_windows を使っていない"
    # 旧実装（monthly_windows を丸ごと使って [-1:] で切る）が残っていないこと
    assert not re.search(r"windows\s*=\s*monthly_windows\(\)\s*\n\s*if args\.tail_only",
                         src), f"{path} に旧 tail 実装が残っている"


def test_reconcileが当日復元の安全網を持つ():
    sh = (REPO / "scripts" / "reconcile_walkforward_tail.sh").read_text(encoding="utf-8")
    assert "write_candidates_wt.py" in sh, (
        "tail 再構築後に当日候補を復元する安全網が外れている")
