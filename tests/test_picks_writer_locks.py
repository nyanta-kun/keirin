"""picks_history を書き換える cron スクリプトが共有ロックを持つことを固定する。

背景（2026-08-08 レビュー）:
  多重起動防止の flock は daily_picks_wt / evening_picks_wt / intraday_results_wt /
  results_check_wt の4本にあったが、**それぞれ別のロックファイル**なので
  「自分自身の二重起動」しか防げていなかった。当月分を DELETE→INSERT で
  作り直す reconcile_walkforward_tail.sh に至っては flock が1つも無く、
  唯一の対策が「実行時刻を 00:50 → 08:40 へずらす」という時間差頼みだった。

  daily_picks_wt.sh にはリトライ待機（最大3回×5分）があり 08:40 に食い込みうる。
  intraday_results_wt.sh は15分毎なので 08:40 にも必ず走る。実際 2026-08-06 には
  7A(rebuild 18 / live 26)・7B(25 / 14) で rebuild行×live行の混在が起きている。

  → 全員が同じ `wt_picks_writer.lock` を取るようにした。ロックは1つだけなので
    デッドロックしない（必ず「自分のロック → 共有ロック」の順で取る）。

このテストは「新しく picks_history を書く cron スクリプトを足したときに
共有ロックを付け忘れる」のを防ぐためのもの。付け忘れは**症状が出るまで
気づけない**（片方が黙って上書きするだけでエラーにならない）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# picks_history を書き換える cron スクリプト。増えたらここへ足すこと。
PICKS_WRITER_SCRIPTS = [
    "daily_picks_wt.sh",
    "evening_picks_wt.sh",
    "intraday_results_wt.sh",
    "results_check_wt.sh",
    "reconcile_walkforward_tail.sh",
]

SHARED_LOCK_NAME = "wt_picks_writer.lock"


@pytest.mark.parametrize("name", PICKS_WRITER_SCRIPTS)
def test_holds_shared_writer_lock(name: str) -> None:
    s = (_SCRIPTS / name).read_text(encoding="utf-8")
    assert SHARED_LOCK_NAME in s, (
        f"{name} が共有ロック {SHARED_LOCK_NAME} を取っていない。"
        " picks_history を同時に書き換えて rebuild行×live行の混在を起こす")
    assert re.search(r"exec 201>", s), f"{name}: 共有ロックの fd 201 を開いていない"


@pytest.mark.parametrize("name", PICKS_WRITER_SCRIPTS)
def test_shared_lock_waits_instead_of_skipping(name: str) -> None:
    """共有ロックは `-w`（待つ）であること。

    `-n`（取れなければ即スキップ）にすると、待てば済む競合で
    **朝の予想生成や当月の再構築が丸ごと黙って落ちる**。それは本ロックが
    防ごうとしている状態そのもの（＝データが古いまま放置される）。
    """
    s = (_SCRIPTS / name).read_text(encoding="utf-8")
    lines = [ln for ln in s.splitlines()
             if "flock" in ln and re.search(r"\b201\b", ln) and not ln.lstrip().startswith("#")]
    assert lines, f"{name}: fd 201 に対する flock 呼び出しが見つからない"
    for ln in lines:
        assert re.search(r"flock\s+-w\b", ln), (
            f"{name}: 共有ロックが待ちになっていない → {ln.strip()!r}。-w にすること")


@pytest.mark.parametrize("name", PICKS_WRITER_SCRIPTS)
def test_own_lock_is_acquired_before_shared_lock(name: str) -> None:
    """自分のロック(200) → 共有ロック(201) の順であること（デッドロック回避）。"""
    s = (_SCRIPTS / name).read_text(encoding="utf-8")
    own = s.find("exec 200>")
    shared = s.find("exec 201>")
    assert own != -1, f"{name}: 自分用ロック(fd 200)が無い"
    assert own < shared, (
        f"{name}: 共有ロックを自分のロックより先に取っている。"
        " 取得順が揃っていないとデッドロックしうる")
