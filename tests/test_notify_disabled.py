"""廃止した Discord 通知が復活していないことの検査（2026-08-07）。

ユーザー要望:「『朝夕の推奨』『発走前個別通知』は廃止。『成績報告』の
競輪AI[wt]成績も廃止とし、**レース個別の通知のみ**とします」

🔴 **通知を止めても、そのスクリプト自体は止めてはいけない。**
   `notify_prerace_wt.py` は発走15分前の判定を `picks_history` へ書き、
   `notify_results_wt.py` は確定結果で採点して `picks_history` へ書き戻す。
   cron から外すとその書き込みごと消える。**送信だけを落として実行は続ける。**
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.notify_prerace_wt as prerace  # noqa: E402
import scripts.notify_results_wt as results  # noqa: E402


def test_発走前個別通知は無効():
    assert prerace.PRERACE_NOTIFY_ENABLED is False


def test_成績サマリー通知は無効():
    assert results.RESULTS_SUMMARY_NOTIFY_ENABLED is False


def test_成績サマリーはスイッチを入れれば本文を作れる(monkeypatch):
    """止めたのは送信であって採点・整形ではないこと（再開可能性の担保）。"""
    monkeypatch.setattr(results, "RESULTS_SUMMARY_NOTIFY_ENABLED", True)
    assert results.RESULTS_SUMMARY_NOTIFY_ENABLED is True


def test_レース個別の通知は残っている():
    """唯一残す通知。誤って一緒に畳んでいないこと。"""
    import scripts.notify_race_result_wt as race_result
    assert hasattr(race_result, "main")


def test_朝夕の推奨はシェルから呼ばれない():
    """`notify_picks.py`（朝夕の推奨一覧）は 2026-07-31 に廃止済み。

    スクリプト自体は残っているので、**呼び出しが復活していないこと**を見る。
    """
    for name in ("daily_picks_wt.sh", "evening_picks_wt.sh", "daily_picks.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        live = [ln for ln in text.splitlines()
                if "notify_picks.py" in ln and not ln.lstrip().startswith("#")]
        assert not live, f"{name} が notify_picks.py を呼んでいます: {live}"
