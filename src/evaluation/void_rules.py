"""欠車無効化ルール（void_by_dns）の共通実装。

`scripts/notify_results_wt._void_by_dns` と同一ロジックをここに定義し、
バックテスト (`backtest_wt.py`) と通知スクリプト (notify_results_wt.py) の
両方から参照できるようにする。notify_results_wt.py は変更せず、
バックテスト側がこのモジュールを使う。

ルール（本番 notify_results_wt._void_by_dns と同一）:
  - runners = そのレースで実際に出走した車 (finish_order >= 1) の集合
  - 軸(p1/p2)が欠車      → レース無効（返還）。 returns (True, [])
  - 相手(thirds)が欠車   → その目のみ除外。     returns (False, 有効thirds)
  - 相手が全員欠車       → 買える目なし→無効。  returns (True, [])
  - ワイドは2車とも軸扱い（どちらか欠車で無効）。
"""
from __future__ import annotations


def void_by_dns(
    p1: int,
    p2: int,
    thirds: list[int],
    runners: set[int],
    is_wide: bool = False,
) -> tuple[bool, list[int]]:
    """欠車の無効化ルールを適用する。

    Parameters
    ----------
    p1, p2  : 軸選手の車番
    thirds  : 相手選手の車番リスト
    runners : 出走した車番の集合 (finish_order >= 1)
    is_wide : True のとき p1/p2 を両方軸扱い（ワイド）

    Returns
    -------
    (skip_race, valid_thirds)
      skip_race=True  → レース無効（返還・不計上）
      skip_race=False → 有効な thirds で採点続行
    """
    if p1 not in runners or p2 not in runners:
        return True, []
    if is_wide:
        return False, []
    valid = [t for t in thirds if t in runners]
    return (not valid), valid
