"""欠車無効化ルール（void_by_dns）の共通実装。

`scripts/notify_results_wt._void_by_dns`（変更禁止・2026-07-15仕様確定）と
同一ロジックをここに定義し、バックテスト (`backtest_wt.py`) と通知スクリプト
(notify_results_wt.py) の両方から参照できるようにする。notify_results_wt.py は
変更せず、バックテスト側がこのモジュールを使う。

## 基準は「オッズ盤面掲載車」(board) であり「完走者」(finish_order>=1) ではない
【2026-07-31 是正・PMタスク B-4】本 docstring は従来「runners = 完走者
(finish_order>=1) の集合」と説明していたが、これは本番 notify_results_wt と
一致しない誤記だった。正しい基準は次の通り:

  - board = 最終オッズ盤面（wt_odds の trio 組合せ）に掲載されていた車番の集合
            = 実際に馬券として購入できた車（notify_results_wt._board_frames
            と同一の構築方法）。
  - 発走前に確定した欠車（取消・除外）は wt_entries に行自体が作られず、
    オッズ盤面にも掲載されない → board に含まれない → 返還（無効）対象。
  - 発走後の落車・失格・棄権（finish_order=0 として記録される DNF）は、
    発走前に確定していたオッズ盤面には残ったまま（発走後の事象は盤面を
    書き換えない）→ board に含まれる → 購入は成立したものとして外れ計上
    （没収）する。

  つまり「finish_order による完走/非完走」と「board に載っているか」は別軸の
  情報であり、`finish_order >= 1`（完走者基準）を board の代用にすると、
  DNF を誤って欠車（返還）扱いしてしまい、本番の没収計上と乖離する
  （2026-07-31 是正の発端となった論点。呼び出し側の対応は
  `backtest_wt._load_board_frames_wt` 参照）。

  この関数自体の判定ロジック（軸/相手の無効化）は notify_results_wt._void_by_dns
  と完全に同一であり、これは変更していない。同一性が保たれるかどうかは
  「呼び出し側が正しい board（盤面掲載車の集合）を構築して渡しているか」に
  懸かっている——引数名を board としているのはそれを明示するため。

ルール（本番 notify_results_wt._void_by_dns と同一）:
  - board  = 最終オッズ盤面(trio)に掲載されていた車番の集合
             （欠車は掲載されないため含まれない。DNFは掲載されたまま含まれる）
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
    board: set[int],
    is_wide: bool = False,
) -> tuple[bool, list[int]]:
    """欠車の無効化ルールを適用する。

    Parameters
    ----------
    p1, p2  : 軸選手の車番
    thirds  : 相手選手の車番リスト
    board   : 最終オッズ盤面(trio)に掲載されていた車番の集合
              （＝実際に購入できた車。DNF(finish_order=0)は掲載されたままなので
              含む。事前確定の欠車のみ含まれない）
    is_wide : True のとき p1/p2 を両方軸扱い（ワイド）

    Returns
    -------
    (skip_race, valid_thirds)
      skip_race=True  → レース無効（返還・不計上）
      skip_race=False → 有効な thirds で採点続行
    """
    if p1 not in board or p2 not in board:
        return True, []
    if is_wide:
        return False, []
    valid = [t for t in thirds if t in board]
    return (not valid), valid
