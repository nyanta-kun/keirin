#!/usr/bin/env python3
"""S1(SEVEN_S1) の全期間honest再構築（quarterly walk-forwardモデル使用・VPS PG専用）。

【2026-07-31】S1は全廃済み（commit df31431・ユーザー判断）。picks_historyの
SEVEN_S1行1,504件もVPS PGから削除済み（バックアップ:
data/backup/picks_history_s1_discarded_20260731.csv）。本スクリプトは手動実行
専用であり、reconcile_walkforward_tail.sh からの呼び出しは除去済み（同スクリプト
の00:50毎日cronから自動実行されない）。実行すると本スクリプトのwipe_rows_pg()が
削除済みのSEVEN_S1行をpicks_historyへ再生成してしまうため、意図しない限り実行
しないこと。

rebuild_s1_walkforward.py はローカルSQLite前提（KEIRIN_DB_URLをpopして
ローカル読み取り）だが、2026-07-22にローカルSQLiteは廃止されVPS PGへ
一本化済み（wt_odds含め2024-01-01〜のtrifectaオッズを確認済み）。
S7/S9のrebuild_s7_walkforward_pg.py/rebuild_s9_walkforward_pg.pyと同じ設計で、
環境変数をpopしないPG直読みの単発スクリプトとして実行する（get_connection()
単一経路のみ使用。backfill_s1w_rank_wt.wipe_rows/insert_rowsをそのまま使うと
「ローカル+VPSミラー」二重書き込みが同一PGへ二重に当たるため専用実装）。

2026-07-27: entropyゲート(S1W_ENTROPY_MAX)導入分の再構築にも使用。

【2026-07-29改定】期間定義を`src.wt_vintage_config.monthly_windows()`（月次凍結
vintageモデル・唯一の正本）に統一。詳細は`rebuild_s7_walkforward_pg.py`の
モジュールdocstring参照（同一設計）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s1_walkforward_pg.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_s1w_rank_wt import build_rows
from src.wt_rebuild_common import rebuild_pg_atomic
from src.wt_vintage_config import monthly_windows


#: DELETE/COUNT の WHERE 句（`rebuild_pg_atomic` へ渡す完全な条件文字列）。
_DELETE_COND = "rank='SEVEN_S1' AND race_key LIKE '%#7S1' AND race_date BETWEEN ? AND ?"


def _parse_upto(v: str | None):
    """`--upto` を date へ。未指定なら None（＝当日まで）。"""
    return date.fromisoformat(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tail-only", action="store_true",
                     help="直近月（今月）の窓のみ再構築する日次軽量運用向けオプション。"
                          "確定済み過去月は結果が変わらないため毎日再計算する必要がなく、"
                          "これのみ再実行すれば直近日をhonestな状態に保てる。")
    ap.add_argument("--upto", metavar="YYYY-MM-DD", default=None,
                    help="この日までを再構築する（当日を含めたくないときに使う）。\n"
                         "monthly_windows は既定で当日を含み、結果未確定のレースは\n"
                         "再構築で戻せないため、実行すると当日の行が消える。")
    args = ap.parse_args()

    windows = monthly_windows(_parse_upto(args.upto))
    if args.tail_only:
        windows = windows[-1:]
    wipe_from = windows[0][0]
    wipe_to = windows[-1][1]

    all_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"\n[rebuild-s1-pg] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
        print(f"[rebuild-s1-pg]   S1: {len(rows)}R ({len(rows)/n_days:.1f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-s1-pg] ===== 全期間合計 =====")
    print(f"[rebuild-s1-pg] S1: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    # 🔴 wipe(DELETE)→insert は**単一トランザクション**で行う（2026-08-08 是正）。
    #   従来はここだけ旧方式（DELETE を commit してから INSERT を別 commit）で、
    #   途中で例外が起きると wipe だけ確定して insert が失われ得た
    #   （2026-08-01 に他ランクで実害が出て `rebuild_pg_atomic` が導入された
    #   のに、全廃済みの S1 版だけ移行から漏れていた）。
    rebuild_pg_atomic(
        rank_label="SEVEN_S1",
        delete_cond_sql=_DELETE_COND,
        per_window_rows=[(wipe_from, wipe_to, all_rows)],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
