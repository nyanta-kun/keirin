#!/usr/bin/env python3
"""7A(RANK_7A) の全期間honest再構築（月次凍結vintageモデル使用）。

rebuild_7s_walkforward_pg.py の7A版。VPS PG一本化（2026-07-22〜）に伴い、
環境変数をpopしないPG直読みの単発スクリプトとして実行する。

【2026-07-29改定】期間定義を`src.wt_vintage_config.monthly_windows()`（月次凍結
vintageモデル・唯一の正本）に統一。詳細は`rebuild_7s_walkforward_pg.py`の
モジュールdocstring参照（同一設計）。

【2026-08-01改定・F-4】月初にその月のvintageモデルが未生成だと全期間の計算
（約40分規模）が丸ごと失われる事故が実際に発生したため、`rebuild_7s_walkforward_pg.py`
と同様に事前モデル存在チェック（`--skip-missing-models`）とwipe/insertの
単一トランザクション化（`src.wt_rebuild_common`）を導入。詳細は
`rebuild_7s_walkforward_pg.py`のモジュールdocstring・`docs/vintage_model_policy.md`参照。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_7a_walkforward_pg.py [--dry-run]
    PYTHONPATH=. .venv/bin/python scripts/rebuild_7a_walkforward_pg.py --skip-missing-models
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_7a_rank_wt import build_rows
from src.wt_rebuild_common import (
    format_missing_report,
    notify_discord_warning,
    rebuild_pg_atomic,
    split_by_model_availability,
)
from src.strategy_wt import load_7a_pool_axis_sums
from src.wt_vintage_config import bad_model_name, monthly_windows, tail_windows

_RANK_LABEL = "RANK_7A"
_DELETE_COND = "rank='RANK_7A' AND race_key LIKE '%#7A' AND race_date BETWEEN ? AND ?"
_SCRIPT_NAME = "rebuild_7a_walkforward_pg.py"


def _parse_upto(v: str | None):
    """`--upto` を date へ。未指定なら None（＝当日まで）。"""
    return date.fromisoformat(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-legacy-axis", action="store_true",
                    help="3ヘッド軸選定の適用期間(THREE_HEAD_AXIS_SINCE以降)を"
                         "旧軸で塗り直すことを明示的に許可する。liveの3ヘッド記録が"
                         "失われるため、意図がある場合のみ使う（2026-08-04追加）")
    ap.add_argument("--tail-only", action="store_true",
                     help="直近月（今月）の窓のみ再構築する日次軽量運用向けオプション。"
                          "確定済み過去月は結果が変わらないため毎日再計算する必要がなく、"
                          "これのみ再実行すれば直近日をhonestな状態に保てる。")
    ap.add_argument("--skip-missing-models", action="store_true",
                     help="vintageモデルpklが存在しない月をスキップして処理を継続する。"
                          "指定しない場合、モデル不足を検出した時点で計算を一切開始せず"
                          "即座にエラー終了する（全期間計算後に失敗して結果を失うのを防ぐ・"
                          "2026-08-01のm2608不足によるFileNotFoundError実害を踏まえた対応）。")
    ap.add_argument("--upto", metavar="YYYY-MM-DD", default=None,
                    help="この日までを再構築する（当日を含めたくないときに使う）。\n"
                         "monthly_windows は既定で当日を含み、結果未確定のレースは\n"
                         "再構築で戻せないため、実行すると当日の行が消える。")
    args = ap.parse_args()

    # --tail-only は当日を含めない（tail_windows の docstring 参照）。
    # 当日分を削除すると再構築では戻せず、Web から推奨が消えるため。
    windows = (tail_windows() if args.tail_only
               else monthly_windows(_parse_upto(args.upto)))

    # --- 事前チェック: build_rows(重い計算)を始める前に全窓のモデル存在を検証 ---
    # 7A は 3ヘッド軸（軸2 = argmax z(3着内率) − 0.3×z(大敗率)）で再構築するため、
    # 大敗モデルの vintage（lgbm_wt_bad_mYYMM）も存在チェックの対象に含める。
    available, missing = split_by_model_availability(windows, require_bad=True)
    if missing:
        report = format_missing_report(_RANK_LABEL, missing)
        print(report)
        if not args.skip_missing_models:
            notify_discord_warning(
                f"🚨 **[{_SCRIPT_NAME}] vintageモデル不足のため計算を開始せず中断しました**\n"
                f"{report}\n"
                f"`train_monthly_vintage_models.py --only-missing` で不足月を学習するか、"
                f"`--skip-missing-models` を指定して当該月を除外して続行してください。"
            )
            print(f"[rebuild-7a-pg] --skip-missing-models 未指定のため処理を中断します"
                  f"（計算は一切行っていません）。")
            sys.exit(1)
        notify_discord_warning(
            f"⚠️ **[{_SCRIPT_NAME}] vintageモデル不足を検出、"
            f"--skip-missing-models により当該窓を除外して続行します**\n{report}"
        )
        print(f"[rebuild-7a-pg] --skip-missing-models 指定のため{len(missing)}窓を"
              f"除外して続行します。")

    windows = available
    if not windows:
        print("[rebuild-7a-pg] 処理対象の窓がありません（全窓でモデル不足）。終了します。")
        sys.exit(1)

    per_window_rows: list[tuple[str, str, list[dict]]] = []
    all_rows: list[dict] = []
    # 低配当見送りゲートの母集団。窓をまたいで時系列に引き継ぐ（引き継がないと
    # 各窓の先頭がフォールバック閾値で走り、live と閾値がずれる）。
    # 先頭の窓は live が書いた `_s7a_pool.json` があればそこから種を貰う。
    pool_history: list[tuple[str, float]] = []
    if windows:
        seed = load_7a_pool_axis_sums(Path(__file__).parent.parent / "data" / "picks",
                                      windows[0][0])
        pool_history.extend((windows[0][0], v) for v in seed)
        if seed:
            print(f"[rebuild-7a-pg] ゲート閾値の種: プールJSONから{len(seed)}件", flush=True)
    for date_from, date_to, eval_model, win_model in windows:
        bad_model = bad_model_name(eval_model)
        print(f"\n[rebuild-7a-pg] {date_from}〜{date_to}  eval={eval_model} "
              f"win={win_model} bad={bad_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model,
                          bad_model_name=bad_model, pool_history=pool_history)
        n_hit = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        n_days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
        print(f"[rebuild-7a-pg]   7A: {len(rows)}R ({len(rows)/n_days:.1f}R/日) 的中{n_hit} "
              f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%", flush=True)
        per_window_rows.append((date_from, date_to, rows))
        all_rows.extend(rows)

    total_hit = sum(r["hit"] for r in all_rows)
    total_bet = sum(r["bet_amount"] for r in all_rows)
    total_pay = sum(r["payout"] for r in all_rows)
    print(f"\n[rebuild-7a-pg] ===== 全期間合計 =====")
    print(f"[rebuild-7a-pg] 7A: {len(all_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_rows) * 100 if all_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    # axis_is_three_head=True: build_rows に月次vintageの大敗モデルを渡して
    # **本番と同じ3ヘッド軸**で作り直しているため、「旧軸で3ヘッド期間を塗り潰す」
    # ガードの対象外。allow_legacy_axis（旧軸のまま強行）とは意味が異なる。
    rebuild_pg_atomic(_RANK_LABEL, _DELETE_COND, per_window_rows, args.dry_run,
                      allow_legacy_axis=args.allow_legacy_axis,
                      axis_is_three_head=True)


if __name__ == "__main__":
    main()
