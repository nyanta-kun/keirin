#!/bin/bash
# 毎日00:50実行: S7/S9の直近ウィンドウ（2026-04-13〜昨日）のみを
# 現行eval model（lgbm_wt_eval/lgbm_wt_win_eval）でhonestに再構築する。
#
# 背景: daily_picks_wt.sh/evening_picks_wt.sh が書き込む当日の候補行
# （write_candidates_wt.py・gap12条件のみの広い候補プール）は、
# rebuild_7s_walkforward_pg.py/rebuild_9s_walkforward_pg.py が計算する最終選出後の honest な
# picks（axis選定・entropy/日次capゲート適用済み）より件数が多い。
# 放置すると「直近日だけ候補数が多い」状態が積み上がり、四半期vintage
# モデルで再構築済みの過去期間と条件が食い違う（2026-07-27 ユーザー指摘）。
#
# 四半期分のvintageモデルは確定済みで結果が変わらないため毎日再計算する
# 必要はなく、--tail-only で末尾ウィンドウのみ再構築すれば足りる
# （backfill_missing_prerace_wt.py の 00:40 実行の後、daily_picks_wt.sh の
# 08:00 より前に完了させる）。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/reconcile_tail_${DATE}.log"

echo "[$(date '+%H:%M:%S')] === walk-forward tail再構築 開始 ===" | tee -a "$LOG"

# 2026-07-31: S1(SEVEN_S1)はユーザー判断により全廃（commit df31431）・
# picks_historyのSEVEN_S1行1,504件もVPS PGから削除済み
# （バックアップ: data/backup/picks_history_s1_discarded_20260731.csv）。
# 本行はレビューで「削除済みのSEVEN_S1行をrebuild_s1_walkforward_pg.pyが
# tail-only再構築のたびにDELETE→INSERTで自動再生成してしまう経路」として
# 検出されたため呼び出しを除去した（backfill_missing_prerace_wt.pyのS1除外
# 対応と同型の事故予防・CLAUDE.mdの「ランク全廃時は候補生成/ライブ判定/
# 欠損自動補完の3箇所すべて停止」に加えて本スクリプトが第4の経路だった）。
# rebuild_s1_walkforward_pg.py本体は過去日再採点・分析用に残置し、手動実行
# 専用とする（同スクリプトのdocstring参照）。

.venv/bin/python3 scripts/rebuild_7s_walkforward_pg.py --tail-only 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] S7 tail再構築 失敗（継続）" | tee -a "$LOG"

.venv/bin/python3 scripts/rebuild_9s_walkforward_pg.py --tail-only 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] S9 tail再構築 失敗（継続）" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === walk-forward tail再構築 完了 ===" | tee -a "$LOG"
