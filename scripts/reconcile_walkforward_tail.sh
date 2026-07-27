#!/bin/bash
# 毎日00:50実行: S1/S4/S9の直近ウィンドウ（2026-04-13〜昨日）のみを
# 現行eval model（lgbm_wt_eval/lgbm_wt_win_eval）でhonestに再構築する。
#
# 背景: daily_picks_wt.sh/evening_picks_wt.sh が書き込む当日の候補行
# （write_candidates_wt.py・gap12条件のみの広い候補プール）は、
# rebuild_s1/s4/s9_walkforward_pg.py が計算する最終選出後の honest な
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

.venv/bin/python3 scripts/rebuild_s1_walkforward_pg.py --tail-only 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] S1 tail再構築 失敗（継続）" | tee -a "$LOG"

.venv/bin/python3 scripts/rebuild_s4_walkforward_pg.py --tail-only 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] S4 tail再構築 失敗（継続）" | tee -a "$LOG"

.venv/bin/python3 scripts/rebuild_s9_walkforward_pg.py --tail-only 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] S9 tail再構築 失敗（継続）" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === walk-forward tail再構築 完了 ===" | tee -a "$LOG"
