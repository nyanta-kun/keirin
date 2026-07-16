#!/bin/bash
# 30分ごと実行: 当日の確定結果を kiseki に反映
#
# crontab 設定例（8:30〜23:00 の毎時 00分と 30分）:
#   0,30 8-23 * * * cd /Users/ysuzuki/GitHub/keirin && KEIRIN_DB_URL=... .venv/bin/bash scripts/results_check_wt.sh >> data/logs/results_check.log 2>&1
#
# 処理内容:
#   1. collect-wt --date TODAY: 確定済みレースの finish_order / wt_odds を更新
#   2. notify_results_wt.py TODAY --silent: picks_history を採点・更新（Discord 通知なし）
#      ※ _write_miwokuri は finish_order > 0 の確定レースのみ miwokuri=TRUE にする（未来レース不変）
#   3. write_candidates_wt.py TODAY: notify_results_wt.py が DELETE した未来レースの #CAND を復元
#   4. migrate_sqlite_to_pg.py: VPS PostgreSQL に同期
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] === 当日結果確認 $TODAY ==="

# 1. 確定済みレースのデータ再収集（finish_order/wt_odds 更新）
echo "[$(date '+%H:%M:%S')] 当日($TODAY) 結果収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 収集に失敗（継続）"

# 2. picks_history 採点・更新（Discord通知なし）
echo "[$(date '+%H:%M:%S')] picks_history 採点・更新..."
.venv/bin/python3 scripts/notify_results_wt.py "$TODAY" --silent \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 採点に失敗（継続）"

# 3. 未来レースの #CAND を復元（notify_results_wt.py が DELETE した分を戻す）
echo "[$(date '+%H:%M:%S')] 未来レース候補を復元..."
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 候補復元に失敗（継続）"

# 4. VPS PostgreSQL 同期
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS PostgreSQL 同期..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py \
    >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] === 当日結果確認 完了 ==="
