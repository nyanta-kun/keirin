#!/bin/bash
# 日中(レース開催時間帯)に1時間毎実行: 当日のレース結果を逐次取得し VPS に同期する。
# collect-wt は finish_order>=1 のレースをスキップするため、終了済みは再取得せず
# 未終了レースのみ取りに行く（時間が進むほど軽くなる）。
# 注: wt_odds は最新オッズに更新されるが、朝オッズは wt_odds_snapshot に保全済（別テーブル）。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %H:%M:%S')] 日中 当日結果取得 $TODAY ..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/intraday_${TODAY}.log"
echo "[$(date '+%H:%M:%S')] 日中取得 完了"

# 採点（Discord通知なし）: picks_history.payout を SQLite で更新
echo "[$(date '+%H:%M:%S')] 日中採点（--silent）..."
.venv/bin/python3 scripts/notify_results_wt.py "$TODAY" --silent \
  2>&1 >> "$LOG_DIR/intraday_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 日中採点に失敗（継続）"

# VPS PostgreSQL 同期（wt_races.status / wt_entries.finish_order / picks_history.payout を反映）
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS 同期（wt_races + wt_entries + picks_history）..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py --skip wt_odds_snapshot \
    2>&1 >> "$LOG_DIR/migrate_pg_intraday_${TODAY}.log" \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] 日中処理 完了"
