#!/bin/bash
# 日中(レース開催時間帯)に1時間毎実行: 当日のレース結果を逐次取得する。
# collect-wt は finish_order>=1 のレースをスキップするため、終了済みは再取得せず
# 未終了レースのみ取りに行く（時間が進むほど軽くなる）。通知はしない（採点・通知は翌朝7:00）。
# 注: wt_odds は最新オッズに更新されるが、朝オッズは wt_odds_snapshot に保全済（別テーブル）。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %H:%M:%S')] 日中 当日結果取得 $TODAY ..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/intraday_${TODAY}.log"
echo "[$(date '+%H:%M:%S')] 日中取得 完了"
