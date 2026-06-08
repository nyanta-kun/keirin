#!/bin/bash
# 毎日7:00実行: 前日成績通知 → 当日データ収集 → 予想生成・通知
set -e
# cron環境のPATHには /usr/sbin が無く joblib のCPUコア検出(sysctl)が警告を出すため追加
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" "data/picks"

# 前日の日付（macOS / Linux 対応）
if [[ "$(uname)" == "Darwin" ]]; then
  YESTERDAY=$(date -v-1d +%Y-%m-%d)
else
  YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === 日次処理開始 $TODAY ==="

# --- 1. 前日成績 ---
echo "[$(date '+%H:%M:%S')] 前日($YESTERDAY)データ再収集..."
.venv/bin/python3 -m src.cli.main collect --date "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/collect_${YESTERDAY}.log"

echo "[$(date '+%H:%M:%S')] 前日成績をDiscordへ通知..."
.venv/bin/python3 scripts/notify_results.py "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/notify_${YESTERDAY}.log"

# --- 2. 当日予想 ---
echo "[$(date '+%H:%M:%S')] 当日($TODAY)データ収集..."
.venv/bin/python3 -m src.cli.main collect --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/collect_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] 予想生成..."
.venv/bin/python3 -m src.cli.main wave-picks --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] 予想をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/notify_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] Xポスト用テキスト送信..."
.venv/bin/python3 scripts/tweet_picks.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/notify_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] === 日次処理完了 ==="
