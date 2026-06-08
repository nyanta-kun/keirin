#!/bin/bash
# 毎日7:00実行（winticketルート）: 前日成績通知 → 当日データ収集 → 予想生成・通知
# 2026-06-08 ks→wt 完全移行。ksスクレイピングは廃止。
set -e
# cron環境のPATHには /usr/sbin が無く joblib のCPUコア検出(sysctl)が警告を出すため追加
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" "data/picks"

if [[ "$(uname)" == "Darwin" ]]; then
  YESTERDAY=$(date -v-1d +%Y-%m-%d)
else
  YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket日次処理開始 $TODAY ==="

# --- 1. 前日成績（winticketで結果再収集→採点通知）---
echo "[$(date '+%H:%M:%S')] 前日($YESTERDAY) winticket結果再収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${YESTERDAY}.log"

echo "[$(date '+%H:%M:%S')] 前日成績をDiscordへ通知..."
.venv/bin/python3 scripts/notify_results_wt.py "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${YESTERDAY}.log"

# --- 2. 当日予想 ---
echo "[$(date '+%H:%M:%S')] 当日($TODAY) winticketデータ収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] 予想生成（winticket）..."
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] 予想をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" wave_picks_wt \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] === winticket日次処理完了 ==="
