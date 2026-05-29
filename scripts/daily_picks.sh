#!/bin/bash
# 毎日8:00実行: データ収集 → 波乱PICK生成
set -e
cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" "data/picks"

echo "[$(date '+%H:%M:%S')] === 日次予想生成開始 $DATE ==="
.venv/bin/python3 -m src.cli.main collect --date "$DATE" 2>&1 | tee -a "$LOG_DIR/collect_${DATE}.log"
echo "[$(date '+%H:%M:%S')] データ収集完了"
.venv/bin/python3 -m src.cli.main wave-picks --date "$DATE" 2>&1 | tee -a "$LOG_DIR/picks_${DATE}.log"
echo "[$(date '+%H:%M:%S')] 予想生成完了: data/picks/wave_picks_${DATE}.txt"
