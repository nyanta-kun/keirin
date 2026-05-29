#!/bin/bash
# 毎週日曜23:00実行: 直近7日のデータ収集 → モデル再学習
set -e
cd "$(dirname "$0")/.."
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y-%m-%d)

echo "[$(date '+%H:%M:%S')] === 週次モデル更新開始 ==="

# 直近7日の結果収集
for i in 1 2 3 4 5 6 7; do
  # macOS: date -v, Linux: date -d
  if [[ "$(uname)" == "Darwin" ]]; then
    D=$(date -v-${i}d +%Y-%m-%d)
  else
    D=$(date -d "-${i} days" +%Y-%m-%d)
  fi
  echo "[$(date '+%H:%M:%S')] 収集: $D"
  .venv/bin/python3 -m src.cli.main collect --date "$D" 2>&1 | tee -a "$LOG_DIR/collect_weekly_${DATE}.log" || true
done

# モデル再学習
echo "[$(date '+%H:%M:%S')] モデル再学習開始..."
.venv/bin/python3 -m src.cli.main train \
  --model lgbm \
  --from 2024-06-01 \
  --test-from 2026-03-01 \
  --save-as lgbm_v4 \
  2>&1 | tee -a "$LOG_DIR/train_${DATE}.log"

echo "[$(date '+%H:%M:%S')] === 週次モデル更新完了 ==="
