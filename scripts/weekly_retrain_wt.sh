#!/bin/bash
# 毎週日曜23:30実行（winticketルート）: wtモデル再学習
set -e
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

# テスト分割は直近約3ヶ月前（ホールドアウト確認用）
if [[ "$(uname)" == "Darwin" ]]; then
  TEST_FROM=$(date -v-90d +%Y-%m-%d)
else
  TEST_FROM=$(date -d "90 days ago" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket週次再学習 $DATE (test-from=$TEST_FROM) ==="
.venv/bin/python3 -m src.cli.main train-wt \
  --from 2023-07-01 --test-from "$TEST_FROM" --save-as lgbm_wt_v1 \
  2>&1 | tee -a "$LOG_DIR/train_wt_${DATE}.log"

# 波乱ゲート top3_sum カット定数を新モデルのtrain分布で再計測（test期間は除外）
echo "[$(date '+%H:%M:%S')] 波乱カット定数を再計測..."
.venv/bin/python3 scripts/recompute_upset_cuts_wt.py --to "$TEST_FROM" \
  2>&1 | tee -a "$LOG_DIR/train_wt_${DATE}.log" || \
  echo "[$(date '+%H:%M:%S')] カット再計測に失敗（既定値を維持・処理は継続）"

echo "[$(date '+%H:%M:%S')] === 完了 ==="
