#!/bin/bash
# 毎週日曜23:30実行（winticketルート）: wtモデル再学習
# H-1対応: ①holdout評価(昇格せず・監視用) → ②全データ再学習で配信モデル(lgbm_wt)生成
#          → ③カット再計測 → ④世代退避(ロールバック用)
set -e
set -o pipefail   # L-5: | tee が python の終了コードをマスクしないように
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" data/models/archive
LOG="$LOG_DIR/train_wt_${DATE}.log"

# テスト分割は直近約90日前（ホールドアウト評価用）
if [[ "$(uname)" == "Darwin" ]]; then
  TEST_FROM=$(date -v-90d +%Y-%m-%d)
else
  TEST_FROM=$(date -d "90 days ago" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket週次再学習 $DATE (test-from=$TEST_FROM) ===" | tee -a "$LOG"

# ① ホールドアウト評価モデル（監視用・--no-promote で本番 lgbm_wt は汚さない）
echo "[$(date '+%H:%M:%S')] ① holdout評価（直近90日をテスト）..." | tee -a "$LOG"
.venv/bin/python3 -m src.cli.main train-wt \
  --from 2023-07-01 --test-from "$TEST_FROM" --save-as lgbm_wt_eval --no-promote \
  2>&1 | tee -a "$LOG"

# ② 配信モデル: 全データで再学習して lgbm_wt を更新（H-1）
echo "[$(date '+%H:%M:%S')] ② 配信用: 全データ再学習 → lgbm_wt ..." | tee -a "$LOG"
.venv/bin/python3 -m src.cli.main train-wt \
  --from 2023-07-01 --full-refit --save-as lgbm_wt \
  2>&1 | tee -a "$LOG"

# ③ 波乱ゲート top3_sum カット定数を配信モデルの分布で再計測（test期間除外）
echo "[$(date '+%H:%M:%S')] ③ 波乱カット定数を再計測..." | tee -a "$LOG"
.venv/bin/python3 scripts/recompute_upset_cuts_wt.py --to "$TEST_FROM" \
  2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] カット再計測に失敗/単調性NG（既定値を維持・処理は継続）"

# ④ 世代退避（M-5・ロールバック/再現用。モデル・メタ・カットを日付付きで保存）
echo "[$(date '+%H:%M:%S')] ④ 世代退避 → data/models/archive/ ..." | tee -a "$LOG"
cp -f data/models/lgbm_wt.pkl        "data/models/archive/lgbm_wt_${DATE}.pkl"        2>/dev/null || true
cp -f data/models/lgbm_wt.meta.json  "data/models/archive/lgbm_wt_${DATE}.meta.json"  2>/dev/null || true
cp -f data/models/upset_cuts_wt.json "data/models/archive/upset_cuts_wt_${DATE}.json" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] === 完了 ===" | tee -a "$LOG"
