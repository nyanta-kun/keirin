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

# --- 朝オッズ前向き計測: 収集直後の wt_odds(=朝オッズ) を退避 ---
# 翌日の前日再収集で wt_odds が最終オッズに上書きされる前に保全する。
# 失敗しても日次処理は止めない（計測は補助目的）。
echo "[$(date '+%H:%M:%S')] 朝オッズをスナップショット退避..."
.venv/bin/python3 scripts/snapshot_morning_odds_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/odds_snapshot_${TODAY}.log" || \
  echo "[$(date '+%H:%M:%S')] 朝オッズ退避に失敗（処理は継続）"

echo "[$(date '+%H:%M:%S')] 予想生成（winticket・ガミ回避<5倍）..."
# --gami-skip-odds 5.0: 3点中1点でも朝オッズ<5倍ならレース見送り（鉄板=低価値の除外）。
# 検証: 5倍で総利益ほぼ維持・ROIほぼ倍・落車クッション増（scripts/analyze_gami_threshold_wt.py）。
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" --gami-skip-odds 5.0 \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] 予想をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" wave_picks_wt \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${TODAY}.log"

echo "[$(date '+%H:%M:%S')] === winticket日次処理完了 ==="
