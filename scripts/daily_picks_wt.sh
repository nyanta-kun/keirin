#!/bin/bash
# 毎日7:00実行（winticketルート）: 前日成績通知 → 当日データ収集 → 予想生成・通知
# 2026-06-08 ks→wt 完全移行。ksスクレイピングは廃止。
set -e
set -o pipefail   # L-5: | tee が python の終了コードをマスクしないように
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
# 前日処理は当日予想の前提ではないため、失敗しても継続（pipefailで失敗は可視化）。
echo "[$(date '+%H:%M:%S')] 前日($YESTERDAY) winticket結果再収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${YESTERDAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 前日再収集に失敗（継続）"

echo "[$(date '+%H:%M:%S')] 前日成績をDiscordへ通知..."
.venv/bin/python3 scripts/notify_results_wt.py "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${YESTERDAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 前日成績通知に失敗（継続）"

# --- 2. 当日予想 ---
# 当日収集は予想の前提＝失敗時は中断（pipefail+set -e で異常を握り潰さない）。
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

echo "[$(date '+%H:%M:%S')] 予想生成（winticket・<3倍見送り/3〜5倍未満はBランク）..."
# --gami-skip-odds 3.0: 3点中1点でも朝オッズ<3倍ならレース見送り（明確なガミ）。
# --b-rank-odds 5.0:   最安目が3〜5倍未満ならBランク（鉄板寄り・購入は各自判断）として別枠表示。
# 検証: scripts/analyze_gami_threshold_wt.py（<3倍点含むレースは集団で収支ゼロ）。
# wave-picks-wt は対象レース0件で exit 1（＝静かな日。異常ではない）になり得るため継続。
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  --gami-skip-odds 3.0 --b-rank-odds 5.0 \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 予想生成: 対象レース無し or 失敗（継続）"

echo "[$(date '+%H:%M:%S')] 予想をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" wave_picks_wt \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 予想通知に失敗（継続）"

echo "[$(date '+%H:%M:%S')] === winticket日次処理完了 ==="
