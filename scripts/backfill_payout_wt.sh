#!/bin/bash
# picks_history の payout バックフィル
# wave_picks_wt_YYYY-MM-DD.txt が存在する全日付に対して
# notify_results_wt.py --silent を再実行し、正しい payout を VPS に書き込む。
#
# 使い方:
#   bash scripts/backfill_payout_wt.sh            # 全日付
#   bash scripts/backfill_payout_wt.sh 2026-04    # 指定月のみ

set -o pipefail
cd "$(dirname "$0")/.."

# crontab から KEIRIN_DB_URL を取得
if [[ -z "$KEIRIN_DB_URL" ]]; then
  KEIRIN_DB_URL=$(crontab -l 2>/dev/null | grep "^KEIRIN_DB_URL=" | head -1 | cut -d= -f2-)
fi
if [[ -z "$KEIRIN_DB_URL" ]]; then
  echo "ERROR: KEIRIN_DB_URL が未設定です。crontab に KEIRIN_DB_URL= がありません。"
  exit 1
fi
export KEIRIN_DB_URL

FILTER="${1:-}"   # 例: 2026-04 で月フィルタ
TODAY=$(date +%Y-%m-%d)
LOG="data/logs/backfill_payout_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/logs

echo "=== picks_history payout バックフィル ===" | tee "$LOG"
echo "VPS: $(echo "$KEIRIN_DB_URL" | sed 's/:\/\/[^:]*:[^@]*@/\/\/***:***@/')" | tee -a "$LOG"
echo "フィルタ: ${FILTER:-（なし・全日付）}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 処理対象日付を列挙（当日は除外）
DATES=$(ls data/picks/ \
  | grep "^wave_picks_wt_" \
  | grep -v "_allindex\|_detail\|_night\|_candidates" \
  | sed 's/wave_picks_wt_//' | sed 's/\.txt//' \
  | sort \
  | grep -v "^$TODAY$")

if [[ -n "$FILTER" ]]; then
  DATES=$(echo "$DATES" | grep "^$FILTER")
fi

TOTAL=$(echo "$DATES" | grep -c ".")
echo "対象日数: $TOTAL 日" | tee -a "$LOG"
echo "" | tee -a "$LOG"

OK=0; NG=0; idx=0
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  idx=$((idx+1))
  echo -n "[$idx/$TOTAL] $d ... " | tee -a "$LOG"
  if .venv/bin/python3 scripts/notify_results_wt.py --silent "$d" >> "$LOG" 2>&1; then
    echo "OK" | tee -a "$LOG"
    OK=$((OK+1))
  else
    echo "SKIP/NG" | tee -a "$LOG"
    NG=$((NG+1))
  fi
done <<< "$DATES"

echo "" | tee -a "$LOG"
echo "=== 完了: 成功 $OK / スキップ $NG / 計 $TOTAL ===" | tee -a "$LOG"
echo "ログ: $LOG"
