#!/usr/bin/env bash
# 開催の波ごとの netkeirin 入稿（2026-08-07 新設）
#
#   使い方: scripts/wave_submit_wt.sh {noon|evening} [YYYY-MM-DD]
#
# 三連複の板は「時計時刻」ではなく「発走までの近さ」で埋まる。朝8時台の未確定率は
# 〜10時台発走 0.8% に対し 20時以降発走 **63.4%**。netkeirin は公開後に差し替えが
# できないので、夜の開催は板が育ってから入稿する（`src/meeting_wave.py` が正本）。
#
#   cron 07:00  daily_picks_wt.sh          … モーニング・デイ（第1R < 12時）
#   cron 13:00  wave_submit_wt.sh noon     … ナイター（第1R 12〜17時台・1R 15時に間に合う）
#   cron 18:00  wave_submit_wt.sh evening  … ミッドナイト（第1R 18時〜・1R 20時に間に合う）
#
# ⚠️ この回は**入稿だけ**を行う。予想（picks_history・Discord・Web）は朝の
#    daily_picks_wt.sh が当日全開催ぶんを出し終えている。
# ⚠️ オッズは別途取りに行かない。`intraday_results_wt.sh` が 8:00 以降 15分ごとに
#    wt_odds を更新しているので、この時点の板は最大でも15分前の値。
set -euo pipefail

SESSION="${1:-}"
if [[ "$SESSION" != "noon" && "$SESSION" != "evening" ]]; then
  echo "usage: $0 {noon|evening} [YYYY-MM-DD]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

if [[ -n "${2:-}" ]]; then
  TODAY="$2"
elif [[ "$(uname)" == "Darwin" ]]; then
  TODAY=$(date +%Y-%m-%d)
else
  TODAY=$(date +%Y-%m-%d)
fi

# --- KEIRIN_DB_URL 必須チェック ---
# 開催の波は wt_races.start_at から決めるため、DB が無いと**全開催が「朝」扱いに
# 倒れて**（発走時刻不明のフォールバック）この回で何も出せない。黙って0件に
# なるのが最悪なので、未設定なら明示的に落とす。
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  echo "[$(date '+%H:%M:%S')] [FATAL] KEIRIN_DB_URL が未設定です。wave_submit_wt.sh (${SESSION}) を中断します。"
  PYTHONPATH=. .venv/bin/python3 -c "
from src.notify.discord import send
send('🚨 **[wave_submit_wt.sh] KEIRIN_DB_URL が未設定のため入稿を中断しました。**\ncrontabの環境変数設定を確認してください。', channel='system')
" || true
  exit 1
fi

echo "[$(date '+%H:%M:%S')] === netkeirin 入稿（波: ${SESSION}） $TODAY ==="
PYTHONPATH=. .venv/bin/python3 scripts/netkeirin_submit_wt.py "$TODAY" "$SESSION" \
  2>&1 | tee -a "$LOG_DIR/netkeirin_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] netkeirin入稿(${SESSION})に失敗（継続）"
echo "[$(date '+%H:%M:%S')] === 入稿（波: ${SESSION}） 完了 ==="
