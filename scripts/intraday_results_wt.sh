#!/bin/bash
# 日中(レース開催時間帯)に15分毎実行: 当日のレース結果を逐次取得し VPS に同期する。
# collect-wt は finish_order>=1 のレースをスキップするため、終了済みは再取得せず
# 未終了レースのみ取りに行く（時間が進むほど軽くなる）。15分毎に上げても負荷は小さい。
# 注: wt_odds は最新オッズに更新されるが、朝オッズは wt_odds_snapshot に保全済（別テーブル）。
# 注: 0:00 実行時は前日最終レース（23時台発走分）も追加取得する。
#     最終レースが 23:00 以降に発走した場合、23:00台の実行では結果未確定であり
#     00:00 で TODAY が翌日になって取りこぼされるため、これを防ぐ。
#
# Discordダイジェスト送信について（2026-07-24〜）:
#   picks_history のDB更新（採点）自体は毎回サイレントに行うが、Discordの
#   「results」チャンネルへは DIGEST_HOURS で指定した時刻(毎時0分)のみ送信する
#   （15分毎に毎回送信するとスパムになるため）。当日の最終・正式版ダイジェストは
#   翌朝8:00の daily_picks_wt.sh（前日分・非silent）が引き続き担う。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

# --- 多重起動防止（2026-07-31 D-2）---
# 15分毎cronのため、--full-scan を伴う collect-wt が winticket 側の遅延等で
# 15分を超えると次の */15 発火と重複実行され、両プロセスが wt_races/wt_entries/
# picks_history へ同時に書き込み・削除を行う（write_candidates_wt の候補復元と
# notify_results_wt の未来レース削除が交錯しうる。2026-07-08 prerace_decisions/
# notified 同時消失事故と同型のリスク）。flock は VPS(util-linux)で利用可能と
# 確認済み(2026-07-31)。ロック取得失敗時は「前回が継続中」とみなしスキップする
# （スキップの発生は lock_skips.log に蓄積するので、頻発していれば前回がハング
# していないか確認すること）。
LOCK_FILE="$LOG_DIR/intraday_results_wt.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[$(date '+%F %H:%M:%S')] [intraday_results_wt] 前回実行がロック中のためスキップします（${LOCK_FILE}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 0
fi

# --- 共有ロック: picks_history へ書く処理どうしの競合を防ぐ（2026-08-08）---
# 上の 200 番は「自分自身の多重起動」しか防がない。picks_history を書き換える
# 処理は本スクリプトのほかに reconcile_walkforward_tail.sh（毎日08:30・当月分を
# DELETE→INSERT で再構築）があり、こちらが遅延して 08:30 に食い込むと同じ行を
# 同時に触る。2026-08-06 の rebuild行×live行 混在と同型の事故になる。
# 対策は「実行時刻をずらす」だけだったので、構造的な排他を足す。
# ⚠️ **待つ（-w）**。-n でスキップすると朝の予想生成が黙って丸ごと落ちる。
#    ロックは単一なのでデッドロックしない（必ず自分のロック→共有ロックの順）。
SHARED_LOCK="$LOG_DIR/wt_picks_writer.lock"
exec 201>"$SHARED_LOCK"
if ! flock -w 1800 201; then
  echo "[$(date '+%H:%M:%S')] [intraday_results_wt] 共有ロック待ちが30分を超えました（${SHARED_LOCK}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 1
fi

# --- KEIRIN_DB_URL 必須チェック（2026-07-31 D-1）---
# database.py の get_connection() は KEIRIN_DB_URL 未設定時に RuntimeError を送出する
# 設計だが、本スクリプトの各処理は `|| echo "...失敗（継続）"` で握り潰しているため、
# crontab 編集ミス等でこの変数が消えると15分毎の結果反映が全て空振りしつつ script
# 全体は exit 0 で完走してしまう。ここで早期に検知して中断する。
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  echo "[$(date '+%F %H:%M:%S')] [FATAL] KEIRIN_DB_URL が未設定です。intraday_results_wt.sh を中断します。" \
    | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  # Discord Webhook URL は .env から直接読む実装（src/notify/discord.py::_load_webhook_url）
  # のため、DB接続が無くても通知は送信できる（通知経路はKEIRIN_DB_URLに依存しない）。
  if .venv/bin/python3 -c "
from src.notify.discord import send
ok = send('🚨 **[intraday_results_wt.sh] KEIRIN_DB_URL が未設定のため処理を中断しました。**\ncrontabの環境変数設定を確認してください。', channel='system')
raise SystemExit(0 if ok else 1)
" 2>&1 | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log"; then
    echo "[$(date '+%F %H:%M:%S')] Discordへ中断を通知しました。" \
      | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  else
    echo "[$(date '+%F %H:%M:%S')] [FATAL] Discord通知にも失敗しました（.envのDISCORD_WEBHOOK_URL_SYSTEM未設定などが原因の可能性）。cronログ（標準エラー）で検知してください。" \
      | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  fi
  exit 1
fi

CURRENT_HOUR=$(date +%H)
CURRENT_MIN=$(date +%M)

# 日中Discordダイジェストを送信する時刻（毎時0分実行時のみ判定・1日4回）
DIGEST_HOURS=" 12 15 18 21 "

# 0:00 実行時: 前日最終レース（23時台発走分）の結果を取得する
# cron スケジュール `*/15 10-23,0 * * *` のうち hour=0 かつ分0の実行が該当する
if [[ "$CURRENT_HOUR" == "00" && "$CURRENT_MIN" == "00" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    PREV=$(date -v-1d +%Y-%m-%d)
  else
    PREV=$(date -d "yesterday" +%Y-%m-%d)
  fi
  echo "[$(date '+%F %H:%M:%S')] 前日最終レース取得 $PREV (23時台以降発走分の取りこぼし回収)..."
  .venv/bin/python3 -m src.cli.main collect-wt --date "$PREV" \
    >> "$LOG_DIR/intraday_${PREV}.log" 2>&1 \
    || echo "[$(date '+%F %H:%M:%S')] 前日最終レース取得に失敗（継続）"
  .venv/bin/python3 scripts/notify_results_wt.py "$PREV" --silent \
    >> "$LOG_DIR/intraday_${PREV}.log" 2>&1 \
    || echo "[$(date '+%F %H:%M:%S')] 前日採点に失敗（継続）"
fi

echo "[$(date '+%F %H:%M:%S')] 日中 当日結果取得 $TODAY ..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/intraday_${TODAY}.log"
echo "[$(date '+%H:%M:%S')] 日中取得 完了"

# 採点: picks_history.payout を更新する。DIGEST_HOURS の毎時0分実行時のみ
# Discordの「results」チャンネルへ日中ダイジェスト（その時点までの当日確定分）を送信する。
NOTIFY_ARGS=(--silent)
if [[ "$CURRENT_MIN" == "00" && "$DIGEST_HOURS" == *" $CURRENT_HOUR "* ]]; then
  NOTIFY_ARGS=()
  echo "[$(date '+%H:%M:%S')] 日中採点（Discordダイジェスト送信あり: ${CURRENT_HOUR}:00）..."
else
  echo "[$(date '+%H:%M:%S')] 日中採点（--silent）..."
fi
.venv/bin/python3 scripts/notify_results_wt.py "$TODAY" "${NOTIFY_ARGS[@]}" \
  2>&1 >> "$LOG_DIR/intraday_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 日中採点に失敗（継続）"

# 未来レース候補を復元（notify_results_wt.py の DELETE で消えた #CAND を戻す）
echo "[$(date '+%H:%M:%S')] 未来レース候補復元..."
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  2>&1 >> "$LOG_DIR/intraday_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 候補復元に失敗（継続）"

# VPS PostgreSQL 同期（wt_races.status / wt_entries.finish_order / picks_history.payout を反映）
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS 同期（wt_races + wt_entries + picks_history）..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py --skip wt_odds_snapshot \
    2>&1 >> "$LOG_DIR/migrate_pg_intraday_${TODAY}.log" \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] 日中処理 完了"
