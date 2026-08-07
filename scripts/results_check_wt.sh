#!/bin/bash
# 30分ごと実行: 当日の確定結果を kiseki に反映
#
# crontab 設定例（8:30〜23:00 の毎時 00分と 30分）:
#   0,30 8-23 * * * cd /Users/ysuzuki/GitHub/keirin && KEIRIN_DB_URL=... .venv/bin/bash scripts/results_check_wt.sh >> data/logs/results_check.log 2>&1
#
# 処理内容:
#   1. collect-wt --date TODAY: 確定済みレースの finish_order / wt_odds を更新
#   2. notify_results_wt.py TODAY --silent: picks_history を採点・更新（Discord 通知なし）
#      ※ _write_miwokuri は finish_order > 0 の確定レースのみ miwokuri=TRUE にする（未来レース不変）
#   3. write_candidates_wt.py TODAY: notify_results_wt.py が DELETE した未来レースの #CAND を復元
#   4. migrate_sqlite_to_pg.py: VPS PostgreSQL に同期
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

# --- 多重起動防止（2026-07-31 D-2）---
# 30分毎の想定実行のため、collect-wt が長引くと次回発火と重複実行され、
# wt_races/wt_entries/picks_history への同時書き込み・削除が競合する
# （2026-07-08 prerace_decisions/notified 同時消失事故と同型のリスク）。
# flock は VPS(util-linux)で利用可能と確認済み(2026-07-31)。ロック取得失敗時は
# 「前回が継続中」とみなしスキップする（スキップの発生は lock_skips.log に
# 蓄積するので、頻発していれば前回がハングしていないか確認すること）。
LOCK_FILE="$LOG_DIR/results_check_wt.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[$(date '+%H:%M:%S')] [results_check_wt] 前回実行がロック中のためスキップします（${LOCK_FILE}）。" \
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
  echo "[$(date '+%H:%M:%S')] [results_check_wt] 共有ロック待ちが30分を超えました（${SHARED_LOCK}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 1
fi

# --- KEIRIN_DB_URL 必須チェック（2026-07-31 D-1）---
# database.py の get_connection() は KEIRIN_DB_URL 未設定時に RuntimeError を送出する
# 設計だが、本スクリプトの各処理は `|| echo "...失敗（継続）"` で握り潰しているため、
# crontab 編集ミス等でこの変数が消えると結果反映が全て空振りしつつ script 全体は
# exit 0 で完走してしまう。ここで早期に検知して中断する。
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  echo "[$(date '+%H:%M:%S')] [FATAL] KEIRIN_DB_URL が未設定です。results_check_wt.sh を中断します。" \
    | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  # Discord Webhook URL は .env から直接読む実装（src/notify/discord.py::_load_webhook_url）
  # のため、DB接続が無くても通知は送信できる（通知経路はKEIRIN_DB_URLに依存しない）。
  if .venv/bin/python3 -c "
from src.notify.discord import send
ok = send('🚨 **[results_check_wt.sh] KEIRIN_DB_URL が未設定のため処理を中断しました。**\ncrontabの環境変数設定を確認してください。', channel='system')
raise SystemExit(0 if ok else 1)
" 2>&1 | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log"; then
    echo "[$(date '+%H:%M:%S')] Discordへ中断を通知しました。" \
      | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  else
    echo "[$(date '+%H:%M:%S')] [FATAL] Discord通知にも失敗しました（.envのDISCORD_WEBHOOK_URL_SYSTEM未設定などが原因の可能性）。cronログ（標準エラー）で検知してください。" \
      | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  fi
  exit 1
fi

echo "[$(date '+%H:%M:%S')] === 当日結果確認 $TODAY ==="

# 1. 確定済みレースのデータ再収集（finish_order/wt_odds 更新）
echo "[$(date '+%H:%M:%S')] 当日($TODAY) 結果収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 収集に失敗（継続）"

# 2. picks_history 採点・更新（Discord通知なし）
echo "[$(date '+%H:%M:%S')] picks_history 採点・更新..."
.venv/bin/python3 scripts/notify_results_wt.py "$TODAY" --silent \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 採点に失敗（継続）"

# 3. 未来レースの #CAND を復元（notify_results_wt.py が DELETE した分を戻す）
echo "[$(date '+%H:%M:%S')] 未来レース候補を復元..."
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 候補復元に失敗（継続）"

# 4. VPS PostgreSQL 同期
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS PostgreSQL 同期..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py \
    >> "$LOG_DIR/results_check_${TODAY}.log" 2>&1 \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] === 当日結果確認 完了 ==="
