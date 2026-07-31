#!/bin/bash
# 夕方再生成（2段階生成の第2段・cron 16:00想定）
# 朝(daily_picks_wt.sh)は --start-to-hour 19 で昼〜夕レースのみ推奨する。
# 夜レース(19時〜)はwtのライン構成が朝未公開→精度低下するため(docs B検証)、
# ラインが公開される午後に当日を再収集し、全レース(夜含む)で推奨を再生成・上書きして
# Discordへ「確定版」を再通知する。翌朝 notify_results_wt は最終(この夕方版)ファイルを採点。
# ※ksは合算バックテストで wt単独 に劣後と判明→稼働再開しない(wt単独・docs 2026-06-10)。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" "data/picks"

# --- 多重起動防止（2026-07-31 D-2）---
# 前段処理の遅延等で重複実行されると wt_races/wt_entries/picks_history への
# 同時書き込み・削除が競合する（2026-07-08 prerace_decisions/notified 同時消失
# 事故と同型のリスク）。flock は VPS(util-linux)で利用可能と確認済み(2026-07-31)。
# ロック取得失敗時は「前回が継続中」とみなしスキップする（スキップの発生は
# lock_skips.log に蓄積するので、頻発していれば前回がハングしていないか確認すること）。
LOCK_FILE="$LOG_DIR/evening_picks_wt.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[$(date '+%H:%M:%S')] [evening_picks_wt] 前回実行がロック中のためスキップします（$LOCK_FILE）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 0
fi

# --- KEIRIN_DB_URL 必須チェック（2026-07-31 D-1）---
# database.py の get_connection() は KEIRIN_DB_URL 未設定時に RuntimeError を送出する
# 設計だが、本スクリプトの各処理は `|| echo "...失敗（継続）"` で握り潰しているため、
# crontab 編集ミス等でこの変数が消えると夜の部の収集・予想生成・通知・netkeirin入稿が
# 全て空振りしつつ script 全体は exit 0 で完走してしまう。ここで早期に検知して中断する。
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  echo "[$(date '+%H:%M:%S')] [FATAL] KEIRIN_DB_URL が未設定です。evening_picks_wt.sh を中断します。" \
    | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  # Discord Webhook URL は .env から直接読む実装（src/notify/discord.py::_load_webhook_url）
  # のため、DB接続が無くても通知は送信できる（通知経路はKEIRIN_DB_URLに依存しない）。
  if .venv/bin/python3 -c "
from src.notify.discord import send
ok = send('🚨 **[evening_picks_wt.sh] KEIRIN_DB_URL が未設定のため処理を中断しました。**\ncrontabの環境変数設定を確認してください。', channel='system')
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

echo "[$(date '+%H:%M:%S')] === winticket 夕方再生成 $TODAY ==="

# 1. 当日再収集（全会場フルスキャン＝午後に公開された夜レースのライン/オッズを取得）
echo "[$(date '+%H:%M:%S')] 当日($TODAY) 再収集（全会場・夜ライン取得）..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}.log"

# 1b. 夕方オッズを退避（夜レースは朝オッズ未確定→夕方が実質「生成時オッズ」。
#     ワイド監視で夜レースの朝相当(夕方)→確定ドリフトを見るための基準。snapshot_type='evening'）
.venv/bin/python3 scripts/snapshot_morning_odds_wt.py "$TODAY" --type evening \
  >> "$LOG_DIR/odds_snapshot_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 夕方オッズ退避に失敗（継続）"

# 2. 夜レース(19時〜)のみ推奨生成→専用ファイル(_night)へ。日中レースは朝に通知済で再生成しない。
#    （全レース指数JSON/PDF は時刻フィルタ対象外＝夜ライン反映の更新版になる）
echo "[$(date '+%H:%M:%S')] 夜レース(19時〜)の推奨を生成..."
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  --min-gap12 0.07 --include-7plus --start-from-hour 19 \
  --output "data/picks/wave_picks_wt_${TODAY}_night.txt" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 夜の部: 対象レース無し or 失敗（継続）"

# 3. 「朝夕の推奨」Discord通知（notify_picks.py）は2026-07-31にユーザー要望により廃止。
# 発走15分前の個別通知（notify_prerace_wt.py）のみ残す。

# 2b. S7（Sランク）朝夜統合再選出（2026-07-22新設計）: 朝夜の生候補プールを合算し
#     axis_sumランキングを組み直す。既に買い判定済み(ロック済み)のレースは変更しない。
#     朝が先着で枠を使い切り夜の優良候補を取りこぼす問題への対処
#     （scripts/reselect_7s_evening.py 参照）。
echo "[$(date '+%H:%M:%S')] S7（Sランク）朝夜統合再選出..."
.venv/bin/python3 scripts/reselect_7s_evening.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] S7統合再選出に失敗（継続）"

# 夜の部 candidates を picks_history に書き込み（日中分は daily_picks_wt.sh 実行済み・
# S7統合再選出後の最終候補で書き込むため、S7再選出の後に実行する）
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 夜候補書き込みに失敗（継続）"

# 3b. netkeirin（ウマい車券）へ全7ランク(S1/7SS/7S/7A/9SS/9S/9A)候補を下書き自動入稿
#     （2026-07-23新設・2026-07-28全ランク対応。ランクごとのON/OFFは
#     keirin.netkeirin_settings＝kiseki側 /keirin/settings で管理）
echo "[$(date '+%H:%M:%S')] netkeirinへ下書き入稿（夕）..."
.venv/bin/python3 scripts/netkeirin_submit_wt.py "$TODAY" evening \
  2>&1 | tee -a "$LOG_DIR/netkeirin_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] netkeirin入稿(夕)に失敗（継続）"

# 4. VPS PostgreSQL 同期（夜の部 wt_entries/picks_history を反映）
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS PostgreSQL 同期..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py \
    2>&1 | tee -a "$LOG_DIR/migrate_pg_${TODAY}.log" \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] === 夕方再生成 完了 ==="
