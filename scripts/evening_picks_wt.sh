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

# 3. 夜の部のみDiscordへ通知（Xポスト省略・日中の重複通知なし・指数PDFは更新版）
echo "[$(date '+%H:%M:%S')] 夜の部をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" wave_picks_wt night \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 夜の部通知に失敗（継続）"

# 夜の部 candidates を picks_history に書き込み（日中分は daily_picks_wt.sh 実行済み）
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 夜候補書き込みに失敗（継続）"

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
