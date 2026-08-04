#!/bin/bash
# 月初vintageモデル自動生成ラッパー（2026-08-01・F-4対応）。
#
# 背景:
#   日付が月初(例: 8/1)に変わった直後、scripts/rebuild_7s_walkforward_pg.py が
#   `FileNotFoundError: data/models/lgbm_wt_eval_m2608.pkl` で失敗した。
#   src/wt_vintage_config.py::monthly_windows() は date.today() 基準で当月の窓を
#   生成するが、その月の凍結vintageモデル(lgbm_wt_eval_mYYMM/lgbm_wt_win_mYYMM)は
#   自動生成されないため、月が替わった瞬間に「まだ存在しないモデル」が要求され
#   落ちる。docs/vintage_model_policy.md は「月初cronは未整備（今後のタスク）」と
#   自認しており、それが実害として顕在化した形。
#
#   scripts/train_monthly_vintage_models.py --only-missing は既存月をスキップし
#   不足月のみ学習する実装が既にあったが、月初に自動実行する経路が無かった。
#
# 設計方針:
#   VPSはメモリ1.9GBで学習不可のため、学習は必ずMacで行う（CLAUDE.md）。
#   本スクリプトは「不足月の学習 → VPSへ配布」を1コマンドで行う。
#   マニフェスト更新(vintage_manifest.json登録)はsave_model()内で学習と同時に
#   自動的に行われるため、本スクリプトが別途行う手順はない。
#
# 使い方:
#   scripts/ensure_monthly_vintage.sh            # 不足月の学習 + VPS配布を実行
#   scripts/ensure_monthly_vintage.sh --dry-run   # 何も学習・転送せず対象を表示するのみ
#
# crontabへの組み込みは別途PM/ユーザー確認の上で実施すること（本スクリプトの新設
# 自体はcrontabを変更しない）。推奨エントリ（月初 00:05 = 週次retrain(日曜23:30)と
# 重ならず、かつreconcile_walkforward_tail.sh(00:50, 現在PAUSED)より確実に前）:
#   5 0 1 * * /Users/ysuzuki/GitHub/keirin/scripts/ensure_monthly_vintage.sh \
#     >> /Users/ysuzuki/GitHub/keirin/data/logs/cron.log 2>&1
set -euo pipefail
export PATH="/usr/sbin:/sbin:$PATH"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/data/logs"
mkdir -p "$LOG_DIR"
DATE_STAMP=$(date +%Y-%m-%d)
LOG="$LOG_DIR/ensure_monthly_vintage_${DATE_STAMP}.log"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "不明な引数: $arg" >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

notify_failure() {
  local msg="$1"
  log "[FAIL] $msg"
  # src/notify/discord.py::send は channel 引数が必須（省略すると別チャンネルに
  # 誤送信される事故を防ぐ設計）。ここでは system チャンネルへ送る。
  if .venv/bin/python3 -c "
from src.notify.discord import send
ok = send('''🚨 **[ensure_monthly_vintage.sh] 月次vintageモデル自動生成に失敗しました**\n${msg}''', channel='system')
raise SystemExit(0 if ok else 1)
" >>"$LOG" 2>&1; then
    log "Discordへ失敗を通知しました。"
  else
    log "[FATAL] Discord通知にも失敗しました（.envのDISCORD_WEBHOOK_URL_SYSTEM未設定などが原因の可能性）。"
  fi
}

# --- 多重起動防止（weekly_retrain_wt.sh/sync_models_to_vps.shと同様、
#     mkdirの原子性を使ったPIDロック。macOSにはflock(1)が無いため代替） ---
LOCK_DIR="$LOG_DIR/ensure_monthly_vintage.lockdir"
if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    OLD_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      log "前回実行(PID $OLD_PID)が継続中のためスキップします（${LOCK_DIR}）。"
      exit 0
    fi
    log "古いロック（PID ${OLD_PID:-不明} は不在）を検出。奪って続行します。"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
  fi
  echo $$ > "$LOCK_DIR/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT
fi

# --- KEIRIN_DB_URL 必須チェック（他のwt系cronスクリプトと同型・D-1踏襲） ---
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  notify_failure "KEIRIN_DB_URL が未設定です。crontabの環境変数設定を確認してください。処理を中断します。"
  exit 1
fi

log "=== ensure_monthly_vintage: 不足月チェック開始 ==="

if [[ "$DRY_RUN" -eq 1 ]]; then
  .venv/bin/python3 scripts/train_monthly_vintage_models.py --only-missing --dry-run 2>&1 | tee -a "$LOG"
  log "=== DRY RUN: 何も学習・転送していません ==="
  exit 0
fi

# --- Step 1: 不足月のみ学習（Mac側。VPSはメモリ1.9GBで学習不可のためMac専用） ---
if ! .venv/bin/python3 scripts/train_monthly_vintage_models.py --only-missing 2>&1 | tee -a "$LOG"; then
  notify_failure "train_monthly_vintage_models.py --only-missing が失敗しました。ログ: $LOG"
  exit 1
fi
log "Step1完了: 不足月の学習（0件学習=既に全月揃っている場合も正常終了）"

# --- Step 2: VPSへモデルファイルを配布（sync_models_to_vps.shが検証まで行う） ---
log "=== ensure_monthly_vintage: VPSへ配布開始 ==="
if ! "$REPO_ROOT/scripts/sync_models_to_vps.sh" 2>&1 | tee -a "$LOG"; then
  notify_failure "sync_models_to_vps.sh が失敗しました。ログ: $LOG"
  exit 1
fi

log "=== ensure_monthly_vintage: 完了 ==="
