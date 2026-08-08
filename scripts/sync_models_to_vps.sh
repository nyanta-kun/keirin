#!/bin/bash
# モデルファイル(Mac→VPS)配布スクリプト（2026-07-31・D-5対応）。
#
# 背景:
#   従来はMac crontabのweekly_retrain_wt.sh行の末尾に直接rsyncコマンドが
#   書かれており、転送対象ファイルを長いコマンドラインで列挙していた。
#   `docs/vintage_model_policy.md`記載の通り、月次vintageモデル
#   （`lgbm_wt_eval_m????.pkl`等）は2026-07-30に一度きり手動rsyncで配布
#   されたのみで、crontabのrsync行には組み込まれていない。そのため
#   `scripts/train_monthly_vintage_models.py --only-missing`で新しい月の
#   モデルをMacで作っても、VPS側は今後永久に受け取れない構造的欠陥がある。
#
# 設計方針:
#   転送対象ファイルの列挙をcrontab（変更漏れの温床）からこのスクリプトに
#   一本化する。月次vintageモデルはファイル名パターン（`lgbm_wt_eval_m????.pkl`
#   / `lgbm_wt_win_m????.pkl`とその`.meta.json`）で動的に検出するため、
#   新しい月のモデルが追加されても本スクリプトの変更は不要。
#
# 使い方:
#   scripts/sync_models_to_vps.sh --dry-run   # 転送対象一覧を表示するのみ（何も送信しない）
#   scripts/sync_models_to_vps.sh             # 実際にrsyncで転送し、転送後に検証する
#
# crontabへの組み込み例（このスクリプト自体はcrontabを変更しない。
# 変更は別途PM/ユーザー判断で実施すること）:
#   30 23 * * 0 /Users/ysuzuki/GitHub/keirin/scripts/weekly_retrain_wt.sh \
#     >> /Users/ysuzuki/GitHub/keirin/data/logs/cron.log 2>&1 && \
#     /Users/ysuzuki/GitHub/keirin/scripts/sync_models_to_vps.sh \
#     >> /Users/ysuzuki/GitHub/keirin/data/logs/cron.log 2>&1
#
# 注意: 本スクリプトは実際のrsync転送を行う。crontabの変更・実運用への
# 組み込みはPM/ユーザーが確認の上で実施すること（このファイルの新設のみで
# 既存crontabは変更していない）。
set -euo pipefail
export PATH="/usr/sbin:/sbin:$PATH"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
MODEL_DIR="$REPO_ROOT/data/models"
LOG_DIR="$REPO_ROOT/data/logs"
mkdir -p "$LOG_DIR"
DATE_STAMP=$(date +%Y-%m-%d)
LOG="$LOG_DIR/sync_models_to_vps_${DATE_STAMP}.log"

REMOTE_HOST="sekito"
REMOTE_DIR="~/keirin/data/models/"

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
ok = send('''🚨 **[sync_models_to_vps.sh] モデル配布に失敗しました**\n${msg}''', channel='system')
raise SystemExit(0 if ok else 1)
" >>"$LOG" 2>&1; then
    log "Discordへ失敗を通知しました。"
  else
    log "[FATAL] Discord通知にも失敗しました（DISCORD_WEBHOOK_URL_SYSTEM未設定などの可能性）。"
  fi
}

# --- 多重起動防止（weekly_retrain_wt.shと同様、mkdirの原子性を使ったロック） ---
LOCK_DIR="$LOG_DIR/sync_models_to_vps.lockdir"
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

# --- 転送対象ファイルの収集 ---
# 1) 本番モデル（従来crontabのrsync行と同一の明示列挙。今後増える予定は薄いため固定リスト）
PROD_FILES=(
  "lgbm_wt.pkl" "lgbm_wt.meta.json"
  "lgbm_wt_train_only.pkl" "lgbm_wt_train_only.meta.json"
  "lgbm_wt_win.pkl" "lgbm_wt_win.meta.json"
  "lgbm_wt_bad.pkl" "lgbm_wt_bad.meta.json"
  "lgbm_wt_eval.pkl" "lgbm_wt_eval.meta.json"
  "lgbm_wt_win_eval.pkl" "lgbm_wt_win_eval.meta.json"
  "lgbm_wt_favbust.pkl" "lgbm_wt_favbust.meta.json"
  # 2026-08-08 追加: RANK_9H1 の波乱スコアモデル（レース単位・6/7/9車統合学習・31特徴）。
  # meta.json は無い（save_model は .pkl のみ書く）。
  # 配布されないと build_9h1_candidates.py が落ち、daily_picks_wt.sh は
  # `|| echo ...継続` で握り潰すので **ログ1行だけ残して 9H1 が永久に0件**になる。
  # 抜けは tests/test_model_sync_coverage.py が機械的に検出する。
  "lgbm_upset_screen.pkl"
)
# CI（GitHub Actions）がデプロイ時に取得する最小セット。
# GitHub Actions は Mac のローカルファイルへ到達できないため、
# ここで Releases へ上げておくことで **マージと同時にコードとモデルが揃う**。
# rsync（VPSへの直接配布）はそのまま残す＝二重経路で、どちらか一方が
# 失敗してもVPSが不整合にならないようにする。
RELEASE_TAG="models-latest"
RELEASE_FILES=(
  "lgbm_wt.pkl" "lgbm_wt.meta.json"
  "lgbm_wt_win.pkl" "lgbm_wt_win.meta.json"
  "lgbm_wt_bad.pkl" "lgbm_wt_bad.meta.json"
)
EXTRA_FILES=("upset_cuts_wt.json")

FILES=()
MISSING=()
for name in "${PROD_FILES[@]}" "${EXTRA_FILES[@]}"; do
  if [[ -f "$MODEL_DIR/$name" ]]; then
    FILES+=("$MODEL_DIR/$name")
  else
    MISSING+=("$name")
  fi
done

# 2) 月次vintageモデル（命名規則: lgbm_wt_{eval,win,bad}_mYYMM.{pkl,meta.json}。
#    docs/vintage_model_policy.md参照）。ファイル名パターンでの動的検出のため、
#    新しい「月」が増えてもこのスクリプトの変更は不要。
#
#    ⚠️ **新しい「種類」を増やしたらここへ足すこと。** 2026-08-05 に大敗モデルの
#    月次vintage（lgbm_wt_bad_mYYMM）を新設した際にここへの追加が漏れ、VPSへ
#    1本も配布されていなかった。結果、2026-08-06 に再開した tail 再構築が
#    7車4ランクすべて「lgbm_wt_bad_m2608 が無い」で中断した
#    （ガードが計算前に止めたので実害はゼロ。設計どおり）。
#    同種の「一覧の手書き二重管理」は同日 netkeirin の RANK_ORDER・
#    _THREE_HEAD_RANKS・ガードテストの対象リストでも事故を起こしている。
shopt -s nullglob
VINTAGE_FILES=(
  "$MODEL_DIR"/lgbm_wt_eval_m[0-9][0-9][0-9][0-9].pkl
  "$MODEL_DIR"/lgbm_wt_eval_m[0-9][0-9][0-9][0-9].meta.json
  "$MODEL_DIR"/lgbm_wt_win_m[0-9][0-9][0-9][0-9].pkl
  "$MODEL_DIR"/lgbm_wt_win_m[0-9][0-9][0-9][0-9].meta.json
  "$MODEL_DIR"/lgbm_wt_bad_m[0-9][0-9][0-9][0-9].pkl
  "$MODEL_DIR"/lgbm_wt_bad_m[0-9][0-9][0-9][0-9].meta.json
  # 2026-08-06 追加: 穴推奨 RANK_7H1 のバスト予測モデル（レース単位・67特徴）。
  # ⚠️ 「モデルの新しい"種類"を足したときに配布glob が取り残される」事故は
  #    lgbm_wt_bad で一度踏んでいる（同日）。種別を増やしたら必ずここも見ること。
  "$MODEL_DIR"/lgbm_wt_favbust_m[0-9][0-9][0-9][0-9].pkl
  "$MODEL_DIR"/lgbm_wt_favbust_m[0-9][0-9][0-9][0-9].meta.json
)
shopt -u nullglob
FILES+=("${VINTAGE_FILES[@]}")

N_PROD=$(( ${#PROD_FILES[@]} + ${#EXTRA_FILES[@]} - ${#MISSING[@]} ))
N_VINTAGE=${#VINTAGE_FILES[@]}
N_TOTAL=${#FILES[@]}

log "=== sync_models_to_vps: 転送対象 本番${N_PROD}件 + vintage${N_VINTAGE}件 = 計${N_TOTAL}件 ==="
if [[ ${#MISSING[@]} -gt 0 ]]; then
  log "[警告] 未検出のためスキップされた本番/追加ファイル: ${MISSING[*]}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "--- DRY RUN: 転送対象ファイル一覧 (${N_TOTAL}件) ---"
  for f in "${FILES[@]}"; do
    echo "  $(basename "$f")"
  done
  echo "--- DRY RUN: 何も転送していません ---"
  exit 0
fi

if [[ "$N_TOTAL" -eq 0 ]]; then
  notify_failure "転送対象ファイルが1件も見つかりませんでした（${MODEL_DIR}）。"
  exit 1
fi

# --- 実転送 ---
log "rsync 実行中 → ${REMOTE_HOST}:${REMOTE_DIR}"
if ! rsync -av "${FILES[@]}" "${REMOTE_HOST}:${REMOTE_DIR}" >>"$LOG" 2>&1; then
  notify_failure "rsync がゼロ以外の終了コードで失敗しました。ログ: $LOG"
  exit 1
fi

# --- 転送後の検証 ---
# (a) チェックサム照合: --checksum --dry-run で「転送したはずなのに内容が
#     一致しないファイル」が無いかを確認する（サイズ・mtime一致で誤ってskipされる
#     ケースを検出。VPS側で追加でハッシュ計算プロセスを起動しないためVPSの
#     メモリ負荷は増やさない＝rsyncプロトコル内のチェックサム比較のみ）。
log "検証(1/2): チェックサム照合中..."
CHECKSUM_DIFF=$(rsync -av --checksum --dry-run --itemize-changes "${FILES[@]}" "${REMOTE_HOST}:${REMOTE_DIR}" \
  | grep -E '^[<>ch]' || true)
if [[ -n "$CHECKSUM_DIFF" ]]; then
  log "[検証NG] 転送後もチェックサムが一致しないファイルがあります:"
  echo "$CHECKSUM_DIFF" | tee -a "$LOG"
  notify_failure "転送後チェックサム照合で不一致を検出しました。詳細はログ($LOG)参照。"
  exit 1
fi
log "検証(1/2) OK: 全ファイルのチェックサムが一致しました。"

# (b) ファイル数照合: VPS側の対象パターンファイル数がローカルの想定数と一致するか
# ⚠️ **転送 glob と この検証 regex は必ずセットで更新すること。**
#    モデルの新しい"種類"を足すと、転送は成功しているのにここだけ取り残されて
#    「VPS側ファイル数が下回っています」と誤報する。2026-08-06 に
#    `lgbm_wt_bad`（差64）と `lgbm_wt_favbust`（差60）で2回踏んだ。
log "検証(2/2): VPS側ファイル数を照合中..."
REMOTE_COUNT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
  "ls ${REMOTE_DIR} 2>/dev/null | grep -E '^(lgbm_wt(_train_only|_win|_bad|_eval|_win_eval|_favbust)?\.(pkl|meta\.json)|lgbm_wt_(eval|win|bad|favbust)_m[0-9]{4}\.(pkl|meta\.json)|upset_cuts_wt\.json)$' | wc -l | tr -d ' '" \
  2>>"$LOG" || echo "")
if [[ -z "$REMOTE_COUNT" ]]; then
  notify_failure "VPS側ファイル数の取得に失敗しました（SSH到達不可の可能性）。転送自体は完了している可能性があります。"
  exit 1
fi
if [[ "$REMOTE_COUNT" -lt "$N_TOTAL" ]]; then
  notify_failure "VPS側ファイル数(${REMOTE_COUNT})がローカル転送対象数(${N_TOTAL})を下回っています。"
  exit 1
fi
log "検証(2/2) OK: VPS側ファイル数=${REMOTE_COUNT}（ローカル転送対象=${N_TOTAL}、VPS側は他バージョンも含み得るため >= であれば正常）"

# --- GitHub Releases へのアップロード（CI がデプロイ時に取得する） ---
# ここを忘れると CI 側は古いモデルを配ってしまうため、rsync と同じ実行で必ず行う。
# gh 不在・認証なし・ネットワーク断でも rsync 自体は成功しているので、
# ここでの失敗は警告に留めて全体は成功扱いにする（VPSへの配布は完了している）。
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] GitHub Releases へのアップロードはスキップします"
elif ! command -v gh >/dev/null 2>&1; then
  log "[警告] gh CLI が無いため Releases へのアップロードをスキップしました。"
  log "        CI デプロイ時は古いモデルが配られる可能性があります。"
else
  UP=()
  for name in "${RELEASE_FILES[@]}"; do
    [[ -f "$MODEL_DIR/$name" ]] && UP+=("$MODEL_DIR/$name")
  done
  if [[ ${#UP[@]} -eq 0 ]]; then
    log "[警告] Releases へ上げるファイルが1つも見つかりませんでした。"
  else
    if ! gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
      log "Release '$RELEASE_TAG' が無いため作成します..."
      gh release create "$RELEASE_TAG" --title "本番モデル（最新）" \
        --notes "CI デプロイが取得する本番モデル。sync_models_to_vps.sh が毎回上書きする。" \
        >>"$LOG" 2>&1 || log "[警告] Release の作成に失敗しました。"
    fi
    log "Releases へアップロード中（${#UP[@]}件 → ${RELEASE_TAG}）..."
    if gh release upload "$RELEASE_TAG" "${UP[@]}" --clobber >>"$LOG" 2>&1; then
      log "Releases アップロード OK（${#UP[@]}件）"
    else
      log "[警告] Releases へのアップロードに失敗しました。VPSへのrsyncは成功しています。"
    fi
  fi
fi

log "=== sync_models_to_vps: 完了（本番${N_PROD}件 + vintage${N_VINTAGE}件 = 計${N_TOTAL}件を配布・検証OK） ==="
