#!/bin/bash
# 16:00 第2パス: 朝8:00に情報不足で候補にできなかったレースだけを再算出する。
#
# 【2026-08-04・U-1】ユーザー判断により「8:00単一バッチ」へ2段構成を戻した。ただし
# 旧2段階生成（7:00 日中 + 16:00 夜）とは目的が違う:
#   旧: 時刻で分割し、夜レース(19時〜)を作り直す
#   新: **朝に◎◯が未公開だったレースだけ**を作り直す（時刻では分けない）
#
# 背景: 8:00一本化(2026-08-01)の副作用で、開催情報の公開が遅い会場が毎日まるごと
# 推奨から消えていた。実測で WINTICKET公式の印は 13:16 まで、ライン予想は 11:35 まで
# 出揃わない。**夜開催だからではなく会場ごとの事情**で、京王閣は同じ夜発走でも朝5時
# には公開済み。2026-08-04 実測では朝の生候補 64件中 **19件(30%)** が◎◯未公開だった
# （四日市 1-7R・奈良 3-7R 等、発走 20:41〜23:30）。
#
# 「判定時刻を全体的に後ろへ倒す」案は 08:30 発走の防府が間に合わないため採れない。
# データ収集自体は intraday_results_wt.sh が15分毎に回しており、足りないのは
# 「もう一度選び直す処理」だけだった。
#
# 対象レースの抽出は scripts/list_deferred_races_wt.py（朝の生候補 JSON で
# wt_overlap_n が null の race_key）。**朝に情報が揃っていてゲートで落ちたレースは
# 対象外**＝同じレースを条件を変えて引き直さない（prerace decisions の
# 「発走前判定を事後変更しない」方針と整合）。
#
# 前提: daily_picks_wt.sh(8:00) が完了していること（朝の生候補ファイルを読むため）。
# 生候補ファイルが無い場合は対象0件として正常終了する。
#
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
  echo "[$(date '+%H:%M:%S')] [evening_picks_wt] 前回実行がロック中のためスキップします（${LOCK_FILE}）。" \
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
  echo "[$(date '+%H:%M:%S')] [evening_picks_wt] 共有ロック待ちが30分を超えました（${SHARED_LOCK}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 1
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

# 2026-08-01: 8:00一本化に伴い、朝バッチ(daily_picks_wt.lock)のロック解放を待つ
# ガードを本セッションで一時追加したが、cronからの自動連続起動が無くなった
# （本スクリプトは手動実行専用になった）ため撤去した。手動実行する場合は
# daily_picks_wt.sh の完了を目視確認してから実行すること（ヘッダコメント参照）。

echo "[$(date '+%H:%M:%S')] === winticket 夕方再生成 $TODAY ==="

# 1. 当日再収集（全会場フルスキャン＝午後に公開された夜レースのライン/オッズを取得）
echo "[$(date '+%H:%M:%S')] 当日($TODAY) 再収集（全会場・夜ライン取得）..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}.log"

# --- ライン情報充足チェック（2026-08-01導入・本スクリプトが手動実行された場合用）---
# 夜レース(19時〜)のライン予想(linePrediction)がまだ公開されていないレースが
# 多い場合に備えたリトライ。対象は夜レースのみ（--start-from-hour 19。
# 8:00一本化後の通常運用では日中/夜とも daily_picks_wt.sh 側（時刻指定なし＝
# 全レース対象）で判定するため、本チェックは本スクリプトを手動実行した場合のみ
# 意味を持つ）。最大3回試行しても解消しない場合はDiscordへ警告するのみで
# 処理は継続し、取得できた範囲のレースで推奨を生成する（exitしない）。
LINE_READY=0
for attempt in 1 2 3; do
  if .venv/bin/python3 scripts/check_line_readiness.py --date "$TODAY" --start-from-hour 19 \
      2>&1 | tee -a "$LOG_DIR/line_readiness_${TODAY}.log"; then
    LINE_READY=1
    break
  fi
  echo "[$(date '+%H:%M:%S')] ライン情報不足検知（試行${attempt}/3）。5分待機して再収集..."
  sleep 300
  .venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
    2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}_line_retry${attempt}.log"
done

if [[ "$LINE_READY" != "1" ]]; then
  echo "[$(date '+%H:%M:%S')] ライン情報不足が解消せず。取得できた範囲で推奨生成を継続します。"
  .venv/bin/python3 -c "
from src.notify.discord import send
send('⚠️ **[$TODAY] 夕方の部: ライン情報(winticket linePrediction)不足が3回の再収集(5分間隔)後も解消しませんでした。** 取得できた範囲のレースで推奨は生成します。手動確認を推奨します。', channel='system')
" 2>&1 | tee -a "$LOG_DIR/line_readiness_${TODAY}.log" || true
fi

# 1b. 夕方オッズを退避（夜レースは朝オッズ未確定→夕方が実質「生成時オッズ」。
#     ワイド監視で夜レースの朝相当(夕方)→確定ドリフトを見るための基準。snapshot_type='evening'）
.venv/bin/python3 scripts/snapshot_morning_odds_wt.py "$TODAY" --type evening \
  >> "$LOG_DIR/odds_snapshot_${TODAY}.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] 夕方オッズ退避に失敗（継続）"

# 1c. 7H1（穴推奨・本命バスト型）の候補を夕方の最新データで作り直す。
#     ⚠️ **この早期 exit より前に置くこと。** 下の「朝に◎◯未公開だったレース」の
#     抽出が0件だと 2. で exit 0 して以降が丸ごと走らないため、後ろに置くと
#     7H1 が夕方に一度も再生成されない（既存ランクと同じ穴を踏む）。
#     7H1 も軸1==◎ を母集団条件にするので、朝に印が無かったレースは朝の生成では
#     拾えていない。夕方の全会場再収集（1.）でラインと印が揃った状態で作り直す。
#     出力は _night 側。notify_prerace_wt.py は昼→夜の順に読み race_key で重複排除
#     するので、**朝に出た分の買い目は上書きされない**（既に入稿済みのため正しい）。
echo "[$(date '+%H:%M:%S')] 7H1（穴推奨）候補を夕方データで再生成..."
.venv/bin/python3 scripts/build_7h1_candidates.py --date "$TODAY" \
  --out "data/picks/wave_picks_wt_${TODAY}_night_s7h1_candidates.json" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 7H1候補の夕方再生成に失敗（他ランクには影響しないため継続）"

# 9H1（穴推奨・9車高配当）も夕方データで作り直す。9H1 の選別に使う波乱スコアは
# ライン構成と WT公式印を見るので、朝に印・ラインが未確定だったレースは朝の生成で
# 拾えていない。出力は _night 側で、昼→夜の順に読んで race_key で重複排除される。
echo "[$(date '+%H:%M:%S')] 9H1（穴推奨・9車）候補を夕方データで再生成..."
.venv/bin/python3 scripts/build_9h1_candidates.py --date "$TODAY" \
  --out "data/picks/wave_picks_wt_${TODAY}_night_s9h1_candidates.json" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 9H1候補の夕方再生成に失敗（他ランクには影響しないため継続）"

# 2. 朝に情報不足だったレースだけを抽出（0件なら以降を行わず正常終了）
echo "[$(date '+%H:%M:%S')] 朝8:00に◎◯未公開だったレースを抽出..."
DEFERRED="data/picks/deferred_${TODAY}.txt"
if ! .venv/bin/python3 scripts/list_deferred_races_wt.py "$TODAY" --out "$DEFERRED" \
    2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log"; then
  echo "[$(date '+%H:%M:%S')] 再算出の対象なし。第2パスを終了します（正常）。"
  exit 0
fi

# 2a. 対象レースのみ推奨生成 → 専用ファイル(_night)へ。
#     朝に評価済みのレースは作り直さない（既に公開した推奨の軸が後から変わらないように）。
echo "[$(date '+%H:%M:%S')] 不足分の推奨を生成..."
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  --min-gap12 0.07 --include-7plus --only-races-file "$DEFERRED" \
  --output "data/picks/wave_picks_wt_${TODAY}_night.txt" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 第2パス: 対象レース無し or 失敗（継続）"

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

# 3b. 🔴 **netkeirin への入稿はここでは行わない**（2026-08-07 に分離）。
#     このスクリプトは 16:00 に走るが、ミッドナイト開催（第1R 20時）の三連複は
#     16時台でもまだ2割前後が未確定で、18時まで待つと 10.8% まで下がる。
#     netkeirin は公開後に差し替えられないので、入稿は板が育ってからにする。
#     入稿は `scripts/wave_submit_wt.sh evening`（cron 18:00）が担当する。
#     ここは候補の再生成（ライン予想が午後に公開される夜レース向け）に専念する。

# 4. VPS PostgreSQL 同期（夜の部 wt_entries/picks_history を反映）
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS PostgreSQL 同期..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py \
    2>&1 | tee -a "$LOG_DIR/migrate_pg_${TODAY}.log" \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] === 第2パス(不足分再算出) 完了 ==="
