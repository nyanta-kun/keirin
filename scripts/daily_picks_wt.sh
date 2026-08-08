#!/bin/bash
# 毎日 07:00 実行（winticketルート）: 前日成績通知 → 当日データ収集 → 予想生成・通知
# ※2026-08-07: 入稿を開催単位の3波（07:00 モーニング/デイ・13:00 ナイター・
# 18:00 ミッドナイト）へ分けた際に発火を 08:00 → 07:00 へ前倒し済み。
# 本ヘッダーだけ 8:00 のまま取り残されていたので是正（2026-08-08）。
# ※2026-08-01: 「7:00(日中)+16:00(夜)」の2段階生成を廃止し、8:00の単一バッチへ
# 一本化（ユーザー判断）。PM実測（直近92日・2026-05-01〜07-31）で「1日の最初の
# 発走は全92日とも08:30（最早・中央値・最遅すべて08:30）・8:00より前に発走する
# 日は0日」と判明したため、8:00の1回で当日全レース（1日平均7車64.8R+9車5.9R）を
# 収集・厳選できると判断した。
# ⚠️ 過去の経緯（2026-06-09 commit 4b8ddd2）: 「朝7時はオッズが揃わずガミ3段階
# 判定の精度が落ちる(夜レースで9999.9倍等)」という理由で7:00→8:00へ変更した実績が
# ある。本ファイルは8:00運用を維持しているため、この懸念は再燃しない
# （2026-08-01時点で一度検討された「7:00化」は撤回済み）。
# ライン情報不足時のリトライは check_line_readiness.py 参照（後段のrace_point
# 健全性チェックと同じ「5分待機→再収集」パターンを流用）。
# ※2026-08-01: evening_picks_wt.sh(夕方バッチ)はcronから撤去し本スクリプトへ
# 一本化。夜レース分もここで生成するため --start-to-hour 指定は行わない
# （全レース対象）。S7の日次件数上限(RANK_7S_DAILY_CAP)適用のため
# reselect_7s_evening.py を本スクリプト末尾から呼ぶ（詳細は同ステップのコメント参照）。
# 2026-06-08 ks→wt 完全移行。ksスクレイピングは廃止。
set -e
set -o pipefail   # L-5: | tee が python の終了コードをマスクしないように
# cron環境のPATHには /usr/sbin が無く joblib のCPUコア検出(sysctl)が警告を出すため追加
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
TODAY=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" "data/picks"

# --- 多重起動防止（2026-07-31 D-2）---
# 前段処理の遅延等で cron の次回発火と重複実行されると wt_races/wt_entries/
# picks_history への同時書き込み・削除が競合する（2026-07-08 prerace_decisions/
# notified 同時消失事故と同型のリスク）。flock は VPS(util-linux)で利用可能と
# 確認済み(2026-07-31, `ssh sekito "which flock && flock --version"`)。
# ロック取得失敗時は「前回が継続中」とみなしスキップする（スキップの発生は
# lock_skips.log に蓄積するので、頻発していれば前回がハングしていないか確認すること）。
LOCK_FILE="$LOG_DIR/daily_picks_wt.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[$(date '+%H:%M:%S')] [daily_picks_wt] 前回実行がロック中のためスキップします（${LOCK_FILE}）。" \
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
  echo "[$(date '+%H:%M:%S')] [daily_picks_wt] 共有ロック待ちが30分を超えました（${SHARED_LOCK}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 1
fi

# --- KEIRIN_DB_URL 必須チェック（2026-07-31 D-1）---
# database.py の get_connection() は KEIRIN_DB_URL 未設定時に RuntimeError を送出する
# 設計だが、本スクリプトの各処理は `|| echo "...失敗（継続）"` で握り潰しているため、
# crontab 編集ミス等でこの変数が消えると当日分の収集・予想生成・通知・netkeirin入稿が
# 全て空振りしつつ script 全体は exit 0 で完走してしまう（Discordに何も来ないことでしか
# 気付けない）。ここで早期に検知して中断する。
if [[ -z "${KEIRIN_DB_URL:-}" ]]; then
  echo "[$(date '+%H:%M:%S')] [FATAL] KEIRIN_DB_URL が未設定です。daily_picks_wt.sh を中断します。" \
    | tee -a "$LOG_DIR/db_url_missing_${TODAY}.log" >&2
  # Discord Webhook URL は .env から直接読む実装（src/notify/discord.py::_load_webhook_url）
  # のため、DB接続が無くても通知は送信できる（通知経路はKEIRIN_DB_URLに依存しない）。
  if .venv/bin/python3 -c "
from src.notify.discord import send
ok = send('🚨 **[daily_picks_wt.sh] KEIRIN_DB_URL が未設定のため処理を中断しました。**\ncrontabの環境変数設定を確認してください。', channel='system')
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

if [[ "$(uname)" == "Darwin" ]]; then
  YESTERDAY=$(date -v-1d +%Y-%m-%d)
else
  YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket日次処理開始 $TODAY ==="

# ---------------------------------------------------------------------------
# 前日処理（結果再収集・採点通知・取りこぼしバックフィル）
#
# 🔴 **当日の入稿より後ろで実行する。** 実測（2026-08-07）でこの区間は 9分28秒
#    かかっており、日次処理全体 15分32秒の 61% を占めていた。当日予想にも
#    netkeirin 入稿にも一切必要ないのに、これが先にあるせいで商品の公開が
#    10分近く遅れ、8:30 の第1レースまでの販売時間を削っていた。
#    関数にまとめて後ろへ回す（中身は変えていない）。
# ---------------------------------------------------------------------------
run_previous_day_tasks() {
# --- 前日成績（winticketで結果再収集→採点通知）---
# 前日処理は当日予想の前提ではないため、失敗しても継続（pipefailで失敗は可視化）。
echo "[$(date '+%H:%M:%S')] 前日($YESTERDAY) winticket結果再収集..."
# --full-scan: midnight の前日取得で拾いきれなかった分（Mac スリープ等）を確実に回収するため全会場走査。
.venv/bin/python3 -m src.cli.main collect-wt --date "$YESTERDAY" --full-scan \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${YESTERDAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 前日再収集に失敗（継続）"

echo "[$(date '+%H:%M:%S')] 前日成績をDiscordへ通知..."
.venv/bin/python3 scripts/notify_results_wt.py "$YESTERDAY" \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${YESTERDAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 前日成績通知に失敗（継続）"

# ワイド朝→直前(確定)ドリフト監視（前日分を記録・しばらく監視・通知なし）
# 朝≥2.5倍で推奨したW12が確定で2.5未満に落ちる問題(6/10:平均-63%)を継続計測。
.venv/bin/python3 scripts/monitor_wide_wt.py "$YESTERDAY" \
  >> "$LOG_DIR/wide_monitor_run.log" 2>&1 \
  || echo "[$(date '+%H:%M:%S')] ワイド監視に失敗（継続）"

# --- 結果バックフィル（直近数日の取りこぼし回収）---
# cron不発(Macスリープ等)で日次が飛ぶと、結果再収集は「前日のみ」なのでその日の
# 結果が永久に取り残される（6/6で39R未取得→勝ち予想が消える事象が発生）。
# 直近2〜4日前の未確定レースを再収集し（collect-wtは結果確定済みのみスキップ＝安価）、
# picks_history を --silent で静かに修復（Discord通知はしない＝重複通知を避ける）。
echo "[$(date '+%H:%M:%S')] 結果バックフィル（T-2〜T-4の取りこぼし回収）..."
for n in 2 3 4; do
  if [[ "$(uname)" == "Darwin" ]]; then
    BD=$(date -v-${n}d +%Y-%m-%d)
  else
    BD=$(date -d "$n days ago" +%Y-%m-%d)
  fi
  .venv/bin/python3 -m src.cli.main collect-wt --date "$BD" --full-scan \
    >> "$LOG_DIR/backfill_wt.log" 2>&1 || echo "  backfill collect $BD 失敗（継続）"
  .venv/bin/python3 scripts/notify_results_wt.py "$BD" --silent \
    >> "$LOG_DIR/backfill_wt.log" 2>&1 || echo "  backfill rescore $BD 失敗（継続）"
done
}

# --- 2. 当日予想 ---
# 当日収集は予想の前提＝失敗時は中断（pipefail+set -e で異常を握り潰さない）。
echo "[$(date '+%H:%M:%S')] 当日($TODAY) winticketデータ収集（全会場走査=初日開催の取りこぼし防止）..."
# --full-scan: 全VENUE_SLUGSを走査。旧実装は停止済みksのracesに依存し、ks停止後に
# 始まった初日開催（宇都宮/別府のミッドナイト等）を取りこぼした（2026-06-09修正）。
# 予想収集は漏れが致命的なため当日は常に全会場走査する。
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}.log"

# --- race_point健全性チェック（2026-07-23導入）---
# WINTICKET側がその日のrace_pointをまだ確定しておらず異常な暫定値
# （実例: 平均62-67→4.33に急落）を収集してしまう事象への対策。
# 直近7日の中央値の50%を下回る場合は異常とみなし、5分待って再収集を最大3回試行する。
# 解消しなければ指数算出・推奨提示をスキップしDiscordへ通知する（システムの根幹となる
# 特徴量のため、誤ったデータでの推奨生成を優先せず安全側に倒す）。
RP_SANE=0
for attempt in 1 2 3; do
  if .venv/bin/python3 scripts/check_race_point_sanity.py --date "$TODAY" \
      2>&1 | tee -a "$LOG_DIR/race_point_sanity_${TODAY}.log"; then
    RP_SANE=1
    break
  fi
  echo "[$(date '+%H:%M:%S')] race_point異常検知（試行${attempt}/3）。5分待機して再収集..."
  sleep 300
  .venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
    2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}_retry${attempt}.log"
done

if [[ "$RP_SANE" != "1" ]]; then
  echo "[$(date '+%H:%M:%S')] race_point異常が解消せず。本日の指数算出・推奨提示をスキップします。"
  .venv/bin/python3 -c "
from src.notify.discord import send
send('⚠️ **[$TODAY] race_point(競走得点)異常のため本日の指数算出・推奨提示をスキップしました。**\n3回の再収集(5分間隔)後も解消せず。手動確認が必要です。', channel='system')
" 2>&1 | tee -a "$LOG_DIR/race_point_sanity_${TODAY}.log"
  exit 0
fi

# --- ライン情報充足チェック（2026-08-01導入）---
# winticketのライン予想(linePrediction)がまだ公開されていないレースが多い場合に
# 備えたリトライ。8:00一本化により日中/夜の時刻分割が無くなったため、
# --start-from-hour / --start-to-hour は指定せず当日の全レースを対象に判定する
# （引数自体は将来また分割運用に戻す可能性を考慮しcheck_line_readiness.py側には
# 残してある）。
# race_point健全性チェックとは別懸念（データ欠損 vs 異常値）のため独立したループ
# とする。最大3回試行しても解消しない場合はDiscordへ警告するのみで処理は継続し、
# 取得できた範囲のレースで推奨を生成する（race_point異常時と異なりexitしない）。
LINE_READY=0
for attempt in 1 2 3; do
  if .venv/bin/python3 scripts/check_line_readiness.py --date "$TODAY" \
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
send('⚠️ **[$TODAY] ライン情報(winticket linePrediction)不足が3回の再収集(5分間隔)後も解消しませんでした。** 取得できた範囲のレースで推奨は生成します。手動確認を推奨します。', channel='system')
" 2>&1 | tee -a "$LOG_DIR/line_readiness_${TODAY}.log" || true
fi

# --- 朝オッズ前向き計測: 収集直後の wt_odds(=朝オッズ) を退避 ---
# 翌日の前日再収集で wt_odds が最終オッズに上書きされる前に保全する。
# 失敗しても日次処理は止めない（計測は補助目的）。
echo "[$(date '+%H:%M:%S')] 朝オッズをスナップショット退避..."
.venv/bin/python3 scripts/snapshot_morning_odds_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/odds_snapshot_${TODAY}.log" || \
  echo "[$(date '+%H:%M:%S')] 朝オッズ退避に失敗（処理は継続）"

echo "[$(date '+%H:%M:%S')] 予想生成（winticket・7+車専用 gami≥5倍+gap12≥0.07）..."
# 7+車専用モード: gami≥5.0倍(三連複最安目) + gap12≥0.07 のレースのみ推奨
# Sランク: gap12≥0.10(HOLD~143%) / Aランク: gap12[0.07,0.10)(HOLD~138%)
# 2026-08-01: 8:00一本化により --start-to-hour 指定を撤去。当日の全レース
# （夜レース含む）を対象に生成する（旧evening_picks_wt.shの役割を統合）。
# wave-picks-wt は対象レース0件でも継続（静かな日は正常終了）。
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  --min-gap12 0.07 --include-7plus \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 予想生成: 対象レース無し or 失敗（継続）"

# --- 7H1（穴推奨・本命バスト型）候補生成（2026-08-06 新設）---
# 7H1 は既存6ランクと**入口が違う**。既存は wave-picks-wt が作る選手単位の予測から
# 軸2車を選ぶが、7H1 はレース単位のバスト予測モデル（lgbm_wt_favbust）を使うため
# 独立したスクリプトで候補を作る。出力先は data/picks/ で、notify_prerace_wt.py が
# 他ランクと同じように読む。
# ⚠️ ここは本番モデル（全期間学習）を使う。当日のレースは未来なので honest。
#    過去分の再構築で本番モデルを使うと in-sample になるので backfill 側は vintage を使うこと。
# 0件でも継続する（絶対閾値による選別なので該当なしの日が約7%ある＝正常）。
.venv/bin/python3 scripts/build_7h1_candidates.py --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 7H1候補生成に失敗（他ランクには影響しないため継続）"

# --- 9H1（穴推奨・9車高配当）候補生成（2026-08-08 新設）---
# 7H1 と同じく入口が独立している。こちらはレース単位の波乱スコア
# （lgbm_upset_screen・6/7/9車の統合学習）でレースを選ぶ。
# ⚠️ ここも本番モデル（全期間学習）を使う。当日のレースは未来なので honest。
# 9車立ては1日10件前後しかなく、選別後は 0〜3件/日。0件の日は正常。
.venv/bin/python3 scripts/build_9h1_candidates.py --date "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 9H1候補生成に失敗（他ランクには影響しないため継続）"

# 「朝夕の推奨」Discord通知（notify_picks.py）は2026-07-31にユーザー要望により廃止。
# 発走15分前の個別通知（notify_prerace_wt.py）のみ残す。

# --- S7（Sランク）日次件数上限（RANK_7S_DAILY_CAP）適用（2026-08-01: 8:00一本化に伴い移設）---
# rank_7s_daily_select()（src/strategy_wt.py）自体は日次上限を適用しない設計
# （「朝夕どちらか一方のバッチだけでは日次合計が分からないため」— 元々2バッチ
# 構成を前提にしたコメントがコード内に残っている）。RANK_7S_DAILY_CAP=12は
# reselect_7s_evening.py が呼ぶ rank_7s_evening_reselect() の中でのみ適用される。
# 8:00一本化で朝夕2バッチが無くなった後も、この安全網（entropyゲート通過が
# 異常発生した日に日次12件へトリムする仕組み）を掛け続けるため、旧
# evening_picks_wt.sh が担っていたこの呼び出しを本スクリプトへ移設する
# （reselect_7s_evening.py 自体は無編集。night_raw用ファイルが存在しない/空の
# ため day_raw のみでの日次トリムとして機能する＝1バッチでも安全網は有効）。
# 7A/S9/9A/S1等の他ランクにはそもそも同種の日次合計マージ処理が無い
# （各々 rank_*_daily_select() を1回呼ぶだけで完結する設計のため、8:00一本化で
# 対応不要）。
echo "[$(date '+%H:%M:%S')] S7（Sランク）日次件数上限を適用..."
.venv/bin/python3 scripts/reselect_7s_evening.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] S7日次上限適用に失敗（継続）"

# 候補レース（gap12条件のみ・gamiフィルタなし）を picks_history に即時書き込み
# → 同日中から推奨ページに候補レースを表示するため
# （reselect_7s_evening.py の後に実行＝S7はトリム後の最終候補で書き込まれる）
echo "[$(date '+%H:%M:%S')] 候補レースを picks_history に書き込み..."
.venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 候補書き込みに失敗（継続）"

# --- 2b. netkeirin（ウマい車券）へ現行4ランク(7S/7A/9S/9A)候補を下書き自動入稿
#     （2026-07-23新設・2026-07-28全ランク対応。ランクごとのON/OFFは
#     keirin.netkeirin_settings＝kiseki側 /keirin/settings で管理）
#     2026-08-01: 8:00一本化によりsession="morning"の1回で当日全レース分
#     （旧・夜レース分含む）を入稿する（`_load_candidates()`はsession="morning"
#     時は非"_night_"サフィックスの候補ファイルを読む実装のため、当日の
#     全候補が単一ファイルに書かれる本構成と整合する）。
#
#     🔴 2026-08-07: 入稿は**開催（会場×日）単位で3波に分かれた**（`src/meeting_wave.py`）。
#        この朝の回が出すのは **モーニング・デイ（第1R < 12時）だけ**で、
#        ナイターは昼13:00・ミッドナイトは夕方18:00 の回が出す。
#        理由は板が「時計時刻ではなく発走までの近さ」で埋まるため——朝8時台の
#        三連複 未確定率は 〜10時台発走 0.8% に対し **20時以降発走 63.4%**。
#        netkeirin は公開後に差し替えられないので、夜の開催を朝に出すと
#        傾斜配分（src/stake_allocation.py）がほぼ効かないまま確定してしまう。
#        ⚠️ **予想そのもの（picks_history・Discord・Web）は当日全開催ぶんを
#           この朝の回で出す**。後ろへ回すのは netkeirin への入稿だけ。---
echo "[$(date '+%H:%M:%S')] netkeirinへ下書き入稿（朝の波: モーニング・デイ）..."
.venv/bin/python3 scripts/netkeirin_submit_wt.py "$TODAY" morning \
  2>&1 | tee -a "$LOG_DIR/netkeirin_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] netkeirin入稿に失敗（継続）"

# --- 2c. 前日処理（入稿より後ろ）---
# 当日の商品を1分でも早く出すため、当日予想・入稿に不要な前日処理はここで実行する。
run_previous_day_tasks

# --- 3. VPS PostgreSQL 同期（wt_entries/picks_history 等を反映）---
# wave-picks-wt で race_point(AI確率) が更新された wt_entries と
# write_candidates_wt で書き込まれた picks_history を VPS に同期する。
# KEIRIN_DB_URL 未設定時はスキップ（エラー非致命）。
if [[ -n "$KEIRIN_DB_URL" ]]; then
  echo "[$(date '+%H:%M:%S')] VPS PostgreSQL 同期..."
  .venv/bin/python3 scripts/migrate_sqlite_to_pg.py \
    2>&1 | tee -a "$LOG_DIR/migrate_pg_${TODAY}.log" \
    || echo "[$(date '+%H:%M:%S')] VPS 同期に失敗（継続）"
else
  echo "[$(date '+%H:%M:%S')] KEIRIN_DB_URL 未設定のため VPS 同期をスキップ"
fi

echo "[$(date '+%H:%M:%S')] === winticket日次処理完了 ==="
