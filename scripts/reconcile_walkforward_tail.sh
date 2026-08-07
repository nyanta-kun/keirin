#!/bin/bash
# 毎日 08:40 実行（cron 実測値）: 7車ランク（7SS/7S/7A/7B/7C/7H1）と
# 9車（9S/9A）の直近ウィンドウ（当月）のみを月次凍結vintageモデルで
# honest に再構築する。
#
# 背景1（当初の目的・2026-07-27 ユーザー指摘）:
#   daily_picks_wt.sh / evening_picks_wt.sh が書き込む当日の候補行
#   （write_candidates_wt.py・gap12条件のみの広い候補プール）は、
#   rebuild_*_walkforward_pg.py が計算する最終選出後の honest な picks
#   （axis選定・entropy/日次capゲート適用済み）より件数が多い。放置すると
#   「直近日だけ候補数が多い」状態が積み上がり、過去期間と条件が食い違う。
#
# 背景2（実行時刻を 00:50 → 08:30 へ変更した理由・2026-08-06）:
#   daily_picks_wt.sh の 08:03「結果バックフィル（T-2〜T-4）」が
#   `notify_results_wt.py <過去日> --silent` を走らせ、**その日の候補JSONから
#   live 行を作り直す**。この後に走らないと、tail 再構築が消した live 行が
#   同じ朝のうちに復活し、rebuild 行と live 行が混在する。
#   実際 2026-08-06 に 7A(rebuild 18 / live 26)・7B(25 / 14) で混在が発生し、
#   バックフィルより後にコミットした 7S だけが無傷(16 / 0)だった。
#   → **必ず 08:03 のバックフィルより後に実行すること。**
#
# 背景3（2026-07-27〜2026-08-06 停止していた理由と解除）:
#   3ヘッド軸（2026-08-04〜）を旧2ヘッド軸の再構築で塗り潰す危険があったため
#   停止していた（レジスタ I-2）。2026-08-06 に 7S/7A/7SS を3ヘッド化し、
#   7車4ランクすべてが vintage の大敗モデルで3ヘッド軸で再構築するようになった
#   ためこの阻害要因は解消した。9車は3ヘッドを採用していない（掃引で窓別に
#   符号が反転したため）ので従来どおり2ヘッド軸で正しい。
#
# 当月以外のvintageモデルは確定済みで結果が変わらないため毎日再計算する必要はなく、
# --tail-only で末尾ウィンドウ（当月）のみ再構築すれば足りる。
set -e
set -o pipefail
export PATH="/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.."
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/reconcile_tail_${DATE}.log"

# --- 多重起動防止 + 共有ロック（2026-08-08 追加）---
# 本スクリプトは picks_history の当月分を DELETE→INSERT で作り直すのに、
# 他の書き込み系（daily_picks_wt / evening_picks_wt / intraday_results_wt /
# results_check_wt）が全て持っている flock を**唯一持っていなかった**。
# 背景2 の対策が「実行時刻を 00:50→08:30 へずらす」という時間差頼みで、
# daily_picks_wt.sh 側にはリトライ待機（最大3回×5分）があるため
# 08:30 に食い込みうる。intraday_results_wt.sh は15分毎なので 08:30 にも走る。
# ⚠️ 共有ロックは **待つ（-w）**。-n でスキップすると当月の再構築が
#    黙って行われず、まさに本スクリプトが防ごうとしている混在が残る。
LOCK_FILE="$LOG_DIR/reconcile_walkforward_tail.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[$(date '+%H:%M:%S')] [reconcile_tail] 前回実行がロック中のためスキップします（${LOCK_FILE}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 0
fi
SHARED_LOCK="$LOG_DIR/wt_picks_writer.lock"
exec 201>"$SHARED_LOCK"
if ! flock -w 3600 201; then
  echo "[$(date '+%H:%M:%S')] [reconcile_tail] 共有ロック待ちが60分を超えました（${SHARED_LOCK}）。" \
    | tee -a "$LOG_DIR/lock_skips.log" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] === walk-forward tail再構築 開始 ===" | tee -a "$LOG"

# 2026-07-31: S1(SEVEN_S1)はユーザー判断により全廃（commit df31431）・
# picks_historyのSEVEN_S1行1,504件もVPS PGから削除済み
# （バックアップ: data/backup/picks_history_s1_discarded_20260731.csv）。
# 本行はレビューで「削除済みのSEVEN_S1行をrebuild_s1_walkforward_pg.pyが
# tail-only再構築のたびにDELETE→INSERTで自動再生成してしまう経路」として
# 検出されたため呼び出しを除去した（backfill_missing_prerace_wt.pyのS1除外
# 対応と同型の事故予防・CLAUDE.mdの「ランク全廃時は候補生成/ライブ判定/
# 欠損自動補完の3箇所すべて停止」に加えて本スクリプトが第4の経路だった）。
# rebuild_s1_walkforward_pg.py本体は過去日再採点・分析用に残置し、手動実行
# 専用とする（同スクリプトのdocstring参照）。

# 7車4ランク（すべて3ヘッド軸）+ 9車2ランク（2ヘッド軸）。
# 1本失敗しても残りは続行する（失敗はログに残す）。
# ⚠️ **ランクを新設したらここへも足すこと。** 抜けると当該ランクだけ tail が
#    live 行のまま取り残され、過去期間と条件が食い違う（＝本スクリプトが
#    防ごうとしている状態そのもの）。2026-08-06 時点の一覧は
#    strategy_wt.CURRENT_PAPER_RANKS と一致している。
# RANK_7C（ベースモデル・終日の二軸・2026-08-07新設）は rebuild_7c_walkforward_pg.py
# を同時に用意したのでここに含める。7C は eval モデルしか使わないため
# bad の vintage が無い月でも窓が落ちない。
# RANK_7H1（穴推奨・本命バスト型）は rebuild_7h1_walkforward_pg.py の実装
# （2026-08-07 commit 89acd9a）と同時に登録すべきだったが漏れていた。
# 2026-08-08 のレビューで検出し追加（それまで 7H1 の当月だけ live 行が残り、
# 2026-08-06 に 7A/7B で起きた rebuild行×live行の混在と同じ状態だった）。
# 忘れ防止に tests/test_rank_7h1.py::test_reconcile_covers_7h1_once_rebuild_exists
# が「rebuild スクリプトが存在するのにここへ未登録」を検出して落ちる。
# ⚠️ そのテストは**この for 行だけをパースする**。過去、全文の文字列一致で
#    書かれていたため上のコメントに含まれる "7h1:7H1" を拾って未登録のまま
#    PASS していた（＝安全網が丸ごと無効だった）。
for spec in "7ss:7SS" "7s:7S" "7a:7A" "7b:7B" "7c:7C" "9s:9S" "9a:9A" "7h1:7H1"; do
  script="${spec%%:*}"
  label="${spec##*:}"
  .venv/bin/python3 "scripts/rebuild_${script}_walkforward_pg.py" --tail-only 2>&1 | tee -a "$LOG" \
    || echo "[$(date '+%H:%M:%S')] ${label} tail再構築 失敗（継続）" | tee -a "$LOG"
done

# 【保険】当日の候補行を復元する（2026-08-07 追加）。
# tail の窓は当日を含めない設計（src/wt_vintage_config.tail_windows）に直したので
# 本来ここで消えることは無いが、rebuild 側の窓計算を将来また当日込みに戻して
# しまったときに **Web から推奨が消えたまま 10:00 まで気づけない** ため、
# 消えていたら書き戻す安全網を置く。候補JSONからの再生成で冪等。
TODAY=$(date +%Y-%m-%d)
echo "[$(date '+%H:%M:%S')] 当日候補の復元チェック（${TODAY}）..." | tee -a "$LOG"
PYTHONPATH=. .venv/bin/python3 scripts/write_candidates_wt.py "$TODAY" 2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] 当日候補の復元に失敗（継続）" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === walk-forward tail再構築 完了 ===" | tee -a "$LOG"
