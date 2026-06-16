#!/bin/bash
# 毎日8:00実行（winticketルート）: 前日成績通知 → 当日データ収集 → 予想生成・通知
# ※7:00から8:00に変更(2026-06-09): 朝7時は想定オッズが揃わずガミ判定の精度が落ちるため。
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

if [[ "$(uname)" == "Darwin" ]]; then
  YESTERDAY=$(date -v-1d +%Y-%m-%d)
else
  YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket日次処理開始 $TODAY ==="

# --- 1. 前日成績（winticketで結果再収集→採点通知）---
# 前日処理は当日予想の前提ではないため、失敗しても継続（pipefailで失敗は可視化）。
echo "[$(date '+%H:%M:%S')] 前日($YESTERDAY) winticket結果再収集..."
.venv/bin/python3 -m src.cli.main collect-wt --date "$YESTERDAY" \
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

# --- 1b. 結果バックフィル（直近数日の取りこぼし回収）---
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

# --- 2. 当日予想 ---
# 当日収集は予想の前提＝失敗時は中断（pipefail+set -e で異常を握り潰さない）。
echo "[$(date '+%H:%M:%S')] 当日($TODAY) winticketデータ収集（全会場走査=初日開催の取りこぼし防止）..."
# --full-scan: 全VENUE_SLUGSを走査。旧実装は停止済みksのracesに依存し、ks停止後に
# 始まった初日開催（宇都宮/別府のミッドナイト等）を取りこぼした（2026-06-09修正）。
# 予想収集は漏れが致命的なため当日は常に全会場走査する。
.venv/bin/python3 -m src.cli.main collect-wt --date "$TODAY" --full-scan \
  2>&1 | tee -a "$LOG_DIR/collect_wt_${TODAY}.log"

# --- 朝オッズ前向き計測: 収集直後の wt_odds(=朝オッズ) を退避 ---
# 翌日の前日再収集で wt_odds が最終オッズに上書きされる前に保全する。
# 失敗しても日次処理は止めない（計測は補助目的）。
echo "[$(date '+%H:%M:%S')] 朝オッズをスナップショット退避..."
.venv/bin/python3 scripts/snapshot_morning_odds_wt.py "$TODAY" \
  2>&1 | tee -a "$LOG_DIR/odds_snapshot_${TODAY}.log" || \
  echo "[$(date '+%H:%M:%S')] 朝オッズ退避に失敗（処理は継続）"

echo "[$(date '+%H:%M:%S')] 予想生成（winticket・<3倍見送り/3〜5倍未満はBランク）..."
# --gami-skip-odds 3.0: 3点中1点でも朝オッズ<3倍ならレース見送り（明確なガミ）。
# --b-rank-odds 5.0:   最安目が3〜5倍未満ならBランク（鉄板寄り・購入は各自判断）として別枠表示。
# --ss-trifecta-box:   SS層の3連単を pred1,pred2 1-2着BOX(6点)に拡張（前向き検証・docs/analysis/10）。
#                      SS層のみ変更（S/A/Bは不変）。最終オッズ上限値＋高配当帯ドリフト要注意。
# --wide --wide-min-odds 2.5: ワイド1点(指数1-2位W12)を独立プロダクトとして追加（前向き検証・docs/analysis/12）。
#                      オッズ≥2.5倍のみ（value型・的中50-53%/ROI220-271%上限値）。SS/S/Aとは別集計。
# --start-to-hour 19:  朝は〜19時発走のレースのみ推奨（昼〜夕は朝にライン/オッズ確定）。
#                      夜レース(19時〜)はwtラインが朝未公開→精度低下するため、午後に
#                      evening_picks_wt.sh が全件再生成して上書き（2段階生成・docs B検証）。
#                      ※全レース指数(allindex/PDF)は時刻フィルタ対象外＝朝から全89レース掲載。
# 検証: scripts/analyze_gami_threshold_wt.py（<3倍点含むレースは集団で収支ゼロ）。
# wave-picks-wt は対象レース0件で exit 1（＝静かな日。異常ではない）になり得るため継続。
.venv/bin/python3 -m src.cli.main wave-picks-wt --date "$TODAY" \
  --gami-skip-odds 3.0 --b-rank-odds 5.0 --ss-trifecta-box \
  --wide --wide-min-odds 2.5 --start-to-hour 19 \
  --min-gap12 0.07 --include-7plus \
  2>&1 | tee -a "$LOG_DIR/picks_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 予想生成: 対象レース無し or 失敗（継続）"

echo "[$(date '+%H:%M:%S')] 予想をDiscordへ通知..."
.venv/bin/python3 scripts/notify_picks.py "$TODAY" wave_picks_wt \
  2>&1 | tee -a "$LOG_DIR/notify_wt_${TODAY}.log" \
  || echo "[$(date '+%H:%M:%S')] 予想通知に失敗（継続）"

echo "[$(date '+%H:%M:%S')] === winticket日次処理完了 ==="
