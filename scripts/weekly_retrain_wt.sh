#!/bin/bash
# 毎週日曜23:30実行（winticketルート）: wtモデル再学習
# H-1対応: ①holdout評価(昇格せず・監視用) → ②全データ再学習で配信モデル(lgbm_wt)生成
#          → ③カット再計測 → ④世代退避(ロールバック用)
set -e
set -o pipefail   # L-5: | tee が python の終了コードをマスクしないように
export PATH="/usr/sbin:/sbin:$PATH"
# KEIRIN_DB_URL は crontab または実行前に export して設定すること
cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" data/models/archive
LOG="$LOG_DIR/train_wt_${DATE}.log"

# テスト分割は直近約90日前（ホールドアウト評価用）
if [[ "$(uname)" == "Darwin" ]]; then
  TEST_FROM=$(date -v-90d +%Y-%m-%d)
else
  TEST_FROM=$(date -d "90 days ago" +%Y-%m-%d)
fi

echo "[$(date '+%H:%M:%S')] === winticket週次再学習 $DATE (test-from=$TEST_FROM) ===" | tee -a "$LOG"

# ① ホールドアウト評価モデル（監視用・--no-promote で本番 lgbm_wt は汚さない）
echo "[$(date '+%H:%M:%S')] ① holdout評価（直近90日をテスト）..." | tee -a "$LOG"
# 前回 eval の AUC を比較用に退避（初回は存在しなくてよい）
PREV_EVAL_META="data/models/lgbm_wt_eval.meta.json"
PREV_AUC=$(python3 -c "import json,sys; print(json.load(open('$PREV_EVAL_META')).get('test_auc_holdout') or '')" 2>/dev/null || echo "")
# 学習開始 2022-12-01（全期間）。2026-07-18に一時「2024-04-01短縮」としたが、
# 原因はDNF/欠車(finish_order<1)がsb_dyn特徴のローリング計算を汚染するバグで
# あり、ラベル不足（0埋め希釈）ではなかった。バグ修正後は全期間データで
# ΔAUC+0.0127・3着内+1.02pt（短縮版と同等以上・データ量1.6倍）を確認済み
# （2026-07-19・exp_window_ab_48f.py）。以後は全期間で学習する。
.venv/bin/python3 -m src.cli.main train-wt \
  --from 2022-12-01 --test-from "$TEST_FROM" --save-as lgbm_wt_eval --no-promote \
  2>&1 | tee -a "$LOG"

# ①' 品質ゲート: holdout AUC が絶対下限未満 or 前回比で大幅悪化なら本番昇格を中止
#     （正常終了＝無条件で lgbm_wt 上書き→rsync 配布 だった構造への安全弁・2026-07-12）
AUC_GATE_MIN="${AUC_GATE_MIN:-0.75}"       # 絶対下限（直近実績 ~0.77）
AUC_GATE_MAX_DROP="${AUC_GATE_MAX_DROP:-0.02}"  # 前回比の許容悪化幅
python3 - "$AUC_GATE_MIN" "$AUC_GATE_MAX_DROP" "$PREV_AUC" <<'PYGATE' 2>&1 | tee -a "$LOG"
import json, sys
auc_min, max_drop = float(sys.argv[1]), float(sys.argv[2])
prev = float(sys.argv[3]) if sys.argv[3] else None
meta = json.load(open("data/models/lgbm_wt_eval.meta.json"))
auc = meta.get("test_auc_holdout")
if auc is None:
    print(f"[gate] holdout AUC が meta に無い → 昇格中止")
    sys.exit(1)
if auc < auc_min:
    print(f"[gate] AUC {auc:.4f} < 下限 {auc_min} → 昇格中止")
    sys.exit(1)
if prev is not None and prev - auc > max_drop:
    print(f"[gate] AUC {auc:.4f} が前回 {prev:.4f} から {prev-auc:.4f} 悪化 (> {max_drop}) → 昇格中止")
    sys.exit(1)
print(f"[gate] AUC {auc:.4f} OK (下限 {auc_min} / 前回 {prev})")
PYGATE

# ② 配信モデル: 全データで再学習して lgbm_wt を更新（H-1）
echo "[$(date '+%H:%M:%S')] ② 配信用: 全データ再学習 → lgbm_wt ..." | tee -a "$LOG"
.venv/bin/python3 -m src.cli.main train-wt \
  --from 2022-12-01 --full-refit --save-as lgbm_wt \
  2>&1 | tee -a "$LOG"

# ③ 波乱ゲート top3_sum カット定数を配信モデルの分布で再計測（test期間除外）
echo "[$(date '+%H:%M:%S')] ③ 波乱カット定数を再計測..." | tee -a "$LOG"
.venv/bin/python3 scripts/recompute_upset_cuts_wt.py --to "$TEST_FROM" \
  2>&1 | tee -a "$LOG" \
  || echo "[$(date '+%H:%M:%S')] カット再計測に失敗/単調性NG（既定値を維持・処理は継続）"

# ④ 世代退避（M-5・ロールバック/再現用。モデル・メタ・カットを日付付きで保存）
echo "[$(date '+%H:%M:%S')] ④ 世代退避 → data/models/archive/ ..." | tee -a "$LOG"
cp -f data/models/lgbm_wt.pkl        "data/models/archive/lgbm_wt_${DATE}.pkl"        2>/dev/null || true
cp -f data/models/lgbm_wt.meta.json  "data/models/archive/lgbm_wt_${DATE}.meta.json"  2>/dev/null || true
cp -f data/models/upset_cuts_wt.json "data/models/archive/upset_cuts_wt_${DATE}.json" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] === 完了 ===" | tee -a "$LOG"
