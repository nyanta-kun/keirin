#!/bin/bash
# lgbm_v6 再学習・検証スクリプト
# 実行前提: 2023-01〜2024-05 の収集完了済み
#
# 設計:
#   学習: 2023-07-01 〜 2025-05-31 (23ヶ月 / rolling stats burn-in 6ヶ月確保)
#   ホールドアウト: 2025-06-01 〜 2026-02-28 (9ヶ月 / 真の独立テスト)
#   参考: 2026-03-01 〜 2026-05-31 (従来バックテスト期間)

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python3"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo " lgbm_v6 再学習・検証"
echo " $(date)"
echo "========================================"

# 1. データ件数確認
echo ""
echo "[1/5] DB データ確認..."
$PYTHON - << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from src.database import get_connection
with get_connection() as conn:
    r = conn.execute("SELECT MIN(race_date), MAX(race_date), COUNT(DISTINCT race_key) FROM races").fetchone()
    print(f"  DB期間: {r[0]} 〜 {r[1]}  ({r[2]:,}レース)")
    for period, lo, hi in [
        ("学習期間", "2023-07-01", "2025-05-31"),
        ("ホールドアウト", "2025-06-01", "2026-02-28"),
        ("参考(旧BT)", "2026-03-01", "2026-05-31"),
    ]:
        n = conn.execute(
            "SELECT COUNT(DISTINCT race_key) FROM races WHERE race_date BETWEEN ? AND ?",
            (lo, hi)
        ).fetchone()[0]
        print(f"  {period}: {lo} 〜 {hi}  ({n:,}レース)")
EOF

# 2. compute-stats (全期間)
echo ""
echo "[2/5] compute-stats 実行（全期間）..."
$PYTHON -m src.cli.main compute-stats 2>&1 | tail -5

# 3. lgbm_v6 学習
echo ""
echo "[3/5] lgbm_v6 学習..."
echo "  学習: 2023-07-01 〜 2025-05-31"
echo "  テスト分割: 2025-06-01〜"
$PYTHON -m src.cli.main train \
    --model lgbm \
    --from 2023-07-01 \
    --test-from 2025-06-01 \
    --save-as lgbm_v6 \
    2>&1 | tee "$LOG_DIR/train_v6_${TIMESTAMP}.log"

# 4. ホールドアウトバックテスト (2025-06〜2026-02)
echo ""
echo "[4/5] ホールドアウトバックテスト (2025-06-01 〜 2026-02-28)..."
$PYTHON - << 'EOF'
import sys, numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(".").resolve()))
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model
from src.evaluation.backtest import _load_payouts

MIN_DATE, MAX_DATE = "2025-06-01", "2026-02-28"
print(f"  期間: {MIN_DATE} 〜 {MAX_DATE}")

df_raw = load_raw_data(min_date=MIN_DATE, max_date=MAX_DATE)
model = load_model("lgbm_v6")
df = build_features(df_raw)
X = df[FEATURE_COLS].fillna(0)
df["pred_prob"] = model.predict_proba(X)[:, 1]
df_fin = df[df["finish_position"].notna()].copy()

all_keys = df_fin["race_key"].unique().tolist()
payout_map = _load_payouts(all_keys)

stats = defaultdict(lambda: {"n":0,"hits":0,"bets":0,"returns":0,"payouts":[]})
monthly = defaultdict(lambda: {"n":0,"hits":0,"bets":0,"returns":0})

for race_key in sorted(all_keys):
    grp = df_fin[df_fin["race_key"] == race_key].sort_values("pred_prob", ascending=False).reset_index(drop=True)
    n = len(grp)
    if n > 6: continue
    if len(grp) < 2: continue

    top1, top2p = grp.iloc[0]["pred_prob"], grp.iloc[1]["pred_prob"]
    gap12 = top1 - top2p
    ratio = top1 / (3 / n)

    if gap12 >= 0.15 and ratio < 1.3:   rank = "SS"
    elif gap12 >= 0.15:                   rank = "S"
    elif gap12 >= 0.06:                   rank = "A"
    else: continue

    top3_set = frozenset(grp[grp["finish_position"] <= 3]["frame_no"].tolist())
    if len(top3_set) != 3: continue

    actual_order = list(grp[grp["finish_position"].isin([1,2,3])].sort_values("finish_position")["frame_no"].astype(int))
    pivot1, pivot2 = int(grp.iloc[0]["frame_no"]), int(grp.iloc[1]["frame_no"])
    thirds = [int(grp.iloc[i]["frame_no"]) for i in range(2, min(5,n))]
    bet_amt = len(thirds) * 100
    month = race_key[:4] + "-" + race_key[4:6]

    for d in [stats[rank], stats["ALL"], monthly[(month, rank)], monthly[(month, "ALL")]]:
        d["n"] += 1; d["bets"] += bet_amt

    if rank == "SS":
        for t in thirds:
            if actual_order == [pivot1, pivot2, t]:
                p = payout_map.get(race_key,{}).get(("trifecta", f"{pivot1}-{pivot2}-{t}"), 0)
                for d in [stats[rank], stats["ALL"], monthly[(month, rank)], monthly[(month, "ALL")]]:
                    d["hits"] += 1; d["returns"] += p
                stats[rank]["payouts"].append(p)
                break
    else:
        for t in thirds:
            if frozenset([pivot1,pivot2,t]) == top3_set:
                pk = "=".join(map(str,sorted(top3_set)))
                p = payout_map.get(race_key,{}).get(("trifecta_box", pk), 0)
                for d in [stats[rank], stats["ALL"], monthly[(month, rank)], monthly[(month, "ALL")]]:
                    d["hits"] += 1; d["returns"] += p
                stats[rank]["payouts"].append(p)
                break

print("\n  ランク別結果:")
print(f"  {'ランク':>4} {'件数':>5} {'的中':>4} {'的中率':>7} {'ROI':>8} {'損益':>10} {'avg払戻':>8}")
print("  " + "─"*58)
for rank in ["SS","S","A","ALL"]:
    s = stats[rank]
    if s["n"] == 0: continue
    roi = s["returns"]/s["bets"] if s["bets"] else 0
    hp = s["hits"]/s["n"]*100
    avg_p = np.mean(s["payouts"]) if s["payouts"] else 0
    print(f"  [{rank:>3}]  {s['n']:>4}R  {s['hits']:>3}回  {hp:>6.1f}%  {roi:>7.1%}  {s['returns']-s['bets']:>+10,}円  {avg_p:>8,.0f}円")

print("\n  月別:")
months = sorted(set(m for m,r in monthly.keys()))
for m in months:
    s = monthly.get((m,"ALL"), {"n":0,"hits":0,"bets":0,"returns":0})
    if s["n"] == 0: continue
    roi = s["returns"]/s["bets"] if s["bets"] else 0
    ss_s = monthly.get((m,"SS"),{"n":0,"hits":0,"bets":0,"returns":0})
    sr_s = monthly.get((m,"S"), {"n":0,"hits":0,"bets":0,"returns":0})
    ar_s = monthly.get((m,"A"), {"n":0,"hits":0,"bets":0,"returns":0})
    def r(d): return f"{d['returns']/d['bets']:.0%}" if d['bets'] else "  -"
    print(f"  {m}: {s['n']:>3}R  ROI {roi:>6.1%}  (SS:{r(ss_s)} S:{r(sr_s)} A:{r(ar_s)})  損益{s['returns']-s['bets']:>+8,}円")

# 信頼区間
s = stats["SS"]
if s["n"] > 0:
    import math
    p = s["hits"]/s["n"]
    ci = 1.96 * math.sqrt(p*(1-p)/s["n"])
    print(f"\n  SS 的中率 95%CI: {p:.1%} ± {ci:.1%}  ({p-ci:.1%} 〜 {p+ci:.1%})")
EOF

# 5. 旧バックテスト期間での比較 (2026-03〜05)
echo ""
echo "[5/5] 参考: 旧バックテスト期間 (2026-03-01 〜 2026-05-31) での v6 性能..."
$PYTHON - << 'EOF'
import sys, numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(".").resolve()))
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model
from src.evaluation.backtest import _load_payouts

for model_name, label in [("lgbm_v6","v6(新)"), ("lgbm","v5(現行)")]:
    df_raw = load_raw_data(min_date="2026-03-01", max_date="2026-05-31")
    try:
        model = load_model(model_name)
    except:
        print(f"  {label}: モデルなし")
        continue
    df = build_features(df_raw)
    X = df[FEATURE_COLS].fillna(0)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df_fin = df[df["finish_position"].notna()].copy()
    all_keys = df_fin["race_key"].unique().tolist()
    payout_map = _load_payouts(all_keys)

    total = {"n":0,"hits":0,"bets":0,"returns":0}
    for race_key in sorted(all_keys):
        grp = df_fin[df_fin["race_key"] == race_key].sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(grp)
        if n > 6 or len(grp) < 2: continue
        top1, top2p = grp.iloc[0]["pred_prob"], grp.iloc[1]["pred_prob"]
        gap12 = top1 - top2p; ratio = top1 / (3/n)
        if gap12 >= 0.15 and ratio < 1.3: rank = "SS"
        elif gap12 >= 0.15: rank = "S"
        elif gap12 >= 0.06: rank = "A"
        else: continue
        top3_set = frozenset(grp[grp["finish_position"] <= 3]["frame_no"].tolist())
        if len(top3_set) != 3: continue
        actual_order = list(grp[grp["finish_position"].isin([1,2,3])].sort_values("finish_position")["frame_no"].astype(int))
        pivot1, pivot2 = int(grp.iloc[0]["frame_no"]), int(grp.iloc[1]["frame_no"])
        thirds = [int(grp.iloc[i]["frame_no"]) for i in range(2, min(5,n))]
        bet_amt = len(thirds) * 100
        total["n"] += 1; total["bets"] += bet_amt
        if rank == "SS":
            for t in thirds:
                if actual_order == [pivot1, pivot2, t]:
                    p = payout_map.get(race_key,{}).get(("trifecta", f"{pivot1}-{pivot2}-{t}"), 0)
                    total["hits"] += 1; total["returns"] += p; break
        else:
            for t in thirds:
                if frozenset([pivot1,pivot2,t]) == top3_set:
                    pk = "=".join(map(str,sorted(top3_set)))
                    p = payout_map.get(race_key,{}).get(("trifecta_box", pk), 0)
                    total["hits"] += 1; total["returns"] += p; break

    roi = total["returns"]/total["bets"] if total["bets"] else 0
    print(f"  {label}: {total['n']}R  的中{total['hits']}回({total['hits']/total['n']*100:.1f}%)  ROI {roi:.1%}  損益 {total['returns']-total['bets']:+,}円")
EOF

echo ""
echo "========================================"
echo " 完了: $(date)"
echo "========================================"
