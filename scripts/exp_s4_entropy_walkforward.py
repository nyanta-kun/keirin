"""S4選出済みレース内 entropy フィルターの四半期walk-forward追検証（2026-07-26）。

exp_s4_payout_filter_wt.py で見つかった「S4選出済みレースをフィールド全体の
予測確率entropyでさらに輪切りにすると、最下位25%が的中率・ROIともに突出し
30倍+的中の大半を独占する」というパターンについて、直近3ヶ月(2026-04〜07)だけ
でなく、data/models/ に用意済みの四半期別ホールドアウトモデル
（lgbm_wt_eval_q2401〜q2507 / lgbm_wt_win_q2401〜q2507、各四半期を学習から
除外した正真OOSモデル）を使って2024-01〜2025-09の7四半期 + 2026-04〜07の
直近窓の計8窓で検証する。

さらに、最初の窓（2024 Q1）だけで entropy しきい値を決め、それ以降の全窓
（2024 Q2〜2026年直近）へ完全固定・ブラインド適用する「真のwalk-forward」も行う。

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_entropy_walkforward.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_s4_payout_filter_wt import build

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401", "2024Q1"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404", "2024Q2"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407", "2024Q3"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410", "2024Q4"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501", "2025Q1"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504", "2025Q2"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507", "2025Q3"),
    ("2026-04-24", "2026-07-25", "lgbm_wt_eval", "lgbm_wt_win_eval", "2026Q2-3(直近)"),
]


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"    {label}: n=0")
        return
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    upset = sum(1 for r in rows if r["hit"] and r["trio_payout"] >= 3000)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"    {label:<18} n={n:>4} hit={hits:>3}({hits/n:.1%}) ROI={roi:>6.1f}%  30倍+的中={upset}件")


def main():
    quarter_rows = {}
    for f, t, model, win_model, label in QUARTERS:
        rows = build(model, win_model, f, t)
        quarter_rows[label] = rows
        y_upset = sum(1 for r in rows if r["hit"] and r["trio_payout"] >= 3000)
        n = len(rows)
        hits = sum(r["hit"] for r in rows)
        bet = sum(r["bet_amount"] for r in rows)
        pay = sum(r["payout"] for r in rows)
        roi = pay / bet * 100 if bet else float("nan")
        print(f"\n===== {label}（{f}〜{t}）S4全体: n={n} hit={hits}({hits/n:.1%} if n else 0) ROI={roi:.1f}% "
              f"30倍+的中={y_upset}件 =====")
        if n < 20:
            print("    n不足のためスキップ")
            continue
        ent = np.array([r["entropy"] for r in rows])
        qs = np.percentile(ent, [25, 50, 75])
        for i, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)], 1):
            sel = [r for r, v in zip(rows, ent) if (v > lo and (v <= hi if hi != np.inf else True))]
            summarize(sel, f"entQ{i}(窓内四分位)")

    # ── 真のwalk-forward: 2024Q1だけでしきい値決定 → 以降の全窓へ固定適用 ──
    print("\n\n========== 真のwalk-forward検証（2024Q1のみでしきい値決定→以降ブラインド適用） ==========")
    base_rows = quarter_rows.get("2024Q1", [])
    if len(base_rows) < 20:
        print("2024Q1のデータ不足のため実施不可")
        return
    ent_base = np.array([r["entropy"] for r in base_rows])
    thresh = np.percentile(ent_base, 25)
    print(f"2024Q1(n={len(base_rows)})で決定した固定entropyしきい値: {thresh:.4f}\n")

    total_below = []
    total_above = []
    for f, t, model, win_model, label in QUARTERS:
        if label == "2024Q1":
            continue
        rows = quarter_rows[label]
        if len(rows) < 5:
            continue
        below = [r for r in rows if r["entropy"] <= thresh]
        above = [r for r in rows if r["entropy"] > thresh]
        print(f"  --- {label} ---")
        summarize(below, "entropy<=固定しきい値")
        summarize(above, "entropy>固定しきい値")
        total_below.extend(below)
        total_above.extend(above)

    print("\n  === 2024Q2〜2026直近 全期間合算（2024Q1除く・真のOOS） ===")
    summarize(total_below, "entropy<=固定しきい値")
    summarize(total_above, "entropy>固定しきい値")
    all_rows = total_below + total_above
    summarize(all_rows, "フィルタなし全体")


if __name__ == "__main__":
    main()
