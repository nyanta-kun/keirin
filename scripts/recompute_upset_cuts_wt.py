"""波乱ゲート top3_sum カット定数の再計測（週次再学習後に実行）

再学習でモデルの pred_prob 分布が動くと、波乱帯(Q1_loose〜Q4_chalk)を定義する
top3_sum 四分位カットがズレる。本スクリプトは現行 lgbm_wt の train期間の
SS/S/A・6車以下レースの top3_sum 四分位を再計測し data/models/upset_cuts_wt.json に
保存する（strategy_wt._load_cuts がこれを優先採用。無ければ既定値）。

使い方（weekly_retrain_wt.sh から呼ばれる）:
  PYTHONPATH=. .venv/bin/python3 scripts/recompute_upset_cuts_wt.py --to 2026-03-01
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.strategy_wt import UPSET_TOP3SUM_CUTS, UPSET_TOP3SUM_CUTS_DEFAULT, _CUTS_PATH
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _filter_by_n_riders, _assign_tier


def compute_cuts(model_name: str, date_from: str, date_to: str | None):
    model = load_model(model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, 6)
    vals = []
    for _, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        if _assign_tier(p[0] - p[1], p[0] / (3.0 / n)) is None:
            continue
        vals.append(p[0] + p[1] + p[2])
    import pandas as pd
    s = pd.Series(vals)
    q = s.quantile([0.25, 0.5, 0.75]).round(4).tolist()
    return q, len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_wt")
    ap.add_argument("--from", dest="date_from", default="2023-07-01")
    ap.add_argument("--to", dest="date_to", default=None,
                    help="train期間の上限（test期間を除外。週次は test-from を渡す）")
    ap.add_argument("--dry-run", action="store_true", help="保存せず比較のみ")
    args = ap.parse_args()

    cuts, n = compute_cuts(args.model, args.date_from, args.date_to)
    old = tuple(UPSET_TOP3SUM_CUTS)
    print(f"[cuts] 対象 {n:,}R（{args.model} / {args.date_from}〜{args.date_to or 'latest'}）")
    print(f"  現行: {old}")
    print(f"  新規: {tuple(cuts)}")
    drift = [round(cuts[i] - old[i], 4) for i in range(3)]
    print(f"  差分: {drift}  既定値: {UPSET_TOP3SUM_CUTS_DEFAULT}")

    if not (cuts[0] < cuts[1] < cuts[2]):
        print("  !! 単調性NG → 保存スキップ（既定値を維持）")
        return
    if args.dry_run:
        print("  (dry-run: 保存せず)")
        return

    _CUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CUTS_PATH.write_text(json.dumps({
        "cuts": cuts,
        "model": args.model,
        "train_from": args.date_from,
        "train_to": args.date_to,
        "n_races": n,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  保存: {_CUTS_PATH}")


if __name__ == "__main__":
    main()
