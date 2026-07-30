#!/usr/bin/env python3
"""wt_entries.pred_win_pct/pred_top3_pct を過去分にリークなしで一括反映する。

Web表示（単勝指数・複勝指数）を過去レースでも見られるようにするための
バックフィル。月次凍結vintageモデル群（lgbm_wt_eval_mYYMM/lgbm_wt_win_mYYMM等・
`src.wt_vintage_config.monthly_windows()`が唯一の正本）を使い、各月はその月を
test窓として学習済みのモデルでのみスコアリングする（各レース終了当時に
得られたはずの精度を再現・全期間フルリフィットモデルは使わない）。

【2026-07-29改定】
- 旧版はローカルSQLite前提（KEIRIN_DB_URLをpopして読み取り、末尾にSQLite→PG
  ミラー処理）だったが、2026-07-22のVPS PG一本化でローカルSQLiteは廃止済み
  （現在は空ファイル）。本版は他のrebuild_*_walkforward_pg.pyと同じくVPS PG
  へ直接読み書きする（KEIRIN_DB_URLをpopしない）。
- 四半期QUARTERS+静的TAIL_FROMの2層構造（TAIL_FROMとlgbm_wt_evalの実際の
  test_fromが週次で乖離し続けるバグの温床だった）を廃止し、月次凍結モデルの
  みを使う設計に統一（[[keirin_s7_foundational_rethink_2026_07_29]]）。
- 対象は2024-01-01〜今月（`monthly_windows()`が生成する全窓）。7車以外の
  レースも含め全出走馬を対象にする（wave-picks-wtの候補選定とは無関係）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_index_pct_wt.py [--dry-run]
    PYTHONPATH=. .venv/bin/python scripts/backfill_index_pct_wt.py --tail-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.wt_vintage_config import monthly_windows


def backfill_window(date_from: str, date_to: str, eval_model_name: str, win_model_name: str,
                     dry_run: bool) -> int:
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return 0
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]

    import pandas as pd
    rows = [
        (
            round(float(r.pred_win) * 100, 1) if pd.notna(r.pred_win) else None,
            round(float(r.pred_prob) * 100, 1) if pd.notna(r.pred_prob) else None,
            r.race_key, int(r.frame_no),
        )
        for r in df.itertuples(index=False)
    ]
    if not dry_run:
        with get_connection() as conn:
            conn.executemany(
                "UPDATE wt_entries SET pred_win_pct = ?, pred_top3_pct = ? "
                "WHERE race_key = ? AND frame_no = ?",
                rows,
            )
            conn.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tail-only", action="store_true",
                     help="直近月（今月）の窓のみ再計算する日次軽量運用向けオプション。")
    args = ap.parse_args()

    windows = monthly_windows()
    if args.tail_only:
        windows = windows[-1:]

    total = 0
    for date_from, date_to, eval_model_name, win_model_name in windows:
        print(f"\n[backfill-index] {date_from}〜{date_to}  eval={eval_model_name} win={win_model_name}",
              flush=True)
        n = backfill_window(date_from, date_to, eval_model_name, win_model_name, args.dry_run)
        print(f"[backfill-index]   {n}件 更新{'（dry-run）' if args.dry_run else ''}", flush=True)
        total += n

    print(f"\n[backfill-index] ===== 合計 {total}件 =====")


if __name__ == "__main__":
    main()
