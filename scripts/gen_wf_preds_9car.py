#!/usr/bin/env python3
"""9車立ての walk-forward 予測（pp3 / ppw / pbad）を生成してキャッシュする。

## なぜ必要か

既存の `data/exp_cache/wf_preds_*.pkl` は **7車だけにフィルタして作られている**
（`exp_7car_gap_fresh.py` L168 `df = df[df["race_key"].map(ne) == 7]`）。
実際の車数分布は 7車 48,783 / 6車 529 / **9車 0** で、9車の予測が1件も無い。
そのため P-A（本命バスト型）の 9車検証ができない。

## 設計

- **学習は全車数**（7車も9車も使う。母数を減らす理由がない）
- **予測は9車レースのみ**保存する
- 窓は `src/exp_highpay_fav_bust.WF` と同じ四半期区切り。各窓とも
  **窓開始より前のデータだけで学習**する（honest walk-forward）
- 出力: `data/exp_cache/wf_preds9_{from}_{to}.pkl`（race_key, frame_no, pp3, ppw, pbad）

⚠️ SEEDS=5 × 3ターゲット × 窓数 の学習になるため時間がかかる（1窓あたり数分）。
   バックグラウンド実行を想定。**保存失敗は握り潰さず例外を上げる**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_axis_rule_decomposition import fit_predict  # noqa: E402
from scripts.exp_highpay_fav_bust import WF  # noqa: E402
from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)

CACHE_DIR = REPO / "data" / "exp_cache"
DATA_FROM = "2022-12-01"


def main() -> None:
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=DATA_FROM, max_date="2026-08-04"))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    df["win_flag"] = (fo == 1).astype(int)
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races"))
    df["n_entries"] = df["race_key"].map(ne)
    print(f"  全 {len(df):,} 行 / 9車 {int((df['n_entries'] == 9).sum()):,} 行",
          flush=True)

    for w_from, w_to in WF:
        out_path = CACHE_DIR / f"wf_preds9_{w_from}_{w_to}.pkl"
        if out_path.exists():
            print(f"[skip] {out_path.name}", flush=True)
            continue
        train = df[df["race_date"] < w_from]
        test = df[(df["race_date"] >= w_from) & (df["race_date"] <= w_to)
                  & (df["n_entries"] == 9)]
        if len(test) == 0 or len(train) < 20000:
            print(f"[skip] {w_from}〜{w_to} train={len(train)} test={len(test)}",
                  flush=True)
            continue
        print(f"[fit] {w_from}〜{w_to} train={len(train):,} test={len(test):,} ...",
              flush=True)
        out = test[["race_key", "frame_no"]].copy()
        out["pp3"] = fit_predict(train, test, TARGET_COL_WT)
        out["ppw"] = fit_predict(train, test, "win_flag")
        out["pbad"] = fit_predict(train, test, "bad6")
        tmp = out_path.with_suffix(".pkl.tmp")
        out.to_pickle(tmp)              # 失敗すれば例外が上がる
        tmp.replace(out_path)
        print(f"[done] {out_path.name} ({len(out):,} 行)", flush=True)

    files = sorted(CACHE_DIR.glob("wf_preds9_*.pkl"))
    total = sum(len(pd.read_pickle(f)) for f in files)
    print(f"\n完了: {len(files)} ファイル / {total:,} 行", flush=True)


if __name__ == "__main__":
    main()
