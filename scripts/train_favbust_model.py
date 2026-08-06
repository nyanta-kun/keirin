#!/usr/bin/env python3
"""RANK_7H1 のバスト予測モデル（`lgbm_wt_favbust`）を学習する。

## 何を予測するのか

    y = 1 if 「モデル軸1 == WINTICKET◎ の本命」が 4着以下（欠車・失格を含む）

母集団は **7車 ∧ 軸1==◎** のレースのみ。基準率 19.50%・honest AUC 0.6848。

## なぜ月次vintage が要るのか

特徴量に **当方モデルの出力**（`fav_pp3` / `fav_ppw` / `fav_pbad`）が入る。
過去分を作るときに本番モデル（全期間学習）で予測すると in-sample になるため、
**その月より前のデータだけで学習した月次vintage**（`lgbm_wt_eval_mYYMM` /
`lgbm_wt_win_mYYMM` / `lgbm_wt_bad_mYYMM`）で予測を作る。
これは 7S/7A/7SS/7B の walk-forward 再構築と同じ契約。

## 生成物

| 種別 | 名前 | 学習データ | 用途 |
|---|---|---|---|
| 本番 | `lgbm_wt_favbust` | 全期間 | 当日の候補生成（`wave-picks-wt`） |
| vintage | `lgbm_wt_favbust_mYYMM` | M月の前月末まで | 過去分の honest 再構築 |

⚠️ **本番モデルはホールドアウト無し**。これを過去へ遡って使うと in-sample になる。
   `backfill_7h1_rank_wt.py` は必ず vintage を使うこと。

⚠️ 学習に使う特徴量は `src/preprocessing/favbust_features.FAVBUST_FEATURE_COLS`
   （67列）。`load_model` はこの列名で照合するので、列を変えたら必ず再学習する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/train_favbust_model.py            # 本番のみ
    PYTHONPATH=. .venv/bin/python scripts/train_favbust_model.py --vintages # + 月次
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.models.model_io import atomic_pickle_dump, atomic_write_json  # noqa: E402
from src.models.trainer import load_model  # noqa: E402
from src.preprocessing.favbust_features import (  # noqa: E402
    FAVBUST_FEATURE_COLS, build_favbust_row, feature_vector,
)
from src.preprocessing.feature_wt import (  # noqa: E402
    build_features_wt, load_raw_data_wt, prepare_X,
)
from src.wt_vintage_config import bad_model_name, monthly_windows  # noqa: E402

MODEL_DIR = REPO / "data" / "models"
CACHE = REPO / "data" / "exp_cache" / "favbust_trainset.pkl"
N_CAR = 7
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 80, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "verbose": -1, "seed": 42,
}
N_ROUNDS = 300


def _load_context(date_from: str, date_to: str) -> tuple[dict, dict]:
    """(race_key -> meta, race_key -> entries) を返す（7車のみ）。"""
    with get_connection() as c:
        meta = {}
        for r in c.execute(
                "SELECT r.race_key, r.race_date, r.grade, r.race_type, r.day_index, "
                "       r.start_at, r.distance, v.bank_length, v.is_indoor "
                "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
                "WHERE r.n_entries = ? AND r.cancel = 0 "
                "  AND r.race_date BETWEEN ? AND ?", (N_CAR, date_from, date_to)):
            meta[r["race_key"]] = dict(r)
        keys = sorted(meta)
        ents: dict[str, list[dict]] = defaultdict(list)
        for i in range(0, len(keys), 700):
            ch = keys[i:i + 700]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, "
                 "       prediction_mark, race_point, line_group, line_size, line_pos, "
                 "       is_line_leader, n_lines, finish_order, style, prefecture, "
                 "       player_class FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(ch)))
            for r in c.execute(q, ch):
                ents[r["race_key"]].append(dict(r))
    return meta, dict(ents)


def build_trainset(force: bool = False) -> list[dict]:
    """月次vintage で予測を作り、67特徴 + bust ラベルの学習セットを組む。"""
    if CACHE.exists() and not force:
        with CACHE.open("rb") as f:
            rows = pickle.load(f)
        print(f"[cache] 学習セット {len(rows):,}件", flush=True)
        return rows

    windows = monthly_windows()
    print(f"月次窓 {len(windows)}本で予測を作成...", flush=True)
    d_from, d_to = windows[0][0], windows[-1][1]
    meta_all, ents_all = _load_context(d_from, d_to)
    print(f"  7車レース {len(meta_all):,}件", flush=True)

    df_all = build_features_wt(load_raw_data_wt(min_date=d_from, max_date=d_to))
    df_all = df_all[df_all["race_key"].isin(set(meta_all))].copy()

    rows: list[dict] = []
    for w_from, w_to, eval_name, win_name in windows:
        bad_name = bad_model_name(eval_name)
        try:
            ev, wi = load_model(eval_name), load_model(win_name)
            ba = load_model(bad_name)
        except FileNotFoundError:
            print(f"  [skip] {w_from[:7]} vintage 未整備", flush=True)
            continue
        sub = df_all[(df_all["race_date"] >= w_from) & (df_all["race_date"] <= w_to)]
        if sub.empty:
            continue
        X = prepare_X(sub)
        pp3 = ev.predict_proba(X)[:, 1]
        ppw = wi.predict_proba(X)[:, 1]
        pbd = ba.predict_proba(X)[:, 1]
        preds: dict[str, dict[int, tuple]] = defaultdict(dict)
        for rk, fno, a, b, cc in zip(sub["race_key"], sub["frame_no"], pp3, ppw, pbd):
            preds[rk][int(fno)] = (float(a), float(b), float(cc))

        n_add = 0
        for rk, pr in preds.items():
            ents = ents_all.get(rk)
            if not ents or len(ents) != N_CAR:
                continue
            row = build_favbust_row(meta_all[rk], ents, pr)
            if row is None:
                continue                       # 軸1 != ◎（母集団外）
            fav = row.pop("_fav")
            fo = next((e["finish_order"] for e in ents
                       if int(e["frame_no"]) == fav), None)
            if fo is None:
                continue
            row.update(race_key=rk, race_date=meta_all[rk]["race_date"],
                       fav=fav, bust=1 if (fo == 0 or fo >= 4) else 0)
            rows.append(row)
            n_add += 1
        print(f"  {w_from[:7]}  +{n_add:5}件 (累計 {len(rows):,})", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:            # 保存失敗は握り潰さない
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE)
    return rows


def _fit(rows: list[dict], name: str, upto: str | None) -> None:
    """upto（含む）までのデータで学習して保存する。upto=None なら全期間。"""
    use = [r for r in rows if upto is None or r["race_date"] <= upto]
    if len(use) < 3000:
        print(f"  [skip] {name}: 学習データ {len(use)}件は少なすぎます", flush=True)
        return
    X = np.array([feature_vector(r) for r in use], dtype=float)
    y = np.array([r["bust"] for r in use])
    ds = lgb.Dataset(X, label=y, feature_name=list(FAVBUST_FEATURE_COLS))
    model = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
    atomic_pickle_dump(model, MODEL_DIR / f"{name}.pkl")
    atomic_write_json({
        "model": name, "n_train": len(use), "n_features": len(FAVBUST_FEATURE_COLS),
        "train_upto": upto or "all", "bust_rate": float(y.mean()),
        "features": list(FAVBUST_FEATURE_COLS),
        "trained_at": date.today().isoformat(),
    }, MODEL_DIR / f"{name}.meta.json")
    print(f"  [saved] {name}  n={len(use):,} bust率={y.mean() * 100:.2f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", action="store_true", help="月次vintageも学習する")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    rows = build_trainset(force=args.rebuild_cache)
    if not rows:
        print("学習データが空です", file=sys.stderr)
        sys.exit(1)
    y = np.array([r["bust"] for r in rows])
    print(f"\n学習セット {len(rows):,}件 / bust基準率 {y.mean() * 100:.2f}% / "
          f"特徴量 {len(FAVBUST_FEATURE_COLS)}列")

    print("\n本番モデル（全期間）:")
    _fit(rows, "lgbm_wt_favbust", None)

    if args.vintages:
        print("\n月次vintage:")
        for w_from, _w_to, eval_name, _win in monthly_windows():
            tag = eval_name.rsplit("_", 1)[-1]          # 例 m2608
            # M月のモデルは「M月の前月末まで」で学習する（既存vintageと同じ契約）
            prev_end = (date.fromisoformat(w_from) - __import__("datetime")
                        .timedelta(days=1)).isoformat()
            _fit(rows, f"lgbm_wt_favbust_{tag}", prev_end)


if __name__ == "__main__":
    main()
