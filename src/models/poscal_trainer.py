"""隊列位置バイアス補正（2段目モデル）の学習。

`src/preprocessing/position_calib.py` の設計意図・検証結果はそちらの docstring を参照。

学習は 3 本のモデルを作り、そのうち 2 本を保存する:

1. `base_inner` : 内側窓より前だけで学習した3着内モデル（**保存しない**）
2. `b_inner`    : 同上のB取りモデル（**保存しない**）
3. `lgbm_wt_b`      : 全データで学習したB取りモデル（保存・配信）
4. `lgbm_wt_poscal` : 2段目モデル（保存・配信）

2段目は「ベースモデルが**まだ見ていない**期間の予測」を入力に学習しなければ
意味がない（自分の学習データ上の予測は較正が良すぎる）。そのため内側窓
（既定12ヶ月）を切り、その手前までで学習した 1. 2. の予測を入力にする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.preprocessing.feature_wt import FEATURE_COLS_WT
from src.preprocessing.position_calib import (
    POSCAL_FEATURE_COLS,
    add_position_features,
)

# ベース／B取りモデルの学習パラメータ。検証（2026-08-03）で用いた設定と一致させる。
_BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1, n_estimators=600,
    random_state=42, deterministic=True, verbose=-1,
)
# 2段目は入力8列と小さいので木も小さくする（過学習させると順位が壊れる）。
_CALIB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=300,
    colsample_bytree=0.9, n_estimators=300,
    random_state=42, deterministic=True, verbose=-1,
)


def load_res_back(min_date: str) -> pd.DataFrame:
    """`wt_entries.res_back`（そのレースでB＝最終バック先頭を取ったか）を読む。"""
    import os

    sql = (
        "SELECT e.race_key, e.frame_no, e.res_back "
        "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
        "WHERE r.race_date >= :min_date"
    )
    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        from sqlalchemy import create_engine, text as sa_text

        engine = create_engine(db_url)
        pg = (sql.replace("wt_entries", "keirin.wt_entries")
                 .replace("wt_races", "keirin.wt_races"))
        with engine.connect() as conn:
            df = pd.read_sql_query(sa_text(pg), conn, params={"min_date": min_date})
        engine.dispose()
        return df

    from src.database import get_connection

    with get_connection() as conn:
        return pd.read_sql_query(sql.replace(":min_date", "?"), conn, params=(min_date,))


def _fit(df: pd.DataFrame, target: pd.Series) -> LGBMClassifier:
    m = LGBMClassifier(**_BASE_PARAMS)
    m.fit(df[FEATURE_COLS_WT], target)
    return m


def train_position_calibrator(
    df: pd.DataFrame,
    inner_start: str,
    train_end: str | None = None,
) -> tuple[LGBMClassifier, LGBMClassifier, dict]:
    """B取りモデルと2段目モデルを学習して返す。

    Args:
        df: `build_features_wt()` 済み + `finish_order` / `res_back` / `race_date` を持つ行。
            **車数で絞らないこと**（本番 `train-wt` が全車数1本で学習しているため、
            ここで絞ると train/serve skew になる）。
        inner_start: 内側窓の開始日 (YYYY-MM-DD)。ここより前が 1段目の学習に使われる。
        train_end: 内側窓の終了日。省略時は df の最終日。

    Returns:
        (配信用B取りモデル, 2段目モデル, メタ情報)
    """
    end = train_end or str(df["race_date"].max())
    confirmed = df[df["finish_order"].notna()]
    outer = confirmed[confirmed["race_date"] < inner_start]
    inner = confirmed[(confirmed["race_date"] >= inner_start)
                      & (confirmed["race_date"] <= end)]
    if inner.empty or outer.empty:
        raise ValueError(
            f"内側窓の切り方が不正です（outer={len(outer)}行 / inner={len(inner)}行）。"
            f"inner_start={inner_start} train_end={end}"
        )

    y_top3 = outer["finish_order"].between(1, 3).astype(int)
    base_inner = _fit(outer, y_top3)
    ob = outer[outer["res_back"].notna()]
    b_inner = _fit(ob, ob["res_back"].astype(int))

    # 内側窓に「ベースモデルが見ていない」予測を付けて2段目の学習データを作る
    stage2 = inner.copy()
    stage2["pred_prob"] = base_inner.predict_proba(stage2[FEATURE_COLS_WT])[:, 1]
    stage2["pc_p_b"] = b_inner.predict_proba(stage2[FEATURE_COLS_WT])[:, 1]
    stage2 = add_position_features(stage2, prob_col="pred_prob")

    calib = LGBMClassifier(**_CALIB_PARAMS)
    calib.fit(stage2[POSCAL_FEATURE_COLS],
              stage2["finish_order"].between(1, 3).astype(int))

    # 配信用のB取りモデルは全期間で学習し直す（内側窓も使う）
    ab = confirmed[(confirmed["race_date"] <= end) & confirmed["res_back"].notna()]
    b_final = _fit(ab, ab["res_back"].astype(int))

    meta = {
        "inner_start": inner_start,
        "train_end": end,
        "n_outer_races": int(outer["race_key"].nunique()),
        "n_inner_races": int(inner["race_key"].nunique()),
        "n_b_rows": int(len(ab)),
        "feature_cols": POSCAL_FEATURE_COLS,
        "importance": {
            k: int(v) for k, v in
            sorted(zip(POSCAL_FEATURE_COLS, calib.booster_.feature_importance("gain")),
                   key=lambda x: -x[1])
        },
    }
    return b_final, calib, meta


def evaluate_axis_quality(df: pd.DataFrame, before: str, after: str) -> dict:
    """軸2車の的中率など、補正の効き目を測る（学習後の健全性チェック用）。

    Args:
        df: `race_key` / `frame_no` / `finish_order` と 2 つの確率列を持つ行。
        before: 補正前の確率列名。
        after: 補正後の確率列名。
    """
    out: dict[str, float] = {}
    n = 0
    acc = {before: [0, 0], after: [0, 0]}   # [軸1が3着内, 軸2車ともに3着内]
    for _, g in df.groupby("race_key", sort=False):
        top3 = set(g.loc[g["finish_order"].between(1, 3), "frame_no"])
        if len(top3) != 3:
            continue
        n += 1
        for col in (before, after):
            fr = list(g.sort_values(col, ascending=False)["frame_no"])
            acc[col][0] += fr[0] in top3
            acc[col][1] += set(fr[:2]) <= top3
    if n == 0:
        return {"n_races": 0}
    for col in (before, after):
        tag = "before" if col == before else "after"
        out[f"axis1_top3_{tag}"] = acc[col][0] / n
        out[f"axis2_hit_{tag}"] = acc[col][1] / n
    out["n_races"] = n
    out["axis2_hit_delta"] = out["axis2_hit_after"] - out["axis2_hit_before"]
    y = df["finish_order"].between(1, 3).astype(float)
    for col in (before, after):
        tag = "before" if col == before else "after"
        out[f"brier_{tag}"] = float(np.mean((df[col] - y) ** 2))
    return out
