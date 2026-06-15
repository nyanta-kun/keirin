"""venue × grade 限定 rolling WR 特徴量実験（doc37）

仮説:
  現行 venue_wr は選手の会場別生涯勝率（グレード混合）。
  S级 × 小倉 と A級 × 小倉 では競争構造が異なる可能性がある。
  venue_wr を grade 別に分割した venue_grade_wr を追加することで
  AUC・ROI が改善するか検証する。

新特徴量:
  venue_grade_wr: player_id × venue_id × grade_group の expanding 勝率（shift=1・前試合まで）

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）
Phase2 gate: ROI >100% 全3期間（TRAIN/VAL/HOLD）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0

GRADE_MAP = {"S級": "S", "SA混合": "S", "A級": "A", "L級": "L"}


def build_venue_grade_wr(df: pd.DataFrame) -> np.ndarray:
    """player_id × venue_id × grade_group の expanding 勝率（point-in-time）を返す。

    全エントリー（欠車含む）に対して計算する。
    欠車行も「その日の出走履歴」として扱い、as-of 個別ループを避ける。

    Returns:
        ndarray (len=len(df)) の venue_grade_wr 値
    """
    # 全エントリーを time-sorted で取得（欠車除外 = finish_order >= 1）
    with get_connection() as conn:
        H = pd.read_sql_query(
            """
            SELECT e.race_key, e.player_id, e.finish_order,
                   r.race_date, r.venue_id, r.grade
            FROM wt_entries e
            JOIN wt_races r ON e.race_key = r.race_key
            WHERE e.finish_order >= 1
            """,
            conn,
        )

    H["grade_group"] = H["grade"].map(GRADE_MAP).fillna("A")
    H["_dt"] = pd.to_datetime(H["race_date"])
    H["win"] = (H["finish_order"] == 1).astype(float)
    H = H.sort_values(["player_id", "venue_id", "grade_group", "_dt"]).reset_index(drop=True)

    # expanding mean with shift=1（point-in-time: 前レースまでの平均）
    H["venue_grade_wr"] = (
        H.groupby(["player_id", "venue_id", "grade_group"])["win"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=[0, 1, 2], drop=True)
    )

    # merge back to df（左結合）→ 欠車行・履歴未登録はNaN → 0で埋める
    Hmap = H[["race_key", "player_id", "venue_grade_wr"]].copy()
    out = df.merge(Hmap, on=["race_key", "player_id"], how="left")
    out["venue_grade_wr"] = out["venue_grade_wr"].fillna(0.0)
    return out["venue_grade_wr"].values


def compute_roi_records(df: pd.DataFrame, trio_map: dict, actual_trio: dict,
                        n_entries_map: dict) -> list[dict]:
    records = []
    for rk, grp in df.groupby("race_key"):
        if n_entries_map.get(rk, 99) > 6:
            continue
        g = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 3:
            continue
        period = _period_of(str(g["race_date"].iloc[0]))
        if period is None:
            continue
        p1 = int(g.iloc[0]["frame_no"])
        p2 = int(g.iloc[1]["frame_no"])
        thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]
        bd = trio_map.get(rk, {})
        combos = [frozenset({p1, p2, t}) for t in thirds]
        min_odds = min((bd.get(k, 0) for k in combos if bd.get(k, 0) > 0), default=0)
        if min_odds < GAMI_THRESHOLD:
            continue
        actual = actual_trio.get(rk, frozenset())
        pay = 0.0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual == k:
                pay = bd.get(k, 0) * 100
                break
        records.append({
            "period": period, "race_key": rk,
            "pay": pay, "cost": len(thirds) * 100, "hit": int(pay > 0),
        })
    return records


def _period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def roi_str(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return f"  {'—':>9}   {'—':>4}"
    roi = sub["pay"].sum() / sub["cost"].sum() * 100
    mark = "★" if roi >= 100 else " "
    return f"  {roi:>8.1f}%{mark}  {len(sub):>4}"


def main():
    print("venue × grade 限定 rolling WR 特徴量実験（doc37）")
    print()

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    print("venue_grade_wr 計算中...", flush=True)
    df["venue_grade_wr"] = build_venue_grade_wr(df)
    print(f"  venue_grade_wr: mean={df['venue_grade_wr'].mean():.4f}  "
          f"nonzero={( df['venue_grade_wr'] > 0).mean():.1%}  "
          f"zero(履歴なし)={(df['venue_grade_wr'] == 0).mean():.1%}")

    # ── Phase1: AUC ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較（venue_wr のみ vs venue_wr + venue_grade_wr）")
    print("=" * 70)

    # Base モデル（現行）
    fit_base = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit_base), fit_base["top3_flag"].values)

    # 拡張モデル（venue_grade_wr 追加）
    EXT_COLS = FEATURE_COLS_WT + ["venue_grade_wr"]

    def prepare_X_ext(d: pd.DataFrame) -> pd.DataFrame:
        return d.reindex(columns=EXT_COLS).fillna(0)

    m_ext = lgb.LGBMClassifier(**LGB_PARAMS)
    m_ext.fit(prepare_X_ext(fit_base), fit_base["top3_flag"].values)

    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_ext"]  = m_ext.predict_proba(prepare_X_ext(df))[:, 1]

    print(f"\n  {'期間':<10} {'Base AUC':>10} {'拡張 AUC':>10} {'差分':>8}")
    print("  " + "-" * 40)
    for period, s, e in [
        ("VAL",    VAL[0],  VAL[1]),
        ("HOLD",   HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0], HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_x = roc_auc_score(sub["top3_flag"], sub["pred_ext"])
        diff = auc_x - auc_b
        mark = "★" if (period == "VAL+HOLD" and diff >= 0.001) else ""
        print(f"  {period:<10} {auc_b:>10.4f} {auc_x:>10.4f} {diff:>+8.4f}  {mark}")

    # ── 特徴量重要度（上位） ──────────────────────────────────────────
    print("\n  特徴量重要度（拡張モデル・venue_grade_wr 含む上位10）")
    imp = pd.Series(m_ext.feature_importances_, index=EXT_COLS)
    imp_pct = imp / imp.sum() * 100
    for feat, v in imp_pct.sort_values(ascending=False).head(10).items():
        marker = " ←" if feat == "venue_grade_wr" else ""
        print(f"    {feat:<28} {v:>6.1f}%{marker}")

    # ── Phase2: ROI ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 70)

    with get_connection() as conn:
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()
        races_raw = conn.execute(
            "SELECT race_key, n_entries FROM wt_races"
        ).fetchall()

    trio_map: dict = {}
    for rk, comb, ov in trio_raw:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)

    n_entries_map = dict(races_raw)

    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    # Base
    df_b = df.copy()
    df_b["pred_prob"] = df_b["pred_base"]
    rec_b = pd.DataFrame(compute_roi_records(df_b, trio_map, actual_trio, n_entries_map))

    # 拡張
    df_x = df.copy()
    df_x["pred_prob"] = df_x["pred_ext"]
    rec_x = pd.DataFrame(compute_roi_records(df_x, trio_map, actual_trio, n_entries_map))

    print(f"\n  {'期間':<8} {'Base ROI':>10} {'拡張 ROI':>10} {'差分':>8}  n(base)/n(ext)")
    print("  " + "-" * 55)
    for period in ["TRAIN", "VAL", "HOLD"]:
        b = rec_b[rec_b["period"] == period] if len(rec_b) else pd.DataFrame()
        x = rec_x[rec_x["period"] == period] if len(rec_x) else pd.DataFrame()
        roi_b = b["pay"].sum() / b["cost"].sum() * 100 if len(b) > 0 else float("nan")
        roi_x = x["pay"].sum() / x["cost"].sum() * 100 if len(x) > 0 else float("nan")
        mk_b = "★" if roi_b >= 100 else " "
        mk_x = "★" if roi_x >= 100 else " "
        diff = roi_x - roi_b
        print(f"  {period:<8} {roi_b:>9.1f}%{mk_b}  {roi_x:>9.1f}%{mk_x}  "
              f"{diff:>+7.1f}pp  {len(b):>5}/{len(x)}")

    # ── gap12 帯別（拡張） ────────────────────────────────────────────
    if "gap12" in df.columns or True:
        df_x2 = df_x.merge(
            df.groupby("race_key")["pred_ext"]
            .apply(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else np.nan)
            .rename("pred2_max"),
            on="race_key", how="left",
        ) if "gap12" not in df_x.columns else df_x

    print("\n  ※ Phase1/Phase2 の詳細結果を確認してください。")
    print(f"\n  venue_grade_wr 非ゼロ率（TRAIN）: "
          f"{(df.loc[df['race_date'] <= TRAIN[1], 'venue_grade_wr'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
