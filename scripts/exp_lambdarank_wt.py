"""LambdaRank 目的関数実験

LightGBM の LGBMRanker（lambdarank）で着順を直接最適化し、
二値分類ベースラインと AUC・ROI を比較する。

ラベルスキーム:
  Binary : top3=1, others=0（現行と同型）
  Ordinal: 1着=3, 2着=2, 3着=1, others=0（着順情報をフル活用）

Usage:
  python3 scripts/exp_lambdarank_wt.py
"""
import sys, re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0

# ── LGBMRanker パラメータ（LGB_PARAMS から objective を除いて流用）────
RANK_PARAMS = {k: v for k, v in LGB_PARAMS.items() if k != "objective"}


def _period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def build_groups_and_labels(df: pd.DataFrame, label_type: str = "ordinal"):
    """race_key 単位でグループ・ラベルを構築する。

    Returns
    -------
    df_sorted : DataFrame（race_key でソート済み）
    groups    : ndarray (各レースの行数)
    labels    : ndarray (relevance スコア)
    """
    df_sorted = df.sort_values("race_key").reset_index(drop=True)
    groups = df_sorted.groupby("race_key", sort=False)["race_key"].count().values

    fo = df_sorted["finish_order"].values
    if label_type == "binary":
        labels = np.where((fo >= 1) & (fo <= 3), 1, 0).astype(int)
    else:  # ordinal
        labels = np.where(fo == 1, 3,
                 np.where(fo == 2, 2,
                 np.where(fo == 3, 1, 0))).astype(int)
    return df_sorted, groups, labels


def compute_roi(df: pd.DataFrame, pred_col: str, trio_map: dict,
                actual_trio: dict, n_entries_map: dict) -> list[dict]:
    records = []
    for rk, grp in df.groupby("race_key"):
        if n_entries_map.get(rk, 99) > 6:
            continue
        g = grp.sort_values(pred_col, ascending=False).reset_index(drop=True)
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
        records.append({"period": period, "pay": pay, "cost": len(thirds) * 100})
    return records


def main():
    print("LambdaRank 実験\n")

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    # 欠車除外（finish_order=0）
    df = df_all[df_all["finish_order"] >= 1].copy()
    print(f"  全行数: {len(df):,}  レース数: {df['race_key'].nunique():,}")

    # ── TRAIN データ ─────────────────────────────────────────────────
    fit_raw = df[df["race_date"] <= TRAIN[1]].copy()

    # ── ベースライン: LGBMClassifier（現行）────────────────────────
    print("\n[1/3] Baseline LGBMClassifier を学習中...", flush=True)
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit_raw), fit_raw["top3_flag"].values)
    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]

    # ── LGBMRanker Binary ──────────────────────────────────────────
    print("[2/3] LGBMRanker (binary label) を学習中...", flush=True)
    fit_b, grp_b, lbl_b = build_groups_and_labels(fit_raw, "binary")
    m_rank_b = lgb.LGBMRanker(**RANK_PARAMS)
    m_rank_b.fit(prepare_X(fit_b), lbl_b, group=grp_b)
    df["pred_rank_binary"] = m_rank_b.predict(prepare_X(df))

    # ── LGBMRanker Ordinal ─────────────────────────────────────────
    print("[3/3] LGBMRanker (ordinal label: 1着=3, 2着=2, 3着=1) を学習中...", flush=True)
    fit_o, grp_o, lbl_o = build_groups_and_labels(fit_raw, "ordinal")
    m_rank_o = lgb.LGBMRanker(**RANK_PARAMS)
    m_rank_o.fit(prepare_X(fit_o), lbl_o, group=grp_o)
    df["pred_rank_ordinal"] = m_rank_o.predict(prepare_X(df))

    # ── Phase1: AUC ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Phase1: AUC 比較")
    print("=" * 65)
    print(f"\n  {'期間':<10} {'Baseline':>9} {'Δrank_bin':>10} {'Δrank_ord':>10}")
    print("  " + "-" * 42)

    best_delta = -999
    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD",     HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0],  HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b  = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_rb = roc_auc_score(sub["top3_flag"], sub["pred_rank_binary"])
        auc_ro = roc_auc_score(sub["top3_flag"], sub["pred_rank_ordinal"])
        db = auc_rb - auc_b
        do = auc_ro - auc_b
        mk_b = "★" if (period == "VAL+HOLD" and db >= 0.001) else " "
        mk_o = "★" if (period == "VAL+HOLD" and do >= 0.001) else " "
        print(f"  {period:<10} {auc_b:.4f}   {db:>+8.4f}{mk_b}  {do:>+8.4f}{mk_o}")
        if period == "VAL+HOLD":
            best_delta = max(db, do)

    # ── 特徴量重要度 ────────────────────────────────────────────────
    print("\n  特徴量重要度（rank_ordinal・上位12）")
    imp = pd.Series(m_rank_o.feature_importances_, index=FEATURE_COLS_WT)
    for feat, v in (imp / imp.sum() * 100).nlargest(12).items():
        print(f"    {feat:<35} {v:.1f}%")

    # ── Phase1 判定 ─────────────────────────────────────────────────
    print()
    if best_delta >= 0.001:
        print(f"Phase1 通過 ★（最大改善 VAL+HOLD: Δ={best_delta:+.4f}）→ Phase2 評価")
    else:
        print(f"Phase1 不通過（最大改善 Δ={best_delta:+.4f}、閾値 +0.001 未達）→ Phase2 省略")
        return

    # ── Phase2: ROI ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 65)

    with get_connection() as conn:
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()
        races_raw = conn.execute("SELECT race_key, n_entries FROM wt_races").fetchall()

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

    eval_models = [
        ("pred_base",         "Baseline"),
        ("pred_rank_binary",  "rank_binary"),
        ("pred_rank_ordinal", "rank_ordinal"),
    ]

    print(f"\n  {'モデル':<14} {'TRAIN':>9} {'VAL':>9} {'HOLD':>9}  n(TR/VA/HO)")
    print("  " + "-" * 58)
    for col, label in eval_models:
        recs = pd.DataFrame(compute_roi(df, col, trio_map, actual_trio, n_entries_map))
        row = [f"  {label:<14}"]
        ns = []
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = recs[recs["period"] == period] if len(recs) else pd.DataFrame()
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            row.append(f" {roi:>8.1f}%{mk}")
            ns.append(len(sub))
        print("".join(row) + f"  {ns[0]}/{ns[1]}/{ns[2]}")
    print()


if __name__ == "__main__":
    main()
