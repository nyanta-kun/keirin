"""ライン連携コヒージョン特徴量実験（doc38）

仮説:
  競輪のライン（チーム走行）は固定ではなく毎レース異なる組み合わせ。
  過去に同じリーダー(L)に続いた番手選手(F)の top3 率は
  個人の成績（top3_3m 等）では捉えられない「ライン相性」シグナルを持つ可能性がある。

新特徴量:
  partner_top3_rate: リーダー L に続いたとき（line_pos ≥ 2）の F の top3 率
                     (shifting: 前レースまでの expanding mean・リーク無し)
  leader_win_rate:  F が追走したリーダー L の過去勝率（L が強いラインに乗っているか）

  ※ line_pos = 1 の選手（リーダー自身）は partner_top3_rate = NaN → 0 で補完

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）
Phase2 gate: ROI >100% 全3期間

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


def build_line_cohesion_features(df: pd.DataFrame) -> pd.DataFrame:
    """ライン連携コヒージョン特徴量を付与して返す。

    付与される列:
      partner_top3_rate: このレースで一緒のラインを組んだリーダーとの過去 top3 率
      leader_win_rate:   このレースのリーダーの過去勝率（同リーダーと組んだ全レース基準）

    line_pos = 1（リーダー）には 0.0 を割り当て（リーダー視点の指標は別途検討）。
    """
    with get_connection() as conn:
        H = pd.read_sql_query(
            """
            SELECT e.race_key, e.player_id, e.line_group, e.line_pos,
                   e.is_line_leader, e.finish_order, r.race_date
            FROM wt_entries e
            JOIN wt_races r ON e.race_key = r.race_key
            WHERE e.finish_order >= 1
            """,
            conn,
        )

    H["_dt"] = pd.to_datetime(H["race_date"])
    H["top3"] = H["finish_order"].between(1, 3).astype(float)
    H["win"] = (H["finish_order"] == 1).astype(float)

    # リーダーの情報を取得（同一 race_key × line_group の is_line_leader=1）
    leaders = (
        H[H["is_line_leader"] == 1][["race_key", "line_group", "player_id"]]
        .rename(columns={"player_id": "leader_id"})
    )

    # follower（line_pos ≥ 2 または is_line_leader=0）と leader を結合
    H_full = H.merge(leaders, on=["race_key", "line_group"], how="left")
    # リーダー自身の leader_id は自分 → partner 指標は意味なし（0で埋め）
    H_follower = H_full[
        (H_full["leader_id"].notna())
        & (H_full["player_id"] != H_full["leader_id"])
    ].copy()

    # ── 1. partner_top3_rate: (leader_id, follower_id) ペア別 top3 expanding mean
    H_follower = H_follower.sort_values(["player_id", "leader_id", "_dt"]).reset_index(drop=True)
    H_follower["partner_top3_rate"] = (
        H_follower.groupby(["player_id", "leader_id"])["top3"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=[0, 1], drop=True)
    )

    # ── 2. leader_win_rate: (leader_id) の過去勝率（follower が組んだ全レース）
    #    = リーダーとして組んだ全レースで win=1 だった率
    H_follower["leader_win_rate"] = (
        H_follower.groupby(["player_id", "leader_id"])
        .apply(lambda g: pd.Series(
            g.sort_values("_dt")["win"].expanding().mean().shift(1).values,
            index=g.index,
        ))
        .reset_index(level=[0, 1], drop=True)
    )

    pair_map = H_follower[["race_key", "player_id", "partner_top3_rate", "leader_win_rate"]].copy()
    pair_map = pair_map.rename(columns={"player_id": "follower_player_id"})

    # df への merge
    out = df.copy()
    out = out.merge(
        pair_map.rename(columns={"follower_player_id": "player_id"}),
        on=["race_key", "player_id"],
        how="left",
    )
    out["partner_top3_rate"] = out["partner_top3_rate"].fillna(0.0)
    out["leader_win_rate"] = out["leader_win_rate"].fillna(0.0)
    return out


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
            "pay": pay, "cost": len(thirds) * 100,
        })
    return records


def _period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def main():
    print("ライン連携コヒージョン特徴量実験（doc38）")
    print()

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    print("ライン連携コヒージョン特徴量計算中...", flush=True)
    df = build_line_cohesion_features(df)
    print(f"  partner_top3_rate: nonzero={( df['partner_top3_rate'] > 0).mean():.1%}  "
          f"mean(>0)={df.loc[df['partner_top3_rate'] > 0, 'partner_top3_rate'].mean():.4f}")
    print(f"  leader_win_rate:   nonzero={(df['leader_win_rate'] > 0).mean():.1%}  "
          f"mean(>0)={df.loc[df['leader_win_rate'] > 0, 'leader_win_rate'].mean():.4f}")

    # ── Phase1: AUC ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較（Base vs +partner_top3_rate vs +leader_win_rate vs +両方）")
    print("=" * 70)

    EXT2_COLS = FEATURE_COLS_WT + ["partner_top3_rate"]
    EXT3_COLS = FEATURE_COLS_WT + ["leader_win_rate"]
    EXTALL_COLS = FEATURE_COLS_WT + ["partner_top3_rate", "leader_win_rate"]

    def prep(cols):
        return lambda d: d.reindex(columns=cols).fillna(0)

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit), fit["top3_flag"].values)

    m_ext2 = lgb.LGBMClassifier(**LGB_PARAMS)
    m_ext2.fit(prep(EXT2_COLS)(fit), fit["top3_flag"].values)

    m_ext3 = lgb.LGBMClassifier(**LGB_PARAMS)
    m_ext3.fit(prep(EXT3_COLS)(fit), fit["top3_flag"].values)

    m_extall = lgb.LGBMClassifier(**LGB_PARAMS)
    m_extall.fit(prep(EXTALL_COLS)(fit), fit["top3_flag"].values)

    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_p3"]   = m_ext2.predict_proba(prep(EXT2_COLS)(df))[:, 1]
    df["pred_lwr"]  = m_ext3.predict_proba(prep(EXT3_COLS)(df))[:, 1]
    df["pred_all"]  = m_extall.predict_proba(prep(EXTALL_COLS)(df))[:, 1]

    print(f"\n  {'期間':<10} {'Base':>8} {'+ p3r':>8} {'+ lwr':>8} {'+ 両方':>8}")
    print("  " + "-" * 44)
    for period, s, e in [
        ("VAL",    VAL[0],  VAL[1]),
        ("HOLD",   HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0], HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b  = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_p3 = roc_auc_score(sub["top3_flag"], sub["pred_p3"])
        auc_lw = roc_auc_score(sub["top3_flag"], sub["pred_lwr"])
        auc_al = roc_auc_score(sub["top3_flag"], sub["pred_all"])
        def diff_mark(d): return "★" if (period == "VAL+HOLD" and d >= 0.001) else " "
        print(f"  {period:<10} {auc_b:.4f} {auc_p3-auc_b:>+7.4f}{diff_mark(auc_p3-auc_b)} "
              f"{auc_lw-auc_b:>+7.4f}{diff_mark(auc_lw-auc_b)} "
              f"{auc_al-auc_b:>+7.4f}{diff_mark(auc_al-auc_b)}")

    # 特徴量重要度
    print("\n  特徴量重要度（両方追加モデル・上位12）")
    imp = pd.Series(m_extall.feature_importances_, index=EXTALL_COLS)
    imp_pct = imp / imp.sum() * 100
    for feat, v in imp_pct.sort_values(ascending=False).head(12).items():
        marker = " ←" if feat in ("partner_top3_rate", "leader_win_rate") else ""
        print(f"    {feat:<28} {v:>6.1f}%{marker}")

    # ── Phase2: ROI ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 70)

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

    print(f"\n  {'期間':<8} {'Base ROI':>10} {'+ p3r':>10} {'+ 両方':>10}  n")
    print("  " + "-" * 50)
    for pred_col, label in [
        ("pred_base", "Base"),
        ("pred_p3",   "+ p3r"),
        ("pred_all",  "+ 両方"),
    ]:
        df_tmp = df.copy()
        df_tmp["pred_prob"] = df_tmp[pred_col]
        rec = pd.DataFrame(compute_roi_records(df_tmp, trio_map, actual_trio, n_entries_map))
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec[rec["period"] == period] if len(rec) else pd.DataFrame()
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            print(f"  {period:<8} [{label:<7}] {roi:>9.1f}%{mk}  {len(sub)}")
    print()


if __name__ == "__main__":
    main()
