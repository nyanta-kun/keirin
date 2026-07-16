"""師匠情報特徴量実験（doc40）

仮説:
  JKA 登録の師匠（メンター）が同一レースに出走している場合、
  師弟関係がライン形成・走法選択に影響を与える可能性がある。

新特徴量:
  mentor_in_race:       この選手の師匠が同一レースに出走 (0/1)
  is_mentor_of_someone: この選手が同一レースの誰かの師匠 (0/1)

事前評価:
  - 同期≠同ライン（同一養成所同期の同ライン率 13.3% < 異期 22.9%）の実績から
    師匠関係も方向性が不明確（疎・弱シグナル）と予測
  - Coverage: scraping 後に確認（予測 ~40-60%）

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）
Phase2 gate: ROI >100% 全3期間

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15

Requires: data/player_mentors.csv
  → Run: python3 scripts/scrape_mentors_wt.py
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

MENTOR_CSV = Path("data/player_mentors.csv")
GAMI_THRESHOLD = 5.0


def load_mentor_map() -> dict[int, int]:
    """player_id -> mentor_player_id のマップを返す。"""
    if not MENTOR_CSV.exists():
        raise FileNotFoundError(
            f"{MENTOR_CSV} not found.\n"
            "Run: python3 scripts/scrape_mentors_wt.py"
        )
    df = pd.read_csv(MENTOR_CSV)
    df = df[df["mentor_id"].notna() & (df["mentor_id"].astype(str).str.strip() != "")]
    df["player_id"] = df["player_id"].astype(int)
    df["mentor_id"] = df["mentor_id"].astype(float).astype(int)
    return dict(zip(df["player_id"], df["mentor_id"]))


def add_mentor_features(df: pd.DataFrame, mentor_map: dict) -> pd.DataFrame:
    """mentor_in_race / is_mentor_of_someone を付与して返す。"""
    df = df.copy()

    # O(1) set lookup で mentor_in_race
    race_player_set: set = set(zip(df["race_key"], df["player_id"].astype(int)))
    df["_mentor_id"] = df["player_id"].map(mentor_map)

    def _has_mentor(row):
        mid = row["_mentor_id"]
        if pd.isna(mid):
            return 0
        return int((row["race_key"], int(mid)) in race_player_set)

    df["mentor_in_race"] = df.apply(_has_mentor, axis=1)

    # is_mentor_of_someone: 事前に race → player set を構築して効率化
    mentee_by_mentor: dict[int, set] = {}
    for pid, mid in mentor_map.items():
        mentee_by_mentor.setdefault(mid, set()).add(pid)
    race_players: dict = df.groupby("race_key")["player_id"].apply(set).to_dict()

    def _is_mentor(row):
        mentees = mentee_by_mentor.get(int(row["player_id"]), set())
        players = race_players.get(row["race_key"], set())
        return int(bool(mentees & players))

    df["is_mentor_of_someone"] = df.apply(_is_mentor, axis=1)
    df.drop(columns=["_mentor_id"], inplace=True)
    return df


def compute_roi_records(df, trio_map, actual_trio, n_entries_map):
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
        records.append({"period": period, "race_key": rk, "pay": pay, "cost": len(thirds) * 100})
    return records


def _period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def main():
    print("師匠情報特徴量実験（doc40）\n")

    mentor_map = load_mentor_map()
    print(f"師匠データ: {len(mentor_map)} 件ロード済み")

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    print("師匠特徴量計算中...", flush=True)
    df = add_mentor_features(df, mentor_map)
    print(f"  mentor_in_race:       nonzero={(df['mentor_in_race'] > 0).mean():.1%}  "
          f"total={(df['mentor_in_race'] > 0).sum()}")
    print(f"  is_mentor_of_someone: nonzero={(df['is_mentor_of_someone'] > 0).mean():.1%}  "
          f"total={(df['is_mentor_of_someone'] > 0).sum()}")

    # ── Phase1: AUC ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase1: AUC 比較（Base vs +mentor_in_race vs +is_mentor vs +両方）")
    print("=" * 72)

    NEW1 = FEATURE_COLS_WT + ["mentor_in_race"]
    NEW2 = FEATURE_COLS_WT + ["is_mentor_of_someone"]
    NEWALL = FEATURE_COLS_WT + ["mentor_in_race", "is_mentor_of_someone"]

    def prep(cols):
        return lambda d: d.reindex(columns=cols).fillna(0)

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit), fit["top3_flag"].values)

    m_n1 = lgb.LGBMClassifier(**LGB_PARAMS)
    m_n1.fit(prep(NEW1)(fit), fit["top3_flag"].values)

    m_n2 = lgb.LGBMClassifier(**LGB_PARAMS)
    m_n2.fit(prep(NEW2)(fit), fit["top3_flag"].values)

    m_all = lgb.LGBMClassifier(**LGB_PARAMS)
    m_all.fit(prep(NEWALL)(fit), fit["top3_flag"].values)

    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_n1"]   = m_n1.predict_proba(prep(NEW1)(df))[:, 1]
    df["pred_n2"]   = m_n2.predict_proba(prep(NEW2)(df))[:, 1]
    df["pred_all"]  = m_all.predict_proba(prep(NEWALL)(df))[:, 1]

    print(f"\n  {'期間':<10} {'Base':>8} {'+mentor':>9} {'+ismentor':>10} {'+ 両方':>8}")
    print("  " + "-" * 50)
    phase1_pass = False
    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD",     HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0],  HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b  = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_n1 = roc_auc_score(sub["top3_flag"], sub["pred_n1"])
        auc_n2 = roc_auc_score(sub["top3_flag"], sub["pred_n2"])
        auc_al = roc_auc_score(sub["top3_flag"], sub["pred_all"])

        def mk(d): return "★" if (period == "VAL+HOLD" and d >= 0.001) else " "

        print(f"  {period:<10} {auc_b:.4f} {auc_n1-auc_b:>+8.4f}{mk(auc_n1-auc_b)} "
              f"{auc_n2-auc_b:>+9.4f}{mk(auc_n2-auc_b)} "
              f"{auc_al-auc_b:>+7.4f}{mk(auc_al-auc_b)}")
        if period == "VAL+HOLD":
            best = max(auc_n1 - auc_b, auc_n2 - auc_b, auc_al - auc_b)
            if best >= 0.001:
                phase1_pass = True

    # 特徴量重要度（両方追加モデル）
    print("\n  特徴量重要度（両方追加モデル・上位12）")
    imp = pd.Series(m_all.feature_importances_, index=NEWALL)
    imp_pct = imp / imp.sum() * 100
    for feat, v in imp_pct.sort_values(ascending=False).head(12).items():
        mark = " ←" if feat in ("mentor_in_race", "is_mentor_of_someone") else ""
        print(f"    {feat:<32} {v:>6.1f}%{mark}")

    if not phase1_pass:
        print("\nPhase1 不通過 → Phase2 評価省略")
        return

    # ── Phase2: ROI ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 72)

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

    print(f"\n  {'期間':<8} {'Base ROI':>10} {'+ 両方':>10}  n")
    print("  " + "-" * 42)
    for pred_col, label in [("pred_base", "Base"), ("pred_all", "+ 両方")]:
        df_tmp = df.copy()
        df_tmp["pred_prob"] = df_tmp[pred_col]
        rec = pd.DataFrame(
            compute_roi_records(df_tmp, trio_map, actual_trio, n_entries_map)
        )
        if len(rec) == 0:
            print(f"  [{label}] データなし")
            continue
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec[rec["period"] == period]
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            print(f"  {period:<8} [{label:<7}] {roi:>9.1f}%{mk}  {len(sub)}")
    print()


if __name__ == "__main__":
    main()
