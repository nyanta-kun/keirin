"""身体測定特徴量実験（doc43）

仮説:
  JKA 登録の身体測定値（身長・体重・背筋力・肺活量・太もも周径・胸囲）が
  top3 予測の追加情報を持つ可能性がある。
  特に体重あたり背筋力（bsr_per_weight）は「パワーウェイト比」として
  スプリント能力と相関する可能性がある。

新特徴量:
  weight_kg       : 体重 (kg)
  back_strength_kg: 背筋力 (kg)
  lung_capacity_cc: 肺活量 (cc)  ← 欠損率高（公式には記載なし選手多い）
  thigh_cm        : 太もも周径 (cm)
  chest_cm        : 胸囲 (cm)
  bsr_per_weight  : back_strength_kg / weight_kg（欠損は 0.0 で補完）

事前評価:
  - 身体測定は入団時の固定値であり、加齢やコンディション変化を反映しない
  - モデルの既存特徴量（rolling top3 率・オッズ等）に対してどの程度独立情報を持つか不明
  - 肺活量の Coverage が低い場合は欠損補完（0.0 = 平均以下扱い）がバイアスになる可能性あり

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD 平均）
Phase2 gate: ROI >100% 全3期間

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15

Requires: data/player_physicals.csv
  → Run: python3 scripts/scrape_physicals_wt.py
"""
import sys
import re
from pathlib import Path

# ワークツリー内でも本番DBを参照できるよう、リポジトリルートを特定する。
_script_dir = Path(__file__).resolve().parent
_candidates = [
    _script_dir.parent,
    Path("/Users/ysuzuki/GitHub/keirin"),
]
for _repo_root in _candidates:
    _db = _repo_root / "data" / "keirin.db"
    if _db.exists() and _db.stat().st_size > 10_000:
        break
sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)

# ── 期間設定（exp_segment_first_wt.py と同値）────────────────────────────────
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-15")

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

PHYSICALS_CSV = _repo_root / "data" / "player_physicals.csv"
GAMI_THRESHOLD = 5.0

PHYSICAL_COLS = [
    "weight_kg", "back_strength_kg", "lung_capacity_cc",
    "thigh_cm", "chest_cm", "bsr_per_weight",
]


def load_physicals() -> pd.DataFrame:
    """身体測定 CSV を読み込む。なければ FileNotFoundError でガイダンスを表示。"""
    if not PHYSICALS_CSV.exists():
        raise FileNotFoundError(
            f"{PHYSICALS_CSV} が見つかりません。\n"
            "先にスクレイピングを実行してください:\n"
            "  python3 scripts/scrape_physicals_wt.py\n\n"
            "テスト用（5人）:\n"
            "  python3 scripts/scrape_physicals_wt.py --limit 5"
        )
    df = pd.read_csv(PHYSICALS_CSV)
    df["player_id"] = df["player_id"].astype(int)
    for col in ["weight_kg", "back_strength_kg", "lung_capacity_cc", "thigh_cm", "chest_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_physical_features(df: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    """身体測定特徴量を df に結合して返す。欠損は 0.0 で補完。"""
    out = df.copy()
    out = out.merge(
        phys[["player_id", "weight_kg", "back_strength_kg", "lung_capacity_cc", "thigh_cm", "chest_cm"]],
        on="player_id",
        how="left",
    )
    # bsr_per_weight = 背筋力 / 体重（両方 > 0 の場合のみ）
    out["bsr_per_weight"] = np.where(
        (out["weight_kg"] > 0) & (out["back_strength_kg"] > 0),
        out["back_strength_kg"] / out["weight_kg"],
        np.nan,
    )
    for col in PHYSICAL_COLS:
        out[col] = out[col].fillna(0.0)
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
    print("身体測定特徴量実験（doc43）\n")

    # ── データ準備 ─────────────────────────────────────────────────────────
    try:
        phys = load_physicals()
    except FileNotFoundError as e:
        print(e)
        print("\nスクレイプ後に再実行してください。")
        return

    n_total = len(phys)
    print(f"身体測定データ: {n_total} 選手ロード済み")

    # Coverage レポート
    for col in ["weight_kg", "back_strength_kg", "lung_capacity_cc", "thigh_cm", "chest_cm"]:
        nonzero = (phys[col].notna() & (phys[col] > 0)).sum()
        pct = 100 * nonzero / n_total if n_total > 0 else 0.0
        print(f"  {col:<20}: {nonzero:>5} / {n_total}  ({pct:.1f}%)")

    # データが極端に少ない場合は skip
    min_coverage = phys["weight_kg"].notna().sum()
    if min_coverage < 100:
        print(
            f"\n身体測定データが {min_coverage} 件と少なすぎます（100件以上必要）。\n"
            "全件スクレイプ後に再実行してください:\n"
            "  python3 scripts/scrape_physicals_wt.py"
        )
        return

    print("\nデータ準備中（TRAIN〜HOLD）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    print("身体測定特徴量計算中...", flush=True)
    df = add_physical_features(df, phys)
    for col in PHYSICAL_COLS:
        nonzero = (df[col] > 0).mean()
        print(f"  {col:<20}: nonzero={nonzero:.1%}")

    # ── Phase1: AUC 比較 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較（Base vs +weight vs +back_strength vs +lung vs +all）")
    print("=" * 70)

    NEW_W   = FEATURE_COLS_WT + ["weight_kg"]
    NEW_BSR = FEATURE_COLS_WT + ["back_strength_kg", "bsr_per_weight"]
    NEW_LNG = FEATURE_COLS_WT + ["lung_capacity_cc"]
    NEWALL  = FEATURE_COLS_WT + PHYSICAL_COLS

    def prep(cols):
        return lambda d: d.reindex(columns=cols).fillna(0)

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit), fit["top3_flag"].values)

    m_w = lgb.LGBMClassifier(**LGB_PARAMS)
    m_w.fit(prep(NEW_W)(fit), fit["top3_flag"].values)

    m_bsr = lgb.LGBMClassifier(**LGB_PARAMS)
    m_bsr.fit(prep(NEW_BSR)(fit), fit["top3_flag"].values)

    m_lng = lgb.LGBMClassifier(**LGB_PARAMS)
    m_lng.fit(prep(NEW_LNG)(fit), fit["top3_flag"].values)

    m_all = lgb.LGBMClassifier(**LGB_PARAMS)
    m_all.fit(prep(NEWALL)(fit), fit["top3_flag"].values)

    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_w"]    = m_w.predict_proba(prep(NEW_W)(df))[:, 1]
    df["pred_bsr"]  = m_bsr.predict_proba(prep(NEW_BSR)(df))[:, 1]
    df["pred_lng"]  = m_lng.predict_proba(prep(NEW_LNG)(df))[:, 1]
    df["pred_all"]  = m_all.predict_proba(prep(NEWALL)(df))[:, 1]

    print(f"\n  {'期間':<10} {'Base':>8} {'+ weight':>9} {'+ bsr':>8} {'+ lung':>8} {'+ all':>8}")
    print("  " + "-" * 56)
    phase1_pass = False
    phase1_best_col = "pred_base"
    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD",     HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0],  HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b   = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_w   = roc_auc_score(sub["top3_flag"], sub["pred_w"])
        auc_bsr = roc_auc_score(sub["top3_flag"], sub["pred_bsr"])
        auc_lng = roc_auc_score(sub["top3_flag"], sub["pred_lng"])
        auc_al  = roc_auc_score(sub["top3_flag"], sub["pred_all"])

        def mk(d): return "★" if (period == "VAL+HOLD" and d >= 0.001) else " "

        print(f"  {period:<10} {auc_b:.4f} "
              f"{auc_w   - auc_b:>+8.4f}{mk(auc_w   - auc_b)} "
              f"{auc_bsr - auc_b:>+7.4f}{mk(auc_bsr - auc_b)} "
              f"{auc_lng - auc_b:>+7.4f}{mk(auc_lng - auc_b)} "
              f"{auc_al  - auc_b:>+7.4f}{mk(auc_al  - auc_b)}")

        if period == "VAL+HOLD":
            best_delta = max(auc_w - auc_b, auc_bsr - auc_b, auc_lng - auc_b, auc_al - auc_b)
            if best_delta >= 0.001:
                phase1_pass = True
                # 最も改善した列を特定
                best_pred = max(
                    [("pred_w", auc_w), ("pred_bsr", auc_bsr),
                     ("pred_lng", auc_lng), ("pred_all", auc_al)],
                    key=lambda x: x[1],
                )
                phase1_best_col = best_pred[0]

    # 特徴量重要度（全特徴追加モデル）
    print("\n  特徴量重要度（全身体測定追加モデル・上位15）")
    imp = pd.Series(m_all.feature_importances_, index=NEWALL)
    imp_pct = imp / imp.sum() * 100
    for feat, v in imp_pct.sort_values(ascending=False).head(15).items():
        mark = " ←" if feat in PHYSICAL_COLS else ""
        print(f"    {feat:<32} {v:>6.1f}%{mark}")

    if not phase1_pass:
        print("\nPhase1 不通過 → Phase2 評価省略")
        return

    # ── Phase2: ROI ────────────────────────────────────────────────────────
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

    print(f"\n  {'期間':<8} {'Base ROI':>10} {'+ 全身体':>10}  n")
    print("  " + "-" * 36)
    for pred_col, label in [("pred_base", "Base"), (phase1_best_col, "+ best")]:
        df_tmp = df.copy()
        df_tmp["pred_prob"] = df_tmp[pred_col]
        rec = pd.DataFrame(compute_roi_records(df_tmp, trio_map, actual_trio, n_entries_map))
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
