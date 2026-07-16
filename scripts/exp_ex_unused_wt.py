"""EXデータ未使用3列の Phase1 AUC 実験（doc41）

仮説:
  wt_entries に存在するが FEATURE_COLS_WT に未収録の3列
    - ex_left_behind_pct: ちぎられ率  (nonzero 33.3%, corr(top3)=+0.081)
    - ex_split_line_pct:  ちぎり率    (nonzero 43.9%, corr(top3)=-0.070)
    - ex_snatch_pct:      飛びつき成功率 (nonzero 14.7%, corr(top3)=+0.035)
  ※ ex_spurt_pct・ex_thrust_pct は既に FEATURE_COLS_WT 収録済み

  これら3列を個別・組み合わせで追加し AUC 改善 ≥ +0.001 が得られるか検証する。

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）
Phase2 gate: ROI >100% 全3期間（Phase1通過時のみ）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15
"""
import sys
import re
from pathlib import Path

# ワークツリーから実行する場合でもメインリポジトリの src/ を使う
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_CANDIDATE = _SCRIPT_DIR.parent  # worktree root or main repo root
# メインリポジトリの keirin.db が存在するかで判定
if not (_REPO_CANDIDATE / "data" / "keirin.db").exists() or \
   not (_REPO_CANDIDATE / "data" / "keirin.db").stat().st_size > 10_000_000:
    # worktree の keirin.db は空なので main repo を探す
    _MAIN_REPO = Path("/Users/ysuzuki/GitHub/keirin")
    if _MAIN_REPO.exists():
        sys.path.insert(0, str(_MAIN_REPO))
        sys.path.insert(0, str(_MAIN_REPO / "scripts"))
    else:
        sys.path.insert(0, str(_REPO_CANDIDATE))
else:
    sys.path.insert(0, str(_REPO_CANDIDATE))
    sys.path.insert(0, str(_REPO_CANDIDATE / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)

# 実験期間 (exp_segment_first_wt.py と同一設定)
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-15")

LGB_PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, verbose=-1)

GAMI_THRESHOLD = 5.0

# 追加候補列の定義
NEW_COLS = ["ex_left_behind_pct", "ex_split_line_pct", "ex_snatch_pct"]


def normalize_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    """新規追加列を 0-1 に正規化する。
    - ex_left_behind_pct は feature_wt.py で既に /100 済み（0-1）
    - ex_split_line_pct, ex_snatch_pct は raw % (0-100) なので /100 する
    """
    out = df.copy()
    # ex_left_behind_pct: feature_wt.py で /100 済み → そのまま (クリップのみ)
    out["ex_left_behind_pct"] = out["ex_left_behind_pct"].fillna(0.0).clip(0, 1)
    # ex_split_line_pct, ex_snatch_pct: raw % → /100
    out["ex_split_line_pct"] = (out["ex_split_line_pct"].fillna(0.0) / 100.0).clip(0, 1)
    out["ex_snatch_pct"]     = (out["ex_snatch_pct"].fillna(0.0)     / 100.0).clip(0, 1)
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
    print("EXデータ未使用3列の Phase1 AUC 実験（doc41）")
    print()

    print("データ準備中（TRAIN〜HOLD）...", flush=True)
    df_raw = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    df = normalize_new_cols(df_raw)

    # 新規列の基礎統計
    print("\n  新規追加列の基礎統計（finish_order >= 1 対象）:")
    valid = df[df["finish_order"] >= 1]
    for col in NEW_COLS:
        nz = (valid[col] > 0).mean()
        corr = valid[col].corr(valid["top3_flag"])
        print(f"  {col:<25} nonzero={nz:.1%}  corr(top3)={corr:+.4f}")

    # ── モデル設定 ──────────────────────────────────────────────────────
    EXT_LEFT_COLS  = FEATURE_COLS_WT + ["ex_left_behind_pct"]
    EXT_SPLIT_COLS = FEATURE_COLS_WT + ["ex_split_line_pct"]
    EXT_SNATCH_COLS = FEATURE_COLS_WT + ["ex_snatch_pct"]
    EXT_ALL_COLS   = FEATURE_COLS_WT + NEW_COLS

    def prep(cols):
        return lambda d: d.reindex(columns=cols).fillna(0)

    # ── Phase1: AUC ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase1: AUC 比較（Base / +left / +split / +snatch / +all3）")
    print("=" * 72)

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    print(f"  学習サンプル数: {len(fit):,}")

    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit), fit["top3_flag"].values)

    m_left = lgb.LGBMClassifier(**LGB_PARAMS)
    m_left.fit(prep(EXT_LEFT_COLS)(fit), fit["top3_flag"].values)

    m_split = lgb.LGBMClassifier(**LGB_PARAMS)
    m_split.fit(prep(EXT_SPLIT_COLS)(fit), fit["top3_flag"].values)

    m_snatch = lgb.LGBMClassifier(**LGB_PARAMS)
    m_snatch.fit(prep(EXT_SNATCH_COLS)(fit), fit["top3_flag"].values)

    m_all = lgb.LGBMClassifier(**LGB_PARAMS)
    m_all.fit(prep(EXT_ALL_COLS)(fit), fit["top3_flag"].values)

    df["pred_base"]   = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_left"]   = m_left.predict_proba(prep(EXT_LEFT_COLS)(df))[:, 1]
    df["pred_split"]  = m_split.predict_proba(prep(EXT_SPLIT_COLS)(df))[:, 1]
    df["pred_snatch"] = m_snatch.predict_proba(prep(EXT_SNATCH_COLS)(df))[:, 1]
    df["pred_all"]    = m_all.predict_proba(prep(EXT_ALL_COLS)(df))[:, 1]

    header = f"  {'期間':<12} {'Base':>8} {'+ left':>9} {'+ split':>9} {'+ snatch':>10} {'+ all3':>9}"
    print(f"\n{header}")
    print("  " + "-" * 60)

    phase1_pass_any = False
    results = {}
    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD",     HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0],  HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        y = sub["top3_flag"]
        auc_b  = roc_auc_score(y, sub["pred_base"])
        auc_l  = roc_auc_score(y, sub["pred_left"])
        auc_sp = roc_auc_score(y, sub["pred_split"])
        auc_sn = roc_auc_score(y, sub["pred_snatch"])
        auc_al = roc_auc_score(y, sub["pred_all"])

        def diff_mark(d):
            return "★" if (period == "VAL+HOLD" and d >= 0.001) else " "

        results[period] = {
            "base": auc_b, "left": auc_l, "split": auc_sp,
            "snatch": auc_sn, "all": auc_al,
        }
        dl = auc_l  - auc_b
        ds = auc_sp - auc_b
        dn = auc_sn - auc_b
        da = auc_al - auc_b

        if period == "VAL+HOLD" and max(dl, ds, dn, da) >= 0.001:
            phase1_pass_any = True

        print(
            f"  {period:<12} {auc_b:.4f} "
            f"{dl:>+7.4f}{diff_mark(dl)} "
            f"{ds:>+7.4f}{diff_mark(ds)} "
            f"{dn:>+7.4f}{diff_mark(dn)} "
            f"{da:>+7.4f}{diff_mark(da)}"
        )

    # 特徴量重要度（+all3モデル・上位12）
    print("\n  特徴量重要度（+all3 モデル・上位12）")
    imp = pd.Series(m_all.feature_importances_, index=EXT_ALL_COLS)
    imp_pct = imp / imp.sum() * 100
    for feat, v in imp_pct.sort_values(ascending=False).head(12).items():
        marker = " ←" if feat in NEW_COLS else ""
        print(f"    {feat:<30} {v:>6.1f}%{marker}")

    # Phase1 判定
    print()
    if phase1_pass_any:
        print("  Phase1: 通過 ★（VAL+HOLD で AUC ≥ +0.001 の列あり）")
    else:
        print("  Phase1: 不通過（VAL+HOLD で AUC 差 < +0.001・全モデル）")

    # ── Phase2: ROI ────────────────────────────────────────────────────
    if not phase1_pass_any:
        print("\n  Phase2: スキップ（Phase1 不通過）")
        print()
        return

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

    models = [
        ("pred_base",   "Base"),
        ("pred_left",   "+left"),
        ("pred_split",  "+split"),
        ("pred_snatch", "+snatch"),
        ("pred_all",    "+all3"),
    ]

    print(f"\n  {'モデル':<12} {'期間':<8} {'ROI':>10}  n")
    print("  " + "-" * 40)
    for pred_col, label in models:
        df_tmp = df.copy()
        df_tmp["pred_prob"] = df_tmp[pred_col]
        rec = pd.DataFrame(compute_roi_records(df_tmp, trio_map, actual_trio, n_entries_map))
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = rec[rec["period"] == period] if len(rec) else pd.DataFrame()
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            print(f"  {label:<12} {period:<8} {roi:>9.1f}%{mk}  {len(sub)}")
        print()


if __name__ == "__main__":
    main()
