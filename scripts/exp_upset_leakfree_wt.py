"""W01: 波乱モデル × リーク無し再評価（doc18対応）

背景（docs/goals/W01-upset-leakfree-reeval.md）:
  既存 lgbm_upset.pkl は HOLDOUT 期間まで学習済み（リーク）。
  本実験は TRAIN 期間のみで波乱モデルを再学習し、VAL/HOLDOUT で
  「高波乱確率レースを選別するとROIが改善するか」を検証する。

波乱の定義: 三連複(trio)払戻 ≥ 2000円（× 100 = オッズ × 100 なので odds_value ≥ 20）

doc18 セマンティクス:
  - ランキングは全エントリー（欠車=finish_order=0 含む）
  - ≤6車フィルタは出走表基準（race_keyごとの行数）
  - 欠車処理: void_by_dns 準拠（軸欠車=無効・相手欠車=その目除外）
  - モデル: TRAIN期間のみで学習した lgbm_wt_eval（期間限定学習モデル）
  - 払戻: wt_odds の最終オッズ × 100

期間:
  TRAIN   2023-07-01 〜 2025-06-30
  VAL     2025-07-01 〜 2026-02-28
  HOLDOUT 2026-03-01 〜 2026-06-14
"""
import sys
import pickle
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.models.trainer import load_model
from src.database import get_connection
from src.evaluation.backtest_wt import _assign_tier
from src.evaluation.void_rules import void_by_dns

# ============================================================================
# 定数
# ============================================================================
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-14")

UPSET_THRESHOLD = 2000   # trio実際払戻(=odds_value*100) ≥ 2000 → 波乱
# NOTE: 実際の的中trio払戻（≤6車中央値550円・波乱率約19%）を使用。
# wt_oddsは全組み合わせを持つため、MAX払戻だと常に≥2000になるので不可。
# build_features_wt の finish_order から実際の上位3頭を特定してオッズを引く。
LGB_PARAMS_ENTRY = dict(
    objective="binary", n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)
LGB_PARAMS_UPSET = dict(
    objective="binary", metric="auc",
    n_estimators=500, learning_rate=0.03,
    num_leaves=15, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# 波乱モデル特徴量（upset_model.py の UPSET_FEATURE_COLS に対応・wt版）
# racing_score → race_point / recent_top3_rate_6m → top3_6m
UPSET_FEAT_COLS = [
    "grade_enc",        # グレード
    "n_riders",         # 出走頭数（計算）
    "bank_length_enc",  # バンク長/100
    "is_indoor",        # 屋内フラグ
    "score_mean",       # 競走得点の平均（build_features_wt で計算済み）
    "score_std",        # 競走得点の標準偏差
    "score_cv",         # 変動係数（= std/mean）
    "score_range",      # 最大-最小
    "score_top_gap",    # 1位-2位の得点差
    "top3r_mean",       # 3着内率(6m)の平均
    "top3r_std",        # 3着内率(6m)の標準偏差
    "pred_top1",        # 最高予測確率
    "pred_top2",        # 2番目の予測確率
    "pred_gap12",       # top1-top2 の差
    "pred_gap23",       # top2-top3 の差
    "pred_entropy",     # Shannon entropy
    "pred_top3_sum",    # 上位3頭の確率合計
]


# ============================================================================
# Step 1: データロード & 特徴量構築（doc18セマンティクス）
# ============================================================================
def load_data() -> pd.DataFrame:
    print("loading raw data (2023-07〜2026-06-14)...", flush=True)
    df_raw = load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1])
    print(f"  raw: {len(df_raw):,} rows, {df_raw['race_key'].nunique():,} races", flush=True)
    df = build_features_wt(df_raw)
    print(f"  built: {len(df):,} rows", flush=True)
    return df


def apply_entry_model(df: pd.DataFrame) -> pd.DataFrame:
    """TRAIN期間のみで学習したエントリーモデルで pred_prob を付与。
    doc18: 全エントリー（欠車含む）でランキング。"""
    print("training fresh entry LGBM on TRAIN only...", flush=True)
    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)].copy()
    print(f"  fit rows: {len(fit):,}", flush=True)
    m = lgb.LGBMClassifier(**LGB_PARAMS_ENTRY)
    m.fit(prepare_X(fit), fit["top3_flag"])
    df = df.copy()
    df["pred_prob"] = m.predict_proba(prepare_X(df))[:, 1]
    return df, m


def filter_le6_doc18(df: pd.DataFrame) -> pd.DataFrame:
    """出走表基準で≤6車のみ残す（doc18 バイアス②修正）。"""
    sizes = df.groupby("race_key")["frame_no"].count()
    valid = sizes[sizes <= 6].index
    return df[df["race_key"].isin(valid)].copy()


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    """結果確定レースのみ（3頭以上完走）。"""
    done = df.groupby("race_key")["finish_order"].apply(
        lambda s: (s.between(1, 3)).sum() >= 3
    )
    return df[df["race_key"].isin(done[done].index)].copy()


# ============================================================================
# Step 2: レース単位の特徴量構築（wt版）
# ============================================================================
def build_race_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """エントリーレベルdf（pred_prob計算済み）からレース単位の特徴量を構築。

    upset_model.build_race_features のwt対応版。
    racing_score → race_point / recent_top3_rate_6m → top3_6m
    """
    rows = []
    for race_key, grp in df.groupby("race_key"):
        probs = np.sort(grp["pred_prob"].fillna(0).values)[::-1]
        probs_safe = probs + 1e-9
        probs_norm = probs_safe / probs_safe.sum()

        scores = grp["race_point"].dropna().values
        scores_sorted = np.sort(scores) if len(scores) > 0 else np.array([])

        top3r = grp["top3_6m"].dropna().values if "top3_6m" in grp.columns else np.array([])

        row = {
            "race_key": race_key,
            "race_date": grp["race_date"].iloc[0],
            "n_riders": int(len(grp)),
            "grade_enc": float(grp["grade_enc"].iloc[0]) if "grade_enc" in grp.columns else np.nan,
            "bank_length_enc": float(grp["bank_length_enc"].iloc[0]) if "bank_length_enc" in grp.columns else np.nan,
            "is_indoor": float(grp["is_indoor"].iloc[0]) if "is_indoor" in grp.columns else 0.0,

            "score_mean":    float(np.mean(scores)) if len(scores) > 0 else np.nan,
            "score_std":     float(np.std(scores)) if len(scores) > 0 else np.nan,
            "score_cv":      float(np.std(scores) / np.mean(scores))
                             if len(scores) > 0 and np.mean(scores) > 0 else np.nan,
            "score_range":   float(np.ptp(scores)) if len(scores) > 0 else np.nan,
            "score_top_gap": float(scores_sorted[-1] - scores_sorted[-2])
                             if len(scores_sorted) > 1 else np.nan,

            "top3r_mean": float(np.mean(top3r)) if len(top3r) > 0 else np.nan,
            "top3r_std":  float(np.std(top3r)) if len(top3r) > 0 else np.nan,

            "pred_top1":     float(probs[0]) if len(probs) > 0 else np.nan,
            "pred_top2":     float(probs[1]) if len(probs) > 1 else np.nan,
            "pred_gap12":    float(probs[0] - probs[1]) if len(probs) > 1 else np.nan,
            "pred_gap23":    float(probs[1] - probs[2]) if len(probs) > 2 else np.nan,
            "pred_entropy":  float(-np.sum(probs_norm * np.log(probs_norm))),
            "pred_top3_sum": float(np.sum(probs[:3])) if len(probs) >= 3 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Step 3: 波乱ラベル付与（wt版: 実際の的中trio払戻を使用）
# ============================================================================
def add_upset_target_wt(df_race: pd.DataFrame,
                        df_entry: pd.DataFrame,
                        upset_threshold: int = UPSET_THRESHOLD) -> pd.DataFrame:
    """レース単位dfに波乱ラベルを付与（wt版: 実際の的中trio払戻を使用）。

    upset_model.add_upset_target のwt対応版。
    wt_oddsは全組み合わせを持つため MAX払戻だと常に≥2000（≤6車で波乱率99.8%の誤り）。
    修正: df_entry の finish_order から実際の上位3頭を特定し、その組み合わせの払戻を使用。
    波乱の定義: 実際に的中した三連複の払戻 ≥ upset_threshold（≤6車で約19%）
    """
    race_keys = df_race["race_key"].tolist()
    if not race_keys:
        df_race = df_race.copy()
        df_race["trio_actual_payout"] = np.nan
        df_race["is_upset"] = np.nan
        return df_race

    # 実際の上位3頭の組み合わせを df_entry から取得
    top3_map: dict[str, frozenset] = {}
    for rk, grp in df_entry.groupby("race_key"):
        fin = grp[grp["finish_order"].between(1, 3)]
        if len(fin) >= 3:
            top3_map[rk] = frozenset(fin["frame_no"].astype(int).tolist())

    # trio盤面をロード（全組み合わせ）
    trio_board: dict[str, dict] = {}
    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE race_key IN ({ph}) AND bet_type='trio' AND odds_value IS NOT NULL",
                chunk,
            ).fetchall()
            for row in rows:
                rk, comb, ov = row[0], row[1], row[2]
                try:
                    parts = [int(x) for x in re.split(r"[-=]", str(comb)) if x]
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio_board.setdefault(rk, {})[frozenset(parts)] = float(ov)

    # 実際の的中払戻を計算
    payout_map: dict[str, float] = {}
    for rk in race_keys:
        top3 = top3_map.get(rk)
        if top3 is None:
            continue
        bd = trio_board.get(rk, {})
        ov = bd.get(top3)
        if ov is not None:
            payout_map[rk] = float(ov) * 100

    df = df_race.copy()
    df["trio_actual_payout"] = df["race_key"].map(payout_map)
    df["is_upset"] = np.where(
        df["trio_actual_payout"].isna(), np.nan,
        (df["trio_actual_payout"] >= upset_threshold).astype(float),
    )
    return df


# ============================================================================
# Step 4: 波乱モデル学習（TRAIN期間のみ・リーク無し）
# ============================================================================
def train_upset_model_leakfree(df_race: pd.DataFrame) -> lgb.LGBMClassifier:
    """TRAIN期間のデータのみで波乱モデルを学習。"""
    df = df_race[df_race["race_date"] <= TRAIN[1]].copy()
    df = df.dropna(subset=UPSET_FEAT_COLS + ["is_upset"])
    df = df[df["is_upset"].isin([0.0, 1.0])].copy()
    df = df.sort_values("race_date")

    X = df[UPSET_FEAT_COLS].values
    y = df["is_upset"].values.astype(int)

    upset_rate = y.mean()
    scale_pos_weight = (1 - upset_rate) / upset_rate
    print(f"波乱モデル学習: {len(df):,} レース  波乱率: {upset_rate:.1%}  "
          f"scale_pos_weight={scale_pos_weight:.2f}", flush=True)

    params = {**LGB_PARAMS_UPSET, "scale_pos_weight": scale_pos_weight}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        pd.DataFrame(X, columns=UPSET_FEAT_COLS), y,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def predict_upset_prob(model: lgb.LGBMClassifier,
                       df_race: pd.DataFrame) -> np.ndarray:
    """全レース期間の波乱確率を予測。"""
    valid_idx = df_race[UPSET_FEAT_COLS].notna().all(axis=1)
    probs = np.full(len(df_race), np.nan)
    if valid_idx.sum() > 0:
        X = df_race.loc[valid_idx, UPSET_FEAT_COLS].values
        p = model.predict_proba(pd.DataFrame(X, columns=UPSET_FEAT_COLS))[:, 1]
        probs[valid_idx.values] = p
    return probs


def compute_auc_by_period(model: lgb.LGBMClassifier,
                           df_race: pd.DataFrame) -> dict:
    """TRAIN/VAL/HOLD各期間のAUCを計算。"""
    periods = {
        "TRAIN": (TRAIN[0], TRAIN[1]),
        "VAL":   (VAL[0], VAL[1]),
        "HOLD":  (HOLD[0], HOLD[1]),
    }
    aucs = {}
    for name, (d0, d1) in periods.items():
        sub = df_race[
            (df_race["race_date"] >= d0) &
            (df_race["race_date"] <= d1) &
            df_race["is_upset"].isin([0.0, 1.0])
        ].dropna(subset=UPSET_FEAT_COLS)
        if len(sub) < 10:
            aucs[name] = None
            continue
        X = sub[UPSET_FEAT_COLS].values
        y = sub["is_upset"].values.astype(int)
        probs = model.predict_proba(pd.DataFrame(X, columns=UPSET_FEAT_COLS))[:, 1]
        try:
            aucs[name] = roc_auc_score(y, probs)
        except Exception:
            aucs[name] = None
    return aucs


# ============================================================================
# Step 5: ROI評価（SS/S/A 層別 × 波乱スコア四分位）
# ============================================================================
def load_payouts(race_keys: list) -> dict:
    """wt_odds から払戻マップを構築。"""
    payout_map = {}
    if not race_keys:
        return payout_map
    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, bet_type, combination, odds_value "
                f"FROM wt_odds WHERE race_key IN ({ph})",
                chunk,
            ).fetchall()
            for row in rows:
                rk, bt, comb, ov = row[0], row[1], row[2], row[3]
                if ov is None:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=]", str(comb)) if x]
                except ValueError:
                    continue
                ordered = {"trifecta", "exacta"}
                key = tuple(parts) if bt in ordered else frozenset(parts)
                payout_map.setdefault(rk, {})[(bt, key)] = int(round(float(ov) * 100))
    return payout_map


def evaluate_period_by_quartile(df_entry: pd.DataFrame,
                                 df_race_upset: pd.DataFrame,
                                 q_thresholds: list[float],
                                 min_odds_kami: float = 5.0) -> list[dict]:
    """波乱スコア四分位ごとの SS/S/A 層別 ROI を計算。

    Args:
        df_entry: エントリーレベルdf（pred_prob付き・≤6車・結果確定済み）
        df_race_upset: レース単位df（upset_prob付き）
        q_thresholds: [Q1上限, Q2上限, Q3上限] = [25%ile, 50%ile, 75%ile]
        min_odds_kami: 3連単/3連複の最安オッズ閾値（デフォルト5.0=ガミ帯除外）
    """
    # upset_prob を df_entry に結合
    up_map = df_race_upset.set_index("race_key")["upset_prob"]
    df = df_entry.copy()
    df["upset_prob"] = df["race_key"].map(up_map)

    # 四分位ラベル付与
    def q_label(p):
        if np.isnan(p):
            return None
        if p <= q_thresholds[0]:
            return "Q1"
        if p <= q_thresholds[1]:
            return "Q2"
        if p <= q_thresholds[2]:
            return "Q3"
        return "Q4"

    df["q_label"] = df["upset_prob"].apply(q_label)

    # 払戻マップ
    payout_map = load_payouts(df["race_key"].unique().tolist())

    results = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = df[df["q_label"] == q]
        if df_q.empty:
            results.append({"quartile": q, "n_races": 0,
                            "pays": [], "bets": []})
            continue

        pays, bets = [], []
        for race_key, grp in df_q.groupby("race_key"):
            # doc18: 全エントリーでランキング
            grp = grp.sort_values("pred_prob", ascending=False)
            n = len(grp)
            if n < 3:
                continue
            probs = grp["pred_prob"].tolist()
            gap12 = probs[0] - probs[1]
            ratio = probs[0] / (3.0 / n)
            tier = _assign_tier(gap12, ratio)
            if tier is None:
                continue

            frames = grp["frame_no"].astype(int).tolist()
            pivot1, pivot2 = frames[0], frames[1]
            thirds = frames[2:5]
            if not thirds:
                continue

            fin = grp[grp["finish_order"].between(1, 3)]
            top3_set = frozenset(fin["frame_no"].astype(int).tolist())
            if len(top3_set) < 3:
                continue
            actual_order = tuple(
                fin.sort_values("finish_order")["frame_no"].astype(int).tolist()
            )
            rp = payout_map.get(race_key, {})

            # 欠車処理（void_by_dns）
            runners = set(grp[grp["finish_order"] >= 1]["frame_no"].astype(int).tolist())
            skip_race, valid_thirds = void_by_dns(pivot1, pivot2, thirds, runners)
            if skip_race:
                continue

            # ガミ帯フィルタ（最安≥5倍）
            if tier == "SS":
                bets_this = []
                for t in valid_thirds:
                    o = rp.get(("trifecta", (pivot1, pivot2, t)))
                    if o:
                        bets_this.append((o, actual_order == (pivot1, pivot2, t), "trifecta", (pivot1, pivot2, t)))
                if not bets_this:
                    continue
                min_odds = min(o for o, _, _, _ in bets_this)
                if min_odds < min_odds_kami * 100:   # min_odds_kami=5 → 500
                    continue
                pay = sum(o for o, hit, _, _ in bets_this if hit)
                bet = len(bets_this) * 100
            else:
                bets_this = []
                for t in valid_thirds:
                    c = frozenset((pivot1, pivot2, t))
                    o = rp.get(("trio", c))
                    if o:
                        bets_this.append((o, c == top3_set))
                if not bets_this:
                    continue
                min_odds = min(o for o, _ in bets_this)
                if min_odds < min_odds_kami * 100:
                    continue
                pay = sum(o for o, hit in bets_this if hit)
                bet = len(bets_this) * 100

            pays.append(pay)
            bets.append(bet)

        results.append({
            "quartile": q,
            "n_races": len(pays),
            "pays": pays,
            "bets": bets,
        })
    return results


# ============================================================================
# Step 6: bootstrap CI
# ============================================================================
def roi_summary(pays: list, bets: list, n_boot: int = 1000, seed: int = 42) -> dict:
    pay = np.array(pays, dtype=float)
    bet = np.array(bets, dtype=float)
    n = len(pay)
    if n == 0 or bet.sum() == 0:
        return {"n": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "roi_ex_max": 0.0}
    roi = pay.sum() / bet.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = pay[idx].sum(axis=1) / bet[idx].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    order = np.argsort(pay)
    keep = order[:-1] if n > 1 else order
    roi_ex_max = pay[keep].sum() / bet[keep].sum() if bet[keep].sum() > 0 else 0.0
    return {
        "n": n,
        "hits": int((pay > 0).sum()),
        "hit_rate": float((pay > 0).mean()),
        "roi": float(roi),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "roi_ex_max": float(roi_ex_max),
    }


# ============================================================================
# メイン
# ============================================================================
def fmt_row(s: dict) -> str:
    if s["n"] == 0:
        return f"{'--':>5}R  --"
    pass_str = "PASS" if s["ci_lo"] > 1.0 else ("  --" if s["n"] < 30 else "FAIL")
    return (f"{s['n']:>5}R  {s['roi']:>6.0%}"
            f" [{s['ci_lo']:>5.0%},{s['ci_hi']:>5.0%}]"
            f" 除{s['roi_ex_max']:>5.0%}  {pass_str}")


def main():
    # ── Step 1: データロード ──────────────────────────────────────
    df_all = load_data()

    # ── Step 2: エントリーモデル（TRAIN期間限定学習） ─────────────
    df_all, entry_model = apply_entry_model(df_all)

    # ≤6車フィルタ（出走表基準・doc18 バイアス②修正）
    df_le6 = filter_le6_doc18(df_all)
    df_le6 = filter_complete_races(df_le6)
    print(f"≤6車+結果確定: {df_le6['race_key'].nunique():,} races", flush=True)

    # ── Step 3: レース単位特徴量 & 波乱ラベル ────────────────────
    print("building race-level features...", flush=True)
    df_race = build_race_features_wt(df_le6)
    # 実際の的中trio払戻を使用（wt_oddsは全組み合わせなのでMAX払戻不可）
    df_race = add_upset_target_wt(df_race, df_le6)
    print(f"  race-level: {len(df_race):,} races", flush=True)
    upset_rate = df_race["is_upset"].dropna().mean()
    trio_pay_q50 = df_race["trio_actual_payout"].dropna().median()
    print(f"  波乱率 (trio実際払戻≥{UPSET_THRESHOLD}円): {upset_rate:.1%}  "
          f"中央払戻: {trio_pay_q50:.0f}円", flush=True)

    # ── Step 4: 波乱モデル学習（TRAIN期間のみ） ──────────────────
    print("training upset model (TRAIN only)...", flush=True)
    upset_model = train_upset_model_leakfree(df_race)

    # モデル保存
    save_path = MODEL_DIR / "lgbm_upset_eval.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(upset_model, f)
    print(f"saved: {save_path}", flush=True)

    # AUC計算
    df_race["upset_prob"] = predict_upset_prob(upset_model, df_race)
    aucs = compute_auc_by_period(upset_model, df_race)
    print(f"\n  波乱モデルAUC: TRAIN={aucs['TRAIN']:.4f}  "
          f"VAL={aucs['VAL']:.4f}  HOLD={aucs['HOLD']:.4f}", flush=True)

    # ── Step 5: 四分位閾値（TRAINの75パーセンタイルで固定） ──────
    train_probs = df_race[
        (df_race["race_date"] >= TRAIN[0]) &
        (df_race["race_date"] <= TRAIN[1]) &
        df_race["upset_prob"].notna()
    ]["upset_prob"].values

    q1 = float(np.percentile(train_probs, 25))
    q2 = float(np.percentile(train_probs, 50))
    q3 = float(np.percentile(train_probs, 75))
    print(f"\n  TRAIN四分位閾値: Q1≤{q1:.3f} / Q2≤{q2:.3f} / Q3≤{q3:.3f}", flush=True)

    # ── Step 6: 期間別ROI評価 ─────────────────────────────────────
    periods = {
        "TRAIN": (TRAIN[0], TRAIN[1]),
        "VAL":   (VAL[0], VAL[1]),
        "HOLD":  (HOLD[0], HOLD[1]),
    }

    all_results = {}
    for period_name, (d0, d1) in periods.items():
        df_p = df_le6[(df_le6["race_date"] >= d0) & (df_le6["race_date"] <= d1)].copy()
        print(f"\n  {period_name}: {df_p['race_key'].nunique():,} races ...", flush=True)
        res = evaluate_period_by_quartile(df_p, df_race, [q1, q2, q3])
        all_results[period_name] = res

    # ── Step 7: 結果表示 ──────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  W01 波乱モデル × リーク無し再評価（doc18セマンティクス）")
    print(f"  エントリーモデル: TRAIN期間のみ学習  /  波乱モデル: TRAIN期間のみ学習")
    print(f"  戦略: SS=3連単 / S/A=3連複 × pivot1-pivot2-thirds × ガミ≥5倍のみ")
    print(f"{'='*110}")
    print(f"\n  波乱モデルAUC:  TRAIN={aucs['TRAIN']:.4f}  VAL={aucs['VAL']:.4f}  HOLD={aucs['HOLD']:.4f}")
    print(f"  波乱率(trio≥2000): {upset_rate:.1%}")
    print(f"  TRAIN四分位閾値: Q1≤{q1:.3f} Q2≤{q2:.3f} Q3≤{q3:.3f}")
    print(f"\n{'='*110}")

    print(f"\n  {'四分位':<5} {'':^55} {'':^55} {'':^55}")
    print(f"  {'':5} {'TRAIN':^55} {'VAL':^55} {'HOLD':^55}")
    header2 = "     R     ROI   [CI_lo  CI_hi]  除最大  判定"
    print(f"  {'':5} {header2}  {header2}  {header2}")
    print(f"  {'-'*168}")

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row_parts = []
        for period_name in ["TRAIN", "VAL", "HOLD"]:
            res = all_results[period_name]
            q_res = next((r for r in res if r["quartile"] == q), None)
            if q_res is None or q_res["n_races"] == 0:
                s = {"n": 0, "roi": 0, "ci_lo": 0, "ci_hi": 0,
                     "roi_ex_max": 0, "hits": 0, "hit_rate": 0}
            else:
                s = roi_summary(q_res["pays"], q_res["bets"])
            row_parts.append(fmt_row(s))
        print(f"  {q:<5} {row_parts[0]}  {row_parts[1]}  {row_parts[2]}")

    print(f"\n{'='*110}")

    # 合格セルチェック（VAL & HOLD 両方でCI下限>100%）
    pass_cells = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        val_res = next((r for r in all_results["VAL"] if r["quartile"] == q), None)
        hold_res = next((r for r in all_results["HOLD"] if r["quartile"] == q), None)
        if val_res and hold_res and val_res["n_races"] >= 30 and hold_res["n_races"] >= 10:
            vs = roi_summary(val_res["pays"], val_res["bets"])
            hs = roi_summary(hold_res["pays"], hold_res["bets"])
            if vs["ci_lo"] > 1.0 and hs["ci_lo"] > 1.0:
                pass_cells.append(q)

    print(f"\n  合格セル (VAL & HOLD CI下限>100%): {pass_cells or '0件（不通過）'}")
    print(f"\n  判定ルール: VAL n≥30 & HOLD n≥10 & 両期間 bootstrap CI下限>100%")
    print(f"  払戻=最終オッズ(上限値)。実運用はドリフトで下振れ前提。")
    print(f"{'='*110}")

    # doc02との比較（注記）
    print(f"\n  【doc02との比較】")
    print(f"  doc02の波乱Q4 ROI 598%/627% はリーク+欠車生存バイアス+完走者ランキング混在。")
    print(f"  本実験: TRAIN限定モデル×全エントリーランキング×出走表≤6車基準で再評価。")

    # 詳細集計
    print(f"\n{'='*110}")
    print(f"  詳細: 四分位 × 期間 × 的中数")
    print(f"  {'Q':<3} {'期間':<7} {'R数':>5} {'的中':>4} {'的中率':>7} {'ROI':>7} {'CI下限':>7} {'CI上限':>7} {'除最大':>7}")
    print(f"  {'-'*60}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        for period_name in ["TRAIN", "VAL", "HOLD"]:
            res = all_results[period_name]
            q_res = next((r for r in res if r["quartile"] == q), None)
            if q_res is None or q_res["n_races"] == 0:
                print(f"  {q:<3} {period_name:<7} {'0':>5}")
                continue
            s = roi_summary(q_res["pays"], q_res["bets"])
            print(f"  {q:<3} {period_name:<7} {s['n']:>5} {s['hits']:>4} "
                  f"{s['hit_rate']:>7.1%} {s['roi']:>7.1%} "
                  f"{s['ci_lo']:>7.1%} {s['ci_hi']:>7.1%} {s['roi_ex_max']:>7.1%}")

    print(f"{'='*110}")
    print("完了", flush=True)

    # 結果を返す（レポート生成用）
    return aucs, all_results, q1, q2, q3, upset_rate, pass_cells


if __name__ == "__main__":
    main()
