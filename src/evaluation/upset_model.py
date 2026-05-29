"""波乱レース予測モデル

エントリーモデルの確率分布とレース構造特徴量を組み合わせ、
高配当（波乱）が見込めるレースを事前に識別する二値分類器。

【波乱の定義】
  3連複払戻 >= UPSET_THRESHOLD (デフォルト 2000円)

【特徴量】
  - レース構造: grade, bank, 競走得点の分散・変動係数
  - 選手成績のばらつき: 3着内率の標準偏差
  - エントリーモデルの予測分布: entropy, top1確率, 1-2位の差 etc.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from ..database import get_connection

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
UPSET_THRESHOLD_DEFAULT = 2000  # 3連複払戻がこの値以上 → 波乱

UPSET_FEATURE_COLS = [
    # レース条件
    "grade_enc",        # グレード (GP=7..A=1)
    "n_riders",         # 出走頭数
    "bank_length_enc",  # バンク長 / 100
    "is_indoor",        # 屋内フラグ
    # 競走得点の統計（フィールドの拮抗度）
    "score_mean",       # 平均競走得点
    "score_std",        # 標準偏差（小さい = 混戦）
    "score_cv",         # 変動係数（小さい = 実力拮抗）
    "score_range",      # 最大-最小（小さい = 混戦）
    "score_top_gap",    # 1位-2位の得点差
    # 選手成績のばらつき
    "top3r_mean",       # 平均3着内率(6m)
    "top3r_std",        # 3着内率のばらつき（小さい = 実力拮抗）
    # エントリーモデルの予測確率分布
    "pred_top1",        # 最高予測確率（低い = 波乱型）
    "pred_top2",        # 2番目の予測確率
    "pred_gap12",       # top1-top2 の差（小さい = 接戦）
    "pred_gap23",       # top2-top3 の差
    "pred_entropy",     # 確率のShannon entropy（高い = 不確実）
    "pred_top3_sum",    # 上位3頭の確率合計（低い = 波乱型）
]


# ---------------------------------------------------------------------------
# 特徴量構築
# ---------------------------------------------------------------------------

def build_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    エントリーレベルdf（pred_prob計算済み）からレース単位の特徴量を構築。

    Parameters
    ----------
    df : build_features() の出力 + pred_prob カラム
    """
    rows = []
    for race_key, grp in df.groupby("race_key"):
        probs = np.sort(grp["pred_prob"].fillna(0).values)[::-1]
        probs_safe = probs + 1e-9
        probs_norm = probs_safe / probs_safe.sum()

        scores = grp["racing_score"].dropna().values
        scores_sorted = np.sort(scores)

        top3r = grp["recent_top3_rate_6m"].dropna().values if "recent_top3_rate_6m" in grp.columns else np.array([])

        row = {
            "race_key": race_key,
            "race_date": grp["race_date"].iloc[0],
            "venue_code": grp["venue_code"].iloc[0],

            "grade_enc": float(grp["grade_enc"].iloc[0]) if "grade_enc" in grp.columns else np.nan,
            "n_riders": int(len(grp)),
            "bank_length_enc": float(grp["bank_length_enc"].iloc[0]) if "bank_length_enc" in grp.columns else np.nan,
            "is_indoor": float(grp["is_indoor"].iloc[0]) if "is_indoor" in grp.columns else 0.0,

            "score_mean":    float(np.mean(scores)) if len(scores) > 0 else np.nan,
            "score_std":     float(np.std(scores)) if len(scores) > 0 else np.nan,
            "score_cv":      float(np.std(scores) / np.mean(scores)) if len(scores) > 0 and np.mean(scores) > 0 else np.nan,
            "score_range":   float(np.ptp(scores)) if len(scores) > 0 else np.nan,
            "score_top_gap": float(scores_sorted[-1] - scores_sorted[-2]) if len(scores) > 1 else np.nan,

            "top3r_mean": float(np.mean(top3r)) if len(top3r) > 0 else np.nan,
            "top3r_std":  float(np.std(top3r)) if len(top3r) > 0 else np.nan,

            "pred_top1":    float(probs[0]) if len(probs) > 0 else np.nan,
            "pred_top2":    float(probs[1]) if len(probs) > 1 else np.nan,
            "pred_gap12":   float(probs[0] - probs[1]) if len(probs) > 1 else np.nan,
            "pred_gap23":   float(probs[1] - probs[2]) if len(probs) > 2 else np.nan,
            "pred_entropy": float(-np.sum(probs_norm * np.log(probs_norm))),
            "pred_top3_sum": float(np.sum(probs[:3])) if len(probs) >= 3 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_upset_target(df_race: pd.DataFrame,
                     upset_threshold: int = UPSET_THRESHOLD_DEFAULT) -> pd.DataFrame:
    """
    レース単位dfに波乱ラベル is_upset と 3連複払戻を付与。
    3連複払戻 >= upset_threshold → is_upset = 1
    払戻データが存在しないレース → is_upset = NaN（除外）
    """
    race_keys = df_race["race_key"].tolist()
    if not race_keys:
        df_race["trifecta_box_payout"] = np.nan
        df_race["is_upset"] = np.nan
        return df_race

    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        rows = conn.execute(f"""
            SELECT race_key, payout
            FROM odds
            WHERE race_key IN ({placeholders})
              AND bet_type = 'trifecta_box'
              AND payout IS NOT NULL
        """, race_keys).fetchall()

    payout_map: dict[str, int] = {}
    for row in rows:
        rk = row["race_key"]
        if rk not in payout_map or row["payout"] > payout_map[rk]:
            payout_map[rk] = row["payout"]

    df = df_race.copy()
    df["trifecta_box_payout"] = df["race_key"].map(payout_map)
    df["is_upset"] = np.where(
        df["trifecta_box_payout"].isna(), np.nan,
        (df["trifecta_box_payout"] >= upset_threshold).astype(float),
    )
    return df


# ---------------------------------------------------------------------------
# 学習・推論
# ---------------------------------------------------------------------------

def train_upset_model(df_race: pd.DataFrame,
                      n_splits: int = 5) -> lgb.LGBMClassifier:
    """
    レース単位の波乱予測モデルを日付ベース時系列CVで学習。

    前提: df_race は add_upset_target() 済み（is_upset カラムあり）
    """
    df = df_race.dropna(subset=UPSET_FEATURE_COLS + ["is_upset"]).copy()
    df = df[df["is_upset"].isin([0.0, 1.0])].copy()
    df = df.sort_values("race_date")

    X = df[UPSET_FEATURE_COLS].values
    y = df["is_upset"].values.astype(int)
    dates = df["race_date"].values

    upset_rate = y.mean()
    scale_pos_weight = (1 - upset_rate) / upset_rate
    print(f"学習データ: {len(df):,} レース  波乱率: {upset_rate:.1%}  "
          f"scale_pos_weight={scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbose": -1,
    }

    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)
    burnin_end = int(n_dates * 0.6)
    val_size = max(1, (n_dates - burnin_end) // n_splits)

    fold_aucs = []
    oof_preds = np.zeros(len(y))

    for i in range(n_splits):
        val_start_idx = burnin_end + i * val_size
        val_end_idx = min(val_start_idx + val_size, n_dates)
        if val_start_idx >= n_dates:
            break

        tr_dates = unique_dates[:val_start_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        tr_mask = np.isin(dates, tr_dates)
        val_mask = np.isin(dates, val_dates)

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = m.predict_proba(pd.DataFrame(X_val, columns=UPSET_FEATURE_COLS))[:, 1]
        oof_preds[val_mask] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {i}: 〜{tr_dates[-1]} | val {val_dates[0]}〜{val_dates[-1]} "
              f"| AUC={auc:.4f}")

    val_covered = np.isin(dates, unique_dates[burnin_end:])
    print(f"波乱モデル CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    if val_covered.sum() > 0:
        print(f"OOF AUC (val期間): "
              f"{roc_auc_score(y[val_covered], oof_preds[val_covered]):.4f}")

    # 全データで最終学習
    final = lgb.LGBMClassifier(**params)
    final.fit(pd.DataFrame(X, columns=UPSET_FEATURE_COLS), y,
              callbacks=[lgb.log_evaluation(0)])
    return final


def predict_upset_prob(upset_model: lgb.LGBMClassifier,
                       df_race: pd.DataFrame) -> pd.Series:
    """レースごとの波乱確率を返す（race_key をインデックスとするSeries）"""
    valid = df_race.dropna(subset=UPSET_FEATURE_COLS)
    X = valid[UPSET_FEATURE_COLS].values
    probs = upset_model.predict_proba(pd.DataFrame(X, columns=UPSET_FEATURE_COLS))[:, 1]
    result = pd.Series(np.nan, index=df_race.index)
    result.loc[valid.index] = probs
    return result.values


def save_upset_model(model, name: str = "lgbm_upset") -> Path:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")
    return path


def load_upset_model(name: str = "lgbm_upset") -> lgb.LGBMClassifier:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 閾値別バックテスト
# ---------------------------------------------------------------------------

def run_upset_threshold_analysis(
    entry_model, upset_model,
    df: pd.DataFrame,
    upset_thresholds: list[float] | None = None,
    strategy_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    波乱モデルの upset_prob 閾値を変えながらバックテストを比較。

    Parameters
    ----------
    upset_thresholds : 波乱確率の閾値リスト（この値以上のレースを対象）
    strategy_names   : バックテストする戦略名リスト

    Returns
    -------
    dict { label → df_result }
    """
    from .backtest import (
        _apply_pred_prob, run_backtest,
        QUINELLA_STRATEGIES, EXACTA_STRATEGIES, WIDE_STRATEGIES, ALL_STRATEGIES,
    )

    if upset_thresholds is None:
        upset_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    if strategy_names is None:
        strategy_names = ["quinella_23", "exacta_21", "wide_23", "box_top3"]
    strategies = [s for s in ALL_STRATEGIES if s.name in strategy_names]

    # エントリーモデルで pred_prob を付与
    df_prob = _apply_pred_prob(entry_model, df)
    df_eval = df_prob[df_prob["finish_position"].notna()].copy()

    # レース単位特徴量と upset_prob を計算
    df_race = build_race_features(df_prob)
    df_race["upset_prob"] = predict_upset_prob(upset_model, df_race)

    results: dict[str, pd.DataFrame] = {}

    # 比較ベースライン: top1_prob フィルター
    for top1_th in [0.60, 0.65]:
        label = f"top1<{top1_th:.0%}"
        top1_map = df_prob.groupby("race_key")["pred_prob"].max()
        valid_keys = top1_map[top1_map <= top1_th].index
        df_t = df_eval[df_eval["race_key"].isin(valid_keys)]
        if not df_t.empty:
            results[label] = run_backtest(entry_model, df_t, strategies=strategies,
                                          max_top1_prob=None)
            results[label]["フィルター対象R"] = df_t["race_key"].nunique()

    # 波乱モデルフィルター
    for th in upset_thresholds:
        label = f"upset>{th:.0%}"
        valid_keys = df_race[df_race["upset_prob"] >= th]["race_key"].tolist()
        df_t = df_eval[df_eval["race_key"].isin(valid_keys)]
        if not df_t.empty:
            results[label] = run_backtest(entry_model, df_t, strategies=strategies,
                                          max_top1_prob=None)
            results[label]["フィルター対象R"] = df_t["race_key"].nunique()

    return results


def print_upset_analysis(results: dict[str, pd.DataFrame],
                         strategy_names: list[str] | None = None):
    """フィルター別バックテスト比較を表示"""
    if not results:
        print("データなし")
        return

    first_df = next(iter(results.values()))
    if strategy_names is None:
        strategy_names = first_df["戦略名"].tolist()

    short = {
        "quinella_23": "2車複23(1pt)",
        "exacta_21":   "2車単21(1pt)",
        "wide_23":     "W23(1pt)",
        "box_top3":    "3連複3車(1pt)",
    }

    col_w = 12
    header = f"{'フィルター':<15} {'対象R':>6}"
    for sn in strategy_names:
        lbl = short.get(sn, sn[:8])
        header += f"  {lbl:>{col_w}}"
    print(f"\n{'='*90}")
    print(" 波乱フィルター × 戦略 ROI比較")
    print(f"{'='*90}")

    sub_header = f"{'':15} {'':>6}"
    for _ in strategy_names:
        sub_header += f"  {'的中率  回収率':>{col_w}}"
    print(header)
    print("-" * 90)

    for label, df_r in results.items():
        n_races = int(df_r["フィルター対象R"].iloc[0]) if "フィルター対象R" in df_r.columns else 0
        row_str = f"{label:<15} {n_races:>6,}"
        for sn in strategy_names:
            match = df_r[df_r["戦略名"] == sn]
            if match.empty:
                row_str += f"  {'N/A':>{col_w}}"
            else:
                r = match.iloc[0]
                hit_rate = r["的中率"]
                roi = r["回収率"]
                roi_raw = r["回収率_raw"]
                marker = "*" if roi_raw >= 3.0 else ("+" if roi_raw >= 1.5 else " ")
                cell = f"{hit_rate} {roi}{marker}"
                row_str += f"  {cell:>{col_w}}"
        print(row_str)

    print("=" * 90)
    print("  * = ROI 300%以上  + = ROI 150%以上")


def print_upset_feature_importance(upset_model: lgb.LGBMClassifier):
    """波乱モデルの特徴量重要度を表示"""
    importance = pd.Series(
        upset_model.feature_importances_,
        index=UPSET_FEATURE_COLS,
    ).sort_values(ascending=False)

    print(f"\n{'='*50}")
    print(" 波乱モデル 特徴量重要度 (gain)")
    print(f"{'='*50}")
    total = importance.sum()
    for feat, imp in importance.items():
        bar = "█" * int(imp / total * 40)
        print(f"  {feat:<20} {imp/total:>5.1%}  {bar}")
    print("=" * 50)
