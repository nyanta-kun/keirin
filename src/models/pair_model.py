"""
2車複専用ペアモデル（Aランク向け）

エントリー単位の特徴量をペア単位に展開し、
「このペアが1・2着になるか」を予測する二値分類器。

目標: top1=60-85%のレース（人気レース）で 2車複ROI > 120%
"""

import pickle
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from ..database import get_connection
from ..preprocessing.feature_engineer import (
    build_features,
    load_raw_data,
    FEATURE_COLS,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ペアモデルで使う特徴量
# 各選手の個別特徴量 + ペア間の差分特徴量 + レース特徴量
PAIR_FEATURE_COLS = [
    # === 選手A（高得点側）の絶対値特徴量 ===
    "a_racing_score",
    "a_quinella_rate",
    "a_recent_top3_rate_6m",
    "a_score_z",
    "a_is_home",
    "a_line_pos_enc",
    "a_frame_no",
    "a_player_class_enc",
    "a_period_norm",
    "a_days_since_last_race",
    # === 選手B（低得点側）の絶対値特徴量 ===
    "b_racing_score",
    "b_quinella_rate",
    "b_recent_top3_rate_6m",
    "b_score_z",
    "b_is_home",
    "b_line_pos_enc",
    "b_frame_no",
    "b_player_class_enc",
    "b_period_norm",
    "b_days_since_last_race",
    # === ペア間の差分・統合特徴量 ===
    "pair_score_diff",        # racing_score の差（A - B）
    "pair_score_mean",        # racing_score の平均
    "pair_score_max",         # racing_score の最大値
    "pair_score_min",         # racing_score の最小値
    "pair_quinella_diff",     # quinella_rate の差（A - B）
    "pair_quinella_mean",     # quinella_rate の平均
    "pair_top3_rate_diff",    # recent_top3_rate_6m の差
    "pair_top3_rate_mean",    # recent_top3_rate_6m の平均
    "pair_score_z_diff",      # score_z の差（絶対差）
    "pair_line_same",         # 同脚質フラグ（0/1）
    "pair_line_diff",         # 脚質の差（捲りvs先行など）
    "pair_home_count",        # ホーム選手の数（0/1/2）
    "pair_frame_sum",         # 枠番の合計（内枠バイアス）
    "pair_frame_min",         # 内側の枠番
    "pair_inner_count",       # 内枠（1-3）選手の数
    # === レース特徴量（両者共通）===
    "grade_enc",
    "bank_length_enc",
    "is_indoor",
    "n_riders",               # 出走頭数
]


def _load_quinella_payouts(race_keys: list) -> pd.DataFrame:
    """指定レースの2車複配当をDBから取得"""
    placeholders = ",".join(["?" for _ in race_keys])
    query = f"""
        SELECT race_key, combination, payout
        FROM odds
        WHERE bet_type = 'quinella'
          AND race_key IN ({placeholders})
          AND payout IS NOT NULL
          AND payout > 0
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=race_keys)
    return df


def build_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """エントリーDataFrameをペア単位に展開し特徴量を構築

    入力: build_features() の出力（エントリー単位・レース内相対特徴量含む）
    出力: ペア単位 DataFrame（1行 = 1ペア）

    各レースの C(n,2) ペアを全展開し、以下を付与する:
    - 各選手の絶対値特徴量（A=高得点側, B=低得点側）
    - ペア間の差分・統合特徴量
    - レース特徴量（grade, bank 等）
    - is_quinella_hit: 実際に1・2着になったか（結果がある場合のみ）
    """
    df = df.copy()

    # 結果がある場合: is_top2 フラグを付与
    df["is_top2"] = (
        df["finish_position"].notna() & (df["finish_position"] <= 2)
    ).astype(int)

    # 出走頭数
    n_riders_map = df.groupby("race_key")["frame_no"].count().rename("n_riders")
    df = df.join(n_riders_map, on="race_key")

    # ペア展開に使う列
    entry_cols = [
        "race_key", "race_date", "frame_no", "player_id",
        "racing_score", "quinella_rate", "recent_top3_rate_6m",
        "score_z", "is_home", "line_pos_enc", "player_class_enc",
        "period_norm", "days_since_last_race",
        "grade_enc", "bank_length_enc", "is_indoor", "n_riders",
        "is_top2", "finish_position",
    ]
    # 存在しない列を除外
    entry_cols = [c for c in entry_cols if c in df.columns]
    df_slim = df[entry_cols]

    records = []
    for race_key, grp in df_slim.groupby("race_key"):
        grp = grp.reset_index(drop=True)
        riders = grp.to_dict("records")
        n = len(riders)
        if n < 2:
            continue

        race_date = grp["race_date"].iloc[0]
        grade_enc = grp["grade_enc"].iloc[0]
        bank_length_enc = grp["bank_length_enc"].iloc[0]
        is_indoor = grp["is_indoor"].iloc[0]
        n_riders = n

        # has_result: この race に finish_position が存在するか
        has_result = grp["finish_position"].notna().any()

        for i, j in combinations(range(n), 2):
            ri, rj = riders[i], riders[j]

            # 高得点側を A, 低得点側を B
            if ri["racing_score"] >= rj["racing_score"]:
                a, b = ri, rj
            else:
                a, b = rj, ri

            # ペアヒット判定（結果が揃っている場合のみ）
            if has_result:
                is_hit = int(a["is_top2"] == 1 and b["is_top2"] == 1)
            else:
                is_hit = np.nan

            rec = {
                "race_key": race_key,
                "race_date": race_date,
                "frame_a": a["frame_no"],
                "frame_b": b["frame_no"],
                "player_a": a.get("player_id", ""),
                "player_b": b.get("player_id", ""),
                # 選手A特徴量
                "a_racing_score": a["racing_score"],
                "a_quinella_rate": a["quinella_rate"],
                "a_recent_top3_rate_6m": a["recent_top3_rate_6m"],
                "a_score_z": a["score_z"],
                "a_is_home": a["is_home"],
                "a_line_pos_enc": a["line_pos_enc"],
                "a_frame_no": a["frame_no"],
                "a_player_class_enc": a["player_class_enc"],
                "a_period_norm": a["period_norm"],
                "a_days_since_last_race": a["days_since_last_race"],
                # 選手B特徴量
                "b_racing_score": b["racing_score"],
                "b_quinella_rate": b["quinella_rate"],
                "b_recent_top3_rate_6m": b["recent_top3_rate_6m"],
                "b_score_z": b["score_z"],
                "b_is_home": b["is_home"],
                "b_line_pos_enc": b["line_pos_enc"],
                "b_frame_no": b["frame_no"],
                "b_player_class_enc": b["player_class_enc"],
                "b_period_norm": b["period_norm"],
                "b_days_since_last_race": b["days_since_last_race"],
                # ペア差分特徴量
                "pair_score_diff": a["racing_score"] - b["racing_score"],
                "pair_score_mean": (a["racing_score"] + b["racing_score"]) / 2,
                "pair_score_max": a["racing_score"],
                "pair_score_min": b["racing_score"],
                "pair_quinella_diff": a["quinella_rate"] - b["quinella_rate"],
                "pair_quinella_mean": (a["quinella_rate"] + b["quinella_rate"]) / 2,
                "pair_top3_rate_diff": a["recent_top3_rate_6m"] - b["recent_top3_rate_6m"],
                "pair_top3_rate_mean": (a["recent_top3_rate_6m"] + b["recent_top3_rate_6m"]) / 2,
                "pair_score_z_diff": abs(a["score_z"] - b["score_z"]),
                "pair_line_same": int(a["line_pos_enc"] == b["line_pos_enc"] and a["line_pos_enc"] >= 0),
                "pair_line_diff": abs(a["line_pos_enc"] - b["line_pos_enc"]),
                "pair_home_count": a["is_home"] + b["is_home"],
                "pair_frame_sum": a["frame_no"] + b["frame_no"],
                "pair_frame_min": min(a["frame_no"], b["frame_no"]),
                "pair_inner_count": int(a["frame_no"] <= 3) + int(b["frame_no"] <= 3),
                # レース特徴量
                "grade_enc": grade_enc,
                "bank_length_enc": bank_length_enc,
                "is_indoor": is_indoor,
                "n_riders": n_riders,
                # ターゲット
                "is_quinella_hit": is_hit,
            }
            records.append(rec)

    df_pairs = pd.DataFrame(records)
    return df_pairs


def train_pair_model(
    df_pairs: pd.DataFrame,
    n_splits: int = 5,
) -> lgb.LGBMClassifier:
    """日付ベース時系列CV でペアモデルを学習

    注意: 1レースから複数ペアが生成されるため、
    同一レースのペアが訓練・検証に混在しないよう
    race_key ベースで日付を決定する。
    """
    df = df_pairs.dropna(subset=PAIR_FEATURE_COLS + ["is_quinella_hit"]).copy()
    df = df.sort_values("race_date").reset_index(drop=True)

    X = df[PAIR_FEATURE_COLS].values
    y = df["is_quinella_hit"].values
    dates = df["race_date"].values

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        # クラス不均衡対応（1レース21ペアのうち的中は1ペア）
        "scale_pos_weight": 15,
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

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        X_val_df = pd.DataFrame(X_val, columns=PAIR_FEATURE_COLS)
        preds = model.predict_proba(X_val_df)[:, 1]
        oof_preds[val_mask] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(
            f"  Fold {i}: train〜{tr_dates[-1]}  "
            f"val {val_dates[0]}〜{val_dates[-1]}  AUC={auc:.4f}"
        )

    val_covered = np.isin(dates, unique_dates[burnin_end:])
    print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    if val_covered.sum() > 0:
        print(
            f"OOF AUC: {roc_auc_score(y[val_covered], oof_preds[val_covered]):.4f}"
        )

    # 全データで最終モデルを学習
    X_df = pd.DataFrame(X, columns=PAIR_FEATURE_COLS)
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_df, y, callbacks=[lgb.log_evaluation(0)])
    return final_model


def save_pair_model(model: lgb.LGBMClassifier, name: str = "lgbm_pair") -> Path:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")
    return path


def load_pair_model(name: str = "lgbm_pair") -> lgb.LGBMClassifier:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def backtest_pair_model(
    entry_model,
    pair_model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    top1_lo: float = 0.60,
    top1_hi: float = 0.85,
    quinella_payouts: Optional[pd.DataFrame] = None,
) -> dict:
    """ペアモデルのバックテスト

    1. entry_model で各エントリーの pred_prob を計算
    2. 対象レース（top1_lo <= top1 < top1_hi）を抽出
    3. 全ペアの pair_prob を pair_model で計算
    4. pair_prob 最上位ペアに 2車複 1点 100円
    5. 的中率・平均配当・ROI を計算（quinella_payouts が必要）

    Returns:
        dict: {
            'n_races': int, 'n_hits': int, 'hit_rate': float,
            'total_bet': int, 'total_return': int, 'roi': float,
            'avg_payout': float,
            'monthly': pd.DataFrame,
        }
    """
    df = df.copy()

    # --- 1. エントリーモデルで pred_prob を計算 ---
    X = df[FEATURE_COLS]
    df["pred_prob"] = entry_model.predict_proba(X)[:, 1]

    # --- 2. top1 フィルタ ---
    race_top1 = df.groupby("race_key")["pred_prob"].max().rename("top1")
    df = df.join(race_top1, on="race_key")
    target_races = race_top1[(race_top1 >= top1_lo) & (race_top1 < top1_hi)].index
    df_target = df[df["race_key"].isin(target_races)].copy()

    print(
        f"対象レース: {len(target_races)} R  "
        f"(top1: {top1_lo:.0%}-{top1_hi:.0%})"
    )

    # --- 3. ペア展開 + pair_prob 計算 ---
    df_pairs = build_pair_features(df_target)

    # 特徴量にNaNがある場合は補完
    for col in PAIR_FEATURE_COLS:
        if col in df_pairs.columns:
            df_pairs[col] = df_pairs[col].fillna(df_pairs[col].median())

    X_pair = df_pairs[PAIR_FEATURE_COLS]
    df_pairs["pair_prob"] = pair_model.predict_proba(X_pair)[:, 1]

    # --- 4. quinella payout を JOIN ---
    if quinella_payouts is not None:
        df_pay = quinella_payouts.copy()
        # combination を (min_frame)=(max_frame) 形式に正規化
        def normalize_combo(s):
            parts = str(s).split("=")
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                return f"{min(a,b)}={max(a,b)}"
            return s

        df_pay["combination"] = df_pay["combination"].apply(normalize_combo)
    else:
        # DBから取得
        race_keys = df_pairs["race_key"].unique().tolist()
        df_pay = _load_quinella_payouts(race_keys)
        df_pay["combination"] = df_pay["combination"].apply(
            lambda s: "=".join(sorted(s.split("="), key=int))
        )

    # --- 5. 各レースで pair_prob 最上位ペアを選択 ---
    best_pairs = (
        df_pairs.sort_values("pair_prob", ascending=False)
        .groupby("race_key")
        .first()
        .reset_index()
    )

    # 枠番から quinella combo キーを生成
    best_pairs["combo_key"] = best_pairs.apply(
        lambda r: f"{int(min(r['frame_a'], r['frame_b']))}={int(max(r['frame_a'], r['frame_b']))}",
        axis=1,
    )

    # payout を JOIN
    best_pairs = best_pairs.merge(
        df_pay[["race_key", "combination", "payout"]],
        left_on=["race_key", "combo_key"],
        right_on=["race_key", "combination"],
        how="left",
    )

    # is_quinella_hit を JOIN（build_pair_features で付与済みのものを利用）
    hit_map = (
        df_pairs.groupby("race_key")
        .apply(
            lambda g: g.sort_values("pair_prob", ascending=False).iloc[0]["is_quinella_hit"],
            include_groups=False,
        )
        .rename("is_hit")
    )
    best_pairs = best_pairs.join(hit_map, on="race_key")

    # 結果ありレースのみ集計
    valid = best_pairs.dropna(subset=["is_hit"])
    n_races = len(valid)
    n_hits = int(valid["is_hit"].sum())
    total_bet = n_races * 100
    total_return = int(valid.loc[valid["is_hit"] == 1, "payout"].fillna(0).sum())
    hit_rate = n_hits / n_races if n_races > 0 else 0
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    avg_payout = (
        valid.loc[valid["is_hit"] == 1, "payout"].mean()
        if n_hits > 0
        else 0.0
    )

    # 月別集計
    valid = valid.copy()
    valid["ym"] = valid["race_date"].str[:7]
    monthly = (
        valid.groupby("ym")
        .apply(
            lambda g: pd.Series({
                "n_races": len(g),
                "n_hits": int(g["is_hit"].sum()),
                "hit_rate": g["is_hit"].mean() * 100,
                "total_return": int(g.loc[g["is_hit"] == 1, "payout"].fillna(0).sum()),
                "roi": int(g.loc[g["is_hit"] == 1, "payout"].fillna(0).sum()) / (len(g) * 100) * 100,
            }),
            include_groups=False,
        )
        .reset_index()
    )

    return {
        "n_races": n_races,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "total_bet": total_bet,
        "total_return": total_return,
        "roi": roi,
        "avg_payout": avg_payout,
        "monthly": monthly,
        "best_pairs": best_pairs,
    }


def print_backtest_report(result: dict, top1_lo: float, top1_hi: float) -> None:
    """バックテスト結果を表示"""
    print(f"\n{'='*55}")
    print(f"=== ペアモデル バックテスト（top1: {top1_lo:.0%}-{top1_hi:.0%}）===")
    print(f"{'='*55}")
    print(f"対象レース:   {result['n_races']} R")
    print(f"的中率:       {result['hit_rate']*100:.1f}%")
    print(f"平均配当:     {result['avg_payout']:.0f}円")
    print(f"投資総額:     {result['total_bet']:,}円")
    print(f"回収総額:     {result['total_return']:,}円")
    print(f"ROI:          {result['roi']:.1f}%")
    print()
    print("月別安定性:")
    for _, row in result["monthly"].iterrows():
        print(
            f"  {row['ym']}: {int(row['n_races']):4d}R  "
            f"的中率{row['hit_rate']:5.1f}%  "
            f"ROI{row['roi']:6.1f}%"
        )


# ---------------------------------------------------------------------------
# メイン実行ブロック（直接実行時）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Phase 1: データ読み込みと基礎統計")
    print("=" * 60)

    # 学習データ
    print("学習データ読み込み中 (2025-06-01 〜 2026-04-30)...")
    df_raw_tr = load_raw_data(min_date="2025-06-01", max_date="2026-04-30")
    df_tr = build_features(df_raw_tr)
    print(f"  エントリー数: {len(df_tr):,}")
    print(f"  レース数:     {df_tr['race_key'].nunique():,}")

    # テストデータ
    print("テストデータ読み込み中 (2026-05-01 〜)...")
    df_raw_te = load_raw_data(min_date="2026-05-01")
    df_te = build_features(df_raw_te)
    print(f"  エントリー数: {len(df_te):,}")
    print(f"  レース数:     {df_te['race_key'].nunique():,}")

    # ペア展開サイズの確認
    n_riders = df_tr.groupby("race_key")["frame_no"].count()
    n_pairs = n_riders.map(lambda n: n * (n - 1) // 2)
    print(f"\n学習データのペア数:  {n_pairs.sum():,}")
    print(f"出走頭数分布:")
    print(n_riders.value_counts().sort_index().to_string())

    print("\n" + "=" * 60)
    print("Phase 2: ペア特徴量展開")
    print("=" * 60)
    print("ペア特徴量を構築中...")
    df_pairs_tr = build_pair_features(df_tr)
    print(f"  学習ペア数: {len(df_pairs_tr):,}")
    print(f"  的中ペア数: {int(df_pairs_tr['is_quinella_hit'].sum()):,}")
    print(f"  的中率:     {df_pairs_tr['is_quinella_hit'].mean()*100:.2f}%")

    print("\nペア特徴量のNaN確認:")
    nan_counts = df_pairs_tr[PAIR_FEATURE_COLS].isna().sum()
    if nan_counts.sum() > 0:
        print(nan_counts[nan_counts > 0].to_string())
    else:
        print("  NaNなし")

    print("\n" + "=" * 60)
    print("Phase 3: ペアモデル学習")
    print("=" * 60)
    pair_model = train_pair_model(df_pairs_tr, n_splits=5)
    pair_model_path = save_pair_model(pair_model, name="lgbm_pair")

    # 特徴量重要度表示
    fi = pd.Series(
        pair_model.feature_importances_,
        index=PAIR_FEATURE_COLS,
    ).sort_values(ascending=False)
    print("\n特徴量重要度 TOP 15:")
    print(fi.head(15).to_string())

    print("\n" + "=" * 60)
    print("Phase 4: バックテスト（学習データ内 OOF 評価）")
    print("=" * 60)

    from src.models.trainer import load_model as load_entry_model

    entry_model = load_entry_model("lgbm_v4")
    print("エントリーモデル: lgbm_v4")

    # バックテスト: 学習期間後半（2025-12-01以降）
    print("\n--- 学習期間後半（2025-12-01〜2026-04-30）---")
    df_raw_bt = load_raw_data(min_date="2025-12-01", max_date="2026-04-30")
    df_bt = build_features(df_raw_bt)
    result_bt = backtest_pair_model(
        entry_model, pair_model, df_bt,
        top1_lo=0.60, top1_hi=0.85,
    )
    print_backtest_report(result_bt, 0.60, 0.85)

    print("\n" + "=" * 60)
    print("Phase 5: テストデータ バックテスト（2026-05-01〜）")
    print("=" * 60)
    if len(df_te) > 0 and df_te["race_key"].nunique() > 10:
        result_te = backtest_pair_model(
            entry_model, pair_model, df_te,
            top1_lo=0.60, top1_hi=0.85,
        )
        print_backtest_report(result_te, 0.60, 0.85)
    else:
        print("テストデータが少ないためスキップ")

    print("\n" + "=" * 60)
    print("追加: 閾値別ROI比較")
    print("=" * 60)
    for lo, hi in [(0.60, 0.70), (0.60, 0.75), (0.60, 0.80), (0.60, 0.85)]:
        r = backtest_pair_model(
            entry_model, pair_model, df_bt,
            top1_lo=lo, top1_hi=hi,
        )
        print(
            f"  top1 {lo:.0%}-{hi:.0%}: {r['n_races']:5d}R  "
            f"的中{r['hit_rate']*100:5.1f}%  "
            f"ROI{r['roi']:6.1f}%"
        )

    print("\n完了。")
