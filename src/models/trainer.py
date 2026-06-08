"""
モデルの学習・評価・保存
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

from ..preprocessing.feature_engineer import FEATURE_COLS, TARGET_COL

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline(df: pd.DataFrame) -> tuple:
    """ロジスティック回帰ベースラインモデルを学習"""
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    model.fit(X, y)
    return model


def train_lgbm(
    df: pd.DataFrame,
    n_splits: int = 5,
    feature_cols: list[str] | None = None,
    target_col: str | None = None,
    weight_col: str | None = None,
) -> lgb.LGBMClassifier:
    """LightGBMモデルを日付ベース時系列CVで学習（未来漏洩なし）

    weight_col: 指定するとその列を sample_weight として使用。
                頭数バイアス対策（1/n_riders で各レースの寄与を均等化）等に使う。
    """
    fcols = feature_cols if feature_cols is not None else FEATURE_COLS
    tcol  = target_col  if target_col  is not None else TARGET_COL
    subset = fcols + [tcol] + ([weight_col] if weight_col else [])
    df = df.dropna(subset=subset)
    df = df.sort_values("race_date")

    X = df[fcols].values
    y = df[tcol].values
    dates = df["race_date"].values
    w = df[weight_col].values if weight_col else None

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
    }

    # 日付ベース時系列CV: 訓練は常にバリデーションより過去のみ
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)
    # 先頭60%をバーンイン（最低限の訓練期間）とし、残り40%をn_splits等分してroll
    burnin_end = int(n_dates * 0.6)
    val_size   = max(1, (n_dates - burnin_end) // n_splits)

    fold_aucs = []
    oof_preds = np.zeros(len(y))

    for i in range(n_splits):
        val_start_idx = burnin_end + i * val_size
        val_end_idx   = min(val_start_idx + val_size, n_dates)
        if val_start_idx >= n_dates:
            break

        tr_dates  = unique_dates[:val_start_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        tr_mask  = np.isin(dates, tr_dates)
        val_mask = np.isin(dates, val_dates)

        X_tr, y_tr   = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        w_tr = w[tr_mask] if w is not None else None

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(pd.DataFrame(X_val, columns=fcols))[:, 1]
        oof_preds[val_mask] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {i}: train〜{tr_dates[-1]}  val {val_dates[0]}〜{val_dates[-1]}  AUC={auc:.4f}")

    val_covered = np.isin(dates, unique_dates[burnin_end:])
    print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    if val_covered.sum() > 0:
        print(f"OOF AUC: {roc_auc_score(y[val_covered], oof_preds[val_covered]):.4f}")

    # 全データで最終モデルを学習
    df_X = pd.DataFrame(X, columns=fcols)
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(df_X, y, sample_weight=w, callbacks=[lgb.log_evaluation(0)])
    return final_model


def save_model(model, name: str):
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")
    return path


def load_model(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
