"""prepare_X（M-1: 推論用特徴行列の統一生成）の純粋テスト。"""
import numpy as np
import pandas as pd

from src.preprocessing.feature_wt import prepare_X, FEATURE_COLS_WT


def test_prepare_x_columns_and_order():
    # 余分な列＋一部欠損列＋NaN を含む df
    df = pd.DataFrame({
        "race_point": [50.0, np.nan],
        "gear_ratio": [3.92, 4.00],
        "extra_unused": [1, 2],   # FEATURE_COLS_WT 外 → 落ちる
    })
    X = prepare_X(df)
    # 列は FEATURE_COLS_WT と完全一致・同順
    assert list(X.columns) == FEATURE_COLS_WT
    # 余分列は含まれない
    assert "extra_unused" not in X.columns
    # NaN は 0 補完
    assert not X.isna().any().any()
    assert X["race_point"].tolist() == [50.0, 0.0]
    # 存在しなかった特徴列は 0 で作られる
    missing = [c for c in FEATURE_COLS_WT if c not in ("race_point", "gear_ratio")][0]
    assert (X[missing] == 0).all()


def test_prepare_x_rowcount_preserved():
    df = pd.DataFrame({"race_point": [10.0, 20.0, 30.0]})
    X = prepare_X(df)
    assert len(X) == 3   # dropna しない＝行数保持（予測対象を落とさない）
