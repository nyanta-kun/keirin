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


def test_leaky_ex_features_excluded_from_feature_cols_wt():
    """train/serve skew（開催中に値が更新される）が実測された ex_spurt_pct /
    ex_thrust_pct は 2026-07-31 に FEATURE_COLS_WT から除外された（48→46特徴）。

    A/B測定（`scripts/exp_ab_leaky_ex_features.py`・12ヶ月・約194,000サンプル）で
    AUC寄与が事実上ゼロ（eval 0.7732→0.7731 / win 0.8233→0.8233）と確認された上での
    リスク回避判断であり、性能改善が目的ではない。H2H特徴（`add_h2h_features_wt`）が
    一度 FEATURE_COLS_WT に採用→ROI悪化で撤回された前例があるため、将来のアドホック
    実験・特徴追加作業でこの2特徴が誤って再混入しないことを保証する回帰テスト。

    SELECT・0-1正規化自体（`load_raw_data_wt`/`build_features_wt`）は分析用途・将来の
    point-in-time化のため引き続き残っており、このテストは FEATURE_COLS_WT への
    不採用のみを検証する（カラム自体の削除ではない）。
    """
    assert "ex_spurt_pct" not in FEATURE_COLS_WT
    assert "ex_thrust_pct" not in FEATURE_COLS_WT
    # 本数の変遷: 48→46（本除外）→48（隊列推定位置・2026-08-03）
    #             →60（race_type 7 + ライン実力 5・2026-08-04）
    assert len(FEATURE_COLS_WT) == 60
    # 同時に検証したレース単位集約は「AUCは上がるが1位3着内率が窓で符号反転」
    # のため不採用。誤って再混入しないことを保証する（ex_* と同型の回帰テスト）。
    for c in ("rp_mean", "rp_std", "rp_gap_top2", "rp_gap_top_self"):
        assert c not in FEATURE_COLS_WT
