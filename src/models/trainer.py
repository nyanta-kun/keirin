"""
モデルの学習・評価・保存
"""
import pickle
import stat
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

from ..preprocessing.feature_engineer import FEATURE_COLS, TARGET_COL
from . import vintage_manifest
from .model_io import atomic_pickle_dump

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


# 凍結vintageモデルの命名規則（四半期q2401等・旧非標準w2/w3・新月次m2401=YYMM等）。
# 2026-07-28にH2H特徴実験で四半期vintageモデル18本が無断上書きされ、honest ROI
# 検証の再現性が失われた事故（[[keirin_s7_foundational_rethink_2026_07_29]]参照）
# の再発防止。このパターンに一致する名前は、一度保存されたら再度 save_model() で
# 上書きしようとするとエラーになる（force=True明示時のみ許可）。
# 注意: 初回実装時にm\d{6}（6桁=YYYYMM想定）としていたが、実際の命名(m2401=
# YYMM=4桁)と食い違い、書き込み保護が発動しないバグがあった（2026-07-29実データで
# 再実行検証中に発覚・修正）。q/m は4桁、旧wのみ桁数不定のため\d+のまま。
# 正規表現の実体は vintage_manifest.py を単一の情報源とする（循環import回避のため
# ここではそちらをエイリアスするだけに留める）。
_VINTAGE_NAME_RE = vintage_manifest.VINTAGE_NAME_RE


def save_model(model, name: str, force: bool = False):
    """モデルをpickleでアトミック保存する（`data/models/{name}.pkl`）。

    `open(path, "wb")` による直接書き込みは異常終了時にファイルを破損状態の
    まま残すため、`model_io.atomic_pickle_dump()` で一時ファイル経由の
    アトミックrenameを行う（D-3）。

    vintage命名規則（`_VINTAGE_NAME_RE`）に一致する名前は凍結保護の対象。
    以下の**いずれか**に該当する場合、`force=True` を明示しない限り
    `FileExistsError` を送出する:
      1. 同名の `.pkl` が既に存在する（従来からの保護）
      2. `.pkl` は存在しないが `vintage_manifest.json` に登録済み
         （＝ `rm` で削除してから再作成しようとした可能性。2026-07-31強化。
         `keirin_s1_abolition_and_gap_heal_fix_2026_07_31` と同型の
         「消してから作り直す」経路を塞ぐ）

    保存後、vintageモデルは chmod 444（読み取り専用化）と
    `vintage_manifest` への登録/更新の両方を行う。
    """
    path = MODEL_DIR / f"{name}.pkl"
    is_vintage = bool(_VINTAGE_NAME_RE.search(name))

    if is_vintage and not force:
        if path.exists():
            raise FileExistsError(
                f"'{name}' は凍結vintageモデル命名規則に一致し、既にファイルが存在します"
                f"（{path}）。honest walk-forward検証の再現性を守るため、"
                f"save_model(..., force=True) を明示しない限り上書きを拒否します。"
                f"意図的な再作成の場合のみ force=True を指定してください。"
            )
        if vintage_manifest.is_registered(name):
            raise FileExistsError(
                f"'{name}' はファイル実体が存在しませんが、凍結vintageモデルとして"
                f"vintage_manifest.json に登録済みです（{vintage_manifest.MANIFEST_PATH}）。"
                f"`rm` 等でファイルを削除してから再作成しようとした可能性があります。"
                f"honest walk-forward検証の再現性を守るため、"
                f"save_model(..., force=True) を明示しない限り保存を拒否します。"
            )

    if is_vintage and path.exists():
        # force=True で上書きする場合: 読み取り専用化されている場合があるため
        # 書き込み可能に戻す（os.replace自体は親ディレクトリの書き込み権限が
        # あれば読み取り専用の置換先も差し替え可能だが、明示的に緩めておく）。
        path.chmod(stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    atomic_pickle_dump(model, path)

    if is_vintage:
        # 保存後に読み取り専用化（ファイルシステムレベルの第二の防御線。
        # save_model()を経由しない直接書き込みからも保護する）。
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        # マニフェストへ登録/更新（rm耐性のある凍結保護の実体。D-4）。
        vintage_manifest.register(name, path)
    print(f"Saved: {path}")
    return path


def load_model(name: str):
    """モデルを読み込む。winticketルートは特徴量セットの一致も検証する。

    【2026-08-03 追加】特徴量**数**が一致しても**中身**が違うモデルは LightGBM が
    素通しするため、無言で誤った予測を返す。実例: 2026-07-31 に
    `ex_spurt_pct`/`ex_thrust_pct` を除去して48→46、2026-08-03 に
    `formation_pos_frac`/`formation_line_rank` を追加して46→48 としたため、
    **旧48特徴のモデル（`lgbm_wt_eval_w2` 等）は列数だけ一致してエラーにならない**。
    列数が違う46特徴のモデルは LightGBM が Fatal で止めてくれるが、旧48は止まらない。

    そのため列名で照合し、不一致なら明示的に落とす。「通ったのに実は別物を見ていた」
    型の事故（本リポジトリで繰り返し踏んでいる）を構造的に防ぐ。

    Raises:
        ValueError: `lgbm_wt` 系モデルの特徴量セットが FEATURE_COLS_WT と異なる場合。
    """
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)

    if name.startswith("lgbm_wt"):
        # 2026-08-06: 穴推奨 RANK_7H1 のバスト予測モデル（lgbm_wt_favbust*）は
        # **レース単位**の別特徴量セット（67列）を使う。選手単位の FEATURE_COLS_WT
        # と照合すると必ず落ちるため、モデル種別ごとに正しい正本と突き合わせる。
        # 照合そのものを外してはいけない（「通ったのに実は別物を見ていた」型の
        # 事故を防ぐのがこの検証の目的）。
        if name.startswith("lgbm_wt_favbust"):
            from ..preprocessing.favbust_features import FAVBUST_FEATURE_COLS
            expected = list(FAVBUST_FEATURE_COLS)
        else:
            from ..preprocessing.feature_wt import FEATURE_COLS_WT
            expected = list(FEATURE_COLS_WT)

        cols = getattr(model, "feature_name_", None)
        if cols is None and hasattr(model, "feature_name"):
            cols = model.feature_name()          # lgb.Booster（train() の戻り値）
        if cols is not None and list(cols) != expected:
            FEATURE_COLS_WT = expected           # 下のメッセージ生成で使う
            missing = [c for c in expected if c not in cols]
            extra = [c for c in cols if c not in expected]
            raise ValueError(
                f"モデル '{name}' の特徴量セットが現在の FEATURE_COLS_WT と一致しません"
                f"（モデル{len(cols)}列 / 現在{len(FEATURE_COLS_WT)}列）。"
                f" 不足={missing or 'なし'} 余分={extra or 'なし'}。"
                " 再学習するか、当時の特徴量定義のコードで実行してください。"
            )
    return model
