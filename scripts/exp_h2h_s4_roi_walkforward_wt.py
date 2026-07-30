"""H2H(対戦表)特徴のS4(S7)戦略ROI効果を、4半期ごとのwalk-forwardで再検証する
（netkeirin未活用データ調査・続き・2026-07-28）。

exp_h2h_s4_roi_wt.py の単一窓検証（2026-04-01〜07-10・3ヶ月強）は選出数がn=31〜33と
小さすぎ、ROI+84pt改善という結果は多重比較ノイズの疑いが強かった（このプロジェクトで
何度も踏んできたパターン）。より長い期間で信頼できる結論を得るため、4半期ごとに
「その時点までのデータだけで学習し直す」honest walk-forwardを5フォールド実施し、
2025-07-01〜2026-07-10（約12.3ヶ月）を1回ずつ評価してn を積み増す。

各フォールド: TRAIN <= fold境界 で学習 → 次の4半期を評価 → 次フォールドはTRAINを
その4半期まで拡張して再学習（=各評価区間は必ずその区間より前のデータのみで学習した
モデルで評価。model-vintage look-aheadなし）。

win_model は各フォールド共通で本番 lgbm_wt_win を使用（baseline/+h2hで共通なので
delta比較としては妥当・軽量化のため）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import lightgbm as lgb

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from exp_h2h_wt import H2H_COLS, compute_h2h
from exp_h2h_s4_roi_wt import build_rows, summarize

FOLDS = [
    ("2025-06-30", "2025-07-01", "2025-09-30"),
    ("2025-09-30", "2025-10-01", "2025-12-31"),
    ("2025-12-31", "2026-01-01", "2026-03-31"),
    ("2026-03-31", "2026-04-01", "2026-06-30"),
    ("2026-06-30", "2026-07-01", "2026-07-10"),
]
PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)
SEED = 42


def main():
    print("データ構築中（全期間・H2H計算込み）...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=FOLDS[-1][2])
    raw = compute_h2h(raw)
    df = build_features_wt(raw)
    df_fit = df[df["finish_order"] >= 1].copy()
    print(f"  全行数={len(df)} / 完走行数={len(df_fit)}")

    win_model = load_model("lgbm_wt_win")

    variants = {
        "baseline": list(FEATURE_COLS_WT),
        "+h2h": list(FEATURE_COLS_WT) + H2H_COLS,
    }
    all_rows = {v: [] for v in variants}

    for tr_to, ev_from, ev_to in FOLDS:
        tr = df_fit[df_fit["race_date"] <= tr_to].copy()
        ev = df[(df["race_date"] >= ev_from) & (df["race_date"] <= ev_to)].copy()
        if ev.empty:
            print(f"  [{ev_from}〜{ev_to}] eval空 → skip")
            continue
        print(f"\n=== fold TRAIN<={tr_to} EVAL {ev_from}〜{ev_to} "
              f"(TRAIN {tr['race_key'].nunique()}R / EVAL {ev['race_key'].nunique()}R) ===")

        with get_connection() as c:
            ne_map = dict(c.execute(
                "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
                (ev_from, ev_to)))
            date_map = dict(c.execute(
                "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
                (ev_from, ev_to)))

        X_win = ev.reindex(columns=FEATURE_COLS_WT).fillna(0)
        ev = ev.copy()
        ev["predwin_x"] = win_model.predict_proba(X_win)[:, 1]
        win_probs = {(r.race_key, int(r.frame_no)): r.predwin_x
                     for r in ev.itertuples(index=False)}

        for vname, cols in variants.items():
            Xtr = tr[cols].fillna(0).values
            ytr = tr[TARGET_COL_WT].values
            m = lgb.LGBMClassifier(**PARAMS, random_state=SEED)
            m.fit(Xtr, ytr)
            ev_v = ev.copy()
            ev_v["predprob_x"] = m.predict_proba(ev_v[cols].fillna(0).values)[:, 1]
            rows = build_rows(ev_v, "predprob_x", win_probs, date_map, ne_map)
            s = summarize(rows)
            all_rows[vname].extend(rows)
            print(f"  {vname:<10} n={s['n']:>4} 的中{s['hits']:>4} ({s['hit_rate']:.1f}%) "
                  f"投資{s['bet']:>8,} 回収{s['ret']:>8,} ROI {s['roi']:.1f}%")

    print("\n================ 通算（walk-forward全5フォールド合算）================")
    print(f"評価期間: {FOLDS[0][1]}〜{FOLDS[-1][2]}")
    for v in variants:
        s = summarize(all_rows[v])
        print(f"{v:<10} 選出R数={s['n']:>4} 的中{s['hits']:>4} ({s['hit_rate']:.1f}%) "
              f"投資{s['bet']:>9,} 回収{s['ret']:>9,} ROI {s['roi']:.1f}%")
    print("\n(フォールド別の内訳は上のループ出力を参照。符号の安定性はそちらで確認すること)")


if __name__ == "__main__":
    main()
