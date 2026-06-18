"""ライン先頭強度・ライン内得点差 特徴量検証

仮説:
  現行モデルは race_point（個人）と line_pos/is_line_leader を個別に持つが、
  「ライン先頭同士の得点比較」と「ライン内得点差（ちぎれリスク）」は
  明示的に特徴量化されていない。これらが AUC を改善するか検証する。

新規特徴量（3本）:
  leader_rp           : 自ラインの先頭選手の race_point（番手/3番手にも付与）
  leader_rp_gap_vs_best: 自ライン先頭 vs レース内最強ライン先頭の得点差（負=劣位）
  within_line_rp_gap  : ライン内の最大得点差（先頭が番手より弱い場合の「ちぎれリスク」）
                        ※ 単騎（line_size=1）は 0

検証設計（事前登録・Phase1 AUC 検定のみ）:
  Phase1: TRAIN 2023-07〜2025-06 のみで学習したリーク無し LGBM
          → VAL/HOLD の AUC 差を計測
          不通過基準: AUC 差 < ±0.001（両期間とも）→ 無情報と判定しクローズ
  Phase2: AUC 差 ≥ 0.001 の期間が 1 つ以上あった場合のみ ROI 検定（C0 tier×3期間）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-14
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from exp_segment_first_wt import LGB_PARAMS, TRAIN, VAL, HOLD

AUC_THRESHOLD = 0.001

LINE_FEATURE_COLS = [
    "leader_rp",
    "leader_rp_gap_vs_best",
    "within_line_rp_gap",
]


def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """ライン先頭強度・ライン内得点差を付与する。

    入力 df は build_features_wt 適用後（race_key / line_group /
    is_line_leader / race_point / line_size 列が必要）。
    """
    df = df.copy()

    # 1. 先頭の race_point をライン全員に付与（複数 leader フラグがある場合は最大を採用）
    leaders = (
        df[df["is_line_leader"] == 1]
        .groupby(["race_key", "line_group"])["race_point"]
        .max()
        .reset_index()
        .rename(columns={"race_point": "leader_rp"})
    )
    df = df.merge(leaders, on=["race_key", "line_group"], how="left")
    # 単騎（line_size=1）は自分が先頭 → そのまま race_point を使用
    mask_solo = df["line_size"] == 1
    df.loc[mask_solo, "leader_rp"] = df.loc[mask_solo, "race_point"]
    df["leader_rp"] = df["leader_rp"].fillna(df["race_point"])

    # 2. レース内最強ライン先頭の race_point を全員に付与（leader_rp が確定してから計算）
    best_leader = (
        df[df["is_line_leader"] == 1]
        .groupby("race_key")["leader_rp"]
        .max()
        .reset_index()
        .rename(columns={"leader_rp": "best_leader_rp"})
    )
    df = df.merge(best_leader, on="race_key", how="left")
    df["best_leader_rp"] = df["best_leader_rp"].fillna(df["race_point"])

    # 3. ライン先頭 vs 最強先頭の得点差（負 = 自ラインが劣位）
    df["leader_rp_gap_vs_best"] = df["leader_rp"] - df["best_leader_rp"]

    # 4. ライン内の最大得点差（番手の中で自ライン先頭より強い選手の最大超過）
    #    正値 = 番手が先頭より得点が高い（先頭が「使われる」リスク）
    #    leader_rp より自 race_point が高い場合（番手・3番手）の差を表す
    df["within_line_rp_gap"] = (df["race_point"] - df["leader_rp"]).clip(lower=0)
    # 先頭自身は 0（自分 vs 自分）
    df.loc[df["is_line_leader"] == 1, "within_line_rp_gap"] = 0.0
    # ライン全員に「ライン内最大ちぎれリスク」を付与（ライン単位の集約）
    max_gap = (
        df.groupby(["race_key", "line_group"])["within_line_rp_gap"]
        .max()
        .reset_index()
        .rename(columns={"within_line_rp_gap": "within_line_rp_gap_max"})
    )
    df = df.merge(max_gap, on=["race_key", "line_group"], how="left")
    df["within_line_rp_gap"] = df["within_line_rp_gap_max"].fillna(0.0)
    df.drop(columns=["within_line_rp_gap_max", "best_leader_rp"], inplace=True)

    return df


def phase1(df_all: pd.DataFrame) -> bool:
    """Phase1: AUC/logloss 差でライン特徴量の情報量を検定する。

    Returns True if Phase2 should proceed.
    """
    print("\n" + "=" * 60)
    print("Phase1: ライン特徴量 AUC 検定（リーク無し LGBM）")
    print("=" * 60)

    # 1. ライン特徴量を付与
    df_line = add_line_features(df_all)

    # 2. TRAIN期間のみで2モデルを学習
    train_mask = (df_all["race_date"] >= TRAIN[0]) & (df_all["race_date"] <= TRAIN[1])
    fit_base = df_all[train_mask & (df_all["finish_order"] >= 1)]
    fit_line = df_line[train_mask & (df_line["finish_order"] >= 1)]

    X_tr_base = prepare_X(fit_base)
    X_tr_line = pd.concat([
        prepare_X(fit_line).reset_index(drop=True),
        fit_line[LINE_FEATURE_COLS].reset_index(drop=True)
    ], axis=1)
    y_tr_base = fit_base["top3_flag"].values
    y_tr_line = fit_line["top3_flag"].reset_index(drop=True).values

    assert len(X_tr_base) == len(y_tr_base), f"base 行数不一致: X={len(X_tr_base)} y={len(y_tr_base)}"
    assert len(X_tr_line) == len(y_tr_line), f"line 行数不一致: X={len(X_tr_line)} y={len(y_tr_line)}"

    print(f"  TRAIN: base={len(fit_base):,}行 / line={len(fit_line):,}行 / 特徴量: base={X_tr_base.shape[1]} / line={X_tr_line.shape[1]}")

    model_base = lgb.LGBMClassifier(**LGB_PARAMS)
    model_base.fit(X_tr_base, y_tr_base)

    model_line = lgb.LGBMClassifier(**LGB_PARAMS)
    model_line.fit(X_tr_line, y_tr_line)

    # 3. VAL / HOLD で評価
    results = {}
    pass1 = False
    for name, lo, hi in [("VAL", TRAIN[1], VAL[1]), ("HOLD", VAL[1], HOLD[1])]:
        mask = (
            (df_all["race_date"] > lo) & (df_all["race_date"] <= hi)
            & (df_all["finish_order"] >= 1)
        )
        sub_base = df_all[mask]
        sub_line = df_line[mask]

        X_base = prepare_X(sub_base)
        X_line = pd.concat([
            prepare_X(sub_line).reset_index(drop=True),
            sub_line[LINE_FEATURE_COLS].reset_index(drop=True)
        ], axis=1)
        y = sub_base["top3_flag"].reset_index(drop=True).values

        p_base = model_base.predict_proba(X_base)[:, 1]
        p_line = model_line.predict_proba(X_line)[:, 1]

        auc_base  = roc_auc_score(y, p_base)
        auc_line  = roc_auc_score(y, p_line)
        ll_base   = log_loss(y, p_base)
        ll_line   = log_loss(y, p_line)
        auc_diff  = auc_line - auc_base
        ll_diff   = ll_line  - ll_base

        results[name] = {"auc_base": auc_base, "auc_line": auc_line,
                         "auc_diff": auc_diff, "ll_diff": ll_diff}

        flag = (
            f"★ AUC差 ≥ ±{AUC_THRESHOLD} → Phase2候補"
            if abs(auc_diff) >= AUC_THRESHOLD else
            f"AUC差 < ±{AUC_THRESHOLD} → 無情報"
        )
        print(f"\n  [{name}] n={len(y):,}")
        print(f"    AUC  base={auc_base:.4f}  line={auc_line:.4f}  diff={auc_diff:+.4f}")
        print(f"    LogL base={ll_base:.4f}  line={ll_line:.4f}  diff={ll_diff:+.4f}")
        print(f"    → {flag}")

        if abs(auc_diff) >= AUC_THRESHOLD:
            pass1 = True

    print()
    if pass1:
        print("【Phase1 通過】AUC 差が閾値を超えた期間あり → Phase2 へ進む")
    else:
        print("【Phase1 不通過】全期間で AUC 差 < ±0.001 → 無情報・クローズ")
    return pass1


def phase2(df_all: pd.DataFrame):
    """Phase2: ROI 検定（C0 tier × 3期間）。

    doc18 セマンティクス: 全エントリーランキング・出走表基準≤6車・欠車void・最終オッズ上限値。
    exp_leakfree_rescore_wt.py と同じパターンで自己完結実装。
    """
    print("\n" + "=" * 60)
    print("Phase2: ROI 検定（C0 tier × 3期間・ライン特徴量モデル）")
    print("=" * 60)

    from src.database import get_connection
    from src.evaluation.backtest_wt import _assign_tier

    df_line = add_line_features(df_all)

    # TRAIN期間のみでライン特徴量モデルを学習（リーク無し）
    train_mask = (df_line["race_date"] >= TRAIN[0]) & (df_line["race_date"] <= TRAIN[1])
    fit = df_line[train_mask & (df_line["finish_order"] >= 1)]
    X_tr = pd.concat([
        prepare_X(fit).reset_index(drop=True),
        fit[LINE_FEATURE_COLS].reset_index(drop=True)
    ], axis=1)
    y_tr = fit["top3_flag"].reset_index(drop=True).values
    assert len(X_tr) == len(y_tr)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)
    print(f"  モデル学習完了（TRAIN {len(fit):,}行）")

    # 全行に pred_prob 付与
    X_all_line = pd.concat([
        prepare_X(df_line).reset_index(drop=True),
        df_line[LINE_FEATURE_COLS].reset_index(drop=True)
    ], axis=1)
    df_line = df_line.copy().reset_index(drop=True)
    df_line["pred_prob"] = model.predict_proba(X_all_line)[:, 1]

    # オッズ読み込み（trio）
    with get_connection() as conn:
        odds_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'",
            conn
        )

    # trio 結果（着順 1-3）
    result_df = (
        df_line[df_line["finish_order"].between(1, 3)]
        .sort_values(["race_key", "finish_order"])
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .reset_index()
        .rename(columns={"frame_no": "result_key"})
    )

    # オッズ辞書: race_key → {frozenset → pay}
    def _parse_combo(s):
        import re
        parts = re.split(r"[-=]", str(s))
        try:
            return frozenset(int(p) for p in parts)
        except Exception:
            return None

    odds_df["combo_key"] = odds_df["combination"].apply(_parse_combo)
    odds_df = odds_df.dropna(subset=["combo_key"])
    odds_map = {}
    for row in odds_df.itertuples(index=False):
        odds_map.setdefault(row.race_key, {})[row.combo_key] = row.odds_value * 100

    # レース単位で tier 判定・ROI 計算
    period_labels = [
        ("TRAIN", TRAIN[0],  TRAIN[1]),
        ("VAL",   TRAIN[1],  VAL[1]),
        ("HOLD",  VAL[1],    HOLD[1]),
    ]

    # 実際の trio 結果辞書: race_key → frozenset(top3 frame_no)
    actual_result = (
        df_line[df_line["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    def _calc_roi_for_model(df_scored, label_prefix=""):
        """指定モデルの pred_prob で tier 判定・ROI を3期間計算。
        S/A: pred1+pred2 を軸に thirds (pred3〜) へ3点流し
        SS: trio frozenset {p1,p2,p3} 1点（三連複として近似）
        """
        rows = []
        for lo, hi in [(TRAIN[0], TRAIN[1]), (TRAIN[1], VAL[1]), (VAL[1], HOLD[1])]:
            sub = df_scored[
                (df_scored["race_date"] > lo) & (df_scored["race_date"] <= hi)
                & (df_scored["finish_order"] >= 0)
            ]
            total_bet = total_pay = hits = bets = n_races = 0
            for race_key, grp in sub.groupby("race_key"):
                if len(grp) > 6:
                    continue
                if race_key not in actual_result:
                    continue
                n_races += 1
                grp_s = grp.sort_values("pred_prob", ascending=False)
                probs = grp_s["pred_prob"].tolist()
                frames = grp_s["frame_no"].astype(int).tolist()
                if len(frames) < 3:
                    continue
                p1_prob, p2_prob = probs[0], probs[1]
                gap12 = p1_prob - p2_prob
                n = len(grp)
                ratio = p1_prob / (3 / n) if n > 0 else 0
                tier = _assign_tier(gap12, ratio)
                if tier not in ("SS", "S", "A"):
                    continue
                p1f, p2f = frames[0], frames[1]
                thirds = frames[2:]  # pred3, pred4, pred5...
                race_odds = odds_map.get(race_key, {})
                # 2軸流し: {p1,p2,third} × len(thirds) 点
                bet_combos = [frozenset([p1f, p2f, t]) for t in thirds]
                if not bet_combos:
                    continue
                # ガミ足切り: 最安レグ（最低 payout）が 5倍（500円）未満
                min_pay = min((race_odds.get(k, 0) for k in bet_combos), default=0)
                if min_pay > 0 and min_pay / 100 < 5.0:
                    continue
                n_pts = len(bet_combos)
                actual = actual_result[race_key]
                hit_pay = race_odds.get(actual, 0) if actual in bet_combos else 0
                bets += 1
                total_bet += n_pts * 100
                total_pay += hit_pay
                if hit_pay > 0:
                    hits += 1
            roi = total_pay / total_bet * 100 if total_bet > 0 else 0
            rows.append((n_races, bets, hits, roi))
        return rows

    # ベースモデル（line特徴量なし）を学習して比較用 pred_prob を付与
    fit_base = df_all[
        (df_all["race_date"] >= TRAIN[0]) & (df_all["race_date"] <= TRAIN[1])
        & (df_all["finish_order"] >= 1)
    ]
    X_tr_base = prepare_X(fit_base)
    y_tr_base = fit_base["top3_flag"].values
    model_base = lgb.LGBMClassifier(**LGB_PARAMS)
    model_base.fit(X_tr_base, y_tr_base)

    df_base_scored = df_all.copy().reset_index(drop=True)
    df_base_scored["pred_prob"] = model_base.predict_proba(
        prepare_X(df_base_scored)
    )[:, 1]

    # ベースモデルでも ROI を計算（比較用）
    print("\n  ベース（line特徴量なし）vs 新モデル（line特徴量あり）の ROI 比較:")
    print(f"  {'期間':<8} {'R':>6} {'対象':>6} {'的中':>5}  base ROI  line ROI")
    print("  " + "-" * 55)

    rows_base = _calc_roi_for_model(df_base_scored)
    rows_line = _calc_roi_for_model(df_line)

    period_names = ["TRAIN", "VAL", "HOLD"]
    all_pass = True
    for i, name in enumerate(period_names):
        nr, b, h, roi_b = rows_base[i]
        _, _, _, roi_l = rows_line[i]
        v_b = "★" if roi_b >= 100 else " "
        v_l = "★" if roi_l >= 100 else " "
        print(f"  {name:<8} {nr:>6,} {b:>6,} {h:>5}  {roi_b:>6.1f}%{v_b}  {roi_l:>6.1f}%{v_l}")
        if roi_l < 100:
            all_pass = False

    print()
    if all_pass:
        print("  【Phase2 通過】全3期間 ROI ≥ 100% → 本番化検討")
    else:
        print("  【Phase2 不通過】ROI < 100% の期間あり → Phase1 通過でも ROI 優位なし")
    print("\n  ⚠️ 最終オッズ上限値・3点流し前提。live実測が採否の唯一の根拠。")

    print("\n  ⚠️ 最終オッズ上限値。live実測（picks_history）が採否の唯一の根拠。")


def describe_features(df_all: pd.DataFrame):
    """新特徴量の分布・相関サマリを出力する。"""
    df_line = add_line_features(df_all)
    df_r = df_line[df_line["finish_order"] >= 1].copy()

    print("\n" + "=" * 60)
    print("新特徴量サマリ（≤6車・2024年以降）")
    print("=" * 60)

    sub = df_r[df_r["n_entries"] <= 6] if "n_entries" in df_r.columns else df_r
    # n_entries が無ければ race_key 経由
    if "n_entries" not in df_r.columns:
        cnt = df_r.groupby("race_key")["frame_no"].transform("count")
        sub = df_r[cnt <= 6]

    print(f"\n  対象: {len(sub):,} 行")
    for col in LINE_FEATURE_COLS:
        if col not in sub.columns:
            continue
        s = sub[col]
        print(f"\n  {col}:")
        print(f"    mean={s.mean():.2f}  std={s.std():.2f}  "
              f"min={s.min():.2f}  max={s.max():.2f}")
        # top3率との相関
        corr = sub[col].corr(sub["top3_flag"])
        print(f"    top3_flag との Pearson r = {corr:+.4f}")

    # leader_rp_gap_vs_best の四分位 × top3率
    print("\n  leader_rp_gap_vs_best（ライン先頭 vs 最強先頭の得点差）× top3率:")
    bins = [-200, -20, -10, -5, 0, 1]
    labels = ["20点以上差", "10〜20点差", "5〜10点差", "0〜5点差", "最強先頭(0差)"]
    sub2 = sub[sub["is_line_leader"] == 1].copy()
    sub2["gap_bin"] = pd.cut(sub2["leader_rp_gap_vs_best"], bins=bins, labels=labels)
    g = sub2.groupby("gap_bin").agg(
        n=("top3_flag", "count"),
        top3=("top3_flag", lambda x: x.mean() * 100)
    ).round(1)
    print(g.to_string())

    print("\n  within_line_rp_gap（ライン内得点差・番手が先頭より高い幅）× 番手のtop3率:")
    fol = sub[sub["is_line_leader"] == 0].copy()
    fol["gap_bin2"] = pd.cut(fol["within_line_rp_gap"],
                              bins=[-1, 0, 5, 10, 200],
                              labels=["先頭優位(0)", "番手が0〜5点上", "5〜10点上", "10点以上上"])
    g2 = fol.groupby("gap_bin2").agg(
        n=("top3_flag", "count"),
        top3=("top3_flag", lambda x: x.mean() * 100)
    ).round(1)
    print(g2.to_string())


def main():
    print("ライン特徴量検証スクリプト")
    print(f"  期間: TRAIN {TRAIN[0]}〜{TRAIN[1]} / VAL {VAL[0]}〜{VAL[1]} / HOLD {VAL[1]}〜{HOLD[1]}")

    print("\nデータ読み込み中...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    print(f"  全行: {len(df_all):,}")

    # 特徴量サマリ
    describe_features(df_all)

    # Phase1
    passed = phase1(df_all)

    # Phase2（Phase1通過時のみ）
    if passed:
        phase2(df_all)
    else:
        print("\n→ Phase2 スキップ（Phase1 不通過）")


if __name__ == "__main__":
    main()
