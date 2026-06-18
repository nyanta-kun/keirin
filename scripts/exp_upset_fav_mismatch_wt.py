"""波乱スコア × fav_mismatch 交差分析（doc18セマンティクス・リーク無し）

目的:
  upset_prob (TRAIN期間学習モデル) と fav_mismatch (市場本命 vs pivot1) の
  交差条件が単独シグナルより強いエッジを持つか評価する。

4セル定義:
  Cell A: upset Q4 (上位25%) + fav_mismatch=True   ← メイン検証
  Cell B: upset Q4                + fav_mismatch=False
  Cell C: upset Q1-Q3             + fav_mismatch=True
  Cell D: upset Q1-Q3             + fav_mismatch=False  ← ベースライン

各セルで現行戦略 (SS=3連単3点, S/A=3連複3点, ガミ≥5倍) を適用。

doc18セマンティクス:
  - ランキングは全エントリー（欠車含む）で行う
  - ≤6車判定は出走表基準（frame_no の行数）
  - 欠車: 軸欠車→レース無効, 相手欠車→その目のみ除外
  - モデルは TRAIN期間のみで学習 (2023-07〜2025-06)
  - 払戻: wt_odds の最終オッズ（上限値）

実行:
  cd /Users/ysuzuki/GitHub/keirin
  python3 scripts/exp_upset_fav_mismatch_wt.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats as scipy_stats

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.evaluation.upset_model import (
    UPSET_FEATURE_COLS, train_upset_model,
)
from src.evaluation.backtest_wt import _assign_tier
from exp_segment_first_wt import load_boards, market_fav, LGB_PARAMS, TRAIN, VAL, HOLD
from roi_robustness_wt import roi_summary

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

# HOLD期間を本日まで延長
HOLD = ("2026-03-01", "2026-06-14")


def _fmt(s, n):
    if n == 0:
        return f"{'0':>4}R  --"
    return (f"{n:>4}R  ROI={s['roi']:>5.0%}  "
            f"的中={s['hit_rate']:>4.0%}  "
            f"CI=[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]  "
            f"除最大={s['roi_ex_max']:>4.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# wt 用 波乱特徴量構築（upset_model.build_race_features の wt 版）
# wt カラム対応: racing_score→race_point, recent_top3_rate_6m→top3_6m
# ─────────────────────────────────────────────────────────────────────────────

def build_race_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """wt エントリー df（pred_prob 計算済み）からレース単位の波乱特徴量を構築。

    upset_model.UPSET_FEATURE_COLS の全列を wt カラム名で埋める。
    """
    rows = []
    for race_key, grp in df.groupby("race_key"):
        probs = np.sort(grp["pred_prob"].fillna(0).values)[::-1]
        probs_safe = probs + 1e-9
        probs_norm = probs_safe / probs_safe.sum()

        # wt では race_point が競走得点
        scores = grp["race_point"].dropna().values
        scores_sorted = np.sort(scores)

        # wt では top3_6m が3着内率（6ヶ月）
        top3r_col = "top3_6m" if "top3_6m" in grp.columns else None
        top3r = grp[top3r_col].dropna().values if top3r_col else np.array([])

        row = {
            "race_key":   race_key,
            "race_date":  grp["race_date"].iloc[0],

            "grade_enc":       float(grp["grade_enc"].iloc[0]) if "grade_enc" in grp.columns else np.nan,
            "n_riders":        int(len(grp)),
            "bank_length_enc": float(grp["bank_length_enc"].iloc[0]) if "bank_length_enc" in grp.columns else np.nan,
            "is_indoor":       float(grp["is_indoor"].iloc[0]) if "is_indoor" in grp.columns else 0.0,

            "score_mean":    float(np.mean(scores)) if len(scores) > 0 else np.nan,
            "score_std":     float(np.std(scores))  if len(scores) > 0 else np.nan,
            "score_cv":      float(np.std(scores) / np.mean(scores))
                             if len(scores) > 0 and np.mean(scores) > 0 else np.nan,
            "score_range":   float(np.ptp(scores))  if len(scores) > 0 else np.nan,
            "score_top_gap": float(scores_sorted[-1] - scores_sorted[-2])
                             if len(scores) > 1 else np.nan,

            "top3r_mean": float(np.mean(top3r)) if len(top3r) > 0 else np.nan,
            "top3r_std":  float(np.std(top3r))  if len(top3r) > 0 else np.nan,

            "pred_top1":     float(probs[0])               if len(probs) > 0 else np.nan,
            "pred_top2":     float(probs[1])               if len(probs) > 1 else np.nan,
            "pred_gap12":    float(probs[0] - probs[1])    if len(probs) > 1 else np.nan,
            "pred_gap23":    float(probs[1] - probs[2])    if len(probs) > 2 else np.nan,
            "pred_entropy":  float(-np.sum(probs_norm * np.log(probs_norm))),
            "pred_top3_sum": float(np.sum(probs[:3]))      if len(probs) >= 3 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_upset_target_wt(df_entry: pd.DataFrame, df_race: pd.DataFrame,
                        upset_threshold_yen: int = 2000) -> pd.DataFrame:
    """レース単位 df に波乱ラベル is_upset を付与（wt_odds の的中 trio 倍率を使用）。

    的中 trio 組み合わせ（3着内の車番3つ）のオッズ × 100 が upset_threshold_yen 以上 → 波乱。

    wt_odds の odds_value は倍率（例: 20.5x）。100 円賭けの払戻 = odds_value * 100 円。
    upset_threshold_yen=2000 → 的中 trio オッズ >= 20.0。

    Parameters
    ----------
    df_entry : エントリー df（build_features_wt の出力・finish_order 含む）
    df_race  : レース単位 df（build_race_features_wt の出力）
    """
    import re
    from src.database import get_connection

    odds_thr = upset_threshold_yen / 100.0

    race_keys = df_race["race_key"].tolist()
    if not race_keys:
        df_race = df_race.copy()
        df_race["winner_trio_odds"] = np.nan
        df_race["is_upset"] = np.nan
        return df_race

    # 各レースの1-3着選手（車番）を取得
    fin3 = df_entry[df_entry["finish_order"].between(1, 3)].copy()
    winner_comb: dict[str, frozenset] = {}
    for rk, grp in fin3.groupby("race_key"):
        frames = grp["frame_no"].astype(int).tolist()
        if len(frames) >= 3:
            winner_comb[rk] = frozenset(frames[:3])

    # 対象レースの trio オッズを一括ロード
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT race_key, combination, odds_value FROM wt_odds "
            f"WHERE bet_type='trio' AND race_key IN ({placeholders})",
            race_keys,
        ).fetchall()

    # レースごとに的中 trio のオッズを取得
    trio_board: dict[str, dict] = {}
    for row in rows:
        rk = row["race_key"]
        ov = row["odds_value"]
        if ov is None or ov <= 0:
            continue
        try:
            frames = frozenset(int(x) for x in re.split(r"[-=]", str(row["combination"])))
        except ValueError:
            continue
        if len(frames) != 3:
            continue
        if rk not in trio_board:
            trio_board[rk] = {}
        trio_board[rk][frames] = float(ov)

    win_odds_map: dict[str, float] = {}
    for rk, comb in winner_comb.items():
        board = trio_board.get(rk, {})
        odds = board.get(comb)
        if odds is not None:
            win_odds_map[rk] = odds

    df = df_race.copy()
    df["winner_trio_odds"] = df["race_key"].map(win_odds_map)
    df["is_upset"] = np.where(
        df["winner_trio_odds"].isna(), np.nan,
        (df["winner_trio_odds"] >= odds_thr).astype(float),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: データ収集・モデル学習・レース構造体構築
# ─────────────────────────────────────────────────────────────────────────────

def collect():
    """全期間ロード → TRAIN期間モデル学習 → 全レースの upset_prob / fav_mismatch を計算。"""
    print("=== データロード & 特徴量構築 ===", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    # ── エントリーモデル（TRAIN期間のみ学習・リーク無し）──
    fit = df_all[(df_all["race_date"] <= TRAIN[1]) & (df_all["finish_order"] >= 1)]
    print(f"  エントリーモデル学習: {len(fit):,}行 (〜{TRAIN[1]})", flush=True)
    entry_model = lgb.LGBMClassifier(**LGB_PARAMS)
    entry_model.fit(prepare_X(fit), fit["top3_flag"])
    df_all["pred_prob"] = entry_model.predict_proba(prepare_X(df_all))[:, 1]

    # ── 波乱モデル（TRAIN期間のみ学習・リーク無し）──
    df_train = df_all[df_all["race_date"] <= TRAIN[1]].copy()
    df_race_train = build_race_features_wt(df_train)
    df_race_train = add_upset_target_wt(df_train, df_race_train)

    train_race = df_race_train.dropna(subset=UPSET_FEATURE_COLS + ["is_upset"])
    train_race = train_race[train_race["is_upset"].isin([0.0, 1.0])]
    print(f"  波乱モデル学習: {len(train_race):,}レース", flush=True)
    upset_model = train_upset_model(train_race)

    # ── 全期間の upset_prob を計算 ──
    df_race_all = build_race_features_wt(df_all)
    valid_mask = df_race_all[UPSET_FEATURE_COLS].notna().all(axis=1)
    X_all = df_race_all.loc[valid_mask, UPSET_FEATURE_COLS].values
    probs = upset_model.predict_proba(
        pd.DataFrame(X_all, columns=UPSET_FEATURE_COLS)
    )[:, 1]
    df_race_all.loc[valid_mask, "upset_prob"] = probs
    df_race_all["upset_prob"] = df_race_all.get("upset_prob", pd.Series(np.nan,
                                index=df_race_all.index))

    upset_map = df_race_all.set_index("race_key")["upset_prob"].to_dict()

    # ── upset_prob Q4 閾値を TRAIN期間の全≤6車レースで固定 ──
    train_race_keys = set(df_all[df_all["race_date"] <= TRAIN[1]]["race_key"].unique())
    train_probs_all = [
        v for rk, v in upset_map.items()
        if rk in train_race_keys and not np.isnan(v)
    ]
    q4_threshold = float(np.percentile(train_probs_all, 75))
    print(f"  upset Q4 閾値 (TRAIN ≤6車全レース 75パーセンタイル): {q4_threshold:.4f}", flush=True)
    print(f"  TRAIN 波乱確率範囲: [{min(train_probs_all):.4f}, {max(train_probs_all):.4f}]", flush=True)

    # ── ≤6車フィルタ（出走表基準・doc18）──
    sz = df_all.groupby("race_key")["frame_no"].count()
    race_keys_le6 = set(sz[sz <= 6].index)

    # ── 結果確定レースのみ（3着内が3名以上）──
    done = df_all.groupby("race_key")["finish_order"].apply(
        lambda s: s.between(1, 3).sum() >= 3
    )
    race_keys_done = set(done[done].index)

    # ── trio 盤面・三連単盤面をロード ──
    valid_keys = list(race_keys_le6 & race_keys_done)
    print(f"  有効レース数 (≤6車・結果確定): {len(valid_keys):,}", flush=True)
    trio_b, tf_b, _ = load_boards(valid_keys)

    # ── レース構造体を構築 ──
    races = []
    for rk, g0 in df_all[df_all["race_key"].isin(valid_keys)].groupby("race_key"):
        n = len(g0)                       # エントリー数（欠車含む）
        if n < 4:
            continue
        bd = trio_b.get(rk, {})
        if not bd:
            continue
        fin = g0[g0["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue

        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        dns = set(g0[g0["finish_order"] == 0]["frame_no"].astype(int).tolist())

        # 全エントリーでランキング（doc18）
        g = g0.sort_values("pred_prob", ascending=False)
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]

        tier = _assign_tier(p[0] - p[1], p[0] / (3.0 / n))

        # 欠車処理
        axis_void = (p1 in dns) or (p2 in dns)
        trio3, tf3 = [], []
        if not axis_void:
            tfb = tf_b.get(rk, {})
            for x in thirds:
                if x in dns:
                    continue
                c = frozenset((p1, p2, x))
                if c in bd:
                    trio3.append((bd[c], c == top3))
                o = tfb.get((p1, p2, x))
                if o:
                    tf3.append((o, order == (p1, p2, x)))

        # 最安オッズ
        legs_for_min = tf3 if tier == "SS" else trio3
        min3 = min((o for o, _ in legs_for_min), default=None)

        # fav_mismatch
        mf = market_fav(bd)
        fav_mismatch = (mf is not None) and (int(mf) != p1)

        # upset_prob
        up = upset_map.get(rk, np.nan)

        races.append({
            "race_key":    rk,
            "date":        g0["race_date"].iloc[0],
            "tier":        tier,
            "trio3":       trio3,
            "tf3":         tf3,
            "min3":        min3,
            "fav_mismatch": fav_mismatch,
            "upset_prob":  up,
            "axis_void":   axis_void,
        })

    print(f"  レース構造体: {len(races):,}件", flush=True)
    return races, q4_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 現行戦略の評価関数
# ─────────────────────────────────────────────────────────────────────────────

def base_legs(r):
    """現行戦略: tier成立・該当買式3点・最安≥5倍（ガミ除外）。不成立はNone。"""
    if r["tier"] is None or r["axis_void"]:
        return None
    legs = r["tf3"] if r["tier"] == "SS" else r["trio3"]
    if not legs:
        return None
    if r["min3"] is None or r["min3"] < 5.0:
        return None
    return legs


def cell_roi(races):
    """レースリストから ROI 統計を計算。"""
    pays, bets = [], []
    for r in races:
        legs = base_legs(r)
        if not legs:
            continue
        pays.append(sum(o * 100 for o, hit in legs if hit))
        bets.append(len(legs) * 100)
    s = roi_summary(pays, bets)
    return s, len(pays)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 独立性チェック（phi係数・χ²検定）
# ─────────────────────────────────────────────────────────────────────────────

def independence_check(races, q4_thr):
    """upset Q4 と fav_mismatch の phi 係数・χ²検定を計算。

    基本戦略通過レース（tier成立・ガミ≥5倍・欠車除外）を母集団とする。
    """
    a11, a10, a01, a00 = 0, 0, 0, 0  # [upset_q4, fav_mis]
    for r in races:
        if base_legs(r) is None:
            continue
        up = r["upset_prob"]
        if np.isnan(up):
            continue
        q4_flag = up >= q4_thr
        fm = r["fav_mismatch"]
        if q4_flag and fm:
            a11 += 1
        elif q4_flag and not fm:
            a10 += 1
        elif not q4_flag and fm:
            a01 += 1
        else:
            a00 += 1

    total = a11 + a10 + a01 + a00
    if total == 0:
        return {"phi": None, "chi2": None, "p_value": None,
                "a11": 0, "a10": 0, "a01": 0, "a00": 0}

    ct = np.array([[a11, a10], [a01, a00]])
    if np.any(ct == 0):
        # 空セルがある場合はYates補正付き or Fisher exact test
        if np.any(ct == 0) and (a11 == 0 or a10 == 0 or a01 == 0 or a00 == 0):
            # Fisher exact test を使用
            from scipy.stats import fisher_exact
            _, p_val = fisher_exact(ct)
            # phi はゼロセルに注意して計算
            n_row1 = a11 + a10
            n_row2 = a01 + a00
            n_col1 = a11 + a01
            n_col2 = a10 + a00
            chi2 = 0.0
            if n_row1 and n_row2 and n_col1 and n_col2:
                expected_11 = n_row1 * n_col1 / total
                if expected_11 > 0:
                    chi2 = total * (a11 / total - n_row1 * n_col1 / total**2) ** 2 * total / (n_row1 * n_row2 * n_col1 * n_col2 / total**3)
        else:
            chi2, p_val, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
    else:
        chi2, p_val, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
    phi = np.sqrt(chi2 / total) * np.sign(
        a11 * a00 - a10 * a01  # 符号: 正=正の相関, 負=負の相関
    )

    row_q4  = a11 + a10
    row_nq4 = a01 + a00
    col_fm  = a11 + a01
    col_nfm = a10 + a00
    print(f"\n  [独立性] 分割表 (基本戦略通過レース {total:,}R)")
    print(f"               fav_mis=T  fav_mis=F  合計")
    print(f"  upset_Q4:    {a11:>6}     {a10:>6}   {row_q4:>6}")
    print(f"  Q1-Q3:       {a01:>6}     {a00:>6}   {row_nq4:>6}")
    print(f"  合計:        {col_fm:>6}     {col_nfm:>6}   {total:>6}")
    print(f"  φ係数={phi:.4f}  χ²={chi2:.2f}  p={p_val:.4f}")
    print(f"  共起率 (upset_Q4∩fav_mis=T): {a11}/{total} = {a11/total:.1%}")

    return {
        "phi": float(phi), "chi2": float(chi2), "p_value": float(p_val),
        "a11": a11, "a10": a10, "a01": a01, "a00": a00,
        "cooccurrence_rate": float(a11 / total) if total else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    races, q4_thr = collect()

    # 期間分割
    by = {
        "TRAIN": [r for r in races if r["date"] <= TRAIN[1]],
        "VAL":   [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
        "HOLD":  [r for r in races if r["date"] > VAL[1]],
    }
    print(f"\n  レース総数: TRAIN {len(by['TRAIN'])} / VAL {len(by['VAL'])} / HOLD {len(by['HOLD'])}")
    print(f"  upset Q4 閾値（全≤6車）: {q4_thr:.4f}")

    # ── 補助: 戦略通過レース内の upset_prob Q4 閾値 (TRAIN期間)
    # 戦略通過レース内での「相対的に高波乱」を定義するための閾値
    train_strategy_probs = [
        r["upset_prob"] for r in by["TRAIN"]
        if not np.isnan(r["upset_prob"]) and base_legs(r) is not None
    ]
    if train_strategy_probs:
        q4_thr_strategy = float(np.percentile(train_strategy_probs, 75))
    else:
        q4_thr_strategy = q4_thr
    print(f"  upset Q4 閾値（戦略通過内）: {q4_thr_strategy:.4f}", flush=True)
    print(f"  ※ 全レースの Q4 ({q4_thr:.4f}) と戦略通過内 Q4 ({q4_thr_strategy:.4f}) を両方報告")

    # 4セル定義（全≤6車 Q4 閾値）
    cells = [
        ("A: upset_Q4(全体) + fav_mis=T",
         lambda r, q=q4_thr: r["upset_prob"] >= q and r["fav_mismatch"]),
        ("B: upset_Q4(全体) + fav_mis=F",
         lambda r, q=q4_thr: r["upset_prob"] >= q and not r["fav_mismatch"]),
        ("C: Q1-Q3(全体)   + fav_mis=T",
         lambda r, q=q4_thr: r["upset_prob"] <  q and r["fav_mismatch"]),
        ("D: Q1-Q3(全体)   + fav_mis=F",
         lambda r, q=q4_thr: r["upset_prob"] <  q and not r["fav_mismatch"]),
    ]

    # 4セル定義（戦略通過内 Q4 閾値）
    cells_strat = [
        ("A': upset_Q4(戦略内) + fav_mis=T",
         lambda r, q=q4_thr_strategy: r["upset_prob"] >= q and r["fav_mismatch"]),
        ("B': upset_Q4(戦略内) + fav_mis=F",
         lambda r, q=q4_thr_strategy: r["upset_prob"] >= q and not r["fav_mismatch"]),
        ("C': Q1-Q3(戦略内)   + fav_mis=T",
         lambda r, q=q4_thr_strategy: r["upset_prob"] <  q and r["fav_mismatch"]),
        ("D': Q1-Q3(戦略内)   + fav_mis=F",
         lambda r, q=q4_thr_strategy: r["upset_prob"] <  q and not r["fav_mismatch"]),
    ]

    results = {}
    print(f"\n{'='*110}")
    print("  4セル × 3期間 ROI（現行戦略: tier=SS/S/A × 3点 × 最安≥5倍・全エントリーランキング）")
    print(f"{'='*110}")

    print("\n  ▼ 全≤6車 Q4 閾値 ({:.4f}) による4セル".format(q4_thr))
    for cell_name, cond in cells:
        print(f"\n  ◆ {cell_name}")
        row = {}
        for period in ("TRAIN", "VAL", "HOLD"):
            # upset_prob が NaN のレースはセル外扱い
            sub = [r for r in by[period]
                   if not np.isnan(r["upset_prob"]) and cond(r)]
            s, n = cell_roi(sub)
            row[period] = (s, n)
            print(f"    {period:<6}: {_fmt(s, n)}")
        results[cell_name] = row

    print("\n  ▼ 戦略通過内 Q4 閾値 ({:.4f}) による4セル（全セルにデータあり保証）".format(q4_thr_strategy))
    results_strat = {}
    for cell_name, cond in cells_strat:
        print(f"\n  ◆ {cell_name}")
        row = {}
        for period in ("TRAIN", "VAL", "HOLD"):
            sub = [r for r in by[period]
                   if not np.isnan(r["upset_prob"]) and base_legs(r) is not None and cond(r)]
            s, n = cell_roi(sub)
            row[period] = (s, n)
            print(f"    {period:<6}: {_fmt(s, n)}")
        results_strat[cell_name] = row

    # 独立性チェック（全期間合算）
    print(f"\n{'='*110}")
    print("  独立性チェック: upset Q4（戦略内） × fav_mismatch")
    indep = independence_check(races, q4_thr_strategy)

    # 通過判定（戦略内 Q4 を主判定に使用・データあり保証）
    print(f"\n{'='*110}")
    print("  通過判定 (Cell A': VAL & HOLD 両方 CI下限 > 100%)")
    cell_a_val  = results_strat["A': upset_Q4(戦略内) + fav_mis=T"]["VAL"]
    cell_a_hold = results_strat["A': upset_Q4(戦略内) + fav_mis=T"]["HOLD"]
    val_pass  = (cell_a_val[0]["ci_lo"]  > 1.0 and cell_a_val[1]  >= 10)
    hold_pass = (cell_a_hold[0]["ci_lo"] > 1.0 and cell_a_hold[1] >= 10)
    judgment = "★ 通過" if (val_pass and hold_pass) else "✗ 不通過"
    print(f"  VAL  CI下限={cell_a_val[0]['ci_lo']:.0%}  n={cell_a_val[1]}  "
          f"→ {'通過' if val_pass else '不通過'}")
    print(f"  HOLD CI下限={cell_a_hold[0]['ci_lo']:.0%}  n={cell_a_hold[1]}  "
          f"→ {'通過' if hold_pass else '不通過'}")
    print(f"  総合判定: {judgment}")

    # 結果サマリを JSON 保存（レポート作成用）
    summary = {
        "q4_threshold_all": float(q4_thr),
        "q4_threshold_strategy": float(q4_thr_strategy),
        "periods": {"TRAIN": TRAIN, "VAL": VAL, "HOLD": HOLD},
        "cells_all_q4": {},
        "cells_strategy_q4": {},
        "independence": {
            k: (float(v) if v is not None else None)
            for k, v in indep.items()
            if not isinstance(v, int)
        },
        "independence_counts": {k: v for k, v in indep.items() if isinstance(v, int)},
        "judgment": judgment,
    }
    for cell_name, row in results.items():
        summary["cells_all_q4"][cell_name] = {}
        for period, (s, n) in row.items():
            summary["cells_all_q4"][cell_name][period] = {
                "n": n,
                "roi": float(s["roi"]),
                "ci_lo": float(s["ci_lo"]),
                "ci_hi": float(s["ci_hi"]),
                "hit_rate": float(s["hit_rate"]),
                "roi_ex_max": float(s["roi_ex_max"]),
                "hits": int(s["hits"]),
            }
    for cell_name, row in results_strat.items():
        summary["cells_strategy_q4"][cell_name] = {}
        for period, (s, n) in row.items():
            summary["cells_strategy_q4"][cell_name][period] = {
                "n": n,
                "roi": float(s["roi"]),
                "ci_lo": float(s["ci_lo"]),
                "ci_hi": float(s["ci_hi"]),
                "hit_rate": float(s["hit_rate"]),
                "roi_ex_max": float(s["roi_ex_max"]),
                "hits": int(s["hits"]),
            }

    out_path = (Path(__file__).parent.parent
                / "data" / "analysis" / "exp_upset_fav_mismatch_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  結果を {out_path} に保存しました。")
    print(f"{'='*110}")

    return summary


if __name__ == "__main__":
    main()
