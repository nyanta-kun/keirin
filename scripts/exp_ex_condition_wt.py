"""WINTICKET 条件別成績特徴量 Phase1 AUC 実験（G44）

天候別・バンク周長別・時間帯別・位置別成績（scrape_winticket_ex_stats.py で収集）を
特徴量として追加し、Phase1 AUC ゲートを突破するか検証する。

Usage:
  python3 scripts/exp_ex_condition_wt.py

依存:
  data/player_ex_stats.csv  (scrape_winticket_ex_stats.py で事前収集が必要)
"""
import sys, csv, datetime, re
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if ".claude/worktrees" in str(_REPO_ROOT):
    _REPO_ROOT = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

STATS_FILE = _REPO_ROOT / "data" / "player_ex_stats.csv"
MIN_PLAYERS = 100  # 評価に最低限必要な選手数


# ── 条件フィールド定義 ────────────────────────────────────────────────
# (CSV 列名, 説明)
COND_COLS = [
    "weather_sunny_top3_pct",
    "weather_cloudy_top3_pct",
    "weather_rainy_top3_pct",
    "track_333_top3_pct",
    "track_400_top3_pct",
    "track_500_top3_pct",
    "hour_normal_top3_pct",
    "hour_morning_top3_pct",
    "hour_night_top3_pct",
    "hour_midnight_top3_pct",
    "pos_first_top3_pct",
    "pos_second_top3_pct",
    "pos_third_top3_pct",
    "pos_single_top3_pct",
    "pos_compete_top3_pct",
]


def load_stats() -> pd.DataFrame:
    """data/player_ex_stats.csv を読み込む。"""
    if not STATS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(STATS_FILE, dtype={"player_id": int})
    # None/空文字を NaN に変換
    for col in COND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bank_to_col(bank: int) -> str | None:
    return {333: "track_333_top3_pct", 400: "track_400_top3_pct", 500: "track_500_top3_pct"}.get(bank)


def _ts_to_hour_col(ts_str) -> str | None:
    """Unix timestamp → 時間帯列名"""
    try:
        hour = datetime.datetime.fromtimestamp(int(float(ts_str))).hour
    except (ValueError, TypeError, OSError):
        return None
    if hour < 10:
        return "hour_morning_top3_pct"
    elif hour < 18:
        return "hour_normal_top3_pct"
    elif hour < 22:
        return "hour_night_top3_pct"
    else:
        return "hour_midnight_top3_pct"


def load_venue_bank() -> dict[str, int]:
    """venue_code → bank_length のマッピングを返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT venue_code, bank_length FROM venue_info WHERE bank_length IS NOT NULL"
        ).fetchall()
    return {r["venue_code"]: r["bank_length"] for r in rows}


def build_condition_features(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """条件別成績特徴量を df に追加する。"""
    venue_bank = load_venue_bank()

    # player_id をキーに stats をジョイン
    df = df.merge(stats[["player_id"] + COND_COLS], on="player_id", how="left")

    # 1. track_match_top3_pct: このレースのバンク周長での成績
    bank_series = df["venue_id"].map(venue_bank).fillna(0).astype(int)
    track_match = []
    for bank, row_idx in zip(bank_series, df.index):
        col = _bank_to_col(bank)
        track_match.append(df.at[row_idx, col] if col else np.nan)
    df["track_match_top3_pct"] = track_match

    # 2. hour_match_top3_pct: このレースの時間帯での成績
    hour_match = []
    for ts, row_idx in zip(df.get("start_at", pd.Series(dtype=str)), df.index):
        col = _ts_to_hour_col(ts)
        hour_match.append(df.at[row_idx, col] if col else np.nan)
    df["hour_match_top3_pct"] = hour_match

    # 3. rain_vs_sunny_diff: 雨天適性（雨 - 晴天 top3率差）
    df["rain_vs_sunny_diff"] = df["weather_rainy_top3_pct"] - df["weather_sunny_top3_pct"]

    return df


def prep_ext(cols):
    """指定列セットで DataFrame を整形するクロージャ。"""
    def _inner(d):
        return d.reindex(columns=cols).fillna(0)
    return _inner


def main():
    print("WINTICKET 条件別成績特徴量実験（G44）\n")

    # ── データロード ──────────────────────────────────────────────────
    stats = load_stats()
    if len(stats) < MIN_PLAYERS:
        print(f"条件別成績データが {len(stats)} 件と少なすぎます（{MIN_PLAYERS}件以上必要）。")
        print("全件スクレイプ後に再実行してください:")
        print("  python3 scripts/scrape_winticket_ex_stats.py")
        return

    print(f"条件別成績データ: {len(stats)} 選手ロード済み")
    for col in COND_COLS:
        if col in stats.columns:
            cnt = stats[col].notna().sum()
            print(f"  {col:<35}: {cnt:5d} / {len(stats)} ({cnt/len(stats)*100:.1f}%)")

    print("\nデータ準備中（TRAIN〜HOLD）...", flush=True)

    # start_at を wt_races から取得するため raw_data を直接 SQL で補完
    with get_connection() as conn:
        races_meta = pd.read_sql(
            "SELECT race_key, venue_id, start_at FROM wt_races",
            conn
        )

    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    # start_at / venue_id を wt_races からジョイン（build_features_wt では含まれない場合を考慮）
    if "start_at" not in df.columns or "venue_id" not in df.columns:
        df = df.merge(races_meta, on="race_key", how="left", suffixes=("", "_r"))
        if "venue_id_r" in df.columns:
            df["venue_id"] = df["venue_id"].fillna(df["venue_id_r"])
            df.drop(columns=["venue_id_r"], inplace=True)

    print("条件別成績特徴量を計算中...", flush=True)
    df = build_condition_features(df, stats)

    for col in ["track_match_top3_pct", "hour_match_top3_pct", "rain_vs_sunny_diff",
                "pos_first_top3_pct", "pos_second_top3_pct"]:
        nn = df[col].notna().mean()
        print(f"  {col:<35}: nonzero/notna={nn:.1%}")

    # ── Phase1: AUC ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase1: AUC 比較（Base vs 各条件特徴量）")
    print("=" * 72)

    EXT_TRACK   = FEATURE_COLS_WT + ["track_match_top3_pct"]
    EXT_HOUR    = FEATURE_COLS_WT + ["hour_match_top3_pct"]
    EXT_WEATHER = FEATURE_COLS_WT + ["rain_vs_sunny_diff"]
    EXT_POS     = FEATURE_COLS_WT + ["pos_first_top3_pct"]
    EXT_ALL     = FEATURE_COLS_WT + [
        "track_match_top3_pct", "hour_match_top3_pct",
        "rain_vs_sunny_diff", "pos_first_top3_pct", "pos_second_top3_pct",
    ]

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]

    models = {}
    configs = {
        "base":    (FEATURE_COLS_WT,  prepare_X),
        "+track":  (EXT_TRACK,        prep_ext(EXT_TRACK)),
        "+hour":   (EXT_HOUR,         prep_ext(EXT_HOUR)),
        "+weather":(EXT_WEATHER,      prep_ext(EXT_WEATHER)),
        "+pos":    (EXT_POS,          prep_ext(EXT_POS)),
        "+all":    (EXT_ALL,          prep_ext(EXT_ALL)),
    }

    for name, (cols, pfn) in configs.items():
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(pfn(fit), fit["top3_flag"].values)
        df[f"pred_{name}"] = m.predict_proba(pfn(df))[:, 1]
        models[name] = (m, cols, pfn)

    model_names = list(configs.keys())
    header = f"  {'期間':<10} {'Base':>8} " + " ".join(f"{'Δ'+n:>9}" for n in model_names[1:])
    print(header)
    print("  " + "-" * (10 + 9 + 9 * len(model_names)))

    best_delta = -999
    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD",     HOLD[0], HOLD[1]),
        ("VAL+HOLD", VAL[0],  HOLD[1]),
    ]:
        mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_base = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        parts = [f"  {period:<10} {auc_base:.4f}"]
        for name in model_names[1:]:
            d = roc_auc_score(sub["top3_flag"], sub[f"pred_{name}"]) - auc_base
            mark = "★" if (period == "VAL+HOLD" and d >= 0.001) else " "
            parts.append(f" {d:>+8.4f}{mark}")
            if period == "VAL+HOLD":
                best_delta = max(best_delta, d)
        print("".join(parts))

    # ── 特徴量重要度（+all モデル） ──────────────────────────────────
    print("\n  特徴量重要度（+all モデル・上位15）")
    m_all, all_cols, all_pfn = models["+all"]
    imp = pd.Series(m_all.feature_importances_, index=all_cols)
    imp = imp / imp.sum() * 100
    for feat, v in imp.nlargest(15).items():
        marker = " ←" if feat in EXT_ALL else ""
        print(f"    {feat:<40} {v:.1f}%{marker}")

    # ── 判定 ────────────────────────────────────────────────────────
    print()
    if best_delta >= 0.001:
        print(f"Phase1 通過 ★（最大改善 VAL+HOLD: Δ={best_delta:+.4f}）→ Phase2 ROI 評価")
    else:
        print(f"Phase1 不通過（最大改善 VAL+HOLD: Δ={best_delta:+.4f}、閾値 +0.001 未達）→ Phase2 評価省略")
        return

    # ── Phase2: ROI ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase2: ROI 比較（C0戦略・ガミ≥5倍・≤6車・リーク無し）")
    print("=" * 72)

    GAMI_THRESHOLD = 5.0

    with get_connection() as conn:
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()
        races_raw = conn.execute("SELECT race_key, n_entries FROM wt_races").fetchall()

    trio_map: dict = {}
    for rk, comb, ov in trio_raw:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)

    n_entries_map = dict(races_raw)
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    def _period_of(rd: str) -> str | None:
        if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
        if VAL[0]   <= rd <= VAL[1]:   return "VAL"
        if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
        return None

    def compute_roi(pred_col: str) -> dict:
        records = []
        df_tmp = df.copy()
        df_tmp["pred_prob"] = df_tmp[pred_col]
        for rk, grp in df_tmp.groupby("race_key"):
            if n_entries_map.get(rk, 99) > 6:
                continue
            g = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            if len(g) < 3:
                continue
            period = _period_of(str(g["race_date"].iloc[0]))
            if period is None:
                continue
            p1 = int(g.iloc[0]["frame_no"])
            p2 = int(g.iloc[1]["frame_no"])
            thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]
            bd = trio_map.get(rk, {})
            combos = [frozenset({p1, p2, t}) for t in thirds]
            min_odds = min((bd.get(k, 0) for k in combos if bd.get(k, 0) > 0), default=0)
            if min_odds < GAMI_THRESHOLD:
                continue
            actual = actual_trio.get(rk, frozenset())
            pay = 0.0
            for t in thirds:
                k = frozenset({p1, p2, t})
                if actual == k:
                    pay = bd.get(k, 0) * 100
                    break
            records.append({"period": period, "pay": pay, "cost": len(thirds) * 100})
        return records

    eval_models = [
        ("pred_base", "Base"),
        ("pred_+track", "+track"),
        ("pred_+hour", "+hour"),
        ("pred_+all", "+all"),
    ]

    print(f"\n  {'モデル':<10} {'TRAIN':>10} {'VAL':>10} {'HOLD':>10}  n(TRAIN/VAL/HOLD)")
    print("  " + "-" * 65)
    for pred_col, label in eval_models:
        col = pred_col.replace("pred_", "pred_")
        records = pd.DataFrame(compute_roi(col))
        row = [f"  {label:<10}"]
        ns = []
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = records[records["period"] == period] if len(records) else pd.DataFrame()
            roi = sub["pay"].sum() / sub["cost"].sum() * 100 if len(sub) > 0 else float("nan")
            mk = "★" if roi >= 100 else " "
            row.append(f" {roi:>9.1f}%{mk}")
            ns.append(len(sub))
        print("".join(row) + f"  {ns[0]}/{ns[1]}/{ns[2]}")
    print()


if __name__ == "__main__":
    main()
