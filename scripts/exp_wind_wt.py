"""風×バンク特徴のリーク無し検証 (G06)

仮説: 風は日時変動する外生変数であり、同じバンクでも日によって逃げ/差しの有利が
変わる＝レース内相対に乗る可能性がある（docs/analysis/20 の残課題）。

検証設計（事前登録・docs/goals/G06-wind-verification.md）:
  Phase1: AUC/logloss 情報量検定
    - TRAIN 2023-07〜2025-06 のみで学習したリーク無し LGBM（ベース）に
      風特徴を追加し、VAL/HOLDOUT の AUC 差・logloss 差を測る。
    - 不通過基準: AUC 差が ±0.001 未満なら「無情報」と判定し Phase2 に進まない。
    - is_indoor=1 の会場は wind 系特徴を 0 扱い。
  Phase2: ROI 検定（Phase1 通過時のみ）
    - doc18 セマンティクス（全エントリーランキング・出走表基準≤6車・欠車void・上限値注記）
    - 現行 C0 × 風ゲート（強風≥7m/s 屋外のみ / 低風<3m/s 屋外のみ の 2 セル）
    - 3 期間 TRAIN/VAL/HOLDOUT
  決まり手事後分析:
    - 風速帯×決まり手分布のシフトを確認（doc20 のバンク版の風版）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-12
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT
)
from src.database import get_connection
from exp_segment_first_wt import load_boards, market_fav, LGB_PARAMS, TRAIN, VAL, HOLD
from roi_robustness_wt import roi_summary
from src.evaluation.backtest_wt import _assign_tier

# ─── 期間 ────────────────────────────────────────────────────────────────────
# TRAIN/VAL/HOLD は exp_segment_first_wt.py に合わせる

# ─── 不通過閾値 ──────────────────────────────────────────────────────────────
AUC_THRESHOLD = 0.001   # AUC 差がこれ未満なら Phase1 不通過
WIND_STRONG = 7.0       # Phase2 強風ゲート (m/s)
WIND_CALM   = 3.0       # Phase2 低風ゲート (m/s)

WIND_FEATURE_COLS = [
    "wind_speed",
    "wind_gust",
    "wind_x_style",      # wind_speed × style_enc（逃=0/自在=1/追=2）
    "wind_x_straight",   # wind_speed × straight_len（バンク直線長）
    "wind_x_cant",       # wind_speed × cant_deg（カント角）
    "temp",
    "precip",
]


# ─── 気象データ付与 ───────────────────────────────────────────────────────────
def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """df（wt_entries + wt_races 結合済み）に気象特徴を付与する。

    start_at が unix epoch 整数の場合は JST に変換し dt_hour を生成。
    is_indoor=1 の会場は風速・突風を 0 に固定。
    """
    df = df.copy()

    # dt_hour を生成（wt_races.start_at は unix epoch 整数）
    def _to_dt_hour(val):
        try:
            ts = int(float(val))
            import datetime as _dt
            jst = _dt.datetime.utcfromtimestamp(ts) + _dt.timedelta(hours=9)
            return jst.strftime("%Y-%m-%d %H:00")
        except Exception:
            return None

    df["_dt_hour"] = df["start_at"].apply(_to_dt_hour)
    df["_venue_id"] = df["venue_id"].astype(str)

    # 気象テーブルを一括ロード（venue_id × dt_hour をキーに JOIN）
    unique_pairs = df[["_venue_id", "_dt_hour"]].dropna().drop_duplicates()
    weather_dict: dict[tuple, dict] = {}

    with get_connection() as conn:
        for _, row in unique_pairs.iterrows():
            vid, dth = row["_venue_id"], row["_dt_hour"]
            if dth is None:
                continue
            rec = conn.execute(
                "SELECT wind_speed, wind_gust, wind_dir, temp, precip "
                "FROM wt_weather WHERE venue_id=? AND dt_hour=?",
                (vid, dth)
            ).fetchone()
            if rec is not None:
                weather_dict[(vid, dth)] = {
                    "wind_speed": rec[0], "wind_gust": rec[1],
                    "wind_dir": rec[2], "temp": rec[3], "precip": rec[4]
                }
            else:
                # ±1時間で最近傍探索
                import datetime as _dt
                try:
                    base = _dt.datetime.strptime(dth, "%Y-%m-%d %H:00")
                    dth_prev = (base - _dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
                    dth_next = (base + _dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
                    rec2 = conn.execute(
                        "SELECT wind_speed, wind_gust, wind_dir, temp, precip "
                        "FROM wt_weather WHERE venue_id=? AND dt_hour IN (?,?) "
                        "ORDER BY ABS(strftime('%s', dt_hour) - strftime('%s', ?)) LIMIT 1",
                        (vid, dth_prev, dth_next, dth)
                    ).fetchone()
                    if rec2 is not None:
                        weather_dict[(vid, dth)] = {
                            "wind_speed": rec2[0], "wind_gust": rec2[1],
                            "wind_dir": rec2[2], "temp": rec2[3], "precip": rec2[4]
                        }
                except Exception:
                    pass

    # 列を付与
    def _get(vid, dth, key, default=0.0):
        d = weather_dict.get((vid, dth), {})
        v = d.get(key)
        return float(v) if v is not None else default

    df["wind_speed"] = [_get(v, d, "wind_speed") for v, d in zip(df["_venue_id"], df["_dt_hour"])]
    df["wind_gust"]  = [_get(v, d, "wind_gust")  for v, d in zip(df["_venue_id"], df["_dt_hour"])]
    df["temp"]       = [_get(v, d, "temp", 15.0) for v, d in zip(df["_venue_id"], df["_dt_hour"])]
    df["precip"]     = [_get(v, d, "precip")     for v, d in zip(df["_venue_id"], df["_dt_hour"])]

    # 屋内会場の風速を 0 に固定（is_indoor=1）
    is_indoor = df["is_indoor"].fillna(0).astype(int)
    df.loc[is_indoor == 1, "wind_speed"] = 0.0
    df.loc[is_indoor == 1, "wind_gust"]  = 0.0

    # venue_info から straight_len / cant_deg を取得
    with get_connection() as conn:
        vi = pd.read_sql_query(
            "SELECT venue_code, straight_len, cant_deg FROM venue_info", conn
        )
    vi_dict = {
        str(r["venue_code"]): (r["straight_len"], r["cant_deg"])
        for _, r in vi.iterrows()
    }
    df["_straight"] = df["_venue_id"].map(
        lambda v: vi_dict.get(v, (None, None))[0]
    ).fillna(df["_venue_id"].map(lambda v: vi_dict.get(v, (50.0, 30.0))[0]).median())
    df["_cant"] = df["_venue_id"].map(
        lambda v: vi_dict.get(v, (None, None))[1]
    ).fillna(df["_venue_id"].map(lambda v: vi_dict.get(v, (50.0, 30.0))[1]).median())

    # 交互作用特徴
    style_enc = df["style_enc"].fillna(-1).astype(float)
    df["wind_x_style"]    = df["wind_speed"] * style_enc.clip(0)    # 逃=0 で差し方向に+
    df["wind_x_straight"] = df["wind_speed"] * df["_straight"].fillna(50.0)
    df["wind_x_cant"]     = df["wind_speed"] * df["_cant"].fillna(30.0)

    df = df.drop(columns=["_dt_hour", "_venue_id", "_straight", "_cant"], errors="ignore")
    return df


# ─── Phase1: 情報量検定 ───────────────────────────────────────────────────────
def phase1():
    print("=" * 80)
    print("Phase1: 情報量検定 (AUC/logloss差・リーク無しLGBM)")
    print("=" * 80)

    print("  loading & building features...", flush=True)
    df_base = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    print("  attaching weather features...", flush=True)
    df_wind = attach_weather(df_base)

    # 学習データ: TRAIN 期間・欠車除く
    train_mask = (df_base["race_date"] <= TRAIN[1]) & (df_base["finish_order"] >= 1)
    fit_base = df_base[train_mask]
    fit_wind = df_wind[train_mask]

    print(f"  TRAIN rows: {len(fit_base):,}  (finish_order>=1)")

    # ベースモデル
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit_base), fit_base["top3_flag"])

    # 風特徴追加モデル
    FEAT_WIND = FEATURE_COLS_WT + WIND_FEATURE_COLS
    X_wind_train = fit_wind.reindex(columns=FEAT_WIND).fillna(0)
    m_wind = lgb.LGBMClassifier(**LGB_PARAMS)
    m_wind.fit(X_wind_train, fit_wind["top3_flag"])

    # 予測
    p_base = m_base.predict_proba(prepare_X(df_base))[:, 1]
    X_wind_all = df_wind.reindex(columns=FEAT_WIND).fillna(0)
    p_wind = m_wind.predict_proba(X_wind_all)[:, 1]
    df_base["p_base"] = p_base
    df_base["p_wind"] = p_wind

    # 評価: VAL / HOLD（結果確定行のみ）
    results = {}
    for name, lo, hi in [("VAL", TRAIN[1], VAL[1]), ("HOLD", VAL[1], HOLD[1])]:
        mask = (
            (df_base["race_date"] > lo)
            & (df_base["race_date"] <= hi)
            & (df_base["finish_order"] >= 0)  # 欠車(0)含む全レースで評価
        )
        sub = df_base[mask]
        y   = sub["top3_flag"].values
        pb  = sub["p_base"].values
        pw  = sub["p_wind"].values

        if y.sum() == 0 or (1 - y).sum() == 0:
            print(f"  {name}: y に偏りあり（スキップ）")
            continue

        auc_base = roc_auc_score(y, pb)
        auc_wind = roc_auc_score(y, pw)
        ll_base  = log_loss(y, pb)
        ll_wind  = log_loss(y, pw)

        results[name] = {
            "auc_base": auc_base, "auc_wind": auc_wind,
            "auc_diff": auc_wind - auc_base,
            "ll_base":  ll_base,  "ll_wind":  ll_wind,
            "ll_diff":  ll_wind  - ll_base,
            "n": len(sub),
        }

        print(f"\n  ── {name} ({len(sub):,} rows) ──")
        print(f"    AUC  base={auc_base:.4f}  wind={auc_wind:.4f}  diff={auc_wind-auc_base:+.4f}")
        print(f"    LogL base={ll_base:.4f}  wind={ll_wind:.4f}  diff={ll_wind-ll_base:+.4f}")
        flag = "AUC差 ≥±0.001 → Phase2候補" if abs(auc_wind - auc_base) >= AUC_THRESHOLD else "AUC差 <±0.001 → 無情報"
        print(f"    判定: {flag}")

    # Phase1 通過判定: VAL OR HOLD で |AUC diff| >= 0.001
    pass1 = any(abs(v["auc_diff"]) >= AUC_THRESHOLD for v in results.values())

    print(f"\n  ── Phase1 総合判定 ──")
    if pass1:
        print(f"  【通過】AUC 差が閾値 ±{AUC_THRESHOLD} を超えた期間あり → Phase2 へ")
    else:
        print(f"  【不通過】全評価期間で AUC 差 <±{AUC_THRESHOLD} → 無情報と判定・Phase2 省略")

    return pass1, df_base, df_wind, m_base, m_wind


# ─── 決まり手事後分析 ─────────────────────────────────────────────────────────
def factor_analysis():
    print("\n" + "=" * 80)
    print("決まり手事後分析: 風速帯×決まり手分布（屋外レースのみ）")
    print("=" * 80)

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT e.factor, e.style, w.wind_speed, vi.is_indoor
            FROM wt_entries e
            JOIN wt_races r ON e.race_key = r.race_key
            JOIN wt_weather w ON w.venue_id = r.venue_id
                AND w.dt_hour = strftime('%Y-%m-%d %%H:00',
                    datetime(r.start_at, 'unixepoch', '+9 hours'))
            LEFT JOIN venue_info vi ON r.venue_id = vi.venue_code
            WHERE r.race_date >= ? AND r.race_date <= ?
              AND e.finish_order = 1
              AND e.factor IN ('逃', '捲', '差')
            """.replace("%%", "%"),
            conn,
            params=(TRAIN[0], HOLD[1]),
        )

    # 屋外のみ
    df = df[df["is_indoor"].fillna(0) == 0].copy()

    bins  = [0, 3, 7, 12, 999]
    labels = ["0-2m/s", "3-6m/s", "7-11m/s", "12+m/s"]
    df["wind_bin"] = pd.cut(df["wind_speed"], bins=bins, right=False, labels=labels)

    pivot = (
        df.groupby(["wind_bin", "factor"], observed=True)["factor"]
        .count()
        .unstack("factor")
        .fillna(0)
    )
    pivot["total"] = pivot.sum(axis=1)
    for col in ["逃", "捲", "差"]:
        if col in pivot.columns:
            pivot[f"{col}%"] = (pivot[col] / pivot["total"] * 100).round(1)

    print("\n  風速帯  |  逃%   捲%   差%  | 合計")
    print("  --------+-------------------+------")
    for idx, row in pivot.iterrows():
        nige = row.get("逃%", 0.0)
        maki = row.get("捲%", 0.0)
        sash = row.get("差%", 0.0)
        tot  = int(row["total"])
        print(f"  {str(idx):<8}|  {nige:>4.1f}  {maki:>4.1f}  {sash:>4.1f} | {tot:>6}")

    print("\n  ※ 風速帯間の決まり手シフト幅が±2pp以内 → バンク特定成果（doc20）と同様に軽微。")


# ─── Phase2: ROI 検定 ─────────────────────────────────────────────────────────
def phase2(df_base: pd.DataFrame, df_wind: pd.DataFrame, m_wind):
    """doc18 セマンティクスで風ゲート × C0 の 2 セルを評価する。"""
    print("\n" + "=" * 80)
    print("Phase2: ROI 検定（doc18 セマンティクス）")
    print("=" * 80)

    # ── 全期間を一括処理（TRAIN/VAL/HOLD 分割は後で）──
    df = df_wind.copy()
    FEAT_WIND = FEATURE_COLS_WT + WIND_FEATURE_COLS
    X_all = df.reindex(columns=FEAT_WIND).fillna(0)
    df["pred_prob"] = m_wind.predict_proba(X_all)[:, 1]

    # 出走表基準 ≤6車・結果確定レースのみ
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz <= 6].index)].copy()
    done = df.groupby("race_key")["finish_order"].apply(lambda s: (s >= 1).sum() >= 3)
    df = df[df["race_key"].isin(done[done].index)]

    race_keys = df["race_key"].unique().tolist()
    trio_b, tf_b, _ = load_boards(race_keys)

    # 欠車 void / 風速をレース単位で付与
    with get_connection() as conn:
        wind_per_race = {}
        is_indoor_per_race = {}
        for rk, g in df.groupby("race_key"):
            wind_per_race[rk] = g["wind_speed"].iloc[0]
            is_indoor_per_race[rk] = int(g["is_indoor"].fillna(0).iloc[0])

    races = []
    for rk, g0 in df.groupby("race_key"):
        n = len(g0)
        if n < 4:
            continue
        bd = trio_b.get(rk, {})
        if not bd:
            continue
        fin = g0[g0["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3  = frozenset(fin["frame_no"].astype(int).tolist())
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        dns   = set(g0[g0["finish_order"] == 0]["frame_no"].astype(int).tolist())

        g = g0.sort_values("pred_prob", ascending=False)
        p  = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]
        tier = _assign_tier(p[0] - p[1], p[0] / (3.0 / n))

        axis_void = (p1 in dns) or (p2 in dns)
        trio3 = []
        if not axis_void:
            for x in thirds:
                if x in dns:
                    continue
                c = frozenset((p1, p2, x))
                if c in bd:
                    trio3.append((bd[c], c == top3))

        tfb = tf_b.get(rk, {})
        tf3, tf6 = [], []
        if not axis_void:
            for x in thirds:
                if x in dns:
                    continue
                o = tfb.get((p1, p2, x))
                if o:
                    tf3.append((o, order == (p1, p2, x)))
            for a, b in ((p1, p2), (p2, p1)):
                for x in thirds:
                    if x in dns:
                        continue
                    ob = tfb.get((a, b, x))
                    if ob:
                        tf6.append((ob, order == (a, b, x)))

        min3 = min((o for o, _ in (tf3 if tier == "SS" else trio3)), default=None)

        races.append({
            "date":     g0["race_date"].iloc[0],
            "wind":     wind_per_race.get(rk, 0.0),
            "is_indoor": is_indoor_per_race.get(rk, 0),
            "tier": tier,
            "min3": min3,
            "trio3": trio3,
            "tf3":   tf3,
            "tf6":   tf6,
        })

    by = {
        "TRAIN": [r for r in races if r["date"] <= TRAIN[1]],
        "VAL":   [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
        "HOLD":  [r for r in races if r["date"] > VAL[1]],
    }
    print(f"  races: TRAIN {len(by['TRAIN'])} / VAL {len(by['VAL'])} / HOLD {len(by['HOLD'])}")

    def c0_legs(r):
        """C0: tier 成立・3点・最安≥5倍。"""
        if r["tier"] is None or r["min3"] is None or r["min3"] < 5.0:
            return None
        return r["tf3"] if r["tier"] == "SS" else r["trio3"]

    def gate_strong(r):
        """C0 ∩ 強風≥7m/s ∩ 屋外。"""
        if r["is_indoor"] or r["wind"] < WIND_STRONG:
            return None
        return c0_legs(r)

    def gate_calm(r):
        """C0 ∩ 低風<3m/s ∩ 屋外。"""
        if r["is_indoor"] or r["wind"] >= WIND_CALM:
            return None
        return c0_legs(r)

    gates = [
        ("C0 全レース", c0_legs),
        (f"C0 ∩ 強風≥{WIND_STRONG}m/s 屋外", gate_strong),
        (f"C0 ∩ 低風<{WIND_CALM}m/s 屋外",  gate_calm),
    ]

    def cell_roi(race_list, fn):
        pays, bets = [], []
        for r in race_list:
            legs = fn(r)
            if not legs:
                continue
            pays.append(sum(o * 100 for o, hit in legs if hit))
            bets.append(len(legs) * 100)
        return roi_summary(pays, bets), len(pays)

    def fmt(s, n):
        if n == 0:
            return f"{'0':>4}R  --"
        return (f"{n:>4}R {s['roi']:>5.0%} "
                f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] "
                f"除{s['roi_ex_max']:>4.0%}")

    print(f"\n  {'ゲート':<26}  {'TRAIN':<38}  {'VAL':<38}  {'HOLD':<38}")
    print(f"  {'-'*26}  {'-'*38}  {'-'*38}  {'-'*38}")

    for name, fn in gates:
        cols = []
        verdicts = []
        for per in ("TRAIN", "VAL", "HOLD"):
            s, n = cell_roi(by[per], fn)
            cols.append(fmt(s, n))
            if per in ("VAL", "HOLD"):
                verdicts.append("✅" if (n >= 20 and s["roi"] > 1.0 and s["roi_ex_max"] > 1.0) else "✗")
        print(f"  {name:<26}  {cols[0]:<38}  {cols[1]:<38}  {cols[2]:<38}"
              f"  VAL:{verdicts[0]} HOLD:{verdicts[1]}")

    print(f"\n  払戻=最終オッズ上限値（実運用は朝→確定ドリフトで下振れ）")


# ─── メイン ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("G06: 風×バンク特徴のリーク無し検証")
    print("  期間: TRAIN", TRAIN, "/ VAL", VAL, "/ HOLD", HOLD)
    print("=" * 80)

    # 決まり手事後分析（常に実行）
    factor_analysis()

    # Phase1
    passed, df_base, df_wind, m_base, m_wind = phase1()

    # Phase2（Phase1 通過時のみ）
    if passed:
        phase2(df_base, df_wind, m_wind)
    else:
        print("\n  Phase2 省略（Phase1 不通過）")

    print("\n" + "=" * 80)
    print("完了。判定: " + ("Phase1 通過 → Phase2 実施" if passed else "Phase1 不通過（無情報）"))
    print("  ※ 本番 FEATURE_COLS_WT は変更しない。")
    print("=" * 80)


if __name__ == "__main__":
    main()
