"""他式別オッズ（二連単・ワイド・二連複）から market-implied probability を特徴量化する実験

仮説:
  三連複市場は「3選手の組み合わせ」を評価するが、
  二連単市場は「1着の方向性」を独立に価格付けしている。
  この多次元の市場評価をモデル特徴量に加えることで
  現行モデルが捉えていない信号を拾えるか。

追加特徴量（6個）:
  mkt_exacta_win_p   : 二連単から計算した「1着implied確率」（正規化）
  mkt_wide_top3_p    : ワイドから計算した「top3 implied確率」（正規化）
  mkt_quin_top2_p    : 二連複から計算した「top2 implied確率」（正規化）
  mkt_exacta_rank    : レース内でのexacta-win順位（1=高）
  mkt_wide_rank      : レース内でのwide-top3順位（1=高）
  mkt_quin_rank      : レース内でのquinella-top2順位（1=高）

検証プロトコル（リーク無し）:
  Phase1: TRAIN期間(2023-07〜2025-06)のみで学習したモデルでVAL/HOLDのAUCを比較
          閾値: +0.001以上の改善で通過
  Phase2: 同モデルでROI比較（現行C0戦略: pred1+pred2→thirds・ガミ≥5倍）
          TRAIN/VAL/HOLD全期間で100%超えが基準

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15
"""

import sys, re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0

NEW_FEATURES = [
    "mkt_exacta_win_p", "mkt_wide_top3_p", "mkt_quin_top2_p",
    "mkt_exacta_rank",  "mkt_wide_rank",    "mkt_quin_rank",
]


# ─── 市場implied確率の計算 ──────────────────────────────────────────────────

def load_market_features(race_keys: list[str]) -> dict:
    """exacta/quinellaPlace/quinellaから各選手の市場implied確率を計算。

    Returns: {race_key: {frame_no: {mkt_exacta_win_p, mkt_wide_top3_p, mkt_quin_top2_p}}}
    """
    exacta_q: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    wide_q:   dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    quin_q:   dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i: i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                f"WHERE bet_type IN ('exacta','quinellaPlace','quinella') "
                f"AND race_key IN ({ph})",
                chunk,
            ).fetchall()
            for rk, bt, comb, ov in rows:
                if ov is None or ov <= 0:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=]", str(comb))]
                except ValueError:
                    continue
                if bt == "exacta" and len(parts) == 2:
                    winner = parts[0]  # i-j: i は1着
                    exacta_q[rk][winner] += 1.0 / ov
                elif bt == "quinellaPlace" and len(parts) == 2:
                    for f in parts:
                        wide_q[rk][f] += 1.0 / ov
                elif bt == "quinella" and len(parts) == 2:
                    for f in parts:
                        quin_q[rk][f] += 1.0 / ov

    result: dict[str, dict[int, dict[str, float]]] = {}
    for rk in race_keys:
        eq = exacta_q.get(rk, {})
        wq = wide_q.get(rk, {})
        qq = quin_q.get(rk, {})
        eq_tot = sum(eq.values()) or 1.0
        wq_tot = sum(wq.values()) or 1.0
        qq_tot = sum(qq.values()) or 1.0

        frames = set(eq) | set(wq) | set(qq)
        race_data: dict[int, dict[str, float]] = {}
        for f in frames:
            race_data[f] = {
                "mkt_exacta_win_p": eq.get(f, 0.0) / eq_tot,
                "mkt_wide_top3_p":  wq.get(f, 0.0) / wq_tot,
                "mkt_quin_top2_p":  qq.get(f, 0.0) / qq_tot,
            }

        # レース内順位（1=最高スコア）
        def _rank(d, key):
            vals = sorted(d.items(), key=lambda x: -x[1][key])
            return {f: r + 1 for r, (f, _) in enumerate(vals)}

        er = _rank(race_data, "mkt_exacta_win_p")
        wr = _rank(race_data, "mkt_wide_top3_p")
        qr = _rank(race_data, "mkt_quin_top2_p")
        for f in frames:
            race_data[f]["mkt_exacta_rank"] = float(er.get(f, len(frames)))
            race_data[f]["mkt_wide_rank"]   = float(wr.get(f, len(frames)))
            race_data[f]["mkt_quin_rank"]   = float(qr.get(f, len(frames)))

        result[rk] = race_data
    return result


def attach_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """dfに市場特徴量を結合する。欠損は0埋め。"""
    race_keys = df["race_key"].unique().tolist()
    print(f"  市場特徴量をロード中 ({len(race_keys):,} レース)...", flush=True)
    mkt = load_market_features(race_keys)

    rows = []
    for rk, fno in zip(df["race_key"], df["frame_no"]):
        rd = mkt.get(rk, {}).get(int(fno), {})
        rows.append({k: rd.get(k, 0.0) for k in NEW_FEATURES})

    mkt_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, mkt_df], axis=1)


# ─── ヘルパー ───────────────────────────────────────────────────────────────

def period_of(race_date: str) -> str | None:
    if TRAIN[0] <= race_date <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= race_date <= VAL[1]:   return "VAL"
    if HOLD[0]  <= race_date <= HOLD[1]:  return "HOLD"
    return None


def roi_from_records(records: list[dict], period: str) -> tuple[float, int, int]:
    sub = [r for r in records if r["period"] == period]
    if not sub:
        return float("nan"), 0, 0
    pay  = sum(r["pay"] for r in sub)
    cost = sum(r["cost"] for r in sub)
    hits = sum(r["hit"] for r in sub)
    roi  = pay / cost * 100 if cost > 0 else float("nan")
    return roi, len(sub), hits


# ─── Phase1: AUC比較 ────────────────────────────────────────────────────────

def phase1(df_base: pd.DataFrame, df_ext: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("Phase1: AUC 比較（TRAIN期間モデル → VAL+HOLD評価）")
    print("=" * 70)

    # ベースモデル
    fit_base = df_base[(df_base["race_date"] >= TRAIN[0]) & (df_base["race_date"] <= TRAIN[1])
                        & (df_base["finish_order"] >= 1)]
    X_tr_base = prepare_X(fit_base)
    y_tr = fit_base["top3_flag"].values
    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(X_tr_base, y_tr)
    print(f"  ベースモデル学習完了 ({len(fit_base):,} rows)", flush=True)

    # 拡張モデル
    ext_cols = FEATURE_COLS_WT + NEW_FEATURES
    fit_ext = df_ext[(df_ext["race_date"] >= TRAIN[0]) & (df_ext["race_date"] <= TRAIN[1])
                      & (df_ext["finish_order"] >= 1)]
    X_tr_ext = fit_ext.reindex(columns=ext_cols).fillna(0)
    m_ext = lgb.LGBMClassifier(**LGB_PARAMS)
    m_ext.fit(X_tr_ext, fit_ext["top3_flag"].values)
    print(f"  拡張モデル学習完了 ({len(fit_ext):,} rows)", flush=True)

    print()
    print(f"  {'期間':<8} {'ベース AUC':>12} {'拡張 AUC':>12} {'差分':>10}")
    print("  " + "-" * 44)
    passed = False
    for period in ["VAL", "HOLD", "VAL+HOLD"]:
        if period == "VAL+HOLD":
            sub_b = df_base[df_base["race_date"].between(VAL[0], HOLD[1]) & (df_base["finish_order"] >= 1)]
            sub_e = df_ext [df_ext["race_date"].between(VAL[0], HOLD[1])  & (df_ext["finish_order"] >= 1)]
        elif period == "VAL":
            sub_b = df_base[df_base["race_date"].between(VAL[0], VAL[1])  & (df_base["finish_order"] >= 1)]
            sub_e = df_ext [df_ext["race_date"].between(VAL[0], VAL[1])   & (df_ext["finish_order"] >= 1)]
        else:
            sub_b = df_base[df_base["race_date"].between(HOLD[0], HOLD[1]) & (df_base["finish_order"] >= 1)]
            sub_e = df_ext [df_ext["race_date"].between(HOLD[0], HOLD[1])  & (df_ext["finish_order"] >= 1)]

        auc_b = roc_auc_score(sub_b["top3_flag"], m_base.predict_proba(prepare_X(sub_b))[:, 1])
        auc_e = roc_auc_score(sub_e["top3_flag"], m_ext.predict_proba(sub_e.reindex(columns=ext_cols).fillna(0))[:, 1])
        diff  = auc_e - auc_b
        mark  = "★ PASS" if (period == "VAL+HOLD" and diff >= 0.001) else ""
        if period == "VAL+HOLD" and diff >= 0.001:
            passed = True
        print(f"  {period:<8} {auc_b:>12.4f} {auc_e:>12.4f} {diff:>+10.4f}  {mark}")

    threshold = 0.001
    print(f"\n  Phase1 閾値: +{threshold:.3f} (VAL+HOLD)")
    print(f"  Phase1 結果: {'通過 ★' if passed else '不通過'}")
    return passed, m_base, m_ext


# ─── Phase2: ROI比較 ────────────────────────────────────────────────────────

def phase2(df_base: pd.DataFrame, df_ext: pd.DataFrame,
           m_base: lgb.LGBMClassifier, m_ext: lgb.LGBMClassifier) -> None:
    print("\n" + "=" * 70)
    print("Phase2: ROI 比較（C0戦略: pred1+pred2→thirds・ガミ≥5倍）")
    print("=" * 70)

    ext_cols = FEATURE_COLS_WT + NEW_FEATURES

    # 予測確率を付与
    df_base = df_base.copy()
    df_ext  = df_ext.copy()
    df_base["pred_prob"] = m_base.predict_proba(prepare_X(df_base))[:, 1]
    df_ext["pred_prob"]  = m_ext.predict_proba(df_ext.reindex(columns=ext_cols).fillna(0))[:, 1]

    # trio盤面ロード
    all_keys = df_base["race_key"].unique().tolist()
    print(f"  trio盤面ロード ({len(all_keys):,} レース)...", flush=True)
    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries FROM wt_races", conn)
        trio_raw = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()

    trio_map: dict[str, dict] = {}
    for rk, comb, ov in trio_raw:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)

    # actual結果
    actual_trio: dict[str, frozenset] = (
        df_base[df_base["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    races_info_map = dict(zip(races_info["race_key"], races_info["n_entries"]))

    def compute_records(df: pd.DataFrame, label: str) -> list[dict]:
        records = []
        for rk, grp in df.groupby("race_key"):
            period = period_of(str(grp["race_date"].iloc[0]))
            if period is None:
                continue
            n_entries = races_info_map.get(rk, 99)
            if n_entries > 6:
                continue
            grp_s = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            if len(grp_s) < 3:
                continue

            p1 = int(grp_s.iloc[0]["frame_no"])
            p2 = int(grp_s.iloc[1]["frame_no"])
            thirds = [int(grp_s.iloc[i]["frame_no"]) for i in range(2, len(grp_s))]

            trio_board = trio_map.get(rk, {})
            combos = [frozenset({p1, p2, t}) for t in thirds]
            min_odds = min(
                (trio_board.get(k, 0) for k in combos if trio_board.get(k, 0) > 0),
                default=0,
            )
            if min_odds < GAMI_THRESHOLD:
                continue

            actual = actual_trio.get(rk, frozenset())
            pay = 0.0
            for t in thirds:
                k = frozenset({p1, p2, t})
                if actual == k:
                    pay = trio_board.get(k, 0) * 100
                    break

            records.append({
                "period": period,
                "race_key": rk,
                "pay": pay,
                "cost": len(thirds) * 100,
                "hit": int(pay > 0),
            })
        return records

    print("  ベースモデルでROI計算中...", flush=True)
    rec_base = compute_records(df_base, "base")
    print("  拡張モデルでROI計算中...", flush=True)
    rec_ext  = compute_records(df_ext, "ext")

    print()
    print(f"  {'期間':<8} {'ベース ROI':>12} {'拡張 ROI':>12} {'差分':>10}  {'n(base)':>8} {'n(ext)':>8}")
    print("  " + "-" * 64)
    ext_passed = True
    for period in ["TRAIN", "VAL", "HOLD"]:
        roi_b, n_b, h_b = roi_from_records(rec_base, period)
        roi_e, n_e, h_e = roi_from_records(rec_ext,  period)
        diff = roi_e - roi_b
        mk_b = "★" if roi_b >= 100 else ""
        mk_e = "★" if roi_e >= 100 else ""
        print(f"  {period:<8} {roi_b:>10.1f}%{mk_b}  {roi_e:>10.1f}%{mk_e}  {diff:>+9.1f}pp  {n_b:>8} {n_e:>8}")
        if roi_e < 100:
            ext_passed = False

    print(f"\n  Phase2 結果: {'通過 ★' if ext_passed else '不通過'}")
    print("  (TRAIN/VAL/HOLD全期間 >100% が基準)")


# ─── 特徴量重要度 ────────────────────────────────────────────────────────────

def feature_importance(m_ext: lgb.LGBMClassifier) -> None:
    print("\n" + "=" * 70)
    print("追加特徴量の重要度（拡張モデル）")
    print("=" * 70)
    cols = FEATURE_COLS_WT + NEW_FEATURES
    imp = pd.Series(m_ext.feature_importances_, index=cols)
    # 新特徴量のみ抽出
    new_imp = imp[NEW_FEATURES].sort_values(ascending=False)
    all_imp_sorted = imp.sort_values(ascending=False)
    print("\n  新特徴量の重要度:")
    for feat, score in new_imp.items():
        rank = (all_imp_sorted.index.tolist().index(feat) + 1)
        pct = score / imp.sum() * 100
        print(f"    {feat:<28} {score:>6.0f}  ({pct:.1f}%)  全体{rank}位")
    print(f"\n  新特徴量合計: {new_imp.sum():.0f} / {imp.sum():.0f} ({new_imp.sum()/imp.sum()*100:.1f}%)")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("他式別オッズ（二連単・ワイド・二連複）→ 特徴量化実験")
    print(f"  仮説: 三連複以外の市場は独立した方向性シグナルを持つ")
    print()

    print("データ準備中...", flush=True)
    df_base = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    print(f"  ベースデータ: {len(df_base):,} 行")

    df_ext = attach_market_features(df_base.copy())
    print(f"  拡張データ: {len(df_ext):,} 行 (+{len(NEW_FEATURES)} 特徴量)")

    # Phase1
    passed, m_base, m_ext = phase1(df_base, df_ext)

    # 特徴量重要度（常に表示）
    feature_importance(m_ext)

    if not passed:
        print("\n→ Phase1 不通過。Phase2 はスキップ。")
        print("  解釈: 三連複市場が既に他式別の情報を統合しているか、")
        print("        prediction_mark が市場マルチシグナルを既に織り込んでいる可能性が高い。")
        return

    # Phase2
    phase2(df_base, df_ext, m_base, m_ext)


if __name__ == "__main__":
    main()
