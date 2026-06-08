"""
予想販売向け: S/A 3連複 低配当レース見送り分析

確定前情報として使える「AIの軸2頭間の2車複(quinella)配当」を代理指標とし、
閾値を変えながら的中率・回収率・1日あたり推奨レース数の変化を検証する。

設計:
  - quinella(pivot1, pivot2)が低い = 市場がこの2頭を強く支持 = 3連複も低配当になりやすい
  - 閾値より低いレースは「推奨しない」とし、その場合の各指標を確認
  - SS(3連単)は対象外
"""
import sys
import pickle
import collections
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS, TARGET_COL
from src.database import get_connection

HOLDOUT_FROM = "2025-06-01"
HOLDOUT_TO   = "2026-02-28"
MODEL_PATH   = "data/models/lgbm.pkl"
MAX_RIDERS   = 6
GAP12_MIN    = 0.06
SS_GAP12     = 0.15
SS_RATIO_MAX = 1.3
UNIT         = 100
SA_PTS       = 3    # S/A は 3点300円


def load_quinella_all(race_keys):
    if not race_keys:
        return {}
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        conn.row_factory = None
        rows = conn.execute(f"""
            SELECT race_key, combination, payout FROM odds
            WHERE race_key IN ({placeholders}) AND bet_type='quinella' AND payout IS NOT NULL
        """, race_keys).fetchall()
    result = collections.defaultdict(dict)
    for rk, combo, pay in rows:
        a, b = combo.split("=")
        result[rk][frozenset([int(a), int(b)])] = pay
    return result


def load_result_payouts(race_keys):
    if not race_keys:
        return {}
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        conn.row_factory = None
        rows = conn.execute(f"""
            SELECT race_key, bet_type, combination, payout FROM odds
            WHERE race_key IN ({placeholders}) AND payout IS NOT NULL
        """, race_keys).fetchall()
    result = collections.defaultdict(dict)
    for rk, bt, combo, pay in rows:
        result[rk][(bt, combo)] = pay
    return result


def classify_rank(gap12, ratio):
    if gap12 < GAP12_MIN:
        return "SKIP"
    if gap12 >= SS_GAP12 and ratio < SS_RATIO_MAX:
        return "SS"
    if gap12 >= SS_GAP12:
        return "S"
    return "A"


def eval_sa(grp, payout_map):
    """S/A の的中判定と3連複払戻を返す"""
    grp = grp.sort_values("pred_prob", ascending=False)
    ranked = grp["frame_no"].tolist()
    race_key = grp["race_key"].iloc[0]
    race_pay = payout_map.get(race_key, {})

    actual_df = grp[grp["finish_position"].isin([1, 2, 3])].sort_values("finish_position")
    if len(actual_df) < 3:
        return False, 0
    top3_set = frozenset(actual_df["frame_no"].tolist())

    pivot1, pivot2 = ranked[0], ranked[1]
    thirds = [r for r in ranked[2:5] if r not in (pivot1, pivot2)]
    combos = [frozenset([pivot1, pivot2, t]) for t in thirds[:3]]
    if top3_set in combos:
        pk = "=".join(map(str, sorted(top3_set)))
        return True, race_pay.get(("trifecta_box", pk), 0)
    return False, 0


def eval_ss(grp, payout_map):
    grp = grp.sort_values("pred_prob", ascending=False)
    ranked = grp["frame_no"].tolist()
    race_key = grp["race_key"].iloc[0]
    race_pay = payout_map.get(race_key, {})
    actual_df = grp[grp["finish_position"].isin([1, 2, 3])].sort_values("finish_position")
    if len(actual_df) < 3:
        return False, 0
    actual_order = tuple(actual_df["frame_no"].tolist())
    pivot1, pivot2 = ranked[0], ranked[1]
    thirds = [r for r in ranked[2:5] if r not in (pivot1, pivot2)]
    combos = [(pivot1, pivot2, t) for t in thirds[:3]]
    if actual_order in combos:
        pk = "-".join(map(str, actual_order))
        return True, race_pay.get(("trifecta", pk), 0)
    return False, 0


def main():
    print(f"Loading data {HOLDOUT_FROM} ~ {HOLDOUT_TO} ...")
    df_raw = load_raw_data(min_date=HOLDOUT_FROM, max_date=HOLDOUT_TO)
    df = build_features(df_raw)
    df = df[df["finish_position"].notna()].copy()

    model = pickle.load(open(MODEL_PATH, "rb"))
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = pd.DataFrame(df[FEATURE_COLS].values, columns=FEATURE_COLS)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    race_sizes = df.groupby("race_key")["frame_no"].count()
    valid = race_sizes[race_sizes <= MAX_RIDERS].index
    df = df[df["race_key"].isin(valid)]
    race_date_map = df.groupby("race_key")["race_date"].first().to_dict()

    all_keys = df["race_key"].unique().tolist()
    q_map = load_quinella_all(all_keys)
    pay_map = load_result_payouts(all_keys)

    # ─── レース単位で集計 ──────────────────────────────────────────────────
    rows = []
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        ranked = grp["frame_no"].tolist()
        n = len(ranked)
        top1 = grp["pred_prob"].iloc[0]
        top2 = grp["pred_prob"].iloc[1] if n >= 2 else 0.0
        gap12 = top1 - top2
        ratio = top1 / (3 / n)
        rank = classify_rank(gap12, ratio)
        if rank == "SKIP":
            continue

        pivot1, pivot2 = ranked[0], ranked[1]
        q_pivot = q_map.get(race_key, {}).get(frozenset([pivot1, pivot2]))

        if rank == "SS":
            hit, payout = eval_ss(grp, pay_map)
            bet = 3 * UNIT
        else:
            hit, payout = eval_sa(grp, pay_map)
            bet = SA_PTS * UNIT

        rows.append({
            "race_key":   race_key,
            "race_date":  race_date_map[race_key],
            "rank":       rank,
            "hit":        hit,
            "payout":     payout,
            "bet":        bet,
            "q_pivot":    q_pivot,   # 軸2頭間quinella配当（確定前代理指標）
        })

    df_r = pd.DataFrame(rows)
    df_sa = df_r[df_r["rank"].isin(["S", "A"])].copy()
    df_ss = df_r[df_r["rank"] == "SS"].copy()

    n_days_total = df_r["race_date"].nunique()
    n_days_sa    = df_sa["race_date"].nunique()

    print(f"\n対象期間: {HOLDOUT_FROM} ~ {HOLDOUT_TO}  ({n_days_total}日)")
    print(f"S/A 対象: {len(df_sa)}R  quinella判明: "
          f"{df_sa['q_pivot'].notna().sum()}R ({df_sa['q_pivot'].notna().mean():.1%})")

    # ─── quinella × 3連複配当の相関を確認 ────────────────────────────────
    df_sa_hit = df_sa[df_sa["hit"]].copy()
    print(f"\n  S/A 的中レース {len(df_sa_hit)}R の quinella(軸ペア) vs 3連複配当:")
    print(f"  {'quinella帯':<18} {'的中R':>6}  {'3連複avg':>9}  {'3連複中央値':>11}  "
          f"{'<300円%':>7}  {'<400円%':>7}  {'≥600円%':>7}")
    print(f"  {'-'*65}")
    q_bins = [0, 200, 300, 400, 500, 700, 9999]
    q_labels = ["~200円", "201~300", "301~400", "401~500", "501~700", "701円~"]
    df_sa_hit["q_band"] = pd.cut(df_sa_hit["q_pivot"], bins=q_bins, labels=q_labels, right=True)
    for band in q_labels:
        g = df_sa_hit[df_sa_hit["q_band"] == band]
        if g.empty:
            continue
        avg = g["payout"].mean()
        med = g["payout"].median()
        low = (g["payout"] < 300).mean()
        mid = (g["payout"] < 400).mean()
        hi  = (g["payout"] >= 600).mean()
        print(f"  {band:<18} {len(g):>6}  {avg:>9,.0f}円  {med:>11,.0f}円  "
              f"{low:>7.1%}  {mid:>7.1%}  {hi:>7.1%}")

    # ─── quinella閾値別フィルター ─────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  【S/A: quinella(軸ペア)閾値フィルター別 回収率・的中率・レース数】")
    print("  ※ quinella < 閾値 のレースは推奨しない（見送り）")
    print(f"{'='*90}")

    # ヘッダー
    print(f"  {'閾値(推奨条件)':<22} {'推奨R':>6} {'1日avg':>7} "
          f"{'見送R':>6} {'見送%':>6}  "
          f"{'的中':>5} {'的中率':>7} "
          f"{'投資':>10} {'回収':>10} {'ROI':>7} {'損益':>10}  "
          f"avg配当  中央値")
    print(f"  {'-'*88}")

    thresholds = [None, 200, 250, 300, 350, 400, 500]
    baseline_rows = len(df_sa)
    baseline_days = n_days_sa

    for thr in thresholds:
        if thr is None:
            sub = df_sa.copy()
            label = "全件(フィルタなし)"
        else:
            # quinella >= thr OR quinella不明(≒確認できないため推奨)
            sub = df_sa[(df_sa["q_pivot"].isna()) | (df_sa["q_pivot"] >= thr)].copy()
            label = f"quinella≥{thr}円のみ"

        n    = len(sub)
        skip = baseline_rows - n
        skip_pct = skip / baseline_rows if baseline_rows else 0
        hits = int(sub["hit"].sum())
        bet  = sub["bet"].sum()
        ret  = sub["payout"].sum()
        roi  = ret / bet if bet else 0
        hit_rate = hits / n if n else 0
        n_days_sub = sub["race_date"].nunique()
        avg_per_day = n / n_days_sub if n_days_sub else 0
        hit_rows = sub[sub["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        med_pay = hit_rows["payout"].median() if hits else 0

        print(f"  {label:<22} {n:>6} {avg_per_day:>7.1f}R "
              f"{skip:>6} {skip_pct:>6.1%}  "
              f"{hits:>5} {hit_rate:>7.1%} "
              f"{bet:>10,} {ret:>10,} {roi:>7.1%} {ret-bet:>+10,}  "
              f"{avg_pay:>6,.0f}円  {med_pay:>6,.0f}円")

    # ─── ランク別（S/A 分離）閾値フィルター ──────────────────────────────
    print(f"\n{'='*90}")
    print("  【S/A ランク別: quinella閾値フィルター】")
    print(f"{'='*90}")

    for rk in ["S", "A"]:
        df_rk = df_sa[df_sa["rank"] == rk]
        n_base = len(df_rk)
        print(f"\n  ── {rk}ランク (全{n_base}R) ──")
        print(f"  {'閾値':<22} {'推奨R':>6} {'1日avg':>7}  "
              f"{'的中率':>7} {'ROI':>7} {'損益':>10}  avg配当  中央値  <300円%  ≥600円%")
        print(f"  {'-'*85}")

        for thr in thresholds:
            if thr is None:
                sub = df_rk.copy()
                label = "全件"
            else:
                sub = df_rk[(df_rk["q_pivot"].isna()) | (df_rk["q_pivot"] >= thr)].copy()
                label = f"q≥{thr}円"

            n    = len(sub)
            hits = int(sub["hit"].sum())
            bet  = sub["bet"].sum()
            ret  = sub["payout"].sum()
            roi  = ret / bet if bet else 0
            n_days_sub = sub["race_date"].nunique()
            avg_per_day = n / n_days_sub if n_days_sub else 0
            hit_rows = sub[sub["hit"]]
            avg_pay = hit_rows["payout"].mean() if hits else 0
            med_pay = hit_rows["payout"].median() if hits else 0
            low_pct = (hit_rows["payout"] < 300).mean() if hits else 0
            hi_pct  = (hit_rows["payout"] >= 600).mean() if hits else 0

            print(f"  {label:<22} {n:>6} {avg_per_day:>7.1f}R  "
                  f"{hits/n:>7.1%} {roi:>7.1%} {ret-bet:>+10,}  "
                  f"{avg_pay:>6,.0f}円  {med_pay:>6,.0f}円  {low_pct:>7.1%}  {hi_pct:>7.1%}")

    # ─── 月別 (S/A, quinella≥300 フィルター適用) ─────────────────────────
    print(f"\n{'='*90}")
    print("  【月別 S/A (quinella≥300円フィルター適用) ← 推奨ライン候補】")
    print(f"{'='*90}")
    thr_main = 300
    df_sa_f = df_sa[(df_sa["q_pivot"].isna()) | (df_sa["q_pivot"] >= thr_main)].copy()
    df_sa_f["ym"] = df_sa_f["race_date"].str[:7]

    print(f"  {'月':<8} {'推奨R':>6} {'1日avg':>7}  "
          f"{'的中':>5} {'的中率':>7} {'投資':>10} {'回収':>10} {'ROI':>7} {'損益':>10}")
    print(f"  {'-'*80}")
    for ym, g in df_sa_f.groupby("ym"):
        n    = len(g)
        hits = int(g["hit"].sum())
        bet  = g["bet"].sum()
        ret  = g["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = g["race_date"].nunique()
        avg  = n / nd if nd else 0
        print(f"  {ym:<8} {n:>6} {avg:>7.1f}R  "
              f"{hits:>5} {hits/n:>7.1%} {bet:>10,} {ret:>10,} {roi:>7.1%} {ret-bet:>+10,}")
    # 合計
    bet_t = df_sa_f["bet"].sum()
    ret_t = df_sa_f["payout"].sum()
    hits_t = int(df_sa_f["hit"].sum())
    nd_t  = df_sa_f["race_date"].nunique()
    print(f"  {'合計':<8} {len(df_sa_f):>6} {len(df_sa_f)/nd_t:>7.1f}R  "
          f"{hits_t:>5} {hits_t/len(df_sa_f):>7.1%} {bet_t:>10,} {ret_t:>10,} "
          f"{ret_t/bet_t if bet_t else 0:>7.1%} {ret_t-bet_t:>+10,}")
    print(f"{'='*90}")

    # ─── SS も含めた全体サマリ（quinella≥300 フィルター後） ──────────────
    print(f"\n{'='*90}")
    print(f"  【全ランク合計 (S/Aはquinella≥{thr_main}円 / SSは全件)】")
    print(f"{'='*90}")
    df_all_f = pd.concat([df_ss, df_sa_f[["race_key","race_date","rank","hit","payout","bet"]]])
    for rk in ["SS", "S", "A", "合計"]:
        if rk == "合計":
            g = df_all_f
        else:
            g = df_all_f[df_all_f["rank"] == rk]
        n = len(g)
        if n == 0:
            continue
        hits = int(g["hit"].sum())
        bet  = g["bet"].sum()
        ret  = g["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = g["race_date"].nunique()
        print(f"  {rk:<5} {n:>5}R  {n/nd:>5.1f}R/日  的中率 {hits/n:>6.1%}  "
              f"ROI {roi:>7.1%}  損益 {ret-bet:>+10,}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
