"""
予想販売向け: S/A 3連複 低配当フィルター分析（確定前情報のみ使用）

確定後オッズは不使用。モデルの確信度指標（確定前情報）で
低配当レースを予測・見送りできるか検証する。

■ 確定前に使える代理指標
  top2_prob_sum = pivot1 + pivot2 の pred_prob 合計
    → 高いほど AI がこの2頭に集中 = 市場も集中 ≒ 3連複低配当になりやすい
  ratio = top1_prob / (3/n_riders)
    → S ランクで既に ≥1.3。さらに高いほど1頭独占
  top3_prob_sum = 上位3頭の pred_prob 合計
    → 高いほど3頭内に確率集中 = 組み合わせが読まれている = 低配当
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
SA_PTS       = 3


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
    pay_map = load_result_payouts(all_keys)

    # ─── レース単位で集計 ──────────────────────────────────────────────────
    rows = []
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        probs = grp["pred_prob"].tolist()
        ranked = grp["frame_no"].tolist()
        n = len(ranked)
        top1 = probs[0]
        top2 = probs[1] if n >= 2 else 0.0
        gap12 = top1 - top2
        ratio = top1 / (3 / n)
        rank = classify_rank(gap12, ratio)
        if rank == "SKIP":
            continue

        top2_sum  = top1 + top2
        top3_sum  = top1 + top2 + (probs[2] if n >= 3 else 0.0)

        if rank == "SS":
            hit, payout = eval_ss(grp, pay_map)
            bet = 3 * UNIT
        else:
            hit, payout = eval_sa(grp, pay_map)
            bet = SA_PTS * UNIT

        rows.append({
            "race_key":  race_key,
            "race_date": race_date_map[race_key],
            "rank":      rank,
            "hit":       hit,
            "payout":    payout,
            "bet":       bet,
            "top1":      top1,
            "top2":      top2,
            "gap12":     gap12,
            "ratio":     ratio,
            "top2_sum":  top2_sum,
            "top3_sum":  top3_sum,
        })

    df_r  = pd.DataFrame(rows)
    df_sa = df_r[df_r["rank"].isin(["S", "A"])].copy()
    df_ss = df_r[df_r["rank"] == "SS"].copy()
    n_days = df_r["race_date"].nunique()

    # ─── 確定前指標 × 3連複配当の相関確認 ──────────────────────────────
    df_sa_hit = df_sa[df_sa["hit"]].copy()

    print(f"\n対象期間: {HOLDOUT_FROM} ~ {HOLDOUT_TO}  ({n_days}日)")
    print(f"S/A 全体: {len(df_sa)}R / 的中: {len(df_sa_hit)}R")

    print(f"\n  【top2_sum（軸2頭 pred_prob 合計）× 3連複配当 相関】")
    print(f"  ※ 確定前に計算可能な市場集中度の代理指標")
    print(f"  {'top2_sum帯':<16} {'的中R':>6}  {'3連複avg':>9}  {'中央値':>8}  "
          f"{'<300円%':>8}  {'≥600円%':>8}")
    print(f"  {'-'*60}")
    bins   = [0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    labels = ["~0.50", "0.51~0.60", "0.61~0.65", "0.66~0.70",
              "0.71~0.75", "0.76~0.80", "0.81~"]
    df_sa_hit["t2_band"] = pd.cut(df_sa_hit["top2_sum"], bins=bins, labels=labels, right=True)
    for band in labels:
        g = df_sa_hit[df_sa_hit["t2_band"] == band]
        if len(g) < 3:
            continue
        avg = g["payout"].mean()
        med = g["payout"].median()
        low = (g["payout"] < 300).mean()
        hi  = (g["payout"] >= 600).mean()
        print(f"  {band:<16} {len(g):>6}  {avg:>9,.0f}円  {med:>8,.0f}円  "
              f"{low:>8.1%}  {hi:>8.1%}")

    # ─── top2_sum フィルター閾値別 ────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  【S/A: top2_sum（確定前）フィルター別 推奨実績】")
    print("  ※ top2_sum ≤ 閾値 は推奨しない（軸2頭の確信が強すぎる = 低配当リスク）")
    print(f"{'='*90}")
    print(f"  {'見送り条件':<22} {'推奨R':>6} {'1日avg':>7}  "
          f"{'見送R':>6} {'見送%':>6}  "
          f"{'的中率':>7} {'ROI':>7} {'損益':>10}  "
          f"avg配当  中央値  <300円%  ≥600円%")
    print(f"  {'-'*90}")

    base_n = len(df_sa)
    base_nd = df_sa["race_date"].nunique()

    # top2_sum が低い（≤閾値）ほど2頭の確信が強い → 低配当リスク
    # ただし top2_sum はあまり差別化できない可能性もある
    # 代わりに top3_sum も試す
    thresholds_top2 = [None, 0.80, 0.75, 0.70, 0.65, 0.60]

    for thr in thresholds_top2:
        if thr is None:
            sub = df_sa.copy()
            label = "全件(フィルタなし)"
        else:
            # top2_sum が高い（確信強い）レースを除外
            sub = df_sa[df_sa["top2_sum"] < thr].copy()
            label = f"top2_sum < {thr}"

        n    = len(sub)
        if n == 0:
            continue
        hits = int(sub["hit"].sum())
        bet  = sub["bet"].sum()
        ret  = sub["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = sub["race_date"].nunique()
        apd  = n / nd if nd else 0
        skip = base_n - n
        skip_pct = skip / base_n

        hit_rows = sub[sub["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        med_pay = hit_rows["payout"].median() if hits else 0
        low_pct = (hit_rows["payout"] < 300).mean() if hits else 0
        hi_pct  = (hit_rows["payout"] >= 600).mean() if hits else 0

        print(f"  {label:<22} {n:>6} {apd:>7.1f}R  "
              f"{skip:>6} {skip_pct:>6.1%}  "
              f"{hits/n:>7.1%} {roi:>7.1%} {ret-bet:>+10,}  "
              f"{avg_pay:>6,.0f}円  {med_pay:>6,.0f}円  {low_pct:>7.1%}  {hi_pct:>7.1%}")

    # ─── ratio フィルター（Sランク専用、ratioが既に≥1.3）─────────────────
    print(f"\n{'='*90}")
    print("  【Sランク: ratio（確定前）による追加フィルター】")
    print("  ※ ratio = top1_prob / (3/n)。高いほど1頭独占 → 低配当傾向")
    print(f"{'='*90}")
    df_s = df_sa[df_sa["rank"] == "S"].copy()
    print(f"  {'条件':<22} {'推奨R':>6} {'1日avg':>7}  "
          f"{'的中率':>7} {'ROI':>7} {'損益':>10}  avg配当  中央値  <300円%  ≥600円%")
    print(f"  {'-'*85}")

    for thr in [None, 1.5, 1.6, 1.7, 1.8, 2.0]:
        if thr is None:
            sub = df_s.copy()
            label = "S全件"
        else:
            sub = df_s[df_s["ratio"] < thr].copy()
            label = f"ratio < {thr}"
        n = len(sub)
        if n == 0:
            continue
        hits = int(sub["hit"].sum())
        bet  = sub["bet"].sum()
        ret  = sub["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = sub["race_date"].nunique()
        hit_rows = sub[sub["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        med_pay = hit_rows["payout"].median() if hits else 0
        low_pct = (hit_rows["payout"] < 300).mean() if hits else 0
        hi_pct  = (hit_rows["payout"] >= 600).mean() if hits else 0
        print(f"  {label:<22} {n:>6} {n/nd:>7.1f}R  "
              f"{hits/n:>7.1%} {roi:>7.1%} {ret-bet:>+10,}  "
              f"{avg_pay:>6,.0f}円  {med_pay:>6,.0f}円  {low_pct:>7.1%}  {hi_pct:>7.1%}")

    # ─── 組み合わせフィルター（実用候補）───────────────────────────────────
    print(f"\n{'='*90}")
    print("  【実用フィルター候補: S/A 条件組み合わせ】")
    print("  全条件は確定前情報のみ使用")
    print(f"{'='*90}")
    print(f"  {'条件':<30} {'推奨R':>6} {'1日avg':>7}  "
          f"{'的中率':>7} {'ROI':>7} {'損益':>10}  avg配当  中央値")
    print(f"  {'-'*85}")

    candidates = [
        ("全件(現行)",               df_sa),
        ("A全件 + S ratio<1.8",      pd.concat([
            df_sa[df_sa["rank"]=="A"],
            df_sa[(df_sa["rank"]=="S") & (df_sa["ratio"]<1.8)]
        ])),
        ("A全件 + S ratio<1.6",      pd.concat([
            df_sa[df_sa["rank"]=="A"],
            df_sa[(df_sa["rank"]=="S") & (df_sa["ratio"]<1.6)]
        ])),
        ("top2_sum<0.70",            df_sa[df_sa["top2_sum"]<0.70]),
        ("top2_sum<0.75",            df_sa[df_sa["top2_sum"]<0.75]),
        ("top2_sum<0.70 + S r<1.8",  df_sa[
            (df_sa["top2_sum"]<0.70) |
            ((df_sa["rank"]=="S") & (df_sa["ratio"]<1.8))
        ].drop_duplicates("race_key")),
    ]

    for label, sub in candidates:
        n = len(sub)
        if n == 0:
            continue
        hits = int(sub["hit"].sum())
        bet  = sub["bet"].sum()
        ret  = sub["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = sub["race_date"].nunique()
        hit_rows = sub[sub["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        med_pay = hit_rows["payout"].median() if hits else 0
        print(f"  {label:<30} {n:>6} {n/nd:>7.1f}R  "
              f"{hits/n:>7.1%} {roi:>7.1%} {ret-bet:>+10,}  "
              f"{avg_pay:>6,.0f}円  {med_pay:>6,.0f}円")

    # ─── 月別 最良候補（top2_sum < 0.75）─────────────────────────────────
    best_filter_sub = df_sa[df_sa["top2_sum"] < 0.75].copy()
    best_filter_sub["ym"] = best_filter_sub["race_date"].str[:7]

    print(f"\n{'='*90}")
    print(f"  【月別 S/A (top2_sum<0.75 フィルター)】")
    print(f"{'='*90}")
    print(f"  {'月':<8} {'推奨R':>6} {'1日avg':>7}  "
          f"{'的中':>5} {'的中率':>7} {'ROI':>7} {'損益':>10}  avg配当")
    print(f"  {'-'*72}")
    for ym, g in best_filter_sub.groupby("ym"):
        n    = len(g)
        hits = int(g["hit"].sum())
        bet  = g["bet"].sum()
        ret  = g["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = g["race_date"].nunique()
        apd  = n / nd if nd else 0
        hit_rows = g[g["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        print(f"  {ym:<8} {n:>6} {apd:>7.1f}R  "
              f"{hits:>5} {hits/n:>7.1%} {roi:>7.1%} {ret-bet:>+10,}  {avg_pay:>7,.0f}円")

    # 合計
    bet_t = best_filter_sub["bet"].sum()
    ret_t = best_filter_sub["payout"].sum()
    hits_t = int(best_filter_sub["hit"].sum())
    nd_t   = best_filter_sub["race_date"].nunique()
    hr = hits_t / len(best_filter_sub)
    hit_rows_t = best_filter_sub[best_filter_sub["hit"]]
    avg_t = hit_rows_t["payout"].mean() if hits_t else 0
    print(f"  {'合計':<8} {len(best_filter_sub):>6} {len(best_filter_sub)/nd_t:>7.1f}R  "
          f"{hits_t:>5} {hr:>7.1%} {ret_t/bet_t if bet_t else 0:>7.1%} "
          f"{ret_t-bet_t:>+10,}  {avg_t:>7,.0f}円")
    print(f"{'='*90}")

    # ─── 全ランク合計（SS + S/A top2_sum<0.75）───────────────────────────
    print(f"\n{'='*90}")
    print(f"  【全ランク (SS全件 + S/A top2_sum<0.75) — 予想サービス推奨ライン】")
    print(f"{'='*90}")
    df_all = pd.concat([
        df_ss,
        best_filter_sub[["race_key","race_date","rank","hit","payout","bet"]]
    ])
    for rk in ["SS", "S", "A", "合計"]:
        g = df_all if rk == "合計" else df_all[df_all["rank"] == rk]
        n = len(g)
        if n == 0:
            continue
        hits = int(g["hit"].sum())
        bet  = g["bet"].sum()
        ret  = g["payout"].sum()
        roi  = ret / bet if bet else 0
        nd   = g["race_date"].nunique()
        hit_rows = g[g["hit"]]
        avg_pay = hit_rows["payout"].mean() if hits else 0
        med_pay = hit_rows["payout"].median() if hits else 0
        print(f"  {rk:<5} {n:>5}R  {n/nd:>5.1f}R/日  "
              f"的中率 {hits/n:>6.1%}  ROI {roi:>7.1%}  "
              f"損益 {ret-bet:>+10,}  avg配当 {avg_pay:>7,.0f}円  中央値 {med_pay:>6,.0f}円")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
