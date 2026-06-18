"""軸精度・三連複 vs 三連単 条件付き戦略検証

ユーザー指摘:
  「的中/不的中の前に軸が想定通り3着以内になっているかを先に見る」
  「1着が予想できる場合、三連複はオッズが期待できない → 三連単の検討が必要」
  「gap12 大(指数差大) → 三連単 / gap12 小(拮抗) → 見送り or 2・3着軸」

検証内容:
  Part1: 軸精度分析（pred1/pred2 の3着以内率を gap12 帯別に確認）
  Part2: 三連複 vs 三連単 ROI 比較（複数パターン）
  Part3: 条件付き戦略（gap12 帯で三連単/三連複/見送りを切り替え）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-14
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.database import get_connection
from src.evaluation.backtest_wt import _assign_tier
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0  # 最安目 trio オッズ足切り（倍）
GAP12_LARGE = 0.10    # 三連単推奨閾値（指数差が十分大きい）
GAP12_SMALL = 0.06    # 拮抗判定閾値（見送り検討）


# ─── データ準備 ──────────────────────────────────────────────────────────────

def _parse_combo(s, sep="="):
    parts = re.split(r"[-=]", str(s))
    try:
        frames = [int(p) for p in parts]
        return tuple(frames) if sep == "-" else frozenset(frames)
    except Exception:
        return None


def load_all_data():
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries FROM wt_races", conn)
        trio_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'",
            conn,
        )
        tri_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trifecta'",
            conn,
        )

    df = df.merge(races_info, on="race_key", how="left")

    # リーク無しモデル学習
    print("  リーク無しモデル学習中 (TRAIN 期間のみ)...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    X_tr = prepare_X(fit).reset_index(drop=True)
    y_tr = fit["top3_flag"].reset_index(drop=True).values
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)
    print(f"  学習完了 ({len(fit):,} 行)")

    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # trio オッズ辞書（frozenset → 円）
    trio_df["combo_key"] = trio_df["combination"].apply(lambda s: _parse_combo(s, "="))
    trio_df = trio_df.dropna(subset=["combo_key"])
    trio_odds_map = {}
    for row in trio_df.itertuples(index=False):
        trio_odds_map.setdefault(row.race_key, {})[row.combo_key] = row.odds_value * 100

    # trifecta オッズ辞書（tuple (1st,2nd,3rd) → 円）
    tri_df["combo_key"] = tri_df["combination"].apply(lambda s: _parse_combo(s, "-"))
    tri_df = tri_df.dropna(subset=["combo_key"])
    trifecta_odds_map = {}
    for row in tri_df.itertuples(index=False):
        trifecta_odds_map.setdefault(row.race_key, {})[row.combo_key] = row.odds_value * 100

    # 実際の結果（trio 用）
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )
    # 実際の結果（trifecta 用: (1st, 2nd, 3rd) タプル）
    actual_trifecta = {}
    for rk, grp in df[df["finish_order"].between(1, 3)].groupby("race_key"):
        ordered = grp.sort_values("finish_order")["frame_no"].astype(int).tolist()
        if len(ordered) == 3:
            actual_trifecta[rk] = tuple(ordered)

    return df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta


# ─── レース単位の指標計算 ─────────────────────────────────────────────────────

def race_metrics(grp):
    """1レースについて pred1/pred2・gap12・tier 等を返す。"""
    grp = grp.sort_values("pred_prob", ascending=False)
    rows = grp.reset_index(drop=True)
    if len(rows) < 3:
        return None
    p1, p2 = rows.iloc[0]["pred_prob"], rows.iloc[1]["pred_prob"]
    gap12 = p1 - p2
    n = int(rows.iloc[0].get("n_entries", len(rows)))
    ratio = p1 / (3 / n)
    tier = _assign_tier(gap12, ratio)
    pred1_frame = int(rows.iloc[0]["frame_no"])
    pred2_frame = int(rows.iloc[1]["frame_no"])
    thirds = [int(r["frame_no"]) for _, r in rows.iloc[2:].iterrows()]
    return {
        "gap12": gap12,
        "tier": tier,
        "pred1_frame": pred1_frame,
        "pred2_frame": pred2_frame,
        "thirds": thirds,
        "r1_prob": p1,
        "r2_prob": p2,
    }


# ─── Part1: 軸精度分析 ────────────────────────────────────────────────────────

def analyze_axis_accuracy(df, trio_odds_map, actual_trio, label=""):
    """pred1/pred2 の3着以内率を gap12 帯別に集計する。"""
    records = []
    for race_key, grp in df.groupby("race_key"):
        m = race_metrics(grp)
        if m is None:
            continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6:
            continue

        # ガミ足切り（trio 基準）
        combos_trio = [frozenset({m["pred1_frame"], m["pred2_frame"], t})
                       for t in m["thirds"]]
        race_odds = trio_odds_map.get(race_key, {})
        min_odds = min((race_odds.get(k, 0) for k in combos_trio if race_odds.get(k, 0) > 0),
                       default=0)
        if min_odds < GAMI_THRESHOLD * 100:
            continue

        actual = actual_trio.get(race_key, frozenset())
        p1_top3 = int(m["pred1_frame"] in actual)
        p2_top3 = int(m["pred2_frame"] in actual)
        both_top3 = int(p1_top3 and p2_top3)

        # gap12 帯
        g = m["gap12"]
        if g < 0.06:
            band = "<0.06"
        elif g < 0.10:
            band = "0.06-0.10"
        elif g < 0.15:
            band = "0.10-0.15"
        else:
            band = "0.15+"

        records.append({
            "gap12_band": band,
            "p1_top3": p1_top3,
            "p2_top3": p2_top3,
            "both_top3": both_top3,
        })

    recs = pd.DataFrame(records)
    bands_order = ["<0.06", "0.06-0.10", "0.10-0.15", "0.15+"]
    print(f"\n{'=' * 60}")
    print(f"Part1: 軸精度（≤6車・ガミ≥5倍 対象レース）{label}")
    print(f"{'=' * 60}")
    print(f"{'gap12帯':<12} {'n':>6} {'pred1 top3':>12} {'pred2 top3':>12} {'両方 top3':>12}")
    print("-" * 60)
    for band in bands_order:
        sub = recs[recs["gap12_band"] == band]
        if len(sub) == 0:
            continue
        n = len(sub)
        p1r = sub["p1_top3"].mean() * 100
        p2r = sub["p2_top3"].mean() * 100
        br = sub["both_top3"].mean() * 100
        print(f"{band:<12} {n:>6} {p1r:>11.1f}% {p2r:>11.1f}% {br:>11.1f}%")
    n = len(recs)
    p1r = recs["p1_top3"].mean() * 100
    p2r = recs["p2_top3"].mean() * 100
    br = recs["both_top3"].mean() * 100
    print(f"{'合計':<12} {n:>6} {p1r:>11.1f}% {p2r:>11.1f}% {br:>11.1f}%")


# ─── Part2 & 3: 戦略 ROI 計算 ─────────────────────────────────────────────────

class Strategy:
    def __init__(self, name):
        self.name = name
        self.records = []

    def add(self, period, hit_pay, n_bets, trio_cost):
        """hit_pay=的中払戻(0=不的中), n_bets=買い目点数, trio_cost=1点コスト（100円）"""
        self.records.append({"period": period, "hit_pay": hit_pay,
                              "cost": n_bets * trio_cost, "hit": int(hit_pay > 0)})

    def summary(self):
        df = pd.DataFrame(self.records)
        rows = []
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = df[df["period"] == period]
            if len(sub) == 0:
                continue
            total_cost = sub["cost"].sum()
            total_pay = sub["hit_pay"].sum()
            roi = total_pay / total_cost * 100 if total_cost > 0 else 0
            n_races = len(sub[sub["cost"] > 0])
            hits = sub["hit"].sum()
            rows.append({"period": period, "対象R": n_races, "的中": hits, "ROI": roi})
        return pd.DataFrame(rows)


def get_period(race_date):
    if TRAIN[0] <= race_date <= TRAIN[1]:
        return "TRAIN"
    if VAL[0] <= race_date <= VAL[1]:
        return "VAL"
    if HOLD[0] <= race_date <= HOLD[1]:
        return "HOLD"
    return None


def evaluate_strategies(df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta):
    """4戦略の ROI を計算する。"""
    # S0: 現行 trio (pred1+pred2→thirds)
    # S1: trifecta pred1 1着固定 → pred2 2着 → thirds 3着
    # S2: trifecta pred1/pred2 1-2着BOX → thirds (6点)
    # S3: 条件付き(gap12≥0.10→S2, 0.06-0.10→S0, <0.06→見送り)
    strats = {k: Strategy(k) for k in ["S0_trio", "S1_tri_p1fix", "S2_tri_box", "S3_cond"]}

    for race_key, grp in df.groupby("race_key"):
        period = get_period(grp["race_date"].iloc[0])
        if period is None:
            continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6:
            continue

        m = race_metrics(grp)
        if m is None:
            continue

        p1, p2 = m["pred1_frame"], m["pred2_frame"]
        thirds = m["thirds"]
        gap12 = m["gap12"]

        # ガミ足切り（trio 基準）
        race_trio_odds = trio_odds_map.get(race_key, {})
        combos_trio = [frozenset({p1, p2, t}) for t in thirds]
        min_trio = min((race_trio_odds.get(k, 0) for k in combos_trio
                        if race_trio_odds.get(k, 0) > 0), default=0)
        if min_trio < GAMI_THRESHOLD * 100:
            continue

        actual_t = actual_trio.get(race_key, frozenset())
        actual_tri = actual_trifecta.get(race_key)
        race_trifecta_odds = trifecta_odds_map.get(race_key, {})

        # ── S0: trio 2軸流し ──
        pay_s0 = 0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual_t == k:
                pay_s0 = race_trio_odds.get(k, 0)
                break
        strats["S0_trio"].add(period, pay_s0, len(thirds), 100)

        # ── S1: trifecta pred1 1着固定 → pred2 2着 → thirds 3着 ──
        pay_s1 = 0
        s1_bets = [(p1, p2, t) for t in thirds]
        for bet in s1_bets:
            if actual_tri == bet:
                pay_s1 = race_trifecta_odds.get(bet, 0)
                break
        strats["S1_tri_p1fix"].add(period, pay_s1, len(s1_bets), 100)

        # ── S2: trifecta pred1/pred2 1-2着BOX → thirds ──
        pay_s2 = 0
        s2_bets = [(p1, p2, t) for t in thirds] + [(p2, p1, t) for t in thirds]
        for bet in s2_bets:
            if actual_tri == bet:
                pay_s2 = race_trifecta_odds.get(bet, 0)
                break
        strats["S2_tri_box"].add(period, pay_s2, len(s2_bets), 100)

        # ── S3: 条件付き ──
        if gap12 >= GAP12_LARGE:
            # 三連単 BOX
            pay_s3 = pay_s2
            cost_s3 = len(s2_bets)
        elif gap12 >= GAP12_SMALL:
            # 三連複（現行）
            pay_s3 = pay_s0
            cost_s3 = len(thirds)
        else:
            # 見送り
            strats["S3_cond"].add(period, 0, 0, 100)
            continue
        strats["S3_cond"].add(period, pay_s3, cost_s3, 100)

    return strats


# ─── gap12 帯別 ROI 内訳 ─────────────────────────────────────────────────────

def gap12_breakdown(df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta):
    """gap12 帯別に S0 / S2 の ROI を比較（VAL+HOLD のみ）。"""
    bins = [0, 0.06, 0.10, 0.15, 1.0]
    labels = ["<0.06", "0.06-0.10", "0.10-0.15", "0.15+"]

    records = []
    for race_key, grp in df.groupby("race_key"):
        period = get_period(grp["race_date"].iloc[0])
        if period not in ("VAL", "HOLD"):
            continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6:
            continue
        m = race_metrics(grp)
        if m is None:
            continue

        p1, p2, thirds, gap12 = m["pred1_frame"], m["pred2_frame"], m["thirds"], m["gap12"]
        race_trio_odds = trio_odds_map.get(race_key, {})
        combos_trio = [frozenset({p1, p2, t}) for t in thirds]
        min_trio = min((race_trio_odds.get(k, 0) for k in combos_trio
                        if race_trio_odds.get(k, 0) > 0), default=0)
        if min_trio < GAMI_THRESHOLD * 100:
            continue

        actual_t = actual_trio.get(race_key, frozenset())
        actual_tri = actual_trifecta.get(race_key)
        race_trifecta_odds = trifecta_odds_map.get(race_key, {})

        pay_s0 = 0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual_t == k:
                pay_s0 = race_trio_odds.get(k, 0)
                break

        s2_bets = [(p1, p2, t) for t in thirds] + [(p2, p1, t) for t in thirds]
        pay_s2 = 0
        for bet in s2_bets:
            if actual_tri == bet:
                pay_s2 = race_trifecta_odds.get(bet, 0)
                break

        result = pd.cut([gap12], bins=bins, labels=labels)
        if pd.isna(result[0]):
            continue
        band = str(result[0])
        records.append({
            "band": band,
            "s0_pay": pay_s0, "s0_cost": len(thirds) * 100,
            "s0_hit": int(pay_s0 > 0),
            "s2_pay": pay_s2, "s2_cost": len(s2_bets) * 100,
            "s2_hit": int(pay_s2 > 0),
        })

    recs = pd.DataFrame(records)
    print(f"\n{'=' * 72}")
    print("Part2 補足: gap12 帯別 S0(trio) vs S2(三連単BOX)  [VAL+HOLD]")
    print(f"{'=' * 72}")
    header = f"{'gap12帯':<12} {'n':>5}  {'S0-trio hit%':>14} {'S0 ROI':>8}  {'S2-tri_box hit%':>16} {'S2 ROI':>8}"
    print(header)
    print("-" * 72)
    for band in labels:
        sub = recs[recs["band"] == band]
        if len(sub) == 0:
            continue
        n = len(sub)
        s0_roi = sub["s0_pay"].sum() / sub["s0_cost"].sum() * 100
        s2_roi = sub["s2_pay"].sum() / sub["s2_cost"].sum() * 100
        s0_hit = sub["s0_hit"].mean() * 100
        s2_hit = sub["s2_hit"].mean() * 100
        mark_s0 = "★" if s0_roi >= 100 else " "
        mark_s2 = "★" if s2_roi >= 100 else " "
        print(f"{band:<12} {n:>5}  {s0_hit:>12.1f}% {s0_roi:>7.1f}%{mark_s0}  "
              f"{s2_hit:>14.1f}% {s2_roi:>7.1f}%{mark_s2}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    print("軸精度・三連複 vs 三連単 条件付き戦略検証")
    print(f"  三連単推奨閾値 gap12≥{GAP12_LARGE} / 拮抗見送り gap12<{GAP12_SMALL}")
    print()
    print("データ準備中...")
    df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta = load_all_data()

    # Part1: 軸精度
    for period, lo, hi in [("TRAIN+VAL+HOLD", TRAIN[0], HOLD[1]),
                            ("VAL+HOLD のみ", VAL[0], HOLD[1])]:
        sub = df[(df["race_date"] >= lo) & (df["race_date"] <= hi)]
        analyze_axis_accuracy(sub, trio_odds_map, actual_trio, f" [{period}]")

    # Part2/3: 戦略 ROI
    strats = evaluate_strategies(df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta)

    print(f"\n{'=' * 72}")
    print("Part2: 戦略別 ROI（全3期間）")
    print(f"{'=' * 72}")
    print(f"  {'戦略':<24} {'期間':<8} {'対象R':>7} {'的中':>6} {'ROI':>8}")
    print("  " + "-" * 58)
    for key, strat in strats.items():
        summ = strat.summary()
        for _, row in summ.iterrows():
            mark = "★" if row["ROI"] >= 100 else ""
            print(f"  {strat.name:<24} {row['period']:<8} {row['対象R']:>7} "
                  f"{row['的中']:>6} {row['ROI']:>7.1f}%{mark}")
        print()

    print(f"\n{'=' * 72}")
    print("Part3: S3 条件付き戦略の内訳")
    print(f"  gap12≥{GAP12_LARGE}: 三連単 BOX(6点)")
    print(f"  {GAP12_SMALL}≤gap12<{GAP12_LARGE}: 三連複 2軸流し(3点)")
    print(f"  gap12<{GAP12_SMALL}: 見送り")
    print(f"{'=' * 72}")

    # gap12帯別の内訳
    gap12_breakdown(df, trio_odds_map, trifecta_odds_map, actual_trio, actual_trifecta)

    print(f"\n{'=' * 72}")
    print("解釈のヒント")
    print(f"{'=' * 72}")
    print("  ・軸精度 = pred1/pred2 の3着以内率。これが低い帯では的中構造的に低い。")
    print("  ・S0 vs S2 の ROI 比較で三連単が有利な帯を特定する。")
    print("  ・S3 は見送り(gap12小)で損失を回避しつつ gap12大帯に三連単を充当。")
    print("  ・全て最終オッズ上限値。採否判断は live 実測のみ。")


if __name__ == "__main__":
    main()
