"""三連単フォーメーション条件別ROI検証 (doc49)

doc33 では「全体」で三連単が不通過（S1 VAL 55%/HOLD 57%）。
本実験の視点:
  ① 帯別 S1(pred1→pred2→{3,4,5}) ROI — doc33 が S0 vs S2 のみで S1 帯別未測定
  ② 着順精度（1着率・条件付き2着率）— 「着固定」が成立する条件を特定
  ③ 点数削減フォーメーション — pred5 除去(2点)・pred3のみ(1点)がROI改善するか
  ④ gap23(pred2優位性)追加フィルタ — pred2の2着確度を高める追加条件
  ⑤ 特別推奨レースの収益評価 — n_bets / 日 × ROI で実運用インパクト

戦略一覧:
  S0: 現行 三連複 (ベースライン)
  S1_all: pred1→pred2→{3,4,5} 3点 (全対象)         ← doc33 S1 の再確認
  --- gap12 帯別切り出し ---
  T1_g10:  S1型・gap12≥0.10 のみ適用 (3点)
  T2_g12:  S1型・gap12≥0.12 のみ (3点)
  T3_g15:  S1型・gap12≥0.15 のみ (3点)
  --- 点数削減 ---
  T4_trim: pred1→pred2→{3,4} 2点・gap12≥0.10          ← pred5 除去
  T5_snap: pred1→pred2→pred3  1点・gap12≥0.10          ← 最タイト
  --- gap23 追加フィルタ ---
  T6_g23:  pred1→pred2→{3,4,5} 3点・gap12≥0.10 AND gap23≥0.05
  T7_g23t: pred1→pred2→{3,4} 2点・gap12≥0.10 AND gap23≥0.05
  --- open (2着軸を緩める) ---
  T8_open: pred1→{pred2,pred3}→{2,3,4,5} 6点・gap12≥0.10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ─── データ準備 ──────────────────────────────────────────────────────────────

def _parse_combo(s, ordered=False):
    import re
    parts = re.split(r"[-=]", str(s))
    try:
        nums = [int(p) for p in parts]
        return tuple(nums) if ordered else frozenset(nums)
    except Exception:
        return None


def load_all():
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        races_info = pd.read_sql(
            "SELECT race_key, n_entries, grade FROM wt_races", conn
        )
        trio_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'", conn
        )
        tri_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trifecta'", conn
        )

    df = df.merge(races_info, on="race_key", how="left")

    print("  TRAIN 期間のみでリーク無しモデル学習中...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    X_tr = prepare_X(fit).reset_index(drop=True)
    y_tr = fit["top3_flag"].reset_index(drop=True).values
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)
    print(f"  学習完了 ({len(fit):,} 行)", flush=True)

    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # trio オッズマップ {race_key: {frozenset: payout_yen}}
    trio_df["k"] = trio_df["combination"].apply(_parse_combo)
    trio_df = trio_df.dropna(subset=["k"])
    trio_map = {}
    for r in trio_df.itertuples(index=False):
        trio_map.setdefault(r.race_key, {})[r.k] = int(round(r.odds_value * 100))

    # trifecta オッズマップ {race_key: {tuple: payout_yen}}
    tri_df["k"] = tri_df["combination"].apply(lambda s: _parse_combo(s, ordered=True))
    tri_df = tri_df.dropna(subset=["k"])
    tri_map = {}
    for r in tri_df.itertuples(index=False):
        tri_map.setdefault(r.race_key, {})[r.k] = int(round(r.odds_value * 100))

    # 実際の着順
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )
    actual_trifecta = {}
    for rk, grp in df[df["finish_order"].between(1, 3)].groupby("race_key"):
        ordered = grp.sort_values("finish_order")["frame_no"].astype(int).tolist()
        if len(ordered) == 3:
            actual_trifecta[rk] = tuple(ordered)

    return df, trio_map, tri_map, actual_trio, actual_trifecta


# ─── レース指標 ──────────────────────────────────────────────────────────────

def race_metrics(grp):
    grp = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    if len(grp) < 3:
        return None
    probs = grp["pred_prob"].tolist()
    p1, p2, p3 = probs[0], probs[1], probs[2]
    gap12 = p1 - p2
    gap23 = p2 - p3
    n = int(grp["n_entries"].iloc[0]) if "n_entries" in grp else len(grp)
    ratio = p1 / (3 / n) if n > 0 else 0
    tier = _assign_tier(gap12, ratio)
    frames = grp["frame_no"].astype(int).tolist()
    return {
        "gap12": gap12,
        "gap23": gap23,
        "tier": tier,
        "pred1": frames[0],
        "pred2": frames[1],
        "pred3": frames[2],
        "thirds": frames[2:],     # pred3以降
        "thirds_top4": frames[2:6],  # pred3〜pred6（最大4人）
        "r1_prob": p1,
        "r2_prob": p2,
        "r3_prob": p3,
    }


def get_period(d):
    if TRAIN[0] <= d <= TRAIN[1]:
        return "TRAIN"
    if VAL[0] <= d <= VAL[1]:
        return "VAL"
    if HOLD[0] <= d <= HOLD[1]:
        return "HOLD"
    return None


# ─── Part1: 着順精度分析 ──────────────────────────────────────────────────────

def analyze_positional_accuracy(df, trio_map, actual_trio, actual_trifecta):
    """
    pred1 の 1着率、pred2 の 2着率（pred1=1着条件付き・非条件付き）を
    gap12 帯別に集計する。
    """
    bands_order = ["<0.06", "0.06-0.10", "0.10-0.12", "0.12-0.15", "0.15+"]

    def band_label(g):
        if g < 0.06:
            return "<0.06"
        elif g < 0.10:
            return "0.06-0.10"
        elif g < 0.12:
            return "0.10-0.12"
        elif g < 0.15:
            return "0.12-0.15"
        else:
            return "0.15+"

    records = []
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

        # ガミ足切り（trio基準）
        p1, p2, thirds = m["pred1"], m["pred2"], m["thirds"]
        race_trio = trio_map.get(race_key, {})
        combos_trio = [frozenset({p1, p2, t}) for t in thirds]
        min_odds = min((race_trio.get(k, 0) for k in combos_trio
                        if race_trio.get(k, 0) > 0), default=0)
        if min_odds < GAMI_THRESHOLD * 100:
            continue

        actual_tri = actual_trifecta.get(race_key)
        if actual_tri is None:
            continue

        band = band_label(m["gap12"])
        p1_is_1st = int(actual_tri[0] == p1)
        p2_is_2nd = int(actual_tri[1] == p2)
        p2_is_2nd_given_p1_1st = int(actual_tri[0] == p1 and actual_tri[1] == p2)
        actual_t = actual_trio.get(race_key, frozenset())
        p1_top3 = int(p1 in actual_t)
        p2_top3 = int(p2 in actual_t)

        records.append({
            "period": period,
            "band": band,
            "gap12": m["gap12"],
            "gap23": m["gap23"],
            "p1_1st": p1_is_1st,
            "p1_top3": p1_top3,
            "p2_top3": p2_top3,
            "p2_2nd": p2_is_2nd,
            "p1p2_exact": p2_is_2nd_given_p1_1st,  # pred1=1着 AND pred2=2着
        })

    recs = pd.DataFrame(records)
    val_hold = recs[recs["period"].isin(["VAL", "HOLD"])]

    print(f"\n{'=' * 78}")
    print("Part1: 着順精度（≤6車・ガミ≥5倍・VAL+HOLD）")
    print(f"{'=' * 78}")
    hdr = (f"{'gap12帯':<14} {'n':>5}  {'pred1 top3%':>12} {'pred1 1着%':>11}  "
           f"{'pred2 top3%':>12} {'pred2 2着%':>11} {'P1=1着&P2=2着%':>15}")
    print(hdr)
    print("-" * 78)
    for band in bands_order:
        sub = val_hold[val_hold["band"] == band]
        if len(sub) == 0:
            continue
        n = len(sub)
        print(f"{band:<14} {n:>5}  "
              f"{sub['p1_top3'].mean()*100:>11.1f}% "
              f"{sub['p1_1st'].mean()*100:>10.1f}%  "
              f"{sub['p2_top3'].mean()*100:>11.1f}% "
              f"{sub['p2_2nd'].mean()*100:>10.1f}% "
              f"{sub['p1p2_exact'].mean()*100:>14.1f}%")
    n = len(val_hold)
    print(f"{'合計':<14} {n:>5}  "
          f"{val_hold['p1_top3'].mean()*100:>11.1f}% "
          f"{val_hold['p1_1st'].mean()*100:>10.1f}%  "
          f"{val_hold['p2_top3'].mean()*100:>11.1f}% "
          f"{val_hold['p2_2nd'].mean()*100:>10.1f}% "
          f"{val_hold['p1p2_exact'].mean()*100:>14.1f}%")

    # gap23 サブ分析（gap12≥0.10帯のみ）
    sub_g10 = val_hold[val_hold["gap12"] >= 0.10]
    if len(sub_g10) > 0:
        print(f"\n  【gap12≥0.10帯 × gap23 分位】 (VAL+HOLD, n={len(sub_g10)})")
        q33, q67 = np.percentile(sub_g10["gap23"], [33, 67])
        for label, mask in [
            (f"gap23<{q33:.3f}(下位1/3)", sub_g10["gap23"] < q33),
            (f"gap23 {q33:.3f}-{q67:.3f}(中)", (sub_g10["gap23"] >= q33) & (sub_g10["gap23"] < q67)),
            (f"gap23≥{q67:.3f}(上位1/3)", sub_g10["gap23"] >= q67),
        ]:
            s = sub_g10[mask]
            if len(s) == 0:
                continue
            print(f"    {label:<28}  n={len(s):>3}  "
                  f"p1_1st={s['p1_1st'].mean()*100:.1f}%  "
                  f"p2_2nd={s['p2_2nd'].mean()*100:.1f}%  "
                  f"p1&p2_exact={s['p1p2_exact'].mean()*100:.1f}%")

    return recs


# ─── Part2: 戦略ROI ──────────────────────────────────────────────────────────

class Strat:
    def __init__(self, name):
        self.name = name
        self.rec = []

    def add(self, period, pay, n_bets, unit=100):
        if n_bets == 0:
            return
        self.rec.append({"period": period, "pay": pay, "cost": n_bets * unit,
                          "hit": int(pay > 0)})

    def summary(self):
        df = pd.DataFrame(self.rec)
        rows = []
        for p in ["TRAIN", "VAL", "HOLD"]:
            sub = df[df["period"] == p]
            if len(sub) == 0:
                continue
            cost = sub["cost"].sum()
            pay = sub["pay"].sum()
            roi = pay / cost * 100 if cost > 0 else 0
            n_hit = sub["hit"].sum()
            n_races = len(sub)
            hit_pct = n_hit / n_races * 100 if n_races > 0 else 0
            rows.append({"period": p, "n": n_races, "hit%": hit_pct, "ROI%": roi})
        return pd.DataFrame(rows)


def evaluate(df, trio_map, tri_map, actual_trio, actual_trifecta):
    strategies = {
        "S0_trio":       Strat("S0 三連複(現行)"),
        "S1_all":        Strat("S1 三連単P1→P2→{3-5} 全対象"),
        "T1_g10":        Strat("T1 三連単P1→P2→{3-5} gap12≥0.10"),
        "T2_g12":        Strat("T2 三連単P1→P2→{3-5} gap12≥0.12"),
        "T3_g15":        Strat("T3 三連単P1→P2→{3-5} gap12≥0.15"),
        "T4_trim":       Strat("T4 三連単P1→P2→{3,4} gap12≥0.10 (2点)"),
        "T5_snap":       Strat("T5 三連単P1→P2→P3   gap12≥0.10 (1点)"),
        "T6_g23":        Strat("T6 三連単P1→P2→{3-5} gap12≥0.10 & gap23≥0.05"),
        "T7_g23t":       Strat("T7 三連単P1→P2→{3,4} gap12≥0.10 & gap23≥0.05 (2点)"),
        "T8_open":       Strat("T8 三連単P1→{P2,P3}→{2-5} gap12≥0.10 (6点)"),
        "H_hybrid":      Strat("H  ハイブリッド: gap12≥0.10→T4/それ以外→S0"),
    }

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

        p1, p2, p3 = m["pred1"], m["pred2"], m["pred3"]
        thirds = m["thirds"]       # pred3以降 (最大4名)
        thirds_top4 = m["thirds_top4"]
        gap12, gap23 = m["gap12"], m["gap23"]

        # ガミ足切り（trio 基準）
        race_trio = trio_map.get(race_key, {})
        combos_trio = [frozenset({p1, p2, t}) for t in thirds]
        min_odds = min((race_trio.get(k, 0) for k in combos_trio
                        if race_trio.get(k, 0) > 0), default=0)
        if min_odds < GAMI_THRESHOLD * 100:
            continue

        actual_t = actual_trio.get(race_key, frozenset())
        actual_tri = actual_trifecta.get(race_key)
        race_tri = tri_map.get(race_key, {})

        # ── S0: 現行 三連複 ──
        pay_s0 = 0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual_t == k:
                pay_s0 = race_trio.get(k, 0)
                break
        strategies["S0_trio"].add(period, pay_s0, len(thirds))

        # ── 三連単の的中判定ヘルパー ──
        def tri_pay(bets):
            for bet in bets:
                if actual_tri == bet:
                    return race_tri.get(bet, 0)
            return 0

        # ── S1: 全対象で pred1→pred2→{thirds} ──
        s1_bets = [(p1, p2, t) for t in thirds]
        strategies["S1_all"].add(period, tri_pay(s1_bets), len(s1_bets))

        # ── T1-T3: gap12 閾値別 ──
        for key, thr in [("T1_g10", 0.10), ("T2_g12", 0.12), ("T3_g15", 0.15)]:
            if gap12 >= thr:
                strategies[key].add(period, tri_pay(s1_bets), len(s1_bets))

        # ── T4: 2点 (pred5 除去: {pred3,pred4}のみ) ──
        t4_bets = [(p1, p2, t) for t in thirds[:2]]  # pred3・pred4 のみ
        if gap12 >= 0.10 and len(t4_bets) >= 1:
            strategies["T4_trim"].add(period, tri_pay(t4_bets), len(t4_bets))

        # ── T5: 1点 (pred3のみ) ──
        t5_bets = [(p1, p2, p3)]
        if gap12 >= 0.10:
            strategies["T5_snap"].add(period, tri_pay(t5_bets), 1)

        # ── T6/T7: gap23 追加フィルタ ──
        if gap12 >= 0.10 and gap23 >= 0.05:
            strategies["T6_g23"].add(period, tri_pay(s1_bets), len(s1_bets))
            if len(t4_bets) >= 1:
                strategies["T7_g23t"].add(period, tri_pay(t4_bets), len(t4_bets))

        # ── T8: open (pred1 1着固定・2着=pred2 or pred3) ──
        # pred1→pred2→{pred3,pred4,pred5}, pred1→pred3→{pred2,pred4,pred5}
        if gap12 >= 0.10:
            open_bets = (
                [(p1, p2, t) for t in thirds] +
                [(p1, p3, t) for t in [p2] + [x for x in thirds if x != p3]]
            )
            strategies["T8_open"].add(period, tri_pay(open_bets), len(open_bets))

        # ── H: ハイブリッド ──
        if gap12 >= 0.10:
            # T4 (三連単 2点) を特別推奨として使用
            strategies["H_hybrid"].add(period, tri_pay(t4_bets), len(t4_bets))
        else:
            # 現行 三連複
            strategies["H_hybrid"].add(period, pay_s0, len(thirds))

    return strategies


# ─── Part3: gap12 帯別 S0 vs S1 内訳 ────────────────────────────────────────

def gap12_band_breakdown(df, trio_map, tri_map, actual_trio, actual_trifecta):
    """VAL+HOLD で gap12 帯 × S0/S1/T4/T5 の ROI を細かく表示"""
    bands = [
        ("<0.06",      0.00, 0.06),
        ("0.06-0.10",  0.06, 0.10),
        ("0.10-0.12",  0.10, 0.12),
        ("0.12-0.15",  0.12, 0.15),
        ("0.15+",      0.15, 1.00),
    ]
    cols = ["band", "n",
            "s0_hit%", "s0_roi",
            "s1_hit%", "s1_roi",
            "t4_hit%", "t4_roi",
            "t5_hit%", "t5_roi"]
    rows = []

    for band_label, lo, hi in bands:
        rec = {"band": band_label, "n": 0,
               "s0_pay": 0, "s0_cost": 0, "s0_hit": 0,
               "s1_pay": 0, "s1_cost": 0, "s1_hit": 0,
               "t4_pay": 0, "t4_cost": 0, "t4_hit": 0,
               "t5_pay": 0, "t5_cost": 0, "t5_hit": 0}

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
            if not (lo <= m["gap12"] < hi):
                continue

            p1, p2, p3, thirds = m["pred1"], m["pred2"], m["pred3"], m["thirds"]
            race_trio = trio_map.get(race_key, {})
            combos_trio = [frozenset({p1, p2, t}) for t in thirds]
            min_odds = min((race_trio.get(k, 0) for k in combos_trio
                            if race_trio.get(k, 0) > 0), default=0)
            if min_odds < GAMI_THRESHOLD * 100:
                continue

            actual_t = actual_trio.get(race_key, frozenset())
            actual_tri = actual_trifecta.get(race_key)
            race_tri = tri_map.get(race_key, {})

            def tri_pay_b(bets):
                for bet in bets:
                    if actual_tri == bet:
                        return race_tri.get(bet, 0)
                return 0

            rec["n"] += 1

            # S0
            pay = 0
            for t in thirds:
                k = frozenset({p1, p2, t})
                if actual_t == k:
                    pay = race_trio.get(k, 0)
                    break
            rec["s0_pay"] += pay
            rec["s0_cost"] += len(thirds) * 100
            rec["s0_hit"] += int(pay > 0)

            # S1
            s1_bets = [(p1, p2, t) for t in thirds]
            pay = tri_pay_b(s1_bets)
            rec["s1_pay"] += pay
            rec["s1_cost"] += len(s1_bets) * 100
            rec["s1_hit"] += int(pay > 0)

            # T4 (2点)
            t4_bets = [(p1, p2, t) for t in thirds[:2]]
            if t4_bets:
                pay = tri_pay_b(t4_bets)
                rec["t4_pay"] += pay
                rec["t4_cost"] += len(t4_bets) * 100
                rec["t4_hit"] += int(pay > 0)

            # T5 (1点)
            pay = tri_pay_b([(p1, p2, p3)])
            rec["t5_pay"] += pay
            rec["t5_cost"] += 100
            rec["t5_hit"] += int(pay > 0)

        rows.append(rec)

    print(f"\n{'=' * 88}")
    print("Part3: gap12 帯別 ROI（VAL+HOLD）  S0=三連複 / S1=三連単P1→P2→{3-5} / T4=三連単P1→P2→{3,4} / T5=1点")
    print(f"{'=' * 88}")
    print(f"{'帯':<12} {'n':>5}  "
          f"{'S0 hit%':>9}{'S0 ROI':>8}  "
          f"{'S1 hit%':>9}{'S1 ROI':>8}  "
          f"{'T4 hit%':>9}{'T4 ROI':>8}  "
          f"{'T5 hit%':>9}{'T5 ROI':>8}")
    print("-" * 88)
    for rec in rows:
        n = rec["n"]
        if n == 0:
            continue

        def roi_str(pay, cost):
            if cost == 0:
                return "  N/A   "
            v = pay / cost * 100
            return f"{v:>7.1f}%" + ("★" if v >= 100 else " ")

        def hit_str(h, n_r):
            return f"{h/n_r*100:>8.1f}%"

        print(f"{rec['band']:<12} {n:>5}  "
              f"{hit_str(rec['s0_hit'], n)}{roi_str(rec['s0_pay'], rec['s0_cost'])}  "
              f"{hit_str(rec['s1_hit'], n)}{roi_str(rec['s1_pay'], rec['s1_cost'])}  "
              f"{hit_str(rec['t4_hit'], n)}{roi_str(rec['t4_pay'], rec['t4_cost'])}  "
              f"{hit_str(rec['t5_hit'], n)}{roi_str(rec['t5_pay'], rec['t5_cost'])}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def print_strat_table(strats):
    print(f"\n{'=' * 78}")
    print("Part2: 戦略別 ROI サマリ")
    print(f"{'=' * 78}")
    print(f"  {'戦略':<44} {'期間':<6} {'対象R':>7} {'的中%':>7} {'ROI':>8}")
    print("  " + "-" * 70)
    for key, strat in strats.items():
        summ = strat.summary()
        if summ.empty:
            continue
        for _, row in summ.iterrows():
            mark = "★" if row["ROI%"] >= 100 else ""
            print(f"  {strat.name:<44} {row['period']:<6} {row['n']:>7} "
                  f"{row['hit%']:>6.1f}% {row['ROI%']:>7.1f}%{mark}")
        print()


def main():
    print("三連単フォーメーション条件別ROI検証 (doc49)")
    print("データ準備中...")
    df, trio_map, tri_map, actual_trio, actual_trifecta = load_all()

    acc_recs = analyze_positional_accuracy(df, trio_map, actual_trio, actual_trifecta)

    strats = evaluate(df, trio_map, tri_map, actual_trio, actual_trifecta)
    print_strat_table(strats)

    gap12_band_breakdown(df, trio_map, tri_map, actual_trio, actual_trifecta)

    print(f"\n{'=' * 78}")
    print("解釈ポイント")
    print(f"{'=' * 78}")
    print("  ・着順精度: p1_1着%が高い帯 × p2_2着%が高い帯 → 三連単の着固定が成立しやすい")
    print("  ・T4(2点) vs S1(3点): 点数削減がROI改善に繋がるか")
    print("  ・T5(1点): オッズ上振れ前提・的中率が厳しいが大きなリターン")
    print("  ・T8 open: 2着軸を広げて的中率補完するトレードオフ")
    print("  ・H hybrid: 三連単(gap12大)+三連複(中)の混在戦略でポートフォリオ効果")
    print("  ・全て最終オッズ上限値。採否判断は live 実測のみ。")


if __name__ == "__main__":
    main()
