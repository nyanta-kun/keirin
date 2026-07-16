"""相互補完カバレッジ探索 — 5R/日 達成

doc47: C0(trio≥5倍/gap12≥0.07)に補完する戦略を探索し、合計5R/日を目指す。
各戦略は独立して ROI≥100%（VAL+HOLD 両期間）を維持する必要がある。

全体 ≤6車プール: ~4.03R/日（VAL+HOLD 実測）
現行 C0: ~0.25R/日（gami≥5倍 + gap12≥0.07 = 6% 使用）

戦略カテゴリ:
  T0: trio, gami≥5倍, gap12≥0.07    [現行 C0]
  T1: trio, gami [3,5倍), gap12≥0.07 [B-rank 三連複]
  T2: trio, gami≥5倍, gap12[0.04,0.07) [低確信 三連複]
  W0: quinellaPlace W12, gap12≥0.07, qp≥2.5 [ワイド 高確信]
  W1: quinellaPlace W12, gap12[0.04,0.07), qp≥2.5 [ワイド 中確信]
  W2: quinellaPlace W12, gap12≥0.00, qp≥2.5 [ワイド 全体]

Usage:
  python3 scripts/exp_coverage_target_wt.py
"""
import sys
import re
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT
from src.database import get_connection
from exp_segment_first_wt import VAL, HOLD, LGB_PARAMS

TRAIN_EXT = ("2022-12-01", "2025-06-30")
GAMI_MIN = 5.0
GAMI_BRANK_MIN = 3.0
WIDE_MIN_ODDS = 2.5

VAL_DAYS  = (pd.Timestamp(VAL[1])  - pd.Timestamp(VAL[0])).days + 1   # 245
HOLD_DAYS = (pd.Timestamp(HOLD[1]) - pd.Timestamp(HOLD[0])).days + 1  # 103
TOTAL_DAYS = VAL_DAYS + HOLD_DAYS  # 348


# ── データロード ──────────────────────────────────────────────────────

def load_n_entries() -> dict:
    with get_connection() as conn:
        rows = conn.execute("SELECT race_key, n_entries FROM wt_races").fetchall()
    return {r[0]: r[1] for r in rows}


def load_trio_map() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'"
        ).fetchall()
    trio_map: dict = {}
    for rk, comb, ov in rows:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        trio_map.setdefault(rk, {})[fr] = float(ov)
    return trio_map


def load_qp_map() -> dict:
    """quinellaPlace (ワイド) オッズマップ: {race_key: {frozenset({fn1,fn2}): odds}}"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='quinellaPlace'"
        ).fetchall()
    qp_map: dict = {}
    for rk, comb, ov in rows:
        if ov is None or ov <= 0:
            continue
        try:
            fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
        except ValueError:
            continue
        qp_map.setdefault(rk, {})[fr] = float(ov)
    return qp_map


def _period_of(rd: str):
    if VAL[0] <= rd <= VAL[1]:   return "VAL"
    if HOLD[0] <= rd <= HOLD[1]: return "HOLD"
    return None


# ── レース単位集計 ────────────────────────────────────────────────────

def build_race_df(df: pd.DataFrame, n_entries_map: dict, trio_map: dict, qp_map: dict) -> pd.DataFrame:
    """≤6車レース全体を1レース1行に集約。trio/ワイド両方のROI計算情報を保持。"""
    actual_top3 = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    records = []
    for rk, grp in df.groupby("race_key"):
        n_ent = n_entries_map.get(rk)
        if n_ent is None or n_ent > 6:
            continue

        g = grp.sort_values("pred", ascending=False).reset_index(drop=True)
        if len(g) < 3:
            continue
        rd = str(g["race_date"].iloc[0])
        period = _period_of(rd)
        if period is None:
            continue

        preds = g["pred"].values
        gap12 = float(preds[0] - preds[1])
        pred1_val = float(preds[0])

        p1 = int(g.iloc[0]["frame_no"])
        p2 = int(g.iloc[1]["frame_no"])
        thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]

        actual = actual_top3.get(rk, frozenset())

        # ── trio 計算 ──
        trio_bd = trio_map.get(rk, {})
        combos = [frozenset({p1, p2, t}) for t in thirds]
        trio_odds_list = [trio_bd.get(k, 0) for k in combos if trio_bd.get(k, 0) > 0]
        min_trio_odds = min(trio_odds_list, default=0)

        trio_pay = 0.0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual == k:
                trio_pay = trio_bd.get(k, 0) * 100
                break
        trio_cost = len(thirds) * 100

        # ── ワイド W12 計算 ──
        qp_bd = qp_map.get(rk, {})
        w12_key = frozenset({p1, p2})
        w12_odds = qp_bd.get(w12_key, 0.0)
        # W12 的中: p1 と p2 が両方 top3 に入っている
        w12_hit = int(p1 in actual and p2 in actual)
        w12_pay = w12_odds * 100 * w12_hit
        w12_cost = 100

        # grade
        grade_raw = g["grade"].iloc[0] if "grade" in g.columns else None
        if grade_raw in ("S級", "SA混合"):
            grade_bin = "S級"
        elif grade_raw == "A級":
            grade_bin = "A級"
        else:
            grade_bin = "その他"

        records.append({
            "race_key":      rk,
            "race_date":     rd,
            "period":        period,
            "n_entries":     int(n_ent),
            "grade_bin":     grade_bin,
            "gap12":         gap12,
            "pred1":         pred1_val,
            # trio
            "min_trio_odds": min_trio_odds,
            "trio_pay":      trio_pay,
            "trio_cost":     trio_cost,
            "trio_hit":      int(trio_pay > 0),
            # wide
            "w12_odds":      w12_odds,
            "w12_pay":       w12_pay,
            "w12_cost":      w12_cost,
            "w12_hit":       w12_hit,
        })

    return pd.DataFrame(records)


# ── 統計 ──────────────────────────────────────────────────────────────

def stats_trio(sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    roi = sub["trio_pay"].sum() / sub["trio_cost"].sum() * 100
    hit = sub["trio_hit"].mean() * 100
    avg_pay = sub[sub["trio_hit"]==1]["trio_pay"].mean() if sub["trio_hit"].sum()>0 else 0.0
    return n, roi, hit, avg_pay


def stats_wide(sub: pd.DataFrame):
    n = len(sub[sub["w12_odds"] >= WIDE_MIN_ODDS])
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    s = sub[sub["w12_odds"] >= WIDE_MIN_ODDS]
    roi = s["w12_pay"].sum() / s["w12_cost"].sum() * 100
    hit = s["w12_hit"].mean() * 100
    avg_pay = s[s["w12_hit"]==1]["w12_pay"].mean() if s["w12_hit"].sum()>0 else 0.0
    return n, roi, hit, avg_pay


def fmt_roi(roi, n, prefix=""):
    if n == 0 or np.isnan(roi):
        return f"    -    (  0)"
    mk = "★" if roi >= 100 else " "
    return f"{roi:6.1f}%{mk}({n:3d})"


def print_strategy(label, trio_mask, wide_mask, rdf, width=30):
    t_row = rdf[trio_mask] if trio_mask is not None else pd.DataFrame()
    w_row = rdf[wide_mask] if wide_mask is not None else pd.DataFrame()
    parts = [f"  {label:<{width}}"]
    for p in ["VAL", "HOLD"]:
        if len(t_row):
            n, roi, hr, ap = stats_trio(t_row[t_row["period"]==p])
            parts.append(f" T:{fmt_roi(roi, n)}")
        else:
            parts.append(f" T:{'    -    (  0)'}")
        if len(w_row):
            n, roi, hr, ap = stats_wide(w_row[w_row["period"]==p])
            parts.append(f" W:{fmt_roi(roi, n)}")
        else:
            parts.append(f" W:{'    -    (  0)'}")
    print("".join(parts))


def rday(rdf_sub, period, use_trio=True, use_wide=False):
    """R/日計算（ユニーク race_key）"""
    s = rdf_sub[rdf_sub["period"]==period]
    if use_wide:
        s = s[s["w12_odds"] >= WIDE_MIN_ODDS]
    return len(s) / (VAL_DAYS if period=="VAL" else HOLD_DAYS)


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("doc47: 相互補完カバレッジ探索 — 5R/日 達成")
    print("=" * 80)
    print(f"TRAIN拡張: {TRAIN_EXT[0]} 〜 {TRAIN_EXT[1]}")
    print(f"VAL  : {VAL[0]} 〜 {VAL[1]}  ({VAL_DAYS}日)")
    print(f"HOLD : {HOLD[0]} 〜 {HOLD[1]}  ({HOLD_DAYS}日)")
    print(f"全体≤6車プール目標: ~4.03R/日（VAL+HOLD実測）")

    print("\nデータ準備中...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN_EXT[0], max_date=HOLD[1]))
    df = df_all[df_all["finish_order"] >= 1].copy()

    n_entries_map = load_n_entries()
    trio_map      = load_trio_map()
    qp_map        = load_qp_map()

    print("モデル学習中（TRAIN拡張・リーク無し）...", flush=True)
    fit = df[df["race_date"] <= TRAIN_EXT[1]]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"].values)
    df["pred"] = m.predict_proba(prepare_X(df))[:, 1]

    print("レース単位集計中...", flush=True)
    rdf = build_race_df(df, n_entries_map, trio_map, qp_map)
    rdf["ym"] = rdf["race_date"].str[:7]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("▶ Section 1: 全体プール確認（≤6車）")
    print("=" * 80)
    for p in ["VAL", "HOLD"]:
        s = rdf[rdf["period"]==p]
        days = VAL_DAYS if p=="VAL" else HOLD_DAYS
        print(f"  {p}: 全体={len(s)}R / {days}日 = {len(s)/days:.2f}R/日")
        # gami≥5倍 trio
        gc0 = s[s["min_trio_odds"] >= GAMI_MIN]
        print(f"       gami≥5倍 trio: {len(gc0)}R  ({len(gc0)/days:.2f}R/日)")
        # gap12≥0.07 + gami≥5
        gc0g = gc0[gc0["gap12"] >= 0.07]
        print(f"       gami≥5倍 + gap12≥0.07 [C0]: {len(gc0g)}R  ({len(gc0g)/days:.2f}R/日)")
        # ワイド w12≥2.5
        ww = s[s["w12_odds"] >= WIDE_MIN_ODDS]
        print(f"       ワイドW12 odds≥2.5: {len(ww)}R  ({len(ww)/days:.2f}R/日)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("▶ Section 2: 戦略別 ROI（T=trio / W=ワイドW12）")
    print("=" * 80)
    print(f"  ※ ROI≥100%=★  T=trio ROI(n)  W=ワイドW12 ROI(n) (odds≥{WIDE_MIN_ODDS})")
    header = f"  {'戦略':<30}  {'--- VAL ---':>35}  {'--- HOLD ---':>35}"
    header2 = f"  {'':30}  {'T-ROI(n)':>16} {'W-ROI(n)':>16}  {'T-ROI(n)':>16} {'W-ROI(n)':>16}"
    print(header)
    print(header2)
    print("  " + "-" * 88)

    def row(label, t_mask, w_mask):
        """T=trio strategy, W=wide strategy (masks applied to rdf)"""
        tr = rdf[t_mask] if t_mask is not None else pd.DataFrame()
        wr = rdf[w_mask] if w_mask is not None else pd.DataFrame()
        parts = [f"  {label:<30}"]
        for p in ["VAL", "HOLD"]:
            if len(tr) > 0:
                ts = tr[tr["period"]==p]
                n, roi, hr, ap = stats_trio(ts)
                t_str = fmt_roi(roi, n) if n > 0 else "      -    (  0)"
            else:
                t_str = "      -    (  0)"
            if len(wr) > 0:
                ws = wr[wr["period"]==p]
                n, roi, hr, ap = stats_wide(ws)
                w_str = fmt_roi(roi, n) if n > 0 else "      -    (  0)"
            else:
                w_str = "      -    (  0)"
            parts.append(f"  {t_str}  {w_str}")
        print("".join(parts))

    # C0 baseline
    c0_mask  = (rdf["min_trio_odds"] >= GAMI_MIN) & (rdf["gap12"] >= 0.07)
    c0_w     = rdf["gap12"] >= 0.07  # same races, wide bet
    row("T0/W0: C0(gami≥5/gap12≥0.07)",  c0_mask, c0_mask)

    # gap12 variants of C0
    c0_g10_mask = (rdf["min_trio_odds"] >= GAMI_MIN) & (rdf["gap12"] >= 0.10)
    row("T0b: gami≥5 + gap12≥0.10",      c0_g10_mask, None)
    c0_g05_mask = (rdf["min_trio_odds"] >= GAMI_MIN) & (rdf["gap12"] >= 0.05) & (rdf["gap12"] < 0.07)
    row("T2a: gami≥5 + gap12[0.05,0.07)", c0_g05_mask, c0_g05_mask)
    c0_g04_mask = (rdf["min_trio_odds"] >= GAMI_MIN) & (rdf["gap12"] >= 0.04) & (rdf["gap12"] < 0.05)
    row("T2b: gami≥5 + gap12[0.04,0.05)", c0_g04_mask, c0_g04_mask)
    c0_g00_mask = (rdf["min_trio_odds"] >= GAMI_MIN) & (rdf["gap12"] < 0.04)
    row("T3: gami≥5 + gap12<0.04",        c0_g00_mask, c0_g00_mask)

    print()
    # B-rank (gami 3-5)
    b_mask_g07 = (rdf["min_trio_odds"] >= GAMI_BRANK_MIN) & (rdf["min_trio_odds"] < GAMI_MIN) & (rdf["gap12"] >= 0.07)
    row("T1a: B-rank(gami[3,5)) + gap12≥0.07", b_mask_g07, b_mask_g07)
    b_mask_g05 = (rdf["min_trio_odds"] >= GAMI_BRANK_MIN) & (rdf["min_trio_odds"] < GAMI_MIN) & (rdf["gap12"] >= 0.05)
    row("T1b: B-rank + gap12≥0.05",       b_mask_g05, b_mask_g05)
    b_mask_all = (rdf["min_trio_odds"] >= GAMI_BRANK_MIN) & (rdf["min_trio_odds"] < GAMI_MIN)
    row("T1c: B-rank + all gap12",         b_mask_all, b_mask_all)

    print()
    # No-gami trio (all ≤6車)
    ng_g10 = rdf["gap12"] >= 0.10
    row("T_ng: no-gami trio + gap12≥0.10",ng_g10, None)
    ng_g07 = rdf["gap12"] >= 0.07
    row("T_ng: no-gami trio + gap12≥0.07",ng_g07, None)
    ng_g05 = rdf["gap12"] >= 0.05
    row("T_ng: no-gami trio + gap12≥0.05",ng_g05, None)

    print()
    # Wide only strategies
    w_g07 = rdf["gap12"] >= 0.07
    w_g05 = (rdf["gap12"] >= 0.05) & (rdf["gap12"] < 0.07)
    w_g04 = (rdf["gap12"] >= 0.04) & (rdf["gap12"] < 0.05)
    w_g00 = rdf["gap12"] <  0.04
    w_all = pd.Series([True] * len(rdf), index=rdf.index)

    def row_wide(label, mask):
        parts = [f"  {label:<30}"]
        for p in ["VAL", "HOLD"]:
            ws = rdf[mask & (rdf["period"]==p)]
            n, roi, hr, ap = stats_wide(ws)
            w_str = fmt_roi(roi, n) if n > 0 else "      -    (  0)"
            parts.append(f"  {'      -    (  0)':>16}  {w_str}")
        print("".join(parts))

    row_wide("W0: wide + gap12≥0.07",      w_g07)
    row_wide("W1: wide + gap12[0.05,0.07)",w_g05)
    row_wide("W2: wide + gap12[0.04,0.05)",w_g04)
    row_wide("W3: wide + gap12<0.04",      w_g00)
    row_wide("W_all: wide, 全体",          w_all)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("▶ Section 3: R/日 & ROI サマリー（ROI≥100% 両期間のみ掲載）")
    print("=" * 80)
    print(f"  {'戦略':<35} {'VAL':>12} {'HOLD':>12}  {'VAL R/日':>8} {'HOLD R/日':>9}")
    print("  " + "-" * 85)

    strategies = [
        # (label, trio_mask, wide_mask)
        ("T0: trio gami≥5 + gap12≥0.07 [C0]",
         (rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.07), None),
        ("T0b: trio gami≥5 + gap12≥0.10",
         (rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.10), None),
        ("T2a: trio gami≥5 + gap12[0.05,0.07)",
         (rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.05) & (rdf["gap12"]<0.07), None),
        ("T1a: B-rank gami[3,5) + gap12≥0.07",
         (rdf["min_trio_odds"]>=GAMI_BRANK_MIN) & (rdf["min_trio_odds"]<GAMI_MIN) & (rdf["gap12"]>=0.07), None),
        ("T1b: B-rank gami[3,5) + gap12≥0.05",
         (rdf["min_trio_odds"]>=GAMI_BRANK_MIN) & (rdf["min_trio_odds"]<GAMI_MIN) & (rdf["gap12"]>=0.05), None),
        ("T_ng: no-gami trio + gap12≥0.10",
         rdf["gap12"]>=0.10, None),
        ("T_ng: no-gami trio + gap12≥0.07",
         rdf["gap12"]>=0.07, None),
        ("W0: wide + gap12≥0.07 (W12≥2.5)",
         None, rdf["gap12"]>=0.07),
        ("W1: wide + gap12[0.05,0.07) (W12≥2.5)",
         None, (rdf["gap12"]>=0.05) & (rdf["gap12"]<0.07)),
        ("W2: wide + gap12[0.04,0.07) (W12≥2.5)",
         None, (rdf["gap12"]>=0.04) & (rdf["gap12"]<0.07)),
        ("W_all: wide, 全体 (W12≥2.5)",
         None, pd.Series([True]*len(rdf), index=rdf.index)),
    ]

    passed = []
    for label, t_mask, w_mask in strategies:
        if t_mask is not None:
            tr = rdf[t_mask]
            vn, vroi, *_ = stats_trio(tr[tr["period"]=="VAL"])
            hn, hroi, *_ = stats_trio(tr[tr["period"]=="HOLD"])
            vd = vn / VAL_DAYS
            hd = hn / HOLD_DAYS
        else:
            tr = pd.DataFrame()
            vn, vroi = 0, float("nan")
            hn, hroi = 0, float("nan")
            vd = hd = 0.0

        if w_mask is not None:
            wr = rdf[w_mask]
            wvn, wvroi, *_ = stats_wide(wr[wr["period"]=="VAL"])
            whn, whroi, *_ = stats_wide(wr[wr["period"]=="HOLD"])
            # override if wide-only
            if t_mask is None:
                vn, vroi, hn, hroi = wvn, wvroi, whn, whroi
                vd = wvn / VAL_DAYS
                hd = whn / HOLD_DAYS

        v_pass = not np.isnan(vroi) and vroi >= 100 and vn >= 5
        h_pass = not np.isnan(hroi) and hroi >= 100 and hn >= 3

        if v_pass and h_pass:
            mk = "★"
            passed.append((label, vroi, vn, hroi, hn, vd, hd))
        else:
            mk = " "
        v_str = f"{vroi:6.1f}%{mk}({vn:3d})" if not np.isnan(vroi) else f"    -    (  0)"
        h_str = f"{hroi:6.1f}%{mk}({hn:3d})" if not np.isnan(hroi) else f"    -    (  0)"
        print(f"  {label:<35} {v_str:>14} {h_str:>14}  {vd:>7.2f}  {hd:>8.2f}")

    print(f"\n  ★通過戦略 ({len(passed)}件):")
    for lbl, vr, vn, hr, hn, vd, hd in sorted(passed, key=lambda x: -(x[1]+x[3])):
        print(f"    {lbl:<40} V:{vr:5.0f}%(n={vn}) H:{hr:5.0f}%(n={hn})  {vd:.2f}/{hd:.2f}R/日")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("▶ Section 4: 月別 ROI 推移（VAL+HOLD、候補戦略）")
    print("=" * 80)

    months = sorted(rdf[rdf["period"].isin(["VAL","HOLD"])]["ym"].dropna().unique())

    def monthly_roi(sub, kind="trio"):
        vals = []
        for ym in months:
            s = sub[sub["ym"]==ym]
            if kind == "trio":
                n, roi, *_ = stats_trio(s)
            else:
                n, roi, *_ = stats_wide(s)
            if n == 0:
                vals.append(f"{ym}: -(0R)")
            else:
                mk = "★" if roi >= 100 else " "
                vals.append(f"{ym}: {roi:4.0f}%{mk}({n}R)")
        return "    " + "  ".join(vals)

    cands = [
        ("T0 C0 [trio gami≥5 gap12≥0.07]",
         rdf[(rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.07)], "trio"),
        ("T_ng no-gami trio gap12≥0.07",
         rdf[rdf["gap12"]>=0.07], "trio"),
        ("W0 wide gap12≥0.07 (qp≥2.5)",
         rdf[rdf["gap12"]>=0.07], "wide"),
        ("W1 wide gap12[0.05,0.07) (qp≥2.5)",
         rdf[(rdf["gap12"]>=0.05) & (rdf["gap12"]<0.07)], "wide"),
        ("W_all wide 全体 (qp≥2.5)",
         rdf.copy(), "wide"),
    ]

    for label, sub, kind in cands:
        print(f"\n  【{label}】")
        print(monthly_roi(sub, kind))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 80)
    print("▶ Section 5: 組み合わせシミュレーション（相互補完・合計 R/日）")
    print("=" * 80)
    print("  重複なし（同一 race_key は最高優先戦略のみでカウント）")

    combos = [
        # (ラベル, [(mask, name)])  — 優先順に記載。上位がマッチしたらそのrace_keyは下位では数えない
        ("C0 + ワイド broad",
         [(((rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.07)), "T0"),
          (rdf["gap12"]>=0.07, "W0-trio_only"),    # same races, wide
          ((rdf["gap12"]>=0.04), "W_broad")]),
        ("C0 + no-gami trio gap12≥0.07",
         [((rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.07), "T0"),
          (rdf["gap12"]>=0.07, "T_ng")]),
        ("C0 + B-rank + ワイド broad",
         [((rdf["min_trio_odds"]>=GAMI_MIN) & (rdf["gap12"]>=0.07), "T0"),
          ((rdf["min_trio_odds"]>=GAMI_BRANK_MIN) & (rdf["min_trio_odds"]<GAMI_MIN) & (rdf["gap12"]>=0.07), "T1"),
          (rdf["gap12"]>=0.04, "W_broad")]),
    ]

    for combo_label, layers in combos:
        assigned = pd.Series("none", index=rdf.index)
        for mask, name in layers:
            unassigned = assigned == "none"
            assigned[unassigned & mask] = name

        print(f"\n  【{combo_label}】")
        for _, name in layers:
            s_val  = rdf[(rdf["period"]=="VAL")  & (assigned==name)]
            s_hold = rdf[(rdf["period"]=="HOLD") & (assigned==name)]
            nv = len(s_val);  nh = len(s_hold)
            print(f"    {name:<20}  VAL={nv:3d}R ({nv/VAL_DAYS:.2f}/日)  HOLD={nh:3d}R ({nh/HOLD_DAYS:.2f}/日)")
        # total
        total_v  = (assigned[rdf["period"]=="VAL"]  != "none").sum()
        total_h  = (assigned[rdf["period"]=="HOLD"] != "none").sum()
        print(f"    {'合計':<20}  VAL={total_v:3d}R ({total_v/VAL_DAYS:.2f}/日)  HOLD={total_h:3d}R ({total_h/HOLD_DAYS:.2f}/日)")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
