"""7+車レース 条件別ROI探索

doc48: 7+車レース（n_entries≥7）を競輪場×ライン構成×得点順位の条件で絞り、
採算ニッチを探す。

前回の閉鎖判断は「全体」ベース。今回は条件付き分析。

分析軸:
  1. n_lines (ライン数): 2(大型ライン衝突)/3(標準)/7(単騎集合)
  2. 得点順位 (score_rank_pred1): モデルの1位予想が得点何位か
  3. 得点順位一致 (pred1_is_top_scorer): モデル1位 == 得点1位
  4. 得点順位相関 (score_corr): モデル予想順位と得点順位のSpearman相関
  5. ライン種別: pred1 がラインリーダーか、ラインサイズ
  6. 競輪場 (venue) × 条件

TRAIN: 2022-12-01〜2025-06-30 / VAL: 2025-07-01〜2026-02-28 / HOLD: 2026-03-01〜2026-06-12

Usage:
  python3 scripts/exp_7plus_conditional_wt.py
"""
import sys
import re
from pathlib import Path
from scipy.stats import spearmanr

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
GAMI_MIN   = 5.0

VAL_DAYS   = (pd.Timestamp(VAL[1])  - pd.Timestamp(VAL[0])).days + 1
HOLD_DAYS  = (pd.Timestamp(HOLD[1]) - pd.Timestamp(HOLD[0])).days + 1


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


def load_venue_map() -> dict:
    """venue_id -> name"""
    with get_connection() as conn:
        rows = conn.execute("SELECT venue_code, name, bank_length FROM venue_info").fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _period_of(rd: str):
    if TRAIN_EXT[0] <= rd <= TRAIN_EXT[1]: return "TRAIN"
    if VAL[0]  <= rd <= VAL[1]:            return "VAL"
    if HOLD[0] <= rd <= HOLD[1]:           return "HOLD"
    return None


# ── レース単位集計 ────────────────────────────────────────────────────

def build_race_df(df: pd.DataFrame, n_entries_map: dict, trio_map: dict) -> pd.DataFrame:
    """7+車レースを1レース1行に集約。得点順位・ライン情報を付加。"""
    actual_top3 = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    records = []
    for rk, grp in df.groupby("race_key"):
        n_ent = n_entries_map.get(rk)
        if n_ent is None or n_ent < 7:
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
        gap13 = float(preds[0] - preds[2])

        p1 = int(g.iloc[0]["frame_no"])
        p2 = int(g.iloc[1]["frame_no"])
        thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]

        actual = actual_top3.get(rk, frozenset())

        # ── trio ──
        trio_bd = trio_map.get(rk, {})
        combos  = [frozenset({p1, p2, t}) for t in thirds]
        trio_opts = [trio_bd.get(k, 0) for k in combos if trio_bd.get(k, 0) > 0]
        min_trio_odds = min(trio_opts, default=0)

        trio_pay = 0.0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual == k:
                trio_pay = trio_bd.get(k, 0) * 100
                break
        trio_cost = len(thirds) * 100

        # ── 得点順位 ──
        pt = g["race_point"].values   # モデルのpred順に並んでいる
        # 得点の高い順にranking (1=最高得点)
        pt_sorted_desc = np.argsort(-pt)   # pred順の選手を得点降順でsort
        # pred1 (index0) の得点rank = 得点でsortした配列中の位置+1
        score_ranks = np.zeros(len(g), dtype=int)
        for rank_i, idx in enumerate(np.argsort(pt)[::-1]):
            score_ranks[idx] = rank_i + 1  # 得点順位 (1=最高)

        score_rank_pred1 = int(score_ranks[0])   # pred1の得点順位
        score_rank_pred2 = int(score_ranks[1])   # pred2の得点順位

        # モデル予想順位 vs 得点順位のSpearman相関
        model_rank = np.arange(1, len(g)+1)      # 1=pred1, 2=pred2, ...
        try:
            score_corr = float(spearmanr(model_rank, score_ranks).statistic)
        except Exception:
            score_corr = float("nan")

        # ── ライン情報 ──
        # pred1のライン情報
        pred1_row = g.iloc[0]
        pred1_is_leader   = int(pred1_row.get("is_line_leader", 0) or 0)
        pred1_line_size   = int(pred1_row.get("line_size", 1) or 1)
        n_lines           = int(pred1_row.get("n_lines", 0) or 0)

        # 最大ラインのサイズ (line_sizeは各ライングループの最大サイズ)
        max_line_size = int(g["line_size"].max()) if "line_size" in g.columns else 0

        # grade
        grade_raw = g["grade"].iloc[0] if "grade" in g.columns else None
        if grade_raw in ("S級", "SA混合"):
            grade_bin = "S級"
        elif grade_raw == "A級":
            grade_bin = "A級"
        else:
            grade_bin = "その他"

        # venue
        venue_id = str(g["venue_id"].iloc[0]) if "venue_id" in g.columns else "?"

        records.append({
            "race_key":           rk,
            "race_date":          rd,
            "period":             period,
            "n_entries":          int(n_ent),
            "grade_bin":          grade_bin,
            "venue_id":           venue_id,
            "gap12":              gap12,
            "gap13":              gap13,
            "min_trio_odds":      min_trio_odds,
            "trio_pay":           trio_pay,
            "trio_cost":          trio_cost,
            "trio_hit":           int(trio_pay > 0),
            # 得点情報
            "score_rank_pred1":   score_rank_pred1,
            "score_rank_pred2":   score_rank_pred2,
            "pred1_is_top_scorer": int(score_rank_pred1 == 1),
            "score_corr":         score_corr,
            # ライン情報
            "n_lines":            n_lines,
            "pred1_is_leader":    pred1_is_leader,
            "pred1_line_size":    pred1_line_size,
            "max_line_size":      max_line_size,
        })

    return pd.DataFrame(records)


# ── 統計 ──────────────────────────────────────────────────────────────

def stats(sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    cost = sub["trio_cost"].sum()
    if cost == 0:
        return n, float("nan"), float("nan"), float("nan")
    roi  = sub["trio_pay"].sum() / cost * 100
    hit  = sub["trio_hit"].mean() * 100
    avg_pay = sub[sub["trio_hit"]==1]["trio_pay"].mean() if sub["trio_hit"].sum()>0 else 0.0
    return n, roi, hit, avg_pay


def fmt(roi, n):
    if n == 0 or np.isnan(roi): return "     -    (  0)"
    mk = "★" if roi >= 100 else " "
    return f"{roi:6.1f}%{mk}({n:3d})"


def print_row(label, sub, width=35):
    parts = [f"  {label:<{width}}"]
    for p in ["TRAIN","VAL","HOLD"]:
        n, roi, hr, ap = stats(sub[sub["period"]==p])
        parts.append(f"  {fmt(roi,n)}")
    print("".join(parts))


def rday(sub, period):
    days = VAL_DAYS if period=="VAL" else HOLD_DAYS
    return len(sub[sub["period"]==period]) / days


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("doc48: 7+車レース 条件別ROI探索")
    print("=" * 80)
    print(f"TRAIN: {TRAIN_EXT[0]}〜{TRAIN_EXT[1]}  VAL: {VAL[0]}〜{VAL[1]}  HOLD: {HOLD[0]}〜{HOLD[1]}")

    print("\nデータ準備中（7+車含む全体ロード）...", flush=True)
    df_all = build_features_wt(load_raw_data_wt(min_date=TRAIN_EXT[0], max_date=HOLD[1]))
    df = df_all[df_all["finish_order"] >= 1].copy()

    n_entries_map = load_n_entries()
    trio_map      = load_trio_map()
    venue_map     = load_venue_map()

    print("モデル学習中（TRAIN拡張・リーク無し）...", flush=True)
    fit = df[df["race_date"] <= TRAIN_EXT[1]]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"].values)
    df["pred"] = m.predict_proba(prepare_X(df))[:, 1]

    # venue_id を df に付与（build_features_wt が含む場合はスキップ）
    if "venue_id" not in df.columns:
        with get_connection() as conn:
            vm = dict(conn.execute("SELECT race_key, venue_id FROM wt_races").fetchall())
        df["venue_id"] = df["race_key"].map(vm)

    print("レース単位集計中（7+車のみ）...", flush=True)
    rdf = build_race_df(df, n_entries_map, trio_map)
    rdf["ym"] = rdf["race_date"].str[:7]

    for p in ["TRAIN","VAL","HOLD"]:
        s = rdf[rdf["period"]==p]
        days = {
            "TRAIN": (pd.Timestamp(TRAIN_EXT[1])-pd.Timestamp(TRAIN_EXT[0])).days+1,
            "VAL": VAL_DAYS, "HOLD": HOLD_DAYS
        }[p]
        print(f"  {p}: {len(s)}R ({len(s)/days:.1f}R/日)")

    HDR = f"\n  {'条件':<35}  {'TRAIN':>16}  {'VAL':>16}  {'HOLD':>16}"
    SEP = "  " + "-" * 90

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 1: ベースライン（gami≥5倍）")
    print("=" * 80)
    print(HDR); print(SEP)
    r = rdf
    gami_mask = r["min_trio_odds"] >= GAMI_MIN
    print_row("ALL 7+車 (no gami)",          r)
    print_row("ALL 7+車 gami≥5倍",           r[gami_mask])
    print_row("7車のみ gami≥5倍",            r[gami_mask & (r["n_entries"]==7)])
    print_row("9車のみ gami≥5倍",            r[gami_mask & (r["n_entries"]==9)])
    print_row("7+車 gami≥5倍 + gap12≥0.07", r[gami_mask & (r["gap12"]>=0.07)])
    print_row("7+車 gami≥5倍 + gap12≥0.10", r[gami_mask & (r["gap12"]>=0.10)])

    # gami分布確認
    print(f"\n  gami≥5倍のレース率 (全7+車):")
    for p in ["VAL","HOLD"]:
        s = rdf[rdf["period"]==p]
        gs = s[s["min_trio_odds"]>=GAMI_MIN]
        print(f"    {p}: {len(gs)}/{len(s)} = {len(gs)/max(len(s),1)*100:.0f}%  ({len(gs)/(VAL_DAYS if p=='VAL' else HOLD_DAYS):.2f}R/日)")

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 2: n_lines（ライン数）別 ROI")
    print("=" * 80)
    print(HDR); print(SEP)
    for nl in [1,2,3,4,5,7]:
        sub = r[gami_mask & (r["n_lines"]==nl)]
        print_row(f"n_lines={nl} gami≥5倍", sub)
    print()
    for nl in [2,3,4,7]:
        sub = r[gami_mask & (r["n_lines"]==nl) & (r["gap12"]>=0.07)]
        print_row(f"n_lines={nl} gami≥5倍 gap12≥0.07", sub)

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 3: 得点順位（score_rank_pred1）別 ROI")
    print("=" * 80)
    print(HDR); print(SEP)
    for sr in [1,2,3,4]:
        sub = r[gami_mask & (r["score_rank_pred1"]==sr)]
        print_row(f"得点rank={sr} gami≥5倍", sub)
    print()
    # 得点rank1（モデル1位=最高得点者）× gap12
    print("  --- モデル1位=得点rank1（一致） × gap12 ---")
    top_match = gami_mask & (r["score_rank_pred1"]==1)
    print_row("一致 gami≥5倍",              r[top_match])
    print_row("一致 + gap12≥0.07",          r[top_match & (r["gap12"]>=0.07)])
    print_row("一致 + gap12≥0.10",          r[top_match & (r["gap12"]>=0.10)])
    print()
    # 不一致（モデル1位≠得点rank1）
    mismatch = gami_mask & (r["score_rank_pred1"]!=1)
    print_row("不一致 gami≥5倍",            r[mismatch])
    print_row("不一致 + gap12≥0.07",        r[mismatch & (r["gap12"]>=0.07)])

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 4: 得点順位相関（score_corr）別 ROI")
    print("=" * 80)
    print("  score_corr = Spearman相関(モデル予想順位, 得点順位)。高い=モデルが得点を再現")
    print(HDR); print(SEP)
    # corr閾値スイープ
    for thr in [0.9, 0.8, 0.7, 0.5, 0.3, 0.0, -0.3]:
        sub = r[gami_mask & (r["score_corr"] >= thr)]
        print_row(f"score_corr≥{thr:.1f} gami≥5倍", sub)
    print()
    # 逆相関（モデルが得点と逆転を示す）
    sub_neg = r[gami_mask & (r["score_corr"] < 0)]
    print_row("score_corr<0 (逆転) gami≥5倍", sub_neg)

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 5: ライン属性（pred1 のライン情報）")
    print("=" * 80)
    print(HDR); print(SEP)
    print_row("pred1=リーダー gami≥5倍",     r[gami_mask & (r["pred1_is_leader"]==1)])
    print_row("pred1=非リーダー gami≥5倍",   r[gami_mask & (r["pred1_is_leader"]==0)])
    print_row("pred1 linesize≥3 gami≥5倍",  r[gami_mask & (r["pred1_line_size"]>=3)])
    print_row("pred1 linesize=2 gami≥5倍",  r[gami_mask & (r["pred1_line_size"]==2)])
    print_row("pred1 linesize=1 gami≥5倍",  r[gami_mask & (r["pred1_line_size"]==1)])
    print()
    print_row("pred1=リーダー + gap12≥0.07", r[gami_mask & (r["pred1_is_leader"]==1) & (r["gap12"]>=0.07)])
    print_row("pred1=リーダー + gap12≥0.10", r[gami_mask & (r["pred1_is_leader"]==1) & (r["gap12"]>=0.10)])
    print_row("pred1=リーダー + score_rank1", r[gami_mask & (r["pred1_is_leader"]==1) & (r["score_rank_pred1"]==1)])

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 6: n_lines × 得点rank1一致 × gap12 交差")
    print("=" * 80)
    print(HDR); print(SEP)
    cells = [
        ("n_lines=2 + 一致 + gap12≥0.07",
         gami_mask & (r["n_lines"]==2) & (r["pred1_is_top_scorer"]==1) & (r["gap12"]>=0.07)),
        ("n_lines=3 + 一致 + gap12≥0.07",
         gami_mask & (r["n_lines"]==3) & (r["pred1_is_top_scorer"]==1) & (r["gap12"]>=0.07)),
        ("n_lines=7 + 一致 gami≥5倍",
         gami_mask & (r["n_lines"]==7) & (r["pred1_is_top_scorer"]==1)),
        ("n_lines=7 + 一致 + gap12≥0.07",
         gami_mask & (r["n_lines"]==7) & (r["pred1_is_top_scorer"]==1) & (r["gap12"]>=0.07)),
        ("n_lines=7 + gami≥5倍",
         gami_mask & (r["n_lines"]==7)),
        ("n_lines=2 + リーダー + gap12≥0.07",
         gami_mask & (r["n_lines"]==2) & (r["pred1_is_leader"]==1) & (r["gap12"]>=0.07)),
        ("n_lines=3 + リーダー + gap12≥0.07",
         gami_mask & (r["n_lines"]==3) & (r["pred1_is_leader"]==1) & (r["gap12"]>=0.07)),
        ("n_lines≤3 + 一致 + gap12≥0.07",
         gami_mask & (r["n_lines"]<=3) & (r["pred1_is_top_scorer"]==1) & (r["gap12"]>=0.07)),
        ("一致 + リーダー + gap12≥0.07",
         gami_mask & (r["pred1_is_top_scorer"]==1) & (r["pred1_is_leader"]==1) & (r["gap12"]>=0.07)),
        ("一致 + リーダー + n_lines≤3 + gap12≥0.07",
         gami_mask & (r["pred1_is_top_scorer"]==1) & (r["pred1_is_leader"]==1) & (r["n_lines"]<=3) & (r["gap12"]>=0.07)),
    ]
    for label, mask in cells:
        print_row(label, r[mask])

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 7: grade × n_lines × gap12 スイープ")
    print("=" * 80)
    print(HDR); print(SEP)
    for grade in ["S級","A級"]:
        for nl in [2,3,7]:
            sub = r[gami_mask & (r["grade_bin"]==grade) & (r["n_lines"]==nl)]
            print_row(f"{grade} n_lines={nl} gami≥5倍", sub)
        print()
    for grade in ["S級","A級"]:
        sub = r[gami_mask & (r["grade_bin"]==grade) & (r["gap12"]>=0.07)]
        print_row(f"{grade} gap12≥0.07 gami≥5倍", sub)
    print()

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 8: 全候補セルの VAL×HOLD スクリーニング（n≥5/3 両通過）")
    print("=" * 80)

    gami5 = rdf["min_trio_odds"] >= GAMI_MIN
    all_cells = []
    for nl_label, nl_mask in [
        ("nl_any",  pd.Series([True]*len(rdf), index=rdf.index)),
        ("nl=2",    rdf["n_lines"]==2),
        ("nl=3",    rdf["n_lines"]==3),
        ("nl=7",    rdf["n_lines"]==7),
        ("nl≤3",    rdf["n_lines"]<=3),
    ]:
        for sr_label, sr_mask in [
            ("any",    pd.Series([True]*len(rdf), index=rdf.index)),
            ("sr1",    rdf["score_rank_pred1"]==1),
            ("sr≤2",   rdf["score_rank_pred1"]<=2),
        ]:
            for ldr_label, ldr_mask in [
                ("any",    pd.Series([True]*len(rdf), index=rdf.index)),
                ("leader", rdf["pred1_is_leader"]==1),
            ]:
                for g_thr in [0.0, 0.07, 0.10, 0.12]:
                    g_mask = rdf["gap12"] >= g_thr
                    for grd_label, grd_mask in [
                        ("any",  pd.Series([True]*len(rdf), index=rdf.index)),
                        ("S",    rdf["grade_bin"]=="S級"),
                        ("A",    rdf["grade_bin"]=="A級"),
                    ]:
                        mask = gami5 & nl_mask & sr_mask & ldr_mask & g_mask & grd_mask
                        sub  = rdf[mask]
                        vn, vroi, *_ = stats(sub[sub["period"]=="VAL"])
                        hn, hroi, *_ = stats(sub[sub["period"]=="HOLD"])
                        label = f"{nl_label}/{sr_label}/{ldr_label}/g≥{g_thr:.2f}/{grd_label}"
                        all_cells.append((label, vroi, vn, hroi, hn,
                                          rday(sub,"VAL"), rday(sub,"HOLD")))

    passed = [(lbl, vr, vn, hr, hn, vd, hd) for lbl,vr,vn,hr,hn,vd,hd in all_cells
              if not np.isnan(vr) and vr>=100 and not np.isnan(hr) and hr>=100
              and vn>=5 and hn>=3]

    if passed:
        print(f"\n  ★通過セル ({len(passed)}件) — VAL ROI降順")
        print(f"  {'条件':<45} {'VAL':>10} {'HOLD':>10}  {'V R/日':>7} {'H R/日':>7}")
        for lbl, vr, vn, hr, hn, vd, hd in sorted(passed, key=lambda x:-(x[1]+x[3])):
            print(f"  {lbl:<45} {vr:7.1f}%({vn:3d}) {hr:7.1f}%({hn:3d})  {vd:.2f}  {hd:.2f}")
    else:
        print("\n  （通過セルなし）")

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶ Section 9: 月別ROI推移（通過セルor有力候補）")
    print("=" * 80)
    months = sorted(rdf[rdf["period"].isin(["VAL","HOLD"])]["ym"].dropna().unique())

    def monthly(mask):
        vals = []
        for ym in months:
            s = rdf[mask & (rdf["ym"]==ym)]
            n, roi, *_ = stats(s)
            if n == 0:
                vals.append(f"{ym}: -(0R)")
            else:
                mk = "★" if roi>=100 else " "
                vals.append(f"{ym}: {roi:4.0f}%{mk}({n}R)")
        return "    " + "  ".join(vals)

    cands_monthly = [
        ("7+車 ALL gami≥5倍",
         gami5),
        ("7+車 gami≥5倍 + gap12≥0.07",
         gami5 & (rdf["gap12"]>=0.07)),
        ("n_lines=7 gami≥5倍",
         gami5 & (rdf["n_lines"]==7)),
        ("n_lines=2 + gap12≥0.07",
         gami5 & (rdf["n_lines"]==2) & (rdf["gap12"]>=0.07)),
    ]
    # 通過セルの上位3件を追加
    for lbl, vr, vn, hr, hn, vd, hd in sorted(passed, key=lambda x:-(x[1]+x[3]))[:3]:
        parts = lbl.split("/")
        nl_label, sr_label, ldr_label, g_label, grd_label = parts
        nl_m   = {"nl_any": pd.Series([True]*len(rdf),index=rdf.index),
                  "nl=2":   rdf["n_lines"]==2,
                  "nl=3":   rdf["n_lines"]==3,
                  "nl=7":   rdf["n_lines"]==7,
                  "nl≤3":   rdf["n_lines"]<=3}.get(nl_label, pd.Series([True]*len(rdf),index=rdf.index))
        sr_m   = {"any":    pd.Series([True]*len(rdf),index=rdf.index),
                  "sr1":    rdf["score_rank_pred1"]==1,
                  "sr≤2":  rdf["score_rank_pred1"]<=2}.get(sr_label, pd.Series([True]*len(rdf),index=rdf.index))
        ldr_m  = {"any":    pd.Series([True]*len(rdf),index=rdf.index),
                  "leader": rdf["pred1_is_leader"]==1}.get(ldr_label, pd.Series([True]*len(rdf),index=rdf.index))
        g_thr  = float(g_label.replace("g≥",""))
        grd_m  = {"any": pd.Series([True]*len(rdf),index=rdf.index),
                  "S":   rdf["grade_bin"]=="S級",
                  "A":   rdf["grade_bin"]=="A級"}.get(grd_label, pd.Series([True]*len(rdf),index=rdf.index))
        cands_monthly.append((f"★ {lbl}", gami5 & nl_m & sr_m & ldr_m & (rdf["gap12"]>=g_thr) & grd_m))

    for label, mask in cands_monthly:
        print(f"\n  【{label}】")
        print(monthly(mask))

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
