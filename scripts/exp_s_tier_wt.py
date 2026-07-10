"""Sランク設計スイープ（2026-07-10）

前提: SS = レース単位 min(全目)≥7 ∧ gap12≥0.10 ∧ gap23≥1pt・全目購入（doc52のR）。
S = SS落ちレースから条件緩和で対象を増やし、買い目を指数上位k点に絞ってROIを確保する。

スイープ軸:
  - k: 買い目数（指数順位上位k点 / all）
  - gami_kept: 購入目（上位k点）の最安オッズ下限
  - gap12_min: 0.07（候補ゲート） or 0.10
  - gap23_min: 0 or 1pt

評価: 2025年（lgbm_wt_train_only・11-12月=真OOS）+ 2026-06（june_eval・真OOS）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

CAND_GAP12 = 0.07
SS_GAP12 = 0.10
SS_GAMI = 7.0
GAP23_MIN = 1.0


def load_trio_board(race_keys):
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None and 0 < float(od) < 9000:
                    try:
                        key = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                        board[rk][key] = float(od)
                    except ValueError:
                        pass
    return board


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) >= 7})].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    board = load_trio_board(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 5:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < CAND_GAP12:
            continue
        gap23_pt = (p[1] - p[2]) * 100.0
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        frames = g["frame_no"].astype(int).tolist()
        p1, p2 = frames[0], frames[1]
        thirds_ordered = frames[2:]  # 指数(pred_prob)降順

        bd = board.get(rk, {})
        legs = {}
        for t in thirds_ordered:
            key = frozenset({p1, p2, t})
            if key in bd:
                legs[t] = bd[key]
        if not legs:
            continue

        rows.append({
            "rk": rk, "gap12": gap12, "gap23_pt": gap23_pt,
            "p1": p1, "p2": p2, "top3": top3,
            "thirds": [t for t in thirds_ordered if t in legs],  # 指数順・オッズあり
            "legs": legs,
        })
    return rows


def is_ss(r):
    return (min(r["legs"].values()) >= SS_GAMI
            and r["gap12"] >= SS_GAP12
            and r["gap23_pt"] >= GAP23_MIN)


def eval_s(rows, k, gami_kept, gap12_min, gap23_min):
    """SS落ちレースに S 条件を適用して集計。returns (n, hits, bet, pay)"""
    n = h = b = pp = 0
    for r in rows:
        if is_ss(r):
            continue
        if r["gap12"] < gap12_min or r["gap23_pt"] < gap23_min:
            continue
        kept = r["thirds"] if k is None else r["thirds"][:k]
        if not kept:
            continue
        if min(r["legs"][t] for t in kept) < gami_kept:
            continue
        pay = 0
        for t in kept:
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = int(r["legs"][t] * 100)
                break
        n += 1
        h += 1 if pay > 0 else 0
        b += len(kept) * 100
        pp += pay
    return n, h, b, pp


def sweep(rows, label, days):
    print(f"\n===== {label}（候補{len(rows)}R / SS該当{sum(1 for r in rows if is_ss(r))}R / {days}日） =====")
    print(f"{'k':>4} {'gami':>5} {'gap12':>6} {'gap23':>5} {'R数':>5} {'R/日':>5} "
          f"{'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>7}")
    results = []
    for k in (2, 3, 4, None):
        for gami_kept in (4.0, 5.0, 6.0, 7.0):
            for gap12_min in (0.07, 0.10):
                for gap23_min in (0.0, 1.0):
                    n, h, b, pp = eval_s(rows, k, gami_kept, gap12_min, gap23_min)
                    if n == 0:
                        continue
                    results.append((k, gami_kept, gap12_min, gap23_min, n, h, b, pp))
    # 的中率15%以上 & ROI 100%以上を優先表示、次いでROI降順
    for k, gm, g12, g23, n, h, b, pp in sorted(
            results, key=lambda x: -(x[7] / x[6])):
        hr = h / n
        roi = pp / b
        if hr < 0.13 or n / days < 1.0:
            continue
        ks = "all" if k is None else str(k)
        print(f"{ks:>4} {gm:>5.1f} {g12:>6.2f} {g23:>5.1f} {n:>5} {n/days:>5.1f} "
              f"{hr:>6.1%} {b:>9,} {pp:>9,} {roi:>6.1%}")


def main():
    print("2025年: lgbm_wt_train_only（11-12月=真OOS）", flush=True)
    model = load_model("lgbm_wt_train_only")
    rows25 = collect(model, "2025-01-01", "2025-12-31")
    sweep(rows25, "2025年", 365)

    print("\n2026-06: lgbm_wt_june_eval（真OOS）", flush=True)
    model2 = load_model("lgbm_wt_june_eval")
    rows26 = collect(model2, "2026-06-01", "2026-06-19")
    sweep(rows26, "2026-06", 19)


if __name__ == "__main__":
    main()
