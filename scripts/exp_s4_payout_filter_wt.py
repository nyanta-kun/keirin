"""S4(SS+/SS/S)推奨レース内で、entropy/盤面min三連複オッズによる絞り込みが
実際に高配当(30倍+)ヒット・ROIを底上げできるかを検証する（2026-07-26）。

背景: exp_upset_trio30_v2_wt.py で「entropy(指数分散)」「min_trio_odds(盤面min配当)」
のみが2窓一貫でAUC>0.55、他（ライン/単勝率・複勝率ばらつき/抜け出し等）は無効と判明。
本スクリプトはこの2信号を「S4が既に選んだレース群の中」でさらに使い、
的中時配当の低い(3桁円)レースを避けられるかを実精算方式で検証する。

母集団・選出ロジックは scripts/backfill_s4_rank_wt.py と同一
（s4_select_axis / s4_daily_select / s4_gate_label）。

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_payout_filter_wt.py \
      --model lgbm_wt_eval --win-model lgbm_wt_win_eval \
      --windows 2026-04-24:2026-06-10 2026-06-11:2026-07-25
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    S4_DAILY_TOP_N, S4_STAKE, s4_daily_select, s4_gate_label, s4_select_axis, s4_wt_overlap_n,
)
from backfill_s4_rank_wt import _load_trio_boards

UPSET_ODDS = 30.0


def build(model_name, win_model_name, date_from, date_to):
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    candidates = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s4_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)

        # entropy / min_trio_odds（波乱シグナル。exp_upset_trio30_v2_wt.py と同定義）
        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        odds_all = np.array(list(trio.values()), dtype=float)
        min_trio_odds = float(odds_all.min())

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "wt_overlap_n": wt_overlap_n,
            "axis1_class": class_map.get(axis1), "axis2_class": class_map.get(axis2),
            "entropy": ent, "min_trio_odds": min_trio_odds,
        })

    by_day = defaultdict(list)
    for c_ in candidates:
        by_day[c_["race_date"]].append(c_)

    rows = []
    for d, day_cands in by_day.items():
        for c_ in s4_daily_select(day_cands, cap=S4_DAILY_TOP_N):
            axis1, axis2 = c_["axis1"], c_["axis2"]
            trio = c_["trio"]
            combos = []
            for x in c_["others"]:
                key = frozenset({axis1, axis2, x})
                if key in trio:
                    combos.append(key)
            if not combos:
                continue
            rk = c_["race_key"]
            hit = c_["actual_top3"] in combos
            trio_pay = pm.get(rk, {}).get(("trio", c_["actual_top3"]), 0)
            pay = trio_pay * S4_STAKE // 100 if hit else 0
            bet = len(combos) * S4_STAKE
            gate_label = s4_gate_label(c_["wt_overlap_n"], c_.get("axis1_class"), c_.get("axis2_class"))
            if gate_label is None:
                continue
            rows.append({
                "race_date": d, "race_key": rk, "gate_label": gate_label,
                "hit": int(hit), "payout": pay, "bet_amount": bet,
                "trio_payout": trio_pay,
                "entropy": c_["entropy"], "min_trio_odds": c_["min_trio_odds"],
                "axis_sum": c_["axis_sum"],
            })
    return rows


def summarize(rows, label):
    if not rows:
        print(f"  {label}: n=0")
        return
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    roi = pay / bet * 100 if bet else float("nan")
    upset_n = sum(1 for r in rows if r["hit"] and r["trio_payout"] >= UPSET_ODDS * 100)
    med_pay = np.median([r["trio_payout"] for r in rows if r["hit"]]) if hits else float("nan")
    print(f"  {label:<28} n={n:>4}  的中={hits:>3}({hits/n:.1%})  ROI={roi:>6.1f}%  "
          f"うち30倍+的中={upset_n}件  的中時中央値配当={med_pay:>7.0f}円")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--win-model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    all_rows = []
    for w in args.windows:
        f, t = w.split(":")
        rows = build(args.model, args.win_model, f, t)
        all_rows.extend(rows)
        print(f"\n===== {f} 〜 {t} =====")
        summarize(rows, "S4全体（現行本番選出ロジック）")
        for gl in ("SS+", "SS", "S"):
            summarize([r for r in rows if r["gate_label"] == gl], f"  内訳 {gl}")

        # entropy / min_trio_odds 四分位でのROI（S4選出済みレース内で）
        for feat in ("entropy", "min_trio_odds"):
            vals = np.array([r[feat] for r in rows], dtype=float)
            qs = np.percentile(vals, [25, 50, 75])
            print(f"  --- S4内 {feat} 四分位別 ---")
            for i, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)], 1):
                sel = [r for r, v in zip(rows, vals) if (v > lo and (v <= hi if hi != np.inf else True))]
                summarize(sel, f"    Q{i}")

    print("\n===== 全期間合算 =====")
    summarize(all_rows, "S4全体")
    for gl in ("SS+", "SS", "S"):
        summarize([r for r in all_rows if r["gate_label"] == gl], f"  内訳 {gl}")
    for feat in ("entropy", "min_trio_odds"):
        vals = np.array([r[feat] for r in all_rows], dtype=float)
        qs = np.percentile(vals, [25, 50, 75])
        print(f"  --- S4内 {feat} 四分位別（全期間） ---")
        for i, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)], 1):
            sel = [r for r, v in zip(all_rows, vals) if (v > lo and (v <= hi if hi != np.inf else True))]
            summarize(sel, f"    Q{i}")


if __name__ == "__main__":
    main()
