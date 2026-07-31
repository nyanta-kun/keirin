#!/usr/bin/env python3
"""S9/9A: 現行ロジック(entropy+mark3) vs 7S/7Aスタイル(axis_sum+entropy・mark3撤廃)
の3ヶ月サンプル比較（読み取り専用・DB書き込みなし）。

背景: 現行S9/9AはS7/7Aの旧3ゲート設計を踏襲したまま（axis_sum閾値は9車で
未較正のため未導入）。2026-07-31にS7/7Aはmark3ゲートを撤廃しaxis_sum+entropy
の2ゲートへ再設計済み。9車でも同型の再設計が有効か、月次vintageモデルによる
honest walk-forwardで直近3ヶ月をサンプル確認する。

axis_sum閾値は9車では未較正のため、この3ヶ月サンプル内の実候補分布から
複数の候補閾値（percentileベース）を試し、現行ロジックとの比較を行う。
あくまで予備調査であり、全期間honest検証を経てから本番判断すること。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s9_9a_axis_sum_recalibration.py \
        --start 2026-05-01 --end 2026-07-31
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    S7_ENTROPY_MAX, S9_ENTROPY_MAX, S7_MARK3_OVERLAP_MAX, S9_STAKE,
    s7_field_entropy, s7_select_axis, s7_wt_mark3_overlap_n, s7_wt_overlap_n,
)
from src.wt_vintage_config import monthly_windows

N_CAR = 9


def _load_trio_boards(race_keys: list[str]) -> dict:
    import re
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def build_candidates(model_name: str, date_from: str, date_to: str,
                      win_model_name: str) -> tuple[list[dict], dict, dict]:
    """9車立ての生候補（ゲート適用前・axis選定成功分のみ）を返す。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return [], {}, {}
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rksN = [rk for rk, ne in ne_map.items() if ne and int(ne) == N_CAR]
        fins: dict[str, list[tuple[int, int]]] = {}
        marks: dict[str, dict[int, int]] = {}
        for i in range(0, len(rksN), 900):
            chunk = rksN[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rksN))].copy()
    if df.empty:
        return [], {}, {}
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    candidates: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != N_CAR or len(g) != N_CAR:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != N_CAR:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s7_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = s7_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue

        others = sorted(board - {axis1, axis2})
        if len(others) != N_CAR - 2:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3_overlap_n = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others, "wt_overlap_n": wt_overlap_n,
            "wt_mark3_overlap_n": wt_mark3_overlap_n,
            "actual_top3": actual_top3,
        })
    return candidates, trio_bd, pm


def score(selected: list[dict], trio_bd: dict, pm: dict, stake: int = S9_STAKE) -> dict:
    n_hit = 0
    bet = 0
    pay = 0
    for c_ in selected:
        rk = c_["race_key"]
        trio = trio_bd.get(rk, {})
        combos = [frozenset({c_["axis1"], c_["axis2"], x}) for x in c_["others"]
                  if frozenset({c_["axis1"], c_["axis2"], x}) in trio]
        if not combos:
            continue
        hit = c_["actual_top3"] in combos
        trio_pay = pm.get(rk, {}).get(("trio", c_["actual_top3"]), 0)
        p = trio_pay * stake // 100 if hit else 0
        b = len(combos) * stake
        n_hit += int(hit)
        bet += b
        pay += p
    n = len([c_ for c_ in selected if trio_bd.get(c_["race_key"])])
    roi = pay / bet * 100 if bet else 0.0
    return {"n": n, "hit": n_hit, "bet": bet, "pay": pay, "roi": roi}


def current_s9(candidates: list[dict]) -> list[dict]:
    return [c for c in candidates
            if c.get("entropy", float("inf")) <= S9_ENTROPY_MAX
            and c.get("wt_overlap_n") in (0, 1)
            and c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX]


def current_9a(candidates: list[dict]) -> list[dict]:
    out = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        mark3 = c.get("wt_mark3_overlap_n")
        if mark3 is None:
            continue
        ent_ok = c.get("entropy", float("inf")) <= S9_ENTROPY_MAX
        mark3_ok = mark3 <= S7_MARK3_OVERLAP_MAX
        n_fail = (not ent_ok) + (not mark3_ok)
        if n_fail == 1:
            out.append(c)
    return out


def new_s9(candidates: list[dict], axis_sum_max: float, entropy_max: float) -> list[dict]:
    return [c for c in candidates
            if c["axis_sum"] <= axis_sum_max
            and c.get("entropy", float("inf")) <= entropy_max
            and c.get("wt_overlap_n") in (0, 1)]


def new_9a(candidates: list[dict], axis_sum_max: float, entropy_max: float) -> list[dict]:
    out = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        axis_ok = c["axis_sum"] <= axis_sum_max
        ent_ok = c.get("entropy", float("inf")) <= entropy_max
        n_fail = (not axis_ok) + (not ent_ok)
        if n_fail == 1:
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-07-31")
    args = ap.parse_args()

    windows = [w for w in monthly_windows()
               if w[0] >= args.start[:7] + "-01" and w[0] <= args.end]

    all_candidates: list[dict] = []
    all_trio: dict = {}
    all_pm: dict = {}

    for date_from, date_to, eval_model, win_model in windows:
        print(f"\n[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cands, trio_bd, pm = build_candidates(eval_model, date_from, date_to, win_model)
        print(f"[build]   9車axis選定成功候補: {len(cands)}件", flush=True)
        all_candidates.extend(cands)
        all_trio.update(trio_bd)
        all_pm.update(pm)

    print(f"\n{'=' * 100}")
    print(f"3ヶ月サンプル合計: 9車axis選定成功候補 {len(all_candidates)}件 "
          f"({args.start}〜{args.end})")
    print(f"{'=' * 100}")

    base_pool = [c for c in all_candidates if c.get("wt_overlap_n") in (0, 1)]
    print(f"wt_overlap_n∈{{0,1}}のbase pool: {len(base_pool)}件")
    axis_sums = sorted(c["axis_sum"] for c in base_pool)
    if axis_sums:
        def pct(p):
            idx = min(len(axis_sums) - 1, int(len(axis_sums) * p))
            return axis_sums[idx]
        print(f"axis_sum分布(base pool内): min={axis_sums[0]:.3f} "
              f"p25={pct(0.25):.3f} p40={pct(0.40):.3f} p50={pct(0.50):.3f} "
              f"p60={pct(0.60):.3f} p75={pct(0.75):.3f} max={axis_sums[-1]:.3f}")

    print(f"\n{'=' * 100}")
    print("【現行ロジック】S9(entropy+mark3・axis_sum未導入) / 9A(1of2不合格)")
    print(f"{'=' * 100}")
    r = score(current_s9(all_candidates), all_trio, all_pm)
    print(f"  S9(現行): {r['n']}R 的中{r['hit']} ({r['hit']/r['n']*100 if r['n'] else 0:.1f}%) "
          f"投資{r['bet']:,} → 回収{r['pay']:,} ROI {r['roi']:.1f}%")
    r = score(current_9a(all_candidates), all_trio, all_pm)
    print(f"  9A(現行): {r['n']}R 的中{r['hit']} ({r['hit']/r['n']*100 if r['n'] else 0:.1f}%) "
          f"投資{r['bet']:,} → 回収{r['pay']:,} ROI {r['roi']:.1f}%")

    print(f"\n{'=' * 100}")
    print("【7S/7Aスタイル】axis_sum(候補閾値)+entropy 2ゲート・mark3撤廃")
    print(f"{'=' * 100}")
    if axis_sums:
        candidate_thresholds = sorted(set([
            round(pct(0.25), 2), round(pct(0.40), 2), round(pct(0.50), 2),
            round(pct(0.60), 2), 1.5,
        ]))
    else:
        candidate_thresholds = [1.5]

    for th in candidate_thresholds:
        sel_s9 = new_s9(all_candidates, th, S9_ENTROPY_MAX)
        sel_9a = new_9a(all_candidates, th, S9_ENTROPY_MAX)
        r9 = score(sel_s9, all_trio, all_pm)
        ra = score(sel_9a, all_trio, all_pm)
        print(f"  axis_sum<={th:.2f}:")
        print(f"    S9相当: {r9['n']}R 的中{r9['hit']} "
              f"({r9['hit']/r9['n']*100 if r9['n'] else 0:.1f}%) "
              f"投資{r9['bet']:,} → 回収{r9['pay']:,} ROI {r9['roi']:.1f}%")
        print(f"    9A相当: {ra['n']}R 的中{ra['hit']} "
              f"({ra['hit']/ra['n']*100 if ra['n'] else 0:.1f}%) "
              f"投資{ra['bet']:,} → 回収{ra['pay']:,} ROI {ra['roi']:.1f}%")


if __name__ == "__main__":
    main()
