"""S(重なり1)ランク: 日次件数固定(現行 S4_DAILY_TOP_N=10)ではなく、
axis_sum(波乱度指数)しきい値方式に変えた場合の特性を検証する。

ユーザー依頼(2026-07-22): 朝夕2回に分けて生成する現行方式では、朝が先に
axis_sum上位から日次上限まで埋めてしまい、夕方により良い候補があっても
先着順で取りこぼす構造的懸念がある。件数固定ではなく「一定の質(axis_sum)を
満たす候補は全て採用」というしきい値方式に変更した場合の1日あたり対象レース数・
的中率・ROIを検証し、現行の日次10件が実質どの程度のaxis_sumしきい値に相当するかを
あわせて確認する。

collect_candidates() は exp_s4_avoid_chalk_race.py と同一ロジック
（軸選定・WT重なり判定はオッズ非依存・精算のみオッズを使用）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_threshold_daily_cap.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S4_DAILY_TOP_N, S4_STAKE, s4_select_axis, s4_wt_overlap_n

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
    ("2026-04-13", "2026-07-10", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

THRESHOLDS = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.20, 0.25, 0.30]


def _load_trio_boards(race_keys):
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
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


def collect_candidates(eval_model_name, win_model_name, date_from, date_to):
    """S(重なり1)候補（軸選定＋的中判定）を日次選出前の全件収集する。"""
    model = load_model(eval_model_name)
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
        fins: dict[str, list[tuple[int, int]]] = {}
        marks: dict[str, dict[int, int]] = {}
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
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s4_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        if wt_overlap_n != 1:  # S(重なり1)のみ対象
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)

        buy_combos = [frozenset({axis1, axis2, x}) for x in others
                      if frozenset({axis1, axis2, x}) in trio]
        hit = actual_top3 in buy_combos
        bet = len(buy_combos) * S4_STAKE
        pay = trio_pay * S4_STAKE // 100 if hit else 0

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis_sum": axis_sum, "hit": hit, "bet": bet, "pay": pay,
        })
    return candidates


def _settle(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    roi = pay / bet * 100 if bet else 0
    hit_rate = hits / n * 100 if n else 0
    return n, hits, hit_rate, bet, pay, roi


def main():
    all_candidates = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cs = collect_candidates(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(cs)}R（S候補全体）収集", flush=True)
        all_candidates.extend(cs)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        by_day[c["race_date"]].append(c)
    n_days = len(by_day)

    print("\n" + "=" * 100)
    print(f"対象日数: {n_days}日  全S候補: {len(all_candidates)}R\n")

    # ── 現行方式: 日次axis_sum昇順 上位S4_DAILY_TOP_N件（時系列先着順の影響を除いた
    # 「もし朝夕を区別せず1日分をまとめて選出できたら」の理論上限として評価） ──
    baseline_rows: list[dict] = []
    tenth_axis_sums: list[float] = []  # 日毎の「10件目のaxis_sum」＝現行が実質採用しているしきい値
    for d, day_cands in by_day.items():
        day_cands.sort(key=lambda c: c["axis_sum"])
        baseline_rows.extend(day_cands[:S4_DAILY_TOP_N])
        if len(day_cands) >= S4_DAILY_TOP_N:
            tenth_axis_sums.append(day_cands[S4_DAILY_TOP_N - 1]["axis_sum"])

    n, hits, hit_rate, bet, pay, roi = _settle(baseline_rows)
    print(f"[現行方式] 日次上位{S4_DAILY_TOP_N}件（1日分を朝夕分けず理論上限選出した場合）")
    print(f"  n={n}R ({n/n_days:.2f}R/日) 的中={hits}({hit_rate:.1f}%) "
          f"投資={bet:,} 回収={pay:,} ROI={roi:.1f}%")
    if tenth_axis_sums:
        tenth_axis_sums.sort()
        mid = len(tenth_axis_sums) // 2
        median_10th = tenth_axis_sums[mid]
        avg_10th = sum(tenth_axis_sums) / len(tenth_axis_sums)
        print(f"  参考: 日次10件目のaxis_sum(=現行が実質採用しているしきい値相当) "
              f"中央値={median_10th:.4f}  平均={avg_10th:.4f}"
              f"（対象={len(tenth_axis_sums)}日/{n_days}日＝候補が10件以上あった日）")

    # ── しきい値方式: axis_sum <= threshold を満たす候補を件数上限なしで全採用 ──
    print("\n" + "=" * 100)
    print("[しきい値方式] axis_sum <= threshold を満たす候補を件数上限なしで全採用")
    print("=" * 100)
    print(f"{'閾値':>8s} {'対象R':>7s} {'R/日':>7s} {'的中':>6s} {'的中率':>7s} "
          f"{'投資':>10s} {'回収':>10s} {'ROI':>8s}")
    print("-" * 100)
    for th in THRESHOLDS:
        sel = [c for c in all_candidates if c["axis_sum"] <= th]
        n, hits, hit_rate, bet, pay, roi = _settle(sel)
        print(f"{th:8.2f} {n:7d} {n/n_days:7.2f} {hits:6d} {hit_rate:6.1f}% "
              f"{bet:10,d} {pay:10,d} {roi:7.1f}%")


if __name__ == "__main__":
    main()
