"""S(重なり1)候補が「昼の部だけ」で日次上限(10件)に達する日がどれだけあるか確認する。

ユーザー質問(2026-07-22): 朝夕統合で選び直す方式にしても、そもそも朝(19時発走未満)
だけでS候補が10件を超える日があるなら、朝の時点で枠を使い切ってしまい夜の候補が
入る余地がなくなるのでは？ を検証する。

collect_candidates() は exp_s4_threshold_daily_cap.py と同一（軸選定はオッズ非依存）。
各候補のレース発走時刻(JST)を取得し、19時未満=昼の部・19時以降=夜の部に分類して
日別に集計する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_day_night_split_check.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

JST = timezone(timedelta(hours=9))


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
        start_map = dict(c.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date BETWEEN ? AND ?",
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
        if wt_overlap_n != 1:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        buy_combos = [frozenset({axis1, axis2, x}) for x in others
                      if frozenset({axis1, axis2, x}) in trio]
        hit = actual_top3 in buy_combos
        bet = len(buy_combos) * S4_STAKE
        pay = trio_pay * S4_STAKE // 100 if hit else 0

        start_at = start_map.get(rk)
        hour_jst = None
        if start_at:
            try:
                hour_jst = datetime.fromtimestamp(int(start_at), tz=JST).hour
            except (ValueError, TypeError, OSError):
                hour_jst = None

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis_sum": axis_sum, "hit": hit, "bet": bet, "pay": pay,
            "hour_jst": hour_jst,
        })
    return candidates


def main():
    all_candidates = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[collect] {date_from}〜{date_to}", flush=True)
        cs = collect_candidates(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(cs)}R収集", flush=True)
        all_candidates.extend(cs)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        by_day[c["race_date"]].append(c)

    n_days = len(by_day)
    day_ge10 = 0        # 昼の部(19時未満)だけでS候補が10件以上ある日
    day_counts = []
    night_counts = []
    for d, cands in by_day.items():
        day_part = [c for c in cands if c["hour_jst"] is not None and c["hour_jst"] < 19]
        night_part = [c for c in cands if c["hour_jst"] is not None and c["hour_jst"] >= 19]
        day_counts.append(len(day_part))
        night_counts.append(len(night_part))
        if len(day_part) >= S4_DAILY_TOP_N:
            day_ge10 += 1

    day_counts.sort()
    night_counts.sort()

    def _pctl(sorted_list, p):
        if not sorted_list:
            return 0
        idx = min(len(sorted_list) - 1, int(len(sorted_list) * p))
        return sorted_list[idx]

    print("\n" + "=" * 90)
    print(f"対象日数: {n_days}日")
    print(f"昼の部(19時未満発走)だけでS候補が{S4_DAILY_TOP_N}件以上ある日: "
          f"{day_ge10}日 ({day_ge10/n_days*100:.1f}%)")
    print(f"\n昼の部 候補数/日: 平均={sum(day_counts)/n_days:.2f}  中央値={_pctl(day_counts,0.5)}  "
          f"25%tile={_pctl(day_counts,0.25)}  75%tile={_pctl(day_counts,0.75)}  最大={max(day_counts)}")
    print(f"夜の部 候補数/日: 平均={sum(night_counts)/n_days:.2f}  中央値={_pctl(night_counts,0.5)}  "
          f"25%tile={_pctl(night_counts,0.25)}  75%tile={_pctl(night_counts,0.75)}  最大={max(night_counts)}")


if __name__ == "__main__":
    main()
