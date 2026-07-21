"""S(重なり1)ランク: 朝夕それぞれ上限6件（合計最大12件）に分割する方式を検証する。

ユーザー提案(2026-07-22): 朝夕同一週内であればモデルは同じ（週次retrainのため）
なのでaxis_sumのスケール比較は妥当。朝夕の上限をそれぞれ6件とし、夕方の
良い候補を確実に拾えるようにできないか。

3パターンを比較する:
  A. 理論上限:   朝夕を区別せず1日分を統合してaxis_sum昇順 上位10件
                （もし完全に朝夕同時に判定できたら、という上限ベンチマーク）
  B. 現行実装:   朝の生候補から上位10件を選出 → 夕方は「10 - 朝選出数」を
                夜の生候補から追加選出（2026-07-21デプロイ済みの現行ロジック）
  C. 6+6分割案: 朝の生候補から上位6件 + 夜の生候補から上位6件（合計最大12件）
                （今回のユーザー提案）

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_6plus6_split.py
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
HALF_CAP = 6


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
        print(f"[collect] {date_from}〜{date_to}", flush=True)
        cs = collect_candidates(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(cs)}R収集", flush=True)
        all_candidates.extend(cs)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        by_day[c["race_date"]].append(c)
    n_days = len(by_day)

    rows_a, rows_b, rows_c = [], [], []  # A=理論上限 B=現行実装 C=6+6分割
    for d, cands in by_day.items():
        day_part = sorted((c for c in cands if c["hour_jst"] is not None and c["hour_jst"] < 19),
                           key=lambda c: c["axis_sum"])
        night_part = sorted((c for c in cands if c["hour_jst"] is not None and c["hour_jst"] >= 19),
                             key=lambda c: c["axis_sum"])
        combined = sorted(cands, key=lambda c: c["axis_sum"])

        rows_a.extend(combined[:S4_DAILY_TOP_N])

        day_sel_b = day_part[:S4_DAILY_TOP_N]
        remaining = max(0, S4_DAILY_TOP_N - len(day_sel_b))
        rows_b.extend(day_sel_b)
        rows_b.extend(night_part[:remaining])

        rows_c.extend(day_part[:HALF_CAP])
        rows_c.extend(night_part[:HALF_CAP])

    print("\n" + "=" * 100)
    print(f"対象日数: {n_days}日\n")
    for label, rows in (
        ("A. 理論上限（朝夕統合・日次上位10件）", rows_a),
        ("B. 現行実装（朝上位10件→夕方が残り枠を充当）", rows_b),
        (f"C. 6+6分割案（朝上位{HALF_CAP}件+夜上位{HALF_CAP}件・最大{HALF_CAP*2}件）", rows_c),
    ):
        n, hits, hit_rate, bet, pay, roi = _settle(rows)
        print(f"[{label}]")
        print(f"  n={n}R ({n/n_days:.2f}R/日) 的中={hits}({hit_rate:.1f}%) "
              f"投資={bet:,} 回収={pay:,} ROI={roi:.1f}%\n")

    # A(理論上限)と各方式の一致率（同じレースを選べているか）
    set_a = {c["race_key"] for c in rows_a}
    set_b = {c["race_key"] for c in rows_b}
    set_c = {c["race_key"] for c in rows_c}
    print(f"理論上限(A)との一致率: B(現行)={len(set_a & set_b)/len(set_a)*100:.1f}%  "
          f"C(6+6)={len(set_a & set_c)/len(set_a)*100:.1f}%")


if __name__ == "__main__":
    main()
