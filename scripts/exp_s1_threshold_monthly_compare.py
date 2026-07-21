"""S1: top3_gap閾値0.15 vs 0.22 の月次比較（honest全期間・同一walk-forwardモデル使用）。

2026-07-21、ローカルpicks_historyのS1データが旧閾値0.15のまま（27.9R/日）で、
現行閾値0.22（想定15.3R/日）へのhonest再構築が実際にはDB反映されていなかったと
判明。ユーザーから「閾値0.22で高配当レースが対象外になっていないか、月次で
比較したい」との依頼を受け、rebuild_s1_walkforward.pyと同一の四半期
walk-forwardモデル群を使い、ゲート判定前の全レース（axis/p1/p2/top3_gap）を
収集した上で、0.15/0.22 両方の閾値を事後適用して月次集計する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_threshold_monthly_compare.py
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
from src.strategy_wt import S1W_STAKE, s1w_select

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

THRESHOLDS = (0.15, 0.22)


def _load_trifecta_boards(race_keys):
    tri = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trifecta' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    tri[rk][parts] = fv
    return tri


def collect_ungated(eval_model_name, win_model_name, date_from, date_to):
    """ゲート適用前の全レース（top3_gap計算済み）を収集する。"""
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        meta_map = dict(c.execute(
            "SELECT race_key, race_date || '|' || race_no || '|' || venue_id FROM wt_races "
            "WHERE race_date BETWEEN ? AND ?", (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins: dict[str, list[tuple[int, int]]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
        venue_names = dict(c.execute("SELECT venue_code, name FROM venue_info"))
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    tri_bd = _load_trifecta_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        tri = tri_bd.get(rk)
        if not tri:
            continue
        board: set[int] = set()
        for k in tri:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s1w_select(win_probs, top3_probs)
        if sel is None:
            continue
        axis, p1, p2, top3_gap = sel
        if axis not in board or p1 not in board or p2 not in board:
            continue
        combo_a = (axis, p1, p2)
        combo_b = (axis, p2, p1)
        buy = [c for c in (combo_a, combo_b) if tri.get(c) is not None]
        if not buy:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        hit = order3 in buy
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        pay = trifecta_pay * S1W_STAKE // 100 if hit else 0
        bet = len(buy) * S1W_STAKE

        meta = meta_map.get(rk, "||")
        race_date, race_no, venue_id = (meta.split("|") + ["", "", ""])[:3]
        venue_name = venue_names.get(int(venue_id), venue_id) if venue_id else "?"

        races.append({
            "race_key": rk, "race_date": race_date, "race_no": race_no,
            "venue_name": venue_name, "top3_gap": top3_gap,
            "hit": hit, "bet": bet, "pay": pay,
        })
    return races


def main():
    all_races = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rs = collect_ungated(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(rs)}R収集", flush=True)
        all_races.extend(rs)

    print(f"\n収集完了: 全{len(all_races)}R（ゲート適用前）\n")

    for th in THRESHOLDS:
        sel = [r for r in all_races if r["top3_gap"] >= th]
        by_month: dict[str, list[dict]] = defaultdict(list)
        for r in sel:
            ym = r["race_date"][:7]
            by_month[ym].append(r)

        print("=" * 100)
        print(f"閾値 top3_gap >= {th}   （全期間: n={len(sel)}）")
        print("=" * 100)
        print(f"{'年月':8s} {'対象R':>6s} {'的中':>5s} {'的中率':>7s} {'投資':>10s} {'回収':>10s} "
              f"{'ROI':>8s}   最高払戻(日時・レース)")
        print("-" * 100)
        for ym in sorted(by_month):
            rows = by_month[ym]
            n = len(rows)
            hits = sum(r["hit"] for r in rows)
            bet = sum(r["bet"] for r in rows)
            pay = sum(r["pay"] for r in rows)
            roi = pay / bet * 100 if bet else 0
            hit_rate = hits / n * 100 if n else 0
            best = max(rows, key=lambda r: r["pay"])
            best_desc = (f"{best['pay']:,}円 ({best['race_date']} {best['venue_name']}{best['race_no']}R)"
                         if best["pay"] > 0 else "—")
            print(f"{ym:8s} {n:6d} {hits:5d} {hit_rate:6.1f}% {bet:10,d} {pay:10,d} "
                  f"{roi:7.1f}%   {best_desc}")
        print()


if __name__ == "__main__":
    main()
