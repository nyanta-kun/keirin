"""S4(SS/S): 軸2車の級班（player_class）denyフィルターが日次選出（axis_sumランキング）
に与える影響を検証する。

S1で発見した「軸がS1級/A1級(各グレード最上位)だと配当が下がりやすい」現象が
S4のSS/Sでも本番実績データの単純集計で確認できた（SS: ROI158.9%→360.9%等）が、
S4は日次でaxis_sum昇順に上位N件を採用する「相対選出」方式のため、格上軸を
含む候補を除外すると別候補が繰り上がる。この繰り上がりを反映した状態で
効果を検証する。

正規プロトコル: train+val(〜2026-03-31)で選定 → test(2026-04-01〜2026-07-22)で
一度だけ評価。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_axis_class_deny_analysis.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_PATH = Path("/tmp/exp_s4_axis_class_cache.pkl")

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    S4_DAILY_TOP_N, S4_STAKE, s4_daily_select, s4_select_axis, s4_wt_overlap_n,
)

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
    ("2026-04-13", "2026-07-22", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

TRAIN_VAL_END = "2026-03-31"
DENY_CLASS = {"S1", "A1"}


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


def collect(model_name, win_model_name, date_from, date_to):
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

    candidates: list[dict] = []
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

        axis1_class = class_map.get(axis1)
        axis2_class = class_map.get(axis2)
        has_top_class = axis1_class in DENY_CLASS or axis2_class in DENY_CLASS

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "wt_overlap_n": wt_overlap_n,
            "axis1_class": axis1_class, "axis2_class": axis2_class,
            "has_top_class": has_top_class,
        })
    return candidates


def main():
    if CACHE_PATH.exists():
        print(f"[cache] {CACHE_PATH} からロード", flush=True)
        with open(CACHE_PATH, "rb") as f:
            all_candidates, pm_all = pickle.load(f)
    else:
        all_candidates = []
        pm_all = {}
        for date_from, date_to, eval_model, win_model in QUARTERS:
            print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
            cs = collect(eval_model, win_model, date_from, date_to)
            print(f"[collect]   {len(cs)}件収集(選出前の全候補)", flush=True)
            all_candidates.extend(cs)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((all_candidates, pm_all), f)
        print(f"[cache] {CACHE_PATH} に保存", flush=True)

    pm = _load_payouts_wt(list({c["race_key"] for c in all_candidates}))

    train_val_cands = [c for c in all_candidates if c["race_date"] <= TRAIN_VAL_END]
    test_cands = [c for c in all_candidates if c["race_date"] > TRAIN_VAL_END]

    def by_day(cands):
        d = defaultdict(list)
        for c_ in cands:
            d[c_["race_date"]].append(c_)
        return d

    def run(cands, deny_top_class):
        day_map = by_day(cands)
        out = {"SS": [0, 0, 0, 0], "S": [0, 0, 0, 0]}  # n, hits, bet, pay
        for d, day_cands in day_map.items():
            pool = [c for c in day_cands if not (deny_top_class and c["has_top_class"])] \
                if deny_top_class else day_cands
            for c_ in s4_daily_select(pool, cap=S4_DAILY_TOP_N):
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
                gl = {0: "SS", 1: "S"}.get(c_["wt_overlap_n"])
                if gl not in out:
                    continue
                out[gl][0] += 1
                out[gl][1] += int(hit)
                out[gl][2] += bet
                out[gl][3] += pay
        return out

    for label, cands in (("train+val", train_val_cands), ("test", test_cands)):
        print(f"\n{'='*90}\n[{label}]\n{'='*90}")
        base = run(cands, deny_top_class=False)
        deny = run(cands, deny_top_class=True)
        for gate in ("SS", "S"):
            n0, h0, b0, p0 = base[gate]
            n1, h1, b1, p1 = deny[gate]
            roi0 = p0 / b0 * 100 if b0 else 0
            roi1 = p1 / b1 * 100 if b1 else 0
            hr0 = h0 / n0 * 100 if n0 else 0
            hr1 = h1 / n1 * 100 if n1 else 0
            print(f"\n■ {gate}")
            print(f"  ベースライン: n={n0} 的中={h0}({hr0:.1f}%) 投資={b0:,} 回収={p0:,} ROI={roi0:.1f}%")
            print(f"  denyフィルター後: n={n1} 的中={h1}({hr1:.1f}%) 投資={b1:,} 回収={p1:,} ROI={roi1:.1f}%")


if __name__ == "__main__":
    main()
