"""7S/7A 三連単フォーメーション買いの検証（2026-07-31）。

現行の三連複5点均等買い（軸2車＋残り流し）に代えて、三連単の2つのフォーメーション
を均等買いで検証する:

  パターンA（全10点）: 1軸目(axis1)を1着固定・2軸目(axis2)を2着or3着・
    残り5車のいずれかがもう一方の着 → 2(axis2の位置)×5(相手)=10点
  パターンB（全20点）: 1着は軸2車のいずれか・残る軸ともう一方の着に
    残り5車のいずれか → 2(1着候補)×2(残り軸の位置)×5(相手)=20点
    （軸2車がともに上位2着内・3着目は他5車のいずれか、という条件を
      全通りの着順で網羅する構成）

母集団はS7+7A（2026-07-31改定後の現行2ゲート版）。月次凍結vintageモデルに
よるhonest walk-forward。読み取り専用・DB書き込みなし。均等買い（1点あたり
定額）で検証し、傾斜配分は行わない。
"""
import re
import statistics
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    s7_evening_reselect, s7_field_entropy, s7_select_axis, s7_wt_mark3_overlap_n,
    s7_wt_overlap_n, s7a_daily_select,
)
from src.wt_vintage_config import monthly_windows

N_CAR = 7
UNIT_STAKE = 100.0  # 1点あたりの均等賭け金


def _load_trifecta_odds(race_keys: list[str]) -> dict:
    """wt_odds(bet_type='trifecta') → {race_key: {(1着,2着,3着): odds}}（順序あり）。"""
    out: dict = defaultdict(dict)
    if not race_keys:
        return out
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trifecta' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(int(x) for x in str(comb).split("-"))
                except ValueError:
                    continue
                if len(parts) == 3:
                    out[rk][parts] = fv
    return out


def build_candidates(model_name: str, date_from: str, date_to: str, win_model_name: str) -> tuple[list[dict], dict]:
    """S7/7A母集団の生候補（axis選定成功分・実着順order付き）を返す。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return [], {}
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == N_CAR]
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
        return [], {}
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trifecta_odds = _load_trifecta_odds(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    candidates: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != N_CAR or len(g) != N_CAR:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue
        board = set(int(r.frame_no) for r in g.itertuples(index=False))
        if len(board) != N_CAR:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
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

        order3 = tuple(fno for _, fno in fin[:3])  # 実際の着順(1着,2着,3着)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3_overlap_n = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others, "order3": order3,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n,
        })
    return candidates, trifecta_odds


def pattern_a_combos(axis1: int, axis2: int, others: list[int]) -> list[tuple[int, int, int]]:
    """全10点: 1着=axis1固定・2着/3着=axis2+他5車の並び。"""
    combos = []
    for x in others:
        combos.append((axis1, axis2, x))
        combos.append((axis1, x, axis2))
    return combos


def pattern_b_combos(axis1: int, axis2: int, others: list[int]) -> list[tuple[int, int, int]]:
    """全20点: 1着=軸2車のいずれか・残り2着3着=もう一方の軸+他5車の並び。"""
    combos = []
    for first, other_axis in ((axis1, axis2), (axis2, axis1)):
        for x in others:
            combos.append((first, other_axis, x))
            combos.append((first, x, other_axis))
    return combos


def evaluate(selected: list[dict], trifecta_odds: dict, combo_fn) -> dict:
    n = 0
    bet = 0.0
    ret = 0.0
    hit = 0
    for c_ in selected:
        rk = c_["race_key"]
        odds_map = trifecta_odds.get(rk, {})
        combos = combo_fn(c_["axis1"], c_["axis2"], c_["others"])
        avail = [cb for cb in combos if cb in odds_map]
        if len(avail) < 2:
            continue
        n += 1
        b = UNIT_STAKE * len(avail)
        bet += b
        if c_["order3"] in avail:
            hit += 1
            ret += UNIT_STAKE * odds_map[c_["order3"]]
    return {"n": n, "bet": bet, "ret": ret, "hit": hit}


def main(date_from_filter: str | None = None, date_to_filter: str | None = None, label: str = "全期間") -> None:
    windows = monthly_windows()
    if date_from_filter:
        windows = [w for w in windows if w[1] >= date_from_filter and w[0] <= date_to_filter]
    print(f"[main] {label}: 月次窓数={len(windows)}", flush=True)

    patterns = {"A(10点)": pattern_a_combos, "B(20点)": pattern_b_combos}
    totals = {p: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0} for p in patterns}
    monthly_roi = {p: [] for p in patterns}

    for date_from, date_to, eval_model, win_model in windows:
        candidates, trifecta_odds = build_candidates(eval_model, date_from, date_to, win_model)
        if not candidates:
            continue
        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        selected = []
        for _d, day_cands in by_day.items():
            selected.extend(s7_evening_reselect(day_cands, [], set()))
            selected.extend(s7a_daily_select(day_cands))
        if not selected:
            continue

        line = f"[{date_from}〜{date_to}] n_selected={len(selected)}"
        for pname, fn in patterns.items():
            r = evaluate(selected, trifecta_odds, fn)
            totals[pname]["n"] += r["n"]
            totals[pname]["bet"] += r["bet"]
            totals[pname]["ret"] += r["ret"]
            totals[pname]["hit"] += r["hit"]
            roi = r["ret"] / r["bet"] * 100 if r["bet"] else 0.0
            monthly_roi[pname].append(roi if r["bet"] else None)
            line += f"  {pname}: n={r['n']} 的中{r['hit']} ROI={roi:.1f}%"
        print(line, flush=True)

    print("\n" + "=" * 100)
    print("全期間合計")
    print("=" * 100)
    for pname in patterns:
        t = totals[pname]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        hitrate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        vals = [v for v in monthly_roi[pname] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{pname}: {t['n']}R 的中{t['hit']} ({hitrate:.1f}%) "
              f"投資{t['bet']:,.0f} → 回収{t['ret']:,.0f} ROI {roi:.1f}%  月次標準偏差={sd:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--label", default="全期間")
    args = ap.parse_args()
    main(args.date_from, args.date_to, args.label)
