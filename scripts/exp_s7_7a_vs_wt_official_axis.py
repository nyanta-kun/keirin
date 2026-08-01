"""7S/7A推奨レースにおける「推奨2軸」vs「WINTICKET公式◎◯2軸」の比較（2026-07-31）。

同一レース（S7/7A採用済み・母集団は完全に同一）・同一買い目構造（軸2車＋残り
5車流しの三連複5点均等買い）で、軸の選び方だけを差し替えた場合の的中率・ROIを
比較する。

注: S7/7Aの選出条件はwt_overlap_n∈{0,1}（推奨2軸がWT公式◎◯と完全一致=2の
レースは対象外）のため、この母集団では推奨軸とWT軸は構造的に必ず異なる
（部分的に重なる場合はあるが完全一致はしない）。

母集団はS7+7A（2026-07-31改定後の現行2ゲート版）。月次凍結vintageモデルに
よるhonest walk-forward。読み取り専用・DB書き込みなし。
"""
import sys
from collections import defaultdict
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

from scripts.backfill_7s_rank_wt import _load_trio_boards

N_CAR = 7
STAKE_PER_POINT = 2000.0  # 本番と同一（5点×2,000円=10,000円）


def build_candidates(model_name: str, date_from: str, date_to: str, win_model_name: str) -> tuple[list[dict], dict]:
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
            "others": others, "trio": trio, "actual_top3": actual_top3, "board": board,
            "wt_honmei": wt_honmei, "wt_taikou": wt_taikou,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n,
        })
    return candidates, pm


def _flush_result(axis_a: int, axis_b: int, board: set, trio: dict, actual_top3: frozenset) -> tuple[int, float, float] | None:
    """軸(axis_a,axis_b)+残り流しの5点trio均等買い結果を返す (n_points, bet, ret)。
    軸2車がboardに無い/同一車の場合はNone。
    """
    if axis_a == axis_b or axis_a not in board or axis_b not in board:
        return None
    others = sorted(board - {axis_a, axis_b})
    combos = [frozenset({axis_a, axis_b, x}) for x in others if frozenset({axis_a, axis_b, x}) in trio]
    if len(combos) < 2:
        return None
    bet = STAKE_PER_POINT * len(combos)
    ret = 0.0
    if actual_top3 in combos:
        ret = trio[actual_top3] * STAKE_PER_POINT
    return len(combos), bet, ret


def main(date_from_filter: str | None = None, date_to_filter: str | None = None, label: str = "全期間") -> None:
    windows = monthly_windows()
    if date_from_filter:
        windows = [w for w in windows if w[1] >= date_from_filter and w[0] <= date_to_filter]
    print(f"[main] {label}: 月次窓数={len(windows)}", flush=True)

    totals = {
        "推奨2軸(現行)": {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0},
        "WT公式◎◯2軸": {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0},
    }
    skipped_wt_missing = 0

    for date_from, date_to, eval_model, win_model in windows:
        candidates, pm = build_candidates(eval_model, date_from, date_to, win_model)
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

        m_ours = {"bet": 0.0, "ret": 0.0}
        m_wt = {"bet": 0.0, "ret": 0.0}
        n_ours = n_wt = hit_ours = hit_wt = 0
        n_missing = 0

        for c_ in selected:
            r_ours = _flush_result(c_["axis1"], c_["axis2"], c_["board"], c_["trio"], c_["actual_top3"])
            if r_ours is not None:
                _, bet, ret = r_ours
                m_ours["bet"] += bet
                m_ours["ret"] += ret
                n_ours += 1
                if ret > 0:
                    hit_ours += 1

            wt_h, wt_t = c_["wt_honmei"], c_["wt_taikou"]
            if wt_h is None or wt_t is None:
                n_missing += 1
                continue
            r_wt = _flush_result(wt_h, wt_t, c_["board"], c_["trio"], c_["actual_top3"])
            if r_wt is not None:
                _, bet, ret = r_wt
                m_wt["bet"] += bet
                m_wt["ret"] += ret
                n_wt += 1
                if ret > 0:
                    hit_wt += 1

        skipped_wt_missing += n_missing
        totals["推奨2軸(現行)"]["n"] += n_ours
        totals["推奨2軸(現行)"]["bet"] += m_ours["bet"]
        totals["推奨2軸(現行)"]["ret"] += m_ours["ret"]
        totals["推奨2軸(現行)"]["hit"] += hit_ours
        totals["WT公式◎◯2軸"]["n"] += n_wt
        totals["WT公式◎◯2軸"]["bet"] += m_wt["bet"]
        totals["WT公式◎◯2軸"]["ret"] += m_wt["ret"]
        totals["WT公式◎◯2軸"]["hit"] += hit_wt

        roi_ours = m_ours["ret"] / m_ours["bet"] * 100 if m_ours["bet"] else 0.0
        roi_wt = m_wt["ret"] / m_wt["bet"] * 100 if m_wt["bet"] else 0.0
        print(f"[{date_from}〜{date_to}] n_selected={len(selected)}  "
              f"推奨軸: n={n_ours} 的中{hit_ours} ROI={roi_ours:.1f}%  |  "
              f"WT軸: n={n_wt} 的中{hit_wt} ROI={roi_wt:.1f}%  (WT印欠損{n_missing}件)", flush=True)

    print(f"\n[main] WT印(◎ or ◯)欠損によりスキップ: {skipped_wt_missing}件")
    print("\n" + "=" * 100)
    print("全期間合計（同一レース・同一買い目構造・軸の選び方のみ比較）")
    print("=" * 100)
    for label_, t in totals.items():
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        hitrate = t["hit"] / t["n"] * 100 if t["n"] else 0.0
        print(f"{label_:<14}: {t['n']}R 的中{t['hit']} ({hitrate:.1f}%) "
              f"投資{t['bet']:,.0f} → 回収{t['ret']:,.0f} ROI {roi:.1f}%")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--label", default="全期間")
    args = ap.parse_args()
    main(args.date_from, args.date_to, args.label)
