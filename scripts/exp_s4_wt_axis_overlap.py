"""S4: 2軸(axis1,axis2)がWINTICKET◎◯(prediction_mark 1,2)と完全一致する場合の
期待値検証（ユーザー仮説: 一致レースは市場に織り込まれ済みで高配当が出にくいのでは）。

やること:
  1. 現行S4選出（日次axis_sum昇順でS4_DAILY_TOP_N件）をhonest全期間(四半期
     walk-forwardモデル)で再構築する（rebuild_s4_walkforward.pyと同一ロジック）。
  2. 選出された各レースについて {axis1,axis2} が WT{◎,◯}（prediction_mark∈{1,2}
     の2車）と完全一致するかを判定し、一致/不一致で 件数・的中率・ROI・払戻分布
     （平均払戻・最大払戻・高配当件数）を分割集計する。
  3. 「一致」レースを除外し、当日の次点候補（axis_sum昇順で11番目以降）で
     穴埋めした場合のROIをシミュレーションする（除外のみ・穴埋めなし の2パターン）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_wt_axis_overlap.py [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S4_DAILY_TOP_N, S4_STAKE, s4_select_axis

# rebuild_s4_walkforward.py と同一の四半期定義（honest全期間再現のため）
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
]


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


def build_candidates(eval_model_name: str, win_model_name: str, date_from: str, date_to: str):
    """s4_select_axis による候補（採点済み・WT◎◯一致フラグ付き）を構築する。"""
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

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_top2 = {wt_honmei, wt_taikou} if wt_honmei is not None and wt_taikou is not None else None
        axis_set = {axis1, axis2}
        match = wt_top2 is not None and axis_set == wt_top2
        overlap_n = len(axis_set & wt_top2) if wt_top2 is not None else None

        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "trio_pay": trio_pay, "wt_match": match, "wt_overlap_n": overlap_n,
        })
    return candidates


def _settle(c_):
    """1レース分の5点流しを精算する。(bet, hit, payout)"""
    axis1, axis2 = c_["axis1"], c_["axis2"]
    trio = c_["trio"]
    combos = [frozenset({axis1, axis2, x}) for x in c_["others"]
              if frozenset({axis1, axis2, x}) in trio]
    if not combos:
        return None
    bet = len(combos) * S4_STAKE
    hit = c_["actual_top3"] in combos
    pay = c_["trio_pay"] * S4_STAKE // 100 if hit else 0
    return bet, hit, pay


def summarize(label, rows):
    n = hits = bet = pay = 0
    payouts = []
    for r in rows:
        s = _settle(r)
        if s is None:
            continue
        b, h, p = s
        n += 1
        bet += b
        pay += p
        if h:
            hits += 1
            payouts.append(p)
    roi = pay / bet * 100 if bet else 0
    hit_rate = hits / n * 100 if n else 0
    avg_pay = sum(payouts) / len(payouts) if payouts else 0
    max_pay = max(payouts) if payouts else 0
    high_2000 = sum(1 for p in payouts if p >= 2000)  # 単勝的中5点500円に対し4倍(20倍配当)以上
    high_5000 = sum(1 for p in payouts if p >= 5000)
    print(f"  [{label}] n={n:4d} 的中={hits:3d}({hit_rate:5.1f}%) 投資{bet:,} → 回収{pay:,} "
          f"ROI={roi:6.1f}%  払戻: 平均{avg_pay:,.0f}円 最大{max_pay:,}円 "
          f"≥2000円:{high_2000}件 ≥5000円:{high_5000}件")
    return {"n": n, "hits": hits, "bet": bet, "pay": pay, "roi": roi, "payouts": payouts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    if not args.end:
        from datetime import date, timedelta
        args.end = (date.today() - timedelta(days=1)).isoformat()

    quarters = list(QUARTERS)
    quarters.append(("2026-04-13", args.end, "lgbm_wt_eval", "lgbm_wt_win_eval"))

    all_selected: list[dict] = []       # 現行(baseline): 日次TopN選出
    all_excl_only: list[dict] = []      # 一致レースを除外するのみ（穴埋めなし）
    all_excl_fill: list[dict] = []      # 一致レースを除外し次点で穴埋め
    all_priority: list[dict] = []       # 重なり0を最優先で必ず採用→残り枠を重なり1で穴埋め（重なり2は完全除外）
    day_set: set[str] = set()

    for date_from, date_to, eval_model, win_model in quarters:
        print(f"\n[overlap] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cands = build_candidates(eval_model, win_model, date_from, date_to)

        by_day: dict[str, list[dict]] = defaultdict(list)
        for c_ in cands:
            by_day[c_["race_date"]].append(c_)

        for d, day_cands in by_day.items():
            day_set.add(d)
            day_cands.sort(key=lambda c_: c_["axis_sum"])
            baseline = day_cands[:S4_DAILY_TOP_N]
            all_selected.extend(baseline)

            # 除外のみ: baseline のうち wt_match でないものだけ残す
            excl_only = [c_ for c_ in baseline if not c_["wt_match"]]
            all_excl_only.extend(excl_only)

            # 除外+穴埋め: axis_sum順に一致をスキップしてN件確保
            fill: list[dict] = []
            for c_ in day_cands:
                if c_["wt_match"]:
                    continue
                fill.append(c_)
                if len(fill) >= S4_DAILY_TOP_N:
                    break
            all_excl_fill.extend(fill)

            # 優先選出: 重なり0は該当があれば無条件で全件採用（本数上限なし・X件）
            # + 重なり1はaxis_sum昇順で常に上位10件（Y=10・重なり0の件数に関わらず固定）
            # 合計本数は X+10 で可変（重なり2=完全一致は完全除外）
            tier0 = [c_ for c_ in day_cands if c_["wt_overlap_n"] == 0]
            tier1 = sorted((c_ for c_ in day_cands if c_["wt_overlap_n"] == 1),
                           key=lambda c_: c_["axis_sum"])
            priority_picks = tier0 + tier1[:S4_DAILY_TOP_N]
            all_priority.extend(priority_picks)

    print("\n" + "=" * 90)
    print("===== 全期間: WT◎◯一致 vs 不一致（現行の日次Top10選出内） =====")
    match_rows = [r for r in all_selected if r["wt_match"]]
    nomatch_rows = [r for r in all_selected if not r["wt_match"]]
    unknown_rows = [r for r in all_selected if r["wt_overlap_n"] is None]
    print(f"選出総数: {len(all_selected)}R（WT◎◯マーク欠損で判定不能: {len(unknown_rows)}R）")
    summarize("現行(baseline)全体", all_selected)
    summarize("2軸=WT◎◯ 完全一致", match_rows)
    summarize("2軸=WT◎◯ 不一致(片方以下)", nomatch_rows)

    # overlap_n 別内訳(0/1/2)
    for k in (0, 1, 2):
        sub = [r for r in all_selected if r["wt_overlap_n"] == k]
        if sub:
            summarize(f"WT◎◯との重なり数={k}", sub)

    print("\n===== シミュレーション: 一致レースを除外した場合 =====")
    summarize("除外のみ（穴埋めなし・母数減）", all_excl_only)
    summarize("除外+次点で穴埋め（母数維持）", all_excl_fill)

    n_days = len(day_set)
    print(f"\n===== シミュレーション: 重なり0を最優先採用+重なり1で穴埋め（重なり2は完全除外） =====")
    print(f"対象日数: {n_days}日")
    res = summarize("重なり0最優先+重なり1穴埋め", all_priority)
    print(f"  1日あたり平均レース数: {res['n'] / n_days:.2f}R/日")
    tier0_in_priority = [r for r in all_priority if r["wt_overlap_n"] == 0]
    tier1_in_priority = [r for r in all_priority if r["wt_overlap_n"] == 1]
    print(f"  内訳: 重なり0={len(tier0_in_priority)}件 / 重なり1={len(tier1_in_priority)}件")
    if tier0_in_priority:
        summarize("  └ 内訳:重なり0のみ", tier0_in_priority)
    if tier1_in_priority:
        summarize("  └ 内訳:重なり1のみ", tier1_in_priority)


if __name__ == "__main__":
    main()
