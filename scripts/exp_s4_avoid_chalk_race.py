"""S(重なり1)ランク: オッズを使わず「WINTICKET印との関係」で人気決着回避を検証する。

ユーザー方針: 現行のSは実オッズを一切参照せず、軸選定（単勝×複勝指数の重なり）と
WT◎◯との重なり数（0/1/2）だけでレースを選出している。この設計思想を踏襲し、
「人気決着しそうなレースの回避」もオッズではなくWT印（◎◯▲△）との関係性で
判定する。

操作的定義: Sランクは軸2車のうち1車がWT◎またはWT◎(honmei)/◯(taikou)と一致
（wt_overlap_n==1）。この時、**一致しなかった方の軸車**（システムが選んだが
WT◎◯ではない車）のWT印を調べる:
  - その車がWT▲(mark=3)またはWT△(mark=4)である
    → WTも「本命級ではないが上位候補」とみなしている＝システムとWTの評価が
      近い＝「人気決着」寄りと解釈し回避候補とする
  - その車がWT印なし（mark=0/None・「注」以下）である
    → WTからは全く評価されていない車をシステムが軸に選んでいる＝システムと
      WTの評価が大きく乖離＝「軸自体が波乱含み」と解釈し優先採用する

日次選出は現行通りaxis_sum昇順・重なり1は固定S4_DAILY_TOP_N件だが、
「回避方式」ではWT▲△一致（人気決着寄り）の候補をスキップし、次点
（axis_sum昇順で次に良い、無印の候補）へ繰り上げる。

honest全期間・四半期walk-forwardモデルで実施（オッズはあくまで精算にのみ使用、
選出判定には一切使わない）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s4_avoid_chalk_race.py
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
    """S(重なり1)候補（軸選定＋WT印関係＋的中判定）を日次選出前の全件収集する。

    オッズは精算（払戻計算）にのみ使用し、選出判定（is_chalk）には使わない。
    """
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
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())  # 精算専用（選出には未使用）
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

        wt_top2 = {wt_honmei, wt_taikou}
        other_axis = axis2 if axis1 in wt_top2 else axis1  # WT◎◯と一致しなかった方の軸車
        other_mark = mk.get(other_axis, 0) or 0
        # 2026-07-21 精密化: 初回検証で▲(3)のみがROI103.3%(ほぼ収支トントン)の主犯と
        # 判明。△(4)はROI145.3%と優良だったため回避対象から除外し▲のみ回避する。
        is_chalk = other_mark == 3

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
            "axis_sum": axis_sum, "other_mark": other_mark, "is_chalk": is_chalk,
            "hit": hit, "bet": bet, "pay": pay,
        })
    return candidates


def _settle(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    roi = pay / bet * 100 if bet else 0
    payouts = [r["pay"] for r in rows if r["hit"]]
    avg_pay = sum(payouts) / len(payouts) if payouts else 0
    return n, hits, bet, pay, roi, avg_pay


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

    baseline_rows: list[dict] = []   # 現行: axis_sum昇順で日次10件（WT印は見ない）
    avoided_rows: list[dict] = []    # 新方式: 他方軸がWT▲のレースを回避し次点へ繰り上げ
    avoided_dropped: list[dict] = [] # 参考: 単純除外のみ・穴埋めなし

    for d, day_cands in by_day.items():
        day_cands.sort(key=lambda c: c["axis_sum"])
        baseline_rows.extend(day_cands[:S4_DAILY_TOP_N])

        avoided_dropped.extend(
            [c for c in day_cands[:S4_DAILY_TOP_N] if not c["is_chalk"]])

        picked = []
        for c in day_cands:
            if c["is_chalk"]:
                continue
            picked.append(c)
            if len(picked) >= S4_DAILY_TOP_N:
                break
        avoided_rows.extend(picked)

    n_days = len(by_day)
    print("\n" + "=" * 90)
    print(f"対象日数: {n_days}日\n")

    for label, rows in (
        ("現行(baseline): axis_sum昇順 日次10件（WT印は不参照）", baseline_rows),
        ("他方軸がWT▲(人気決着寄り)のレースを回避+次点繰り上げ（日次10件維持）", avoided_rows),
        ("参考: 該当レース単純除外のみ（穴埋めなし）", avoided_dropped),
    ):
        n, hits, bet, pay, roi, avg_pay = _settle(rows)
        hit_rate = hits / n * 100 if n else 0
        print(f"[{label}]")
        print(f"  n={n}R ({n/n_days:.2f}R/日) 的中={hits} ({hit_rate:.1f}%) "
              f"投資={bet:,} 回収={pay:,} ROI={roi:.1f}%  平均払戻(的中時)={avg_pay:,.0f}円\n")

    chalk_n = sum(1 for c in all_candidates if c["is_chalk"])
    print(f"参考: 全S候補中「他方軸がWT▲」の割合 = "
          f"{chalk_n}/{len(all_candidates)} ({chalk_n/len(all_candidates)*100:.1f}%)")

    # 内訳: other_mark別の成績（0=無印, 3=▲, 4=△）
    print("\n他方軸のWT印別 成績（現行baseline採用分のみ）:")
    for m in (0, 3, 4):
        sub = [r for r in baseline_rows if r["other_mark"] == m]
        if not sub:
            continue
        n, hits, bet, pay, roi, avg_pay = _settle(sub)
        label = {0: "無印(注以下)", 3: "▲", 4: "△"}[m]
        print(f"  other_mark={label}: n={n} 的中={hits}({hits/n*100:.1f}%) "
              f"ROI={roi:.1f}% 平均払戻={avg_pay:,.0f}円")


if __name__ == "__main__":
    main()
