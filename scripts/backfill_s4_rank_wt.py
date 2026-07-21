#!/usr/bin/env python3
"""S4（単勝×複勝指数トップ3重なり軸×波乱度選出・SEVEN_S4）の過去分バックフィル。

S4 の検証期間実績を picks_history（SQLite + VPS PG）に構築する。
判定は本番（wave-picks-wt の候補選定 + notify_prerace_wt.judge_s4）と
同一条件を最終オッズ盤面で再現する:

  7車ちょうど ∧ 盤面(trio)7車
  軸2車 = pred_win(単勝指数)上位3 ∩ pred_prob(複勝指数)上位3 の重なりから
          strategy_wt.s4_select_axis() で選定
  波乱度指数(axis_sum) = 軸2車のpred_prob合計。低いほど採用
  選出 = strategy_wt.s4_daily_select()（2026-07-21〜 WT◎◯重なり考慮版）:
         軸2車がWINTICKET公式◎◯(prediction_mark 1,2)と重なる数で3区分し、
         重なり0(全く重ならない)は無条件で全件採用、重なり1(片方一致)は
         axis_sum昇順で固定S4_DAILY_TOP_N件、重なり2(完全一致)・マーク欠損は除外
         （1レース単位の閾値ゲートではなく日次クロスレースの区分選出）
  買い目 = 三連複 軸2車 + 残り5車のいずれか1車（5点・オッズ下限なし）

採点は実精算方式: 盤面7車レースのみ対象・返還処理なし。
払戻 = 的中時 trio 最終オッズ×100。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_s4_rank_wt.py \
        --start 2024-01-01 --end 2026-07-10 [--model lgbm_wt_eval] \
        [--wipe] [--dry-run]
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
from src.strategy_wt import S4_STAKE, s4_daily_select, s4_select_axis, s4_wt_overlap_n


def _load_trio_boards(race_keys: list[str]) -> dict:
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


def build_rows(model_name: str, date_from: str, date_to: str,
                win_model_name: str = "lgbm_wt_win") -> list[dict]:
    """バックフィル対象の S4(#7S4) 行（採点済み）を構築する。"""
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

    # ── 全該当レースの axis1/axis2/axis_sum を先に計算 ──
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

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "wt_overlap_n": wt_overlap_n,
        })

    # ── 日次選出: s4_daily_select()（2026-07-21〜 WT◎◯重なり考慮版） ──
    by_day: dict[str, list[dict]] = defaultdict(list)
    for c_ in candidates:
        by_day[c_["race_date"]].append(c_)

    rows: list[dict] = []
    for d, day_cands in by_day.items():
        for c_ in s4_daily_select(day_cands):
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
            gate_label = {0: "SS", 1: "S"}.get(c_["wt_overlap_n"])
            rows.append({
                "race_date": d,
                "race_key": f"{rk}#7S4", "rank": "SEVEN_S4",
                "pred_combo": f"{axis1}={axis2}-" + ",".join(str(x) for x in c_["others"])
                              + f" (axis_sum={c_['axis_sum']:.1f})",
                "n_combos": len(combos), "hit": int(hit), "payout": pay,
                "trio_payout": trio_pay, "bet_amount": bet, "gate_label": gate_label,
            })
    return rows


def wipe_rows(date_from: str, date_to: str, dry_run: bool) -> None:
    cond = "rank='SEVEN_S4' AND race_key LIKE '%#7S4' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(
            f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
            (date_from, date_to)).fetchone()[0]
        print(f"[backfill] 既存 #7S4 行（{date_from}〜{date_to}）: {n}件 → 削除"
              f"{'（dry-run）' if dry_run else ''}")
        if not dry_run and n:
            conn.execute(f"DELETE FROM picks_history WHERE {cond}", (date_from, date_to))
            conn.commit()

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    import psycopg2
    cond_pg = "rank='SEVEN_S4' AND race_key LIKE %s AND race_date BETWEEN %s AND %s"
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM keirin.picks_history WHERE {cond_pg}",
                        ("%#7S4", date_from, date_to))
            n = cur.fetchone()[0]
            print(f"[backfill] VPS PG 既存 #7S4 行: {n}件 → 削除{'（dry-run）' if dry_run else ''}")
            if not dry_run and n:
                cur.execute(f"DELETE FROM keirin.picks_history WHERE {cond_pg}",
                            ("%#7S4", date_from, date_to))


def insert_rows(rows: list[dict], dry_run: bool) -> None:
    if dry_run or not rows:
        return
    rows_ins = [{**r, "miwokuri": False} for r in rows]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trio_payout,bet_amount,route,miwokuri,gate_label) "
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
            " :payout,:trio_payout,:bet_amount,'wt',:miwokuri,:gate_label)",
            rows_ins)
        conn.commit()
    print(f"[backfill] get_connection先 {len(rows)}件 書き込み完了")

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        print("[backfill] KEIRIN_DB_URL 未設定 → VPS PG ミラーはスキップ")
        return
    import psycopg2
    from psycopg2.extras import execute_batch
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO keirin.picks_history
                  (race_date,race_key,rank,pred_combo,n_combos,hit,payout,
                   trio_payout,bet_amount,route,miwokuri,gate_label)
                VALUES (%(race_date)s,%(race_key)s,%(rank)s,%(pred_combo)s,
                        %(n_combos)s,%(hit)s,%(payout)s,%(trio_payout)s,
                        %(bet_amount)s,'wt',FALSE,%(gate_label)s)
                ON CONFLICT (race_key) DO UPDATE SET
                  race_date=EXCLUDED.race_date, rank=EXCLUDED.rank,
                  pred_combo=EXCLUDED.pred_combo, n_combos=EXCLUDED.n_combos,
                  hit=EXCLUDED.hit, payout=EXCLUDED.payout,
                  trio_payout=EXCLUDED.trio_payout,
                  bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE,
                  gate_label=EXCLUDED.gate_label
            """, rows, page_size=200)
    print(f"[backfill] VPS PG {len(rows)}件 書き込み完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", required=False)
    ap.add_argument("--model", default="lgbm_wt_eval")
    ap.add_argument("--win-model", default="lgbm_wt_win")
    ap.add_argument("--wipe", action="store_true",
                    help="書き込み前に対象期間の既存 #7S4 行を削除")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from datetime import date
    end = args.end or date.today().strftime("%Y-%m-%d")
    print(f"[backfill] model={args.model} win_model={args.win_model} {args.start}〜{end}", flush=True)

    if args.wipe:
        wipe_rows(args.start, end, args.dry_run)

    rows = build_rows(args.model, args.start, end, args.win_model)
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    ret = sum(r["payout"] for r in rows)
    roi = ret / bet * 100 if bet else 0
    print(f"[backfill] S4(波乱度選出): {n}R 的中{hits} ({hits/n*100 if n else 0:.1f}%) "
          f"投資{bet:,} → 回収{ret:,} ROI {roi:.1f}%", flush=True)

    insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
