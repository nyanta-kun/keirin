#!/usr/bin/env python3
"""S1（新設計・win軸1着固定×3着内モデル相手2車・SEVEN_S1）の過去分バックフィル。

S1 の検証期間実績を picks_history（SQLite + VPS PG）に構築する。
判定は本番（wave-picks-wt の候補選定 + notify_prerace_wt.judge_s1）と
同一条件を最終オッズ盤面で再現する:

  7車ちょうど ∧ 盤面(trifecta)7車
  軸 = win model(lgbm_wt_win) レース内1位
  相手 = 3着内モデル（配信モデル）で軸を除いた残り車の上位2頭(p1,p2)
  ゲート: top3_gap(p1-p2の3着内確率差) >= S1W_TOP3_GAP_MIN
  買い目 = 三連単 軸→p1→p2, 軸→p2→p1 の2点（目オッズ下限なし）

採点は実精算方式（S1 live 採点と同一）: 盤面7車レースのみ対象・返還処理なし
（買い目確定後の落車・失格は外れ計上）。払戻 = 的中時 trifecta 最終オッズ×100。

--wipe を付けると、書き込み前に対象期間の既存 #7S1 行を削除する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_s1w_rank_wt.py \
        --start 2022-12-01 --end 2026-07-15 [--model lgbm_wt_eval] \
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
from src.strategy_wt import S1W_STAKE, s1w_gate, s1w_select


def _load_trifecta_boards(race_keys: list[str]) -> dict:
    tri = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trifecta' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
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


def build_rows(model_name: str, date_from: str, date_to: str,
                win_model_name: str = "lgbm_wt_win") -> list[dict]:
    """バックフィル対象の S1(#7S1) 行（採点済み）を構築する。"""
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
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    tri_bd = _load_trifecta_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows: list[dict] = []
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
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s1w_select(win_probs, top3_probs)
        if sel is None:
            continue
        axis, p1, p2, top3_gap = sel
        if not s1w_gate(top3_gap, win_probs[axis], class_map.get(axis)):
            continue
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
        # 表記: axis→p1=p2（2点とも成立時）。片方のみ成立時は該当目のみ明示。
        pred = f"{axis}→{p1}={p2}" if len(buy) == 2 else \
            ",".join("-".join(map(str, c)) for c in buy)
        rows.append({
            "race_date": date_map.get(rk, ""),
            "race_key": f"{rk}#7S1", "rank": "SEVEN_S1",
            "pred_combo": pred, "n_combos": len(buy), "hit": int(hit), "payout": pay,
            "trifecta_payout": trifecta_pay, "bet_amount": bet,
        })
    return rows


def wipe_rows(date_from: str, date_to: str, dry_run: bool) -> None:
    cond = "rank='SEVEN_S1' AND race_key LIKE '%#7S1' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(
            f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
            (date_from, date_to)).fetchone()[0]
        print(f"[backfill] 既存 #7S1 行（{date_from}〜{date_to}）: {n}件 → 削除"
              f"{'（dry-run）' if dry_run else ''}")
        if not dry_run and n:
            conn.execute(f"DELETE FROM picks_history WHERE {cond}", (date_from, date_to))
            conn.commit()

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    import psycopg2
    cond_pg = "rank='SEVEN_S1' AND race_key LIKE %s AND race_date BETWEEN %s AND %s"
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM keirin.picks_history WHERE {cond_pg}",
                        ("%#7S1", date_from, date_to))
            n = cur.fetchone()[0]
            print(f"[backfill] VPS PG 既存 #7S1 行: {n}件 → 削除{'（dry-run）' if dry_run else ''}")
            if not dry_run and n:
                cur.execute(f"DELETE FROM keirin.picks_history WHERE {cond_pg}",
                            ("%#7S1", date_from, date_to))


def insert_rows(rows: list[dict], dry_run: bool) -> None:
    if dry_run or not rows:
        return
    rows_ins = [{**r, "miwokuri": False} for r in rows]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trifecta_payout,bet_amount,route,miwokuri) "
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
            " :payout,:trifecta_payout,:bet_amount,'wt',:miwokuri)",
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
                   trifecta_payout,bet_amount,route,miwokuri)
                VALUES (%(race_date)s,%(race_key)s,%(rank)s,%(pred_combo)s,
                        %(n_combos)s,%(hit)s,%(payout)s,%(trifecta_payout)s,
                        %(bet_amount)s,'wt',FALSE)
                ON CONFLICT (race_key) DO UPDATE SET
                  race_date=EXCLUDED.race_date, rank=EXCLUDED.rank,
                  pred_combo=EXCLUDED.pred_combo, n_combos=EXCLUDED.n_combos,
                  hit=EXCLUDED.hit, payout=EXCLUDED.payout,
                  trifecta_payout=EXCLUDED.trifecta_payout,
                  bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE
            """, rows, page_size=200)
    print(f"[backfill] VPS PG {len(rows)}件 書き込み完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-12-01")
    ap.add_argument("--end", required=False)
    ap.add_argument("--model", default="lgbm_wt_eval")
    ap.add_argument("--win-model", default="lgbm_wt_win")
    ap.add_argument("--wipe", action="store_true",
                    help="書き込み前に対象期間の既存 #7S1 行を削除")
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
    print(f"[backfill] S1(win軸): {n}R 的中{hits} ({hits/n*100 if n else 0:.1f}%) "
          f"投資{bet:,} → 回収{ret:,} ROI {roi:.1f}%", flush=True)

    insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
