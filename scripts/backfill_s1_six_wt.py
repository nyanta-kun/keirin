#!/usr/bin/env python3
"""新S1（6車三連単 モデル1位→2位→{3位,4位}・2点）の過去分バックフィル。

2026-07-16 の旧S1（7PLUS_R・7車三連複）全廃・新S1置換に伴い、
新S1の検証期間実績を picks_history（SQLite + VPS PG）に構築する。

判定（本番 wave-picks-wt / judge_s1 と同一・実精算方式）:
  母集団: 6車ちょうど ∧ trio盤面6車 ∧ gap12 >= S1_GAP12_MIN ∧
          モデル1-4位が盤面内 ∧ 完走3名以上
  買い目: 三連単 (m1,m2,m3), (m1,m2,m4) の2点・100円/点
  払戻:   的中時 trifecta 最終オッズ×100（10円単位切り捨て・_load_payouts_wt）

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_s1_six_wt.py \
        --start 2026-04-13 --end 2026-07-15 [--model lgbm_wt_eval] [--dry-run]
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
from src.strategy_wt import S1_GAP12_MIN, S1_NE, S1_STAKE


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
                if fv is None or not (0 < fv < 9000):
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def build_rows(model_name: str, date_from: str, date_to: str) -> list[dict]:
    """バックフィル対象の新S1行（採点済み）を構築する。"""
    model = load_model(model_name)
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
        rks6 = [rk for rk, ne in ne_map.items() if ne and int(ne) == S1_NE]
        fins: dict[str, list] = {}
        for i in range(0, len(rks6), 900):
            chunk = rks6[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s) AND finish_order >= 1" % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                fins.setdefault(rk, []).append((fo, int(fno)))
    df = df[df["race_key"].isin(set(rks6))].copy()
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if len(g) != S1_NE:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = g["pred_prob"].to_numpy()
        gap12 = float(probs[0] - probs[1])
        if gap12 < S1_GAP12_MIN:
            continue
        frames = g["frame_no"].astype(int).tolist()
        m1, m2, m3, m4 = frames[0], frames[1], frames[2], frames[3]

        # 盤面（最終 trio オッズ掲載車）6車 ∧ モデル1-4位が盤面内（欠車レースは対象外）
        trio = trio_bd.get(rk, {})
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != S1_NE or not {m1, m2, m3, m4} <= board:
            continue

        order3 = tuple(fno for _, fno in f[:3])
        buy = {(m1, m2, m3), (m1, m2, m4)}
        hit = order3 in buy
        tri_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        pay = tri_pay * S1_STAKE // 100 if hit else 0
        trio_pay = pm.get(rk, {}).get(("trio", frozenset(order3)), 0)

        rows.append({
            "race_date": date_map.get(rk, ""),
            "race_key": f"{rk}#6S1",
            "pred_combo": f"{m1}>{m2}>{m3},{m4}",
            "n_combos": 2,
            "hit": int(hit),
            "payout": pay,
            "trio_payout": trio_pay,
            "trifecta_payout": tri_pay,
            "bet_amount": 2 * S1_STAKE,
        })
    return rows


def insert_rows(rows: list[dict], dry_run: bool) -> None:
    if dry_run or not rows:
        return
    rows_ins = [{**r, "miwokuri": False} for r in rows]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trio_payout,trifecta_payout,bet_amount,route,miwokuri) "
            "VALUES (:race_date,:race_key,'SIX_S1',:pred_combo,:n_combos,:hit,"
            " :payout,:trio_payout,:trifecta_payout,:bet_amount,'wt',:miwokuri)",
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
                   trio_payout,trifecta_payout,bet_amount,route,miwokuri)
                VALUES (%(race_date)s,%(race_key)s,'SIX_S1',%(pred_combo)s,
                        %(n_combos)s,%(hit)s,%(payout)s,%(trio_payout)s,
                        %(trifecta_payout)s,%(bet_amount)s,'wt',FALSE)
                ON CONFLICT (race_key) DO UPDATE SET
                  race_date=EXCLUDED.race_date, rank=EXCLUDED.rank,
                  pred_combo=EXCLUDED.pred_combo, n_combos=EXCLUDED.n_combos,
                  hit=EXCLUDED.hit, payout=EXCLUDED.payout,
                  trio_payout=EXCLUDED.trio_payout,
                  trifecta_payout=EXCLUDED.trifecta_payout,
                  bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE
            """, rows, page_size=200)
    print(f"[backfill] VPS PG {len(rows)}件 書き込み完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-13")
    ap.add_argument("--end", required=False)
    ap.add_argument("--model", default="lgbm_wt_eval")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.end:
        from datetime import date, timedelta
        args.end = (date.today() - timedelta(days=1)).isoformat()

    print(f"[backfill] model={args.model} {args.start}〜{args.end}")
    rows = build_rows(args.model, args.start, args.end)
    n_hit = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    print(f"[backfill] S1(6車): {len(rows)}R 的中{n_hit} "
          f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
          f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%")
    insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
