#!/usr/bin/env python3
"""S2（旧U・7PLUS_U）/ S3（旧M・7PLUS_M）の過去分バックフィル。

2026-07-16 のランク体系整理に伴い、S2/S3 の検証期間実績を picks_history
（SQLite + VPS PG）に構築する。判定は本番（wave-picks-wt の候補選定 +
notify_prerace_wt.judge_u / judge_m）と同一条件を最終オッズ盤面で再現する:

  共通: 7車ちょうど ∧ 盤面(trio)7車 ∧ entropy>=U_ENTROPY_MIN ∧ mto>=U_MTO_MIN
  S2(U): 穴=モデル3位内 ∧ (単騎 or ライン先頭/番手) ∧ 市場順位4-7位、
         相方=同ライン「逃」。複数成立は (モデル順位, 車番) 最小の1ペア。
         買い目 = 三連複 {穴,相方,t} のうちオッズ>=U_LEG_MIN_ODDS のみ
  S3(M): WT◎≠システム◎ ∧ 相方=システム◎と同ライン「逃」（lp相補優先→車番最小）。
         S2(buy) と同一ペア集合のレースは S2 優先で S3 は記録しない。
         買い目 = 三連複 {システム◎,相方,t} のうちオッズ>=U_LEG_MIN_ODDS のみ

採点は実精算方式（U/M live 採点と同一）: 盤面7車レースのみ対象・返還処理なし
（買い目確定後の落車・失格は外れ計上）。払戻 = 的中時 trio 最終オッズ×100。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_um_rank_wt.py \
        --start 2026-04-13 --end 2026-07-15 [--model lgbm_wt_eval] [--dry-run]
"""
from __future__ import annotations

import argparse
import math
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
from src.strategy_wt import U_ENTROPY_MIN, U_LEG_MIN_ODDS, U_MTO_MIN, U_STAKE, M_STAKE


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


def _entropy(probs: list[float]) -> float:
    total = sum(probs)
    if total <= 0:
        return 0.0
    ent = 0.0
    for p in probs:
        s = max(p / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def build_rows(model_name: str, date_from: str, date_to: str) -> list[dict]:
    """バックフィル対象の S2(#7U)/S3(#7M) 行（採点済み）を構築する。"""
    import pandas as pd

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
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        marks: dict[str, dict[int, tuple]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, pm_v, fo in c.execute(q, chunk):
                marks.setdefault(rk, {})[int(fno)] = (pm_v, fo)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    def _int(v):
        return None if v is None or pd.isna(v) else int(v)

    rows: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if len(g) != 7:
            continue
        trio = trio_bd.get(rk, {})
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7 or not trio:
            continue
        ent = _entropy([float(x) for x in g["pred_prob"].tolist()])
        if ent < U_ENTROPY_MIN:
            continue
        mto = min(trio.values())
        if mto < U_MTO_MIN:
            continue

        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        mk = marks.get(rk, {})
        fins = sorted((fo, fno) for fno, (_, fo) in mk.items()
                      if fo is not None and fo >= 1)
        if len(fins) < 3:
            continue
        top3 = frozenset(fno for _, fno in fins[:3])
        order3 = tuple(fno for _, fno in fins[:3])
        trio_pay = pm.get(rk, {}).get(("trio", top3), 0)
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0)

        # 市場評価順位（judge_u と同一: q_i = Σ 1/trioオッズ）
        q: dict[int, float] = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        ranked = sorted(board, key=lambda f: (-q[f], f))
        mkt_rank = {f: i + 1 for i, f in enumerate(ranked)}

        rows_g = list(g.itertuples(index=False))

        def _mk_combos(a: int, b: int) -> tuple[list[frozenset], list[int]]:
            combos, thirds = [], []
            for t in sorted(board - {a, b}):
                key = frozenset({a, b, t})
                ov = trio.get(key)
                if ov is not None and ov >= U_LEG_MIN_ODDS:
                    combos.append(key)
                    thirds.append(t)
            return combos, thirds

        # ── S2(U): 穴候補ペア列挙 → 判定 ──
        u_pair: tuple[int, int] | None = None
        eligible: list[tuple[int, int, int]] = []
        for rank_idx, r in enumerate(rows_g[:3], start=1):
            lg = _int(getattr(r, "line_group", None))
            ls = _int(getattr(r, "line_size", None))
            lp = _int(getattr(r, "line_pos", None))
            if not (ls == 1 or lp in (1, 2)) or lg is None:
                continue
            dark = int(r.frame_no)
            if not (4 <= mkt_rank.get(dark, 8) <= 7):
                continue
            for m in rows_g:
                m_fno = int(m.frame_no)
                m_lg = _int(getattr(m, "line_group", None))
                m_style = m.style if isinstance(getattr(m, "style", None), str) else ""
                if m_fno == dark or m_lg is None or m_lg != lg or m_style != "逃":
                    continue
                eligible.append((rank_idx, dark, m_fno))
        if eligible:
            eligible.sort()
            _, dark, mate = eligible[0]
            combos, thirds = _mk_combos(dark, mate)
            if combos:
                u_pair = (dark, mate)
                hit = top3 in combos
                pay = trio_pay * U_STAKE // 100 if hit else 0
                rows.append({
                    "race_date": date_map.get(rk, ""),
                    "race_key": f"{rk}#7U", "rank": "7PLUS_U",
                    "pred_combo": f"{dark}-{mate}-" + ",".join(map(str, thirds)),
                    "n_combos": len(combos), "hit": int(hit), "payout": pay,
                    "trio_payout": trio_pay, "trifecta_payout": trifecta_pay,
                    "bet_amount": len(combos) * U_STAKE,
                })

        # ── S3(M): ◎不一致 × システム◎同ライン逃相方 ──
        wt_tops = [fno for fno, (pm_v, _) in mk.items() if pm_v == 1]
        if not wt_tops:
            continue
        wt_top = min(wt_tops)
        r1 = rows_g[0]
        m1 = int(r1.frame_no)
        if m1 == wt_top:
            continue  # 一致レースは対象外（Aランク側）
        lg1 = _int(getattr(r1, "line_group", None))
        if lg1 is None:
            continue
        lp1 = _int(getattr(r1, "line_pos", None))
        want_lp = 1 if lp1 == 2 else 2
        mates = []
        for r in rows_g:
            fno = int(r.frame_no)
            lg = _int(getattr(r, "line_group", None))
            style = r.style if isinstance(getattr(r, "style", None), str) else ""
            if fno == m1 or lg is None or lg != lg1 or style != "逃":
                continue
            mates.append((fno, _int(getattr(r, "line_pos", None))))
        if not mates:
            continue
        mates.sort()
        mate_m = next((f for f, lp in mates if lp == want_lp), mates[0][0])
        # S2優先の重複排除（同一ペア集合）
        if u_pair is not None and {m1, mate_m} == set(u_pair):
            continue
        combos, thirds = _mk_combos(m1, mate_m)
        if not combos:
            continue
        hit = top3 in combos
        pay = trio_pay * M_STAKE // 100 if hit else 0
        rows.append({
            "race_date": date_map.get(rk, ""),
            "race_key": f"{rk}#7M", "rank": "7PLUS_M",
            "pred_combo": f"{m1}-{mate_m}-" + ",".join(map(str, thirds)),
            "n_combos": len(combos), "hit": int(hit), "payout": pay,
            "trio_payout": trio_pay, "trifecta_payout": trifecta_pay,
            "bet_amount": len(combos) * M_STAKE,
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
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
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
                VALUES (%(race_date)s,%(race_key)s,%(rank)s,%(pred_combo)s,
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
    for rank, label in (("7PLUS_U", "S2(U)"), ("7PLUS_M", "S3(M)")):
        sel = [r for r in rows if r["rank"] == rank]
        n_hit = sum(r["hit"] for r in sel)
        bet = sum(r["bet_amount"] for r in sel)
        pay = sum(r["payout"] for r in sel)
        print(f"[backfill] {label}: {len(sel)}R 的中{n_hit} "
              f"({n_hit / len(sel) * 100 if sel else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%")
    insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
