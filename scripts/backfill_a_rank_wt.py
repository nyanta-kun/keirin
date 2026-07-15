#!/usr/bin/env python3
"""A ランク（◎一致×波乱×別ライン先頭・二連単）の過去分バックフィル。

2026-07-16 のランク体系整理（SS→S1 / U→S2 / M→S3・A 新設）に伴い、
A ランクの過去実績を picks_history（SQLite + VPS PG）に構築する。

判定（本番 wave-picks-wt / judge_a と同一・実精算方式）:
  母集団: 7車ちょうど ∧ WT◎(prediction_mark==1)==システム◎(モデル指数1位)
          ∧ entropy>=U_ENTROPY_MIN ∧ 別ライン先頭（得点最上位）あり
          ∧ 最終オッズ盤面7車
  買い目: 二連単 軸→x のうち最終オッズ ∈ [A_EX_MIN_ODDS, A_EX_MAX_ODDS)
          （出走前の出走状況=オッズ盤面掲載車が母体。欠車=盤面外は買い目に
            含まれない=返還相当。落車・失格は盤面に残るため外れ計上）
  払戻:   的中時 exacta 最終オッズ×100（10円単位切り捨て・_load_payouts_wt）

モデルは OOS 評価モデル（既定 lgbm_wt_eval・学習2023-07〜/test 2026-04-13〜）を
使用する。学習期間内の日付はリークになるため既定開始日は 2026-04-13。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_a_rank_wt.py \
        --start 2026-04-13 --end 2026-07-15 [--model lgbm_wt_eval] [--dry-run]
    # 旧・廃止済み A ランク（7PLUS_A・〜2026-06-19）行の退避のみ実行
    PYTHONPATH=. .venv/bin/python scripts/backfill_a_rank_wt.py --archive-old-only
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
from src.strategy_wt import A_EX_MAX_ODDS, A_EX_MIN_ODDS, A_STAKE, U_ENTROPY_MIN

OLD_A_CUTOFF = "2026-06-30"  # 旧Aランクは2026-06-19終了。これ以前のrank=7PLUS_Aは旧体系


def archive_old_a_rows(dry_run: bool) -> None:
    """旧・廃止済み A ランク行（rank='7PLUS_A'）を退避テーブルへ移動する。

    S/S+ 全廃時の picks_history_st_archive と同じ方式。旧 A は 2025-07〜2026-06-19 の
    旧体系実績で、新 A（2026-07-16 新設・ペーパー）と race_key(#7A) が衝突するため
    picks_history から picks_history_a_archive へ退避して名前空間を空ける。
    """
    with get_connection() as conn:
        _OLD_COND = ("rank='7PLUS_A' AND race_date <= ? "
                     "AND (pred_combo IS NULL OR pred_combo NOT LIKE '%>%')")
        n = conn.execute(
            f"SELECT COUNT(*) FROM picks_history WHERE {_OLD_COND}",
            (OLD_A_CUTOFF,)).fetchone()[0]
        print(f"[archive] SQLite 旧7PLUS_A行: {n}件")
        if not dry_run and n:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS picks_history_a_archive AS "
                "SELECT * FROM picks_history WHERE 0")
            conn.execute(
                "INSERT INTO picks_history_a_archive "
                f"SELECT * FROM picks_history WHERE {_OLD_COND}",
                (OLD_A_CUTOFF,))
            conn.execute(
                f"DELETE FROM picks_history WHERE {_OLD_COND}",
                (OLD_A_CUTOFF,))
            conn.commit()
            print(f"[archive] SQLite {n}件を picks_history_a_archive へ退避完了")

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        print("[archive] KEIRIN_DB_URL 未設定 → VPS PG スキップ")
        return
    import psycopg2
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            _OLD_COND_PG = ("rank='7PLUS_A' AND race_date <= %s "
                            "AND (pred_combo IS NULL OR pred_combo NOT LIKE '%%>%%')")
            cur.execute(
                f"SELECT COUNT(*) FROM keirin.picks_history WHERE {_OLD_COND_PG}",
                (OLD_A_CUTOFF,))
            n = cur.fetchone()[0]
            print(f"[archive] VPS PG 旧7PLUS_A行: {n}件")
            if not dry_run and n:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS keirin.picks_history_a_archive "
                    "(LIKE keirin.picks_history INCLUDING ALL)")
                cur.execute(
                    "INSERT INTO keirin.picks_history_a_archive "
                    "SELECT * FROM keirin.picks_history "
                    f"WHERE {_OLD_COND_PG} "
                    "ON CONFLICT DO NOTHING", (OLD_A_CUTOFF,))
                cur.execute(
                    f"DELETE FROM keirin.picks_history WHERE {_OLD_COND_PG}",
                    (OLD_A_CUTOFF,))
                print(f"[archive] VPS PG {n}件を keirin.picks_history_a_archive へ退避完了")


def _load_boards(race_keys: list[str]) -> tuple[dict, dict]:
    """wt_odds 最終オッズから trio 盤面と exacta 盤面を返す。"""
    trio, exa = defaultdict(dict), defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','exacta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None:
                    continue
                try:
                    fv = float(od)
                except (TypeError, ValueError):
                    continue
                if not (0 < fv < 9000):
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio" and len(parts) == 3:
                    trio[rk][frozenset(parts)] = fv
                elif bt == "exacta" and len(parts) == 2:
                    exa[rk][tuple(parts)] = fv
    return trio, exa


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
    """バックフィル対象の A ランク行（採点済み）を構築する。"""
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
    trio_bd, exa_bd = _load_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    def _int(v):
        return None if v is None or pd.isna(v) else int(v)

    rows: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if len(g) != 7:
            continue
        mk = marks.get(rk, {})
        wt_tops = [fno for fno, (pm_v, _) in mk.items() if pm_v == 1]
        if not wt_tops:
            continue
        wt_top = min(wt_tops)
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        r1 = g.iloc[0]
        sys_top = int(r1["frame_no"])
        if sys_top != wt_top:
            continue  # 一致レースのみ
        ent = _entropy([float(x) for x in g["pred_prob"].tolist()])
        if ent < U_ENTROPY_MIN:
            continue

        # 軸 = 別ライン先頭・得点最上位（wave-picks-wt の A 候補選定と同一決定則）
        lg1 = _int(r1.get("line_group"))
        rivals = []
        for _, r in g.iloc[1:].iterrows():
            if _int(r.get("line_pos")) != 1:
                continue
            lg = _int(r.get("line_group"))
            if lg1 is not None and lg == lg1:
                continue
            rp = r.get("race_point")
            rp_val = float(rp) if rp is not None and rp == rp else -1.0
            rivals.append((rp_val, int(r["frame_no"])))
        if not rivals:
            continue
        _, axis = max(rivals)

        # 盤面（最終 trio オッズ掲載車）7車のレースのみ（欠車発生レースは見送り相当）
        trio = trio_bd.get(rk, {})
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7 or axis not in board:
            continue

        # 買い目 = 二連単 軸→x のうち最終オッズ ∈ [5, 50)
        exa = exa_bd.get(rk, {})
        partners = []
        for x in sorted(board - {axis}):
            ov = exa.get((axis, x))
            if ov is not None and A_EX_MIN_ODDS <= ov < A_EX_MAX_ODDS:
                partners.append(x)
        if not partners:
            continue

        # 着順（1着・2着）
        fins = sorted((fo, fno) for fno, (_, fo) in mk.items()
                      if fo is not None and fo >= 1)
        if len(fins) < 2:
            continue
        f1, f2 = fins[0][1], fins[1][1]

        hit = f1 == axis and f2 in partners
        exacta_pay = pm.get(rk, {}).get(("exacta", (f1, f2)), 0)
        pay = exacta_pay * A_STAKE // 100 if hit else 0
        bet = len(partners) * A_STAKE
        top3 = [fno for _, fno in fins[:3]]
        trio_pay = (pm.get(rk, {}).get(("trio", frozenset(top3)), 0)
                    if len(top3) >= 3 else 0)
        trifecta_pay = (pm.get(rk, {}).get(("trifecta", tuple(top3)), 0)
                        if len(top3) >= 3 else 0)

        rows.append({
            "race_date": date_map.get(rk, str(rk)[:8]),
            "race_key": f"{rk}#7A",
            "pred_combo": f"{axis}>" + ",".join(map(str, partners)),
            "n_combos": len(partners),
            "hit": int(hit),
            "payout": pay,
            "trio_payout": trio_pay,
            "trifecta_payout": trifecta_pay,
            "bet_amount": bet,
        })
    return rows


def insert_rows(rows: list[dict], dry_run: bool) -> None:
    """SQLite + VPS PG へ #7A 行を書き込む（既存の同一キーは上書き）。"""
    if dry_run:
        return
    # miwokuri は PG=boolean / SQLite=integer のためパラメータで False を渡す
    # （get_connection は KEIRIN_DB_URL 設定時 PG 直結・未設定時ローカル SQLite）
    rows_ins = [{**r, "miwokuri": False} for r in rows]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trio_payout,trifecta_payout,bet_amount,route,miwokuri) "
            "VALUES (:race_date,:race_key,'7PLUS_A',:pred_combo,:n_combos,:hit,"
            " :payout,:trio_payout,:trifecta_payout,:bet_amount,'wt',:miwokuri)",
            rows_ins)
        conn.commit()
    print(f"[backfill] get_connection先 {len(rows)}件 書き込み完了")

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        print("[backfill] KEIRIN_DB_URL 未設定 → VPS PG スキップ")
        return
    import psycopg2
    from psycopg2.extras import execute_batch
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO keirin.picks_history
                  (race_date,race_key,rank,pred_combo,n_combos,hit,payout,
                   trio_payout,trifecta_payout,bet_amount,route,miwokuri)
                VALUES (%(race_date)s,%(race_key)s,'7PLUS_A',%(pred_combo)s,
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
    ap.add_argument("--archive-old-only", action="store_true",
                    help="旧7PLUS_A行の退避のみ実行（バックフィルしない）")
    ap.add_argument("--skip-archive", action="store_true",
                    help="旧7PLUS_A行の退避をスキップ（退避済みの再実行時）")
    args = ap.parse_args()

    if not args.skip_archive:
        archive_old_a_rows(args.dry_run)
    if args.archive_old_only:
        return

    if not args.end:
        from datetime import date, timedelta
        args.end = (date.today() - timedelta(days=1)).isoformat()

    print(f"[backfill] model={args.model} {args.start}〜{args.end}")
    rows = build_rows(args.model, args.start, args.end)
    n_hit = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    roi = pay / bet * 100 if bet else 0.0
    print(f"[backfill] 対象 {len(rows)}R 的中{n_hit} ({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
          f"名目投資{bet:,}円 → 回収{pay:,}円 ROI {roi:.1f}%")
    if rows:
        insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
