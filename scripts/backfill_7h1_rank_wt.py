#!/usr/bin/env python3
"""7H1（穴推奨・本命バスト型・RANK_7H1）の過去分バックフィル。

`backfill_7b_rank_wt.py` の 7H1 版だが、**候補生成そのものは
`build_7h1_candidates.build()` を呼ぶ**（7H1 は既存6ランクと入口が違い、
レース単位のバスト予測モデルを使うため専用経路を持つ）。本スクリプトは
その候補に対して、live と同じ盤面フィルタ・採点・記録だけを行う。

## 既存ランクとの違い（採点式）

7H1 は **唯一の2券種ランク**（三連単フォーメーション + 三連複BOX）。
`notify_results_wt.py` の `seven_7h1` 分岐と同じ規則で採点する:

  - 三連複・三連単をそれぞれ独立に判定し、**払戻は合算**して payout に入れる
    （picks_history は1レース1行・ユーザー承認 2026-08-06）
  - 券種別の払戻は trio_payout / trifecta_payout に残す
  - **三連単だけ的中する組み合わせが存在する**（三連単の3着は本命ラインを含む
    総流しだが、三連複はプールのみ）ので、両方を独立に見る
  - 返還処理なし（実精算方式）

## 欠車の扱い

live の `judge_rank_7h1()` と同一にする:
  - 三連単の**1着固定車が盤面に無ければレース無効**（skip）
  - それ以外の欠車は**その目だけを落として購入継続**（点数が減る）
  - 落とした結果どちらかの券種が全滅したら skip
  - 残った点数で `rank_7h1_stakes()` を引き直す（1点100円未満なら skip）

## ⚠️ 必ず vintage モデルで流すこと

本番モデル（`lgbm_wt_favbust` 等）は全期間学習なので、過去へ遡って適用すると
in-sample になる（model-vintage look-ahead）。honest な全期間再構築は
`rebuild_7h1_walkforward_pg.py`（月次凍結vintage）を使うこと。

使い方:
    # 単一窓（デバッグ用。vintage を明示すること）
    PYTHONPATH=. .venv/bin/python scripts/backfill_7h1_rank_wt.py \
        --start 2026-07-01 --end 2026-07-31 \
        --eval-model lgbm_wt_eval_m2607 --win-model lgbm_wt_win_m2607 \
        --bad-model lgbm_wt_bad_m2607 --favbust-model lgbm_wt_favbust_m2607 \
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

from scripts.build_7h1_candidates import build as build_candidates
from src.wt_vintage_config import assert_vintage_for_past
from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.strategy_wt import RANK_7H1_TF_UNIT, rank_7h1_trio_stakes

RANK = "RANK_7H1"
SUFFIX = "#7H1"


def _combo_key(combo: str, ordered: bool):
    """'1-2-3' / '1=2=3' をキーへ。ordered=True なら着順つきタプル。"""
    try:
        nums = [int(p) for p in re.split(r"[-=]", str(combo)) if p != ""]
    except ValueError:
        return None
    if len(nums) != 3:
        return None
    return tuple(nums) if ordered else frozenset(nums)


def _load_boards(race_keys: list[str]) -> tuple[dict[str, set], dict[str, set]]:
    """wt_odds から (三連複の盤面, 三連単の盤面) を返す。

    live の `_build_odds_lookup()` と同じく **odds_value が有効な組み合わせのみ**を
    盤面掲載とみなす（欠車を含む目はオッズが出ない）。
    """
    # 2026-08-07: 三連複は**オッズ値まで**要る（払戻均等配分のため）。
    # 三連単は掲載有無だけ使うが、同じ形にそろえておく。
    trio: dict[str, dict] = defaultdict(dict)
    tf: dict[str, dict] = defaultdict(dict)
    if not race_keys:
        return {}, {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, combo, odds in c.execute(q, chunk):
                if odds is None:
                    continue
                key = _combo_key(combo, ordered=(bt == "trifecta"))
                if key is None:
                    continue
                (tf if bt == "trifecta" else trio)[rk][key] = float(odds)
    return dict(trio), dict(tf)


def _load_finishes(race_keys: list[str]) -> dict[str, list[int]]:
    """{race_key: [1着車, 2着車, 3着車]}。3着まで揃わないレースは含めない。"""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if not race_keys:
        return {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE finish_order BETWEEN 1 AND 3 AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                out[rk].append((int(fo), int(fno)))
    return {rk: [f for _, f in sorted(v)] for rk, v in out.items() if len(v) >= 3}


def build_rows(date_from: str, date_to: str, *, eval_model: str, win_model: str,
               bad_model: str, favbust_model: str) -> list[dict]:
    """バックフィル対象の 7H1（#7H1）行（採点済み）を構築する。

    ⚠️ 4つのモデル名はすべて **その窓の vintage** を渡すこと。本番モデルを渡すと
    in-sample になる（`rebuild_7h1_walkforward_pg.py` がそうしている）。
    """
    cands = build_candidates(date_from, date_to, eval_model, win_model,
                             bad_model, favbust_model)
    if not cands:
        return []

    race_keys = [c["race_key"] for c in cands]
    trio_bd, tf_bd = _load_boards(race_keys)
    fins = _load_finishes(race_keys)
    pm = _load_payouts_wt(race_keys)

    rows: list[dict] = []
    for c in cands:
        rk = c["race_key"]
        trio_lookup, tf_lookup = trio_bd.get(rk), tf_bd.get(rk)
        if not trio_lookup or not tf_lookup:
            continue                      # 盤面が取れていない＝live なら「不明」で再試行
        order = fins.get(rk)
        if not order:
            continue                      # 3着まで確定していない

        legs_trio_all = list(c.get("legs_trio") or [])
        legs_tf_all = list(c.get("legs_tf") or [])
        if not legs_trio_all or not legs_tf_all:
            continue

        # judge_rank_7h1 と同一: 1着固定車が盤面から消えていたらレース無効
        head = int(legs_tf_all[0].split("-")[0])
        if not any(k[0] == head for k in tf_lookup):
            continue

        legs_trio = [t for t in legs_trio_all if _combo_key(t, False) in trio_lookup]
        legs_tf = [t for t in legs_tf_all if _combo_key(t, True) in tf_lookup]
        if not legs_trio or not legs_tf:
            continue                      # 欠車で買い目が全滅

        # 三連単は 1点 RANK_7H1_TF_UNIT 円の均等、残りを三連複へ回し
        # **オッズで払戻が等しくなるよう配分**する（2026-08-07 ユーザー指定）。
        trio_keys = [_combo_key(t, False) for t in legs_trio]
        u_tf = RANK_7H1_TF_UNIT
        trio_stakes = rank_7h1_trio_stakes(
            trio_keys, {k: trio_lookup[k] for k in trio_keys}, len(legs_tf))
        bet = sum(trio_stakes.values()) + u_tf * len(legs_tf)

        top3 = frozenset(order[:3])
        hit_trio = top3 in {_combo_key(t, False) for t in legs_trio}
        hit_tf = "-".join(map(str, order[:3])) in legs_tf
        # pm のオッズは「100円あたりの払戻」なので賭け金で按分する
        trio_odds = pm.get(rk, {}).get(("trio", top3), 0)
        tf_odds = pm.get(rk, {}).get(("trifecta", tuple(order[:3])), 0)
        pay_trio = trio_odds * trio_stakes[top3] // 100 if hit_trio else 0
        pay_tf = tf_odds * u_tf // 100 if hit_tf else 0

        rows.append({
            "race_date": c["race_date"],
            "race_key": rk + SUFFIX,
            "rank": RANK,
            "pred_combo": ("三複:" + ",".join(legs_trio) + " / 三単:" + ",".join(legs_tf)),
            "n_combos": len(legs_trio) + len(legs_tf),
            "hit": int(hit_trio or hit_tf),
            "payout": int(pay_trio + pay_tf),
            # ⚠️ trio_payout / trifecta_payout は **全ランク共通で「100円あたりの確定配当」**
            #    （賭け金非依存の生値）。ここに実払戻額を入れていたため、同じ列が
            #    他ランクと違う意味になり Web が実額と配当を混ぜて表示していた
            #    （2026-08-08 是正・notify_results_wt 側と対）。実額は payout に入る。
            "trio_payout": int(trio_odds),
            "trifecta_payout": int(tf_odds),
            "bet_amount": int(bet),
        })
    return rows


def wipe(date_from: str, date_to: str, dry_run: bool) -> None:
    if dry_run:
        return
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM picks_history WHERE rank=? AND race_key LIKE ? "
            "AND race_date BETWEEN ? AND ?",
            (RANK, "%" + SUFFIX, date_from, date_to))
        conn.commit()


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
    print(f"[backfill-7h1] get_connection先 {len(rows)}件 書き込み完了")

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        print("[backfill-7h1] KEIRIN_DB_URL 未設定 → VPS PG ミラーはスキップ")
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
    print(f"[backfill-7h1] VPS PG {len(rows)}件 書き込み完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--eval-model", default="lgbm_wt_eval")
    ap.add_argument("--win-model", default="lgbm_wt_win")
    ap.add_argument("--bad-model", default="lgbm_wt_bad")
    ap.add_argument("--favbust-model", default="lgbm_wt_favbust")
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 過去日に本番モデルを当てると in-sample になるので落とす（2026-08-08）。
    # 既定値が本番モデル名なので、指定を忘れると**無言で**そうなっていた。
    _end = args.end
    if _end:
        assert_vintage_for_past(_end, {"bad": args.bad_model, "eval": args.eval_model, "favbust": args.favbust_model, "win": args.win_model})

    rows = build_rows(args.start, args.end, eval_model=args.eval_model,
                      win_model=args.win_model, bad_model=args.bad_model,
                      favbust_model=args.favbust_model)
    n_hit = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    print(f"[backfill-7h1] {args.start}〜{args.end}: {len(rows)}R 的中{n_hit} "
          f"({n_hit / len(rows) * 100 if rows else 0:.1f}%) "
          f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%")
    if args.wipe:
        wipe(args.start, args.end, args.dry_run)
    insert_rows(rows, args.dry_run)


if __name__ == "__main__":
    main()
