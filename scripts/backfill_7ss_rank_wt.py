#!/usr/bin/env python3
"""7SS（波乱軸選出・穴レース検知・RANK_7SS）の過去分バックフィル。

S7/S9とは異なりモデル予測に依存しない（wt_entries の公表値のみで判定する）ため、
モデルロード・特徴量ビルドが不要でシンプルに再構築できる。判定は本番
（notify_prerace_wt.py の _build_rank_7ss_candidate + judge_rank_7s 流用）と
同一条件を最終オッズ盤面で再現する:

  7車ちょうど ∧ 盤面(trio)7車
  軸1 = race_point(競走得点)単独top1
  軸2 = WT公式印(prediction_mark 2,3,4=◯△✕)のうち軸1以外でthird_rate最大
  穴指数 = strategy_wt.rank_7ss_score()（TRAIN 2022-2023凍結パラメータ）
  採用条件 = 穴指数 >= RANK_7SS_SCORE_THRESHOLD（TRAIN上位20%点）
  買い目 = 三連複 軸2車 + 残り5車のいずれか1車（5点・オッズ下限なし）

採点は実精算方式: 盤面7車レースのみ対象・返還処理なし。
払戻 = 的中時 trio 最終オッズ×100。

## 欠車判定を void_by_dns へ統一（2026-07-31 是正・PMタスク C-2b-7ss）

【旧実装の問題】従来は `if len(board) != 7: continue`（＝盤面がちょうど7車で
なければ候補プールからレースごと除外）としており、本番 notify_results_wt.
_void_by_dns / src/evaluation/void_rules.py の基準（軸欠車=レース無効・
相手欠車=その目のみ除外して購入継続）と一致していなかった。相手(others)側の
1台だけが盤面から欠けたケースで、本来は「その1台を除いた残り4台で購入継続」
となるべきところ、レース全体を候補プールから除外していた（当時は
backfill_7s_rank_wt.py 等5本の統一作業（PMタスク C-2b）の対象外だった
ためこの乖離が残存し、その後 7SS が本番投入されたため本タスクで是正する）。

【本タスクでの修正】backfill_7s_rank_wt.py（PMタスク C-2b で統一済み）と
同一の方式を採用した:
  - board（欠車判定用の盤面掲載車集合）は `_load_board_frames_wt()` で構築。
    notify_results_wt._board_frames と同一の構築方法（bet_type='trio' の
    combination に現れる車番の和集合。odds_value によるフィルタなし）。
    従来の `_load_trio_boards()`（odds_value フィルタ済み）は「具体的コンボの
    購入可否判定」専用として存続させ、欠車判定には使わないよう分離した。
  - 軸1/軸2 が board に無い場合・相手候補（7車から軸2車を除いた5車）のうち
    board に無い車がある場合は `void_by_dns()`（本番と同一関数、
    src/evaluation/void_rules.py からそのまま import）へ委譲して判定する。
  - 相手が1台だけ欠けた場合、`others`（買い目候補の相手車リスト）が可変長
    （4点）になる。`n_combos`/`bet_amount` は元々 `len(combos)` から算出して
    いたため計算式自体の変更は不要。`pred_combo` は `others` ではなく実際に
    trio に存在した（購入された）目のみ（`bought_thirds`）を列挙するよう
    修正した（旧実装は `others` を直接列挙しており、`others` の要素が trio
    に存在せず実購入されない場合に `pred_combo` の表示と `n_combos`/`combos`
    が食い違う余地があったため。s7/s9/s7a/s9a と同型のバグ）。

  影響規模（読み取り専用DB調査・2026-07-31、n_entries=7 の全レース対象。
  backfill_7s_rank_wt.py と共通の母集団）: 全85,517レース中、盤面7車ちょうど=
  82,939件(97.0%)・盤面6車(1台欠け)=881件(1.03%)・盤面5車(2台欠け)=14件
  (0.02%)・盤面データなし=1,683件(1.97%)。
  「盤面データなし」（trio 行が1件も無い）レースは従来通り対象外のまま
  （`if not board: continue` で除外・void_by_dns 適用前の前提条件）。

  【7SS 固有の連鎖確認】7SS は `rank_7ss_score(feat) >= RANK_7SS_SCORE_THRESHOLD`
  のみで採否を決める独立per-race判定であり、S7本体の `rank_7s_evening_reselect`
  のような日次候補プールのトリム（RANK_7S_DAILY_CAP）は存在しない
  （src/strategy_wt.py の rank_7ss_* 関数・scripts/notify_prerace_wt.py の
  `_process_rank_7ss_candidates` のいずれにも日次件数上限のロジックはない）。
  したがって「相手1台欠け」レースが新たに候補プールへ復帰しても、他レースの
  採否には一切連鎖しない。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_7ss_rank_wt.py \
        --start 2022-01-01 --end 2026-07-30 [--wipe] [--dry-run]
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
from src.evaluation.void_rules import void_by_dns
from src.strategy_wt import (
    RANK_7SS_SCORE_THRESHOLD, RANK_7SS_STAKE, rank_7ss_field_features, rank_7ss_score,
    rank_7ss_select_axis,
)


def _load_trio_boards(race_keys: list[str]) -> dict:
    """具体的コンボの購入可否判定用（odds_value 有効値のみ）。

    欠車判定（void_by_dns）には使わない。欠車判定用の盤面掲載車集合は
    `_load_board_frames_wt()`（odds_value フィルタなし）を使うこと。
    """
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


def _load_board_frames_wt(race_keys: list[str]) -> dict[str, set[int]]:
    """欠車判定用の盤面掲載車集合を返す（notify_results_wt._board_frames /
    backfill_7s_rank_wt.py._load_board_frames_wt と同一の構築方法）。

    bet_type='trio' の combination に現れる車番の和集合。odds_value による
    フィルタは行わない（未確定・異常値でも盤面に車番として存在していれば
    「実際に購入できた車」とみなす本番の判定基準に合わせるため）。
    """
    board_map: dict[str, set[int]] = defaultdict(set)
    if not race_keys:
        return board_map
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb in c.execute(q, chunk):
                for part in re.split(r"[-=]", str(comb)):
                    try:
                        board_map[rk].add(int(part))
                    except ValueError:
                        pass
    return board_map


def build_rows(date_from: str, date_to: str) -> list[dict]:
    """バックフィル対象の 7SS(#7SS) 行（採点済み）を構築する。"""
    with get_connection() as c:
        races = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (date_from, date_to)).fetchall()
        race_keys = [r["race_key"] for r in races]
        date_map = {r["race_key"]: r["race_date"] for r in races}

        entries_by_race: dict[str, list[dict]] = defaultdict(list)
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, race_point, line_group, line_size, n_lines, "
                 "       first_rate, third_rate, finish_order, prediction_mark "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                entries_by_race[r["race_key"]].append(dict(r))

    trio_bd = _load_trio_boards(race_keys)
    board_map = _load_board_frames_wt(race_keys)
    pm = _load_payouts_wt(race_keys)

    candidates: list[dict] = []
    for rk in race_keys:
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = board_map.get(rk)
        if not board:
            continue
        fin = sorted((e["finish_order"], int(e["frame_no"])) for e in ents
                     if e["finish_order"] is not None and e["finish_order"] >= 1)
        if len(fin) < 3:
            continue

        axis = rank_7ss_select_axis(ents)
        if axis is None:
            continue
        feat = rank_7ss_field_features(ents)
        if feat is None:
            continue
        score = rank_7ss_score(feat)
        if score < RANK_7SS_SCORE_THRESHOLD:
            continue
        axis1, axis2 = axis

        # 欠車判定を本番と同一の void_by_dns へ統一（2026-07-31 是正・PMタスク C-2b-7ss）。
        # 軸欠車=レース無効／相手欠車=その目のみ除外して購入継続（可変点数）。
        field_frames = {int(e["frame_no"]) for e in ents}
        thirds_full = sorted(field_frames - {axis1, axis2})
        skip_race, others = void_by_dns(axis1, axis2, thirds_full, board)
        if skip_race:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "score": score,
            "others": others, "trio": trio, "actual_top3": actual_top3,
        })

    rows: list[dict] = []
    for c_ in candidates:
        axis1, axis2 = c_["axis1"], c_["axis2"]
        trio = c_["trio"]
        # combos/bought_thirds を同期して構築する（2026-07-31 是正・PMタスク C-2b-7ss）。
        # pred_combo は実際に買った目(bought_thirds)のみを列挙する（c_["others"]は
        # void_by_dns 後の候補であり、個別コンボのオッズ有効性チェック前のため、
        # 一部が trio に存在せず実購入されないケースがありうる）。
        combos, bought_thirds = [], []
        for x in c_["others"]:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combos.append(key)
                bought_thirds.append(x)
        if not combos:
            continue
        rk = c_["race_key"]
        hit = c_["actual_top3"] in combos
        trio_pay = pm.get(rk, {}).get(("trio", c_["actual_top3"]), 0)
        pay = trio_pay * RANK_7SS_STAKE // 100 if hit else 0
        bet = len(combos) * RANK_7SS_STAKE
        rows.append({
            "race_date": c_["race_date"],
            "race_key": f"{rk}#7SS", "rank": "RANK_7SS",
            "pred_combo": f"{axis1}={axis2}-" + ",".join(str(x) for x in bought_thirds)
                          + f" (score={c_['score']:.2f})",
            "n_combos": len(combos), "hit": int(hit), "payout": pay,
            "trio_payout": trio_pay, "bet_amount": bet,
        })
    return rows


def wipe_rows(date_from: str, date_to: str, dry_run: bool) -> None:
    cond = "rank='RANK_7SS' AND race_key LIKE '%#7SS' AND race_date BETWEEN ? AND ?"
    with get_connection() as conn:
        n = conn.execute(
            f"SELECT COUNT(*) FROM picks_history WHERE {cond}",
            (date_from, date_to)).fetchone()[0]
        print(f"[backfill] 既存 #7SS 行（{date_from}〜{date_to}）: {n}件 → 削除"
              f"{'（dry-run）' if dry_run else ''}")
        if not dry_run and n:
            conn.execute(f"DELETE FROM picks_history WHERE {cond}", (date_from, date_to))
            conn.commit()

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    import psycopg2
    cond_pg = "rank='RANK_7SS' AND race_key LIKE %s AND race_date BETWEEN %s AND %s"
    with psycopg2.connect(db_url) as pg:
        with pg.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM keirin.picks_history WHERE {cond_pg}",
                        ("%#7SS", date_from, date_to))
            n = cur.fetchone()[0]
            print(f"[backfill] VPS PG 既存 #7SS 行: {n}件 → 削除{'（dry-run）' if dry_run else ''}")
            if not dry_run and n:
                cur.execute(f"DELETE FROM keirin.picks_history WHERE {cond_pg}",
                            ("%#7SS", date_from, date_to))


def insert_rows(rows: list[dict], dry_run: bool) -> None:
    if dry_run or not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO picks_history "
            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
            " trio_payout,bet_amount,route,miwokuri) "
            "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
            " :payout,:trio_payout,:bet_amount,'wt',False)",
            rows)
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
                   trio_payout,bet_amount,route,miwokuri)
                VALUES (%(race_date)s,%(race_key)s,%(rank)s,%(pred_combo)s,
                        %(n_combos)s,%(hit)s,%(payout)s,%(trio_payout)s,
                        %(bet_amount)s,'wt',FALSE)
                ON CONFLICT (race_key) DO UPDATE SET
                  race_date=EXCLUDED.race_date, rank=EXCLUDED.rank,
                  pred_combo=EXCLUDED.pred_combo, n_combos=EXCLUDED.n_combos,
                  hit=EXCLUDED.hit, payout=EXCLUDED.payout,
                  trio_payout=EXCLUDED.trio_payout,
                  bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE
            """, rows, page_size=200)
    print(f"[backfill] VPS PG {len(rows)}件 書き込み完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", required=False)
    ap.add_argument("--wipe", action="store_true",
                    help="書き込み前に対象期間の既存 #7SS 行を削除")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from datetime import date
    end = args.end or date.today().strftime("%Y-%m-%d")
    print(f"[backfill] 7SS {args.start}〜{end}", flush=True)

    if args.wipe:
        wipe_rows(args.start, end, args.dry_run)

    rows = build_rows(args.start, end)
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    ret = sum(r["payout"] for r in rows)
    roi = ret / bet * 100 if bet else 0
    max_pay = max((r["trio_payout"] for r in rows if r["hit"]), default=0)
    print(f"[backfill] 7SS(波乱軸選出): {n}R 的中{hits} ({hits/n*100 if n else 0:.1f}%) "
          f"投資{bet:,} → 回収{ret:,} ROI {roi:.1f}%  最高払戻{max_pay/100:.1f}倍", flush=True)

    insert_rows(rows, args.dry_run)
    if args.dry_run:
        print("[backfill] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
