#!/usr/bin/env python3
"""朝時点の候補レースを picks_history に即時書き込む。

gap12 条件を満たす全候補を picks_history に書き込むことで、
同日中から推奨ページに候補レース（miwokuri=False, bet_amount=0）を表示できる。
翌朝 notify_results_wt.py が購入済み/見送りを正確に上書きする。

実行:
    python3 scripts/write_candidates_wt.py [YYYY-MM-DD]

daily_picks_wt.sh および evening_picks_wt.sh から wave-picks-wt の直後に呼ばれる。

初回ガミ判定:
  書き込み後、winticket からその時点の三連複オッズを取得して prerace_gami を設定する。
  最安オッズ < 5.0 の候補は miwokuri=True として早期見送り扱いにする。
  発走 15 分前の notify_prerace_wt.py がリアルタイムオッズで上書き（最終確認）する。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

GAMI_THRESHOLD = 5.0


def _insert_candidates_sqlite(
    target_date: str, rows: list[tuple], existing: set[str]
) -> tuple[int, list[str]]:
    """SQLite に #CAND を INSERT し、新規挿入した race_key (base) リストを返す。"""
    inserted = 0
    newly_inserted_bases: list[str] = []
    with get_connection() as conn:
        for target_date_v, store_key, rank, pred, n_combos in rows:
            base = store_key.rsplit("#", 1)[0]
            if any(k.startswith(base + "#") and k != store_key for k in existing):
                continue
            if store_key in existing:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,'wt',False)",
                (target_date_v, store_key, rank, pred, n_combos),
            )
            existing.add(store_key)
            inserted += 1
            newly_inserted_bases.append(base)
    return inserted, newly_inserted_bases


def _insert_candidates_vps(target_date: str, rows: list[tuple], existing: set[str]) -> None:
    """VPS PostgreSQL に #CAND を直接 INSERT する（hourly sync を待たずに即時反映）。"""
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    to_insert = []
    for target_date_v, store_key, rank, pred, n_combos in rows:
        base = store_key.rsplit("#", 1)[0]
        if any(k.startswith(base + "#") and k != store_key for k in existing):
            continue
        to_insert.append((target_date_v, store_key, rank, pred, n_combos))
    if not to_insert:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                for target_date_v, store_key, rank, pred, n_combos in to_insert:
                    cur.execute(
                        """INSERT INTO keirin.picks_history
                            (race_date, race_key, rank, pred_combo, n_combos,
                             hit, payout, trio_payout, bet_amount, route, miwokuri)
                           VALUES (%s, %s, %s, %s, %s, 0, 0, 0, 0, 'wt', FALSE)
                           ON CONFLICT DO NOTHING""",
                        (target_date_v, store_key, rank, pred, n_combos),
                    )
    except Exception as e:
        print(f"[write_candidates_wt] VPS INSERT 失敗: {e}", flush=True)


def _save_initial_gami(race_key: str, race_date: str, min_odds: float, miwokuri: bool) -> None:
    """prerace_gami（初回）と miwokuri を SQLite + VPS に保存する。

    #CAND エントリのみを更新する。#7SS / #7S などの採点済みエントリは
    notify_results_wt.py が管理するため上書きしない。
    """
    rounded = round(min_odds, 2)
    cand_key = race_key + "#CAND"  # #CAND のみ対象（採点済みエントリを上書きしない）

    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET prerace_gami = ?, miwokuri = ? WHERE race_key = ?",
                (rounded, miwokuri, cand_key),
            )
            conn.commit()
    except Exception as e:
        print(f"[write_candidates_wt] SQLite prerace_gami 更新失敗 {race_key}: {e}", flush=True)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute(
                    "UPDATE keirin.picks_history SET prerace_gami = %s, miwokuri = %s"
                    " WHERE race_key = %s",
                    (rounded, miwokuri, cand_key),
                )
    except Exception as e:
        print(f"[write_candidates_wt] VPS prerace_gami 更新失敗 {race_key}: {e}", flush=True)


def _fetch_initial_gami(candidates: list[dict]) -> None:
    """候補レースのオッズを取得して初回ガミ判定（prerace_gami）を設定する。

    winticket から三連複オッズを取得し最安値を prerace_gami に保存する。
    最安値 < 5.0 なら miwokuri=True（早期見送り）。
    取得失敗時は prerace_gami=NULL のまま notify_prerace_wt.py に委ねる。
    """
    from src.scraper.winticket import WinticketScraper  # noqa: PLC0415

    race_keys = [c["race_key"] for c in candidates if c.get("race_key")]
    if not race_keys:
        return

    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, venue_id, cup_id, day_index, race_date, race_no"
            f" FROM wt_races WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    race_info_map = {r["race_key"]: dict(r) for r in rows}

    scraper = WinticketScraper(request_interval=1.0)

    for cand in candidates:
        rk = cand.get("race_key")
        if not rk:
            continue
        ri = race_info_map.get(rk)
        if not ri:
            print(f"[write_candidates_wt] {rk}: wt_races にレース情報なし（スキップ）", flush=True)
            continue

        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        if not thirds or p1 is None or p2 is None:
            continue

        try:
            odds_data = scraper.fetch_odds(
                venue_id=ri["venue_id"],
                race_date=ri["race_date"],
                race_no=ri["race_no"],
                cup_id=ri["cup_id"],
                day_index=ri["day_index"],
            )
        except Exception as e:
            print(f"[write_candidates_wt] {rk}: オッズ取得失敗: {e}", flush=True)
            time.sleep(0.5)
            continue

        if not odds_data:
            print(f"[write_candidates_wt] {rk}: オッズデータなし（スキップ）", flush=True)
            time.sleep(0.5)
            continue

        # 三連複ルックアップ
        trio_lookup: dict[frozenset, float] = {}
        for item in odds_data.get("trio", []):
            parts = re.split(r"[-=]", str(item.get("combination", "")))
            try:
                key = frozenset(int(p) for p in parts)
                if item.get("odds_value"):
                    trio_lookup[key] = float(item["odds_value"])
            except Exception:
                continue

        valid_odds = []
        for t in thirds:
            key = frozenset({int(p1), int(p2), int(t)})
            if key in trio_lookup:
                valid_odds.append(trio_lookup[key])

        if not valid_odds:
            print(f"[write_candidates_wt] {rk}: 対象組み合わせのオッズなし（スキップ）", flush=True)
            time.sleep(0.5)
            continue

        min_odds = min(valid_odds)
        miwokuri = min_odds < GAMI_THRESHOLD
        _save_initial_gami(rk, ri["race_date"], min_odds, miwokuri)
        status = "⚠️見送り" if miwokuri else "✅OK"
        print(
            f"[write_candidates_wt] {rk}: 初回ガミ判定 最安{min_odds:.1f}倍 {status}",
            flush=True,
        )
        time.sleep(0.5)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    candidates: list[dict] = []
    for fname in (
        f"wave_picks_wt_{target_date}_candidates.json",
        f"wave_picks_wt_{target_date}_night_candidates.json",
    ):
        p = picks_dir / fname
        if p.exists():
            try:
                candidates += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[write_candidates_wt] {fname} 読み込み失敗: {e}", flush=True)

    if not candidates:
        print(f"[write_candidates_wt] {target_date}: candidates なし", flush=True)
        return

    rows: list[tuple] = []
    for cand in candidates:
        rk = cand.get("race_key")
        if not rk:
            continue
        gap12 = cand.get("gap12", 0.0)
        if gap12 < 0.10:
            continue  # Aランク廃止（2026-06-28）
        rank = "7PLUS_S"
        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
        n_combos = len(thirds)
        store_key = f"{rk}#CAND"
        rows.append((target_date, store_key, rank, pred, n_combos))

    with get_connection() as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT race_key FROM picks_history WHERE race_date=? AND route='wt'",
                (target_date,),
            ).fetchall()
        }

    inserted, _ = _insert_candidates_sqlite(target_date, rows, existing)
    _insert_candidates_vps(target_date, rows, existing)

    print(
        f"[write_candidates_wt] {target_date}: {inserted}/{len(rows)} 件書き込み完了",
        flush=True,
    )

    # 初回ガミ判定（INSERT 後に実行・失敗してもメイン処理には影響しない）
    try:
        _fetch_initial_gami(candidates)
    except Exception as e:
        print(f"[write_candidates_wt] 初回ガミ判定 予期しないエラー: {e}", flush=True)


if __name__ == "__main__":
    main()
