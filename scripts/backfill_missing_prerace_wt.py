#!/usr/bin/env python3
"""picks_history の自動完全性保証（2026-07-20）。

notify_prerace_wt.py のT-15分ライブ判定が何らかの理由（scraper障害・
システム停止・rebuild_*_walkforward.py の事故等）で実行されず picks_history
に記録が残らなかった日を検知し、最終オッズを使った build_rows() で
S1(SEVEN_S1) / S3(7PLUS_M) の該当日分を後追いで補完する。

ライブの実際の売買判定（judge_m/judge_u/judge_s1・T-15分オッズ基準）は
一切変更しない。あくまで picks_history という記録の完全性を事後に保証する
バックフィルであり、本来のT-15分判定に対しては最終オッズを使う近似になる
（三連複15倍フィルタ付近で数%〜十数%程度の判定タイミング差がありうる。
2026-07-20 調査）。

常に VPS 本番 PG（cron実行時は KEIRIN_DB_URL が自動的に localhost PG を
指す）を読み取り・書き込み両方に使う。rebuild_*_walkforward.py と異なり
ローカル SQLite には一切触れない。書き込みは対象race_keyが未存在の場合のみ
INSERT する（ON CONFLICT DO NOTHING・既存行の上書きは一切しない）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/backfill_missing_prerace_wt.py [--days 7] [--dry-run]

VPS cron（毎日 00:40 JST・intraday_results_wt.sh の 00:00 実行の後）:
    40 0 * * * cd $KEIRIN_HOME && PYTHONPATH=. .venv/bin/python3 \
        scripts/backfill_missing_prerace_wt.py >> $KEIRIN_HOME/data/logs/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_s1w_rank_wt import build_rows as build_rows_s1
from scripts.backfill_um_rank_wt import build_rows as build_rows_m
from src.database import get_connection
from src.notify.discord import send as discord_send

EVAL_MODEL = "lgbm_wt_eval"
WIN_MODEL = "lgbm_wt_win_eval"
MIN_RACES_FOR_DAY = 10  # この件数以上 wt_races があれば「レースが開催された日」とみなす


def _race_counts(date_from: str, date_to: str) -> dict[str, int]:
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_date, COUNT(*) FROM wt_races "
            "WHERE race_date BETWEEN ? AND ? GROUP BY race_date",
            (date_from, date_to))
        return dict(rows)


def _pick_counts(rank: str, date_from: str, date_to: str) -> dict[str, int]:
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_date, COUNT(*) FROM picks_history "
            "WHERE rank=? AND race_date BETWEEN ? AND ? GROUP BY race_date",
            (rank, date_from, date_to))
        return dict(rows)


def _insert_additive(rows: list[dict], is_m: bool) -> int:
    """race_key が未存在の場合のみ INSERT する（既存行は一切変更しない）。"""
    if not rows:
        return 0
    with get_connection() as c:
        inserted = 0
        for r in rows:
            cur = c.execute(
                "INSERT OR IGNORE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
                " trio_payout,trifecta_payout,bet_amount,route,miwokuri,"
                " gap12,gap23,gap34,gate_label,win_rank,ratio) "
                "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
                " :payout,:trio_payout,:trifecta_payout,:bet_amount,'wt',:miwokuri,"
                " :gap12,:gap23,:gap34,:gate_label,:win_rank,:ratio)",
                {
                    "race_date": r["race_date"], "race_key": r["race_key"],
                    "rank": r["rank"], "pred_combo": r.get("pred_combo"),
                    "n_combos": r.get("n_combos"), "hit": r["hit"],
                    "payout": r["payout"], "trio_payout": r.get("trio_payout", 0) or 0,
                    "trifecta_payout": r.get("trifecta_payout", 0) or 0,
                    "bet_amount": r["bet_amount"], "miwokuri": False,
                    "gap12": r.get("gap12"), "gap23": r.get("gap23"), "gap34": r.get("gap34"),
                    "gate_label": r.get("gate_label") if is_m else None,
                    "win_rank": r.get("win_rank") if is_m else None,
                    "ratio": r.get("ratio") if is_m else None,
                })
            inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        c.commit()
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="遡って確認する日数（既定7日・今日は含めない）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.today()
    date_to = (today - timedelta(days=1)).isoformat()
    date_from = (today - timedelta(days=args.days)).isoformat()

    race_counts = _race_counts(date_from, date_to)
    m_counts = _pick_counts("7PLUS_M", date_from, date_to)
    s1_counts = _pick_counts("SEVEN_S1", date_from, date_to)

    gap_dates_m = sorted(
        d for d, n in race_counts.items() if n >= MIN_RACES_FOR_DAY and m_counts.get(d, 0) == 0
    )
    gap_dates_s1 = sorted(
        d for d, n in race_counts.items() if n >= MIN_RACES_FOR_DAY and s1_counts.get(d, 0) == 0
    )

    print(f"[gap-heal] 確認期間: {date_from}〜{date_to}")
    print(f"[gap-heal] S3(M) 欠損日: {gap_dates_m}")
    print(f"[gap-heal] S1 欠損日: {gap_dates_s1}")

    if not gap_dates_m and not gap_dates_s1:
        print("[gap-heal] 欠損なし。終了。")
        return

    total_new_m = 0
    total_new_s1 = 0
    healed_summary: list[str] = []

    for d in gap_dates_m:
        rows = build_rows_m(EVAL_MODEL, d, d, win_model_name=WIN_MODEL)
        m_rows = [r for r in rows if r["rank"] == "7PLUS_M"]
        n = 0 if args.dry_run else _insert_additive(m_rows, is_m=True)
        if args.dry_run:
            n = len(m_rows)
        total_new_m += n
        print(f"[gap-heal] S3(M) {d}: 候補{len(m_rows)}件 → 挿入{n}件"
              f"{'（dry-run）' if args.dry_run else ''}")
        if n:
            healed_summary.append(f"S3(M) {d}: {n}件")

    for d in gap_dates_s1:
        rows = build_rows_s1(EVAL_MODEL, d, d, win_model_name=WIN_MODEL)
        n = 0 if args.dry_run else _insert_additive(rows, is_m=False)
        if args.dry_run:
            n = len(rows)
        total_new_s1 += n
        print(f"[gap-heal] S1 {d}: 候補{len(rows)}件 → 挿入{n}件"
              f"{'（dry-run）' if args.dry_run else ''}")
        if n:
            healed_summary.append(f"S1 {d}: {n}件")

    print(f"\n[gap-heal] 合計: S3(M) +{total_new_m}件 / S1 +{total_new_s1}件")

    if healed_summary and not args.dry_run:
        msg = (
            "⚠️ **picks_history 欠損を自動補完しました**\n"
            "notify_prerace_wt のライブ判定が実行されなかった日を検知し、"
            "最終オッズで後追い再構築しました（実際の売買判定には影響しません）。\n"
            + "\n".join(f"- {s}" for s in healed_summary)
        )
        discord_send(msg)


if __name__ == "__main__":
    main()
