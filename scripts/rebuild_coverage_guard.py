#!/usr/bin/env python3
"""rebuild_s1_walkforward.py / rebuild_s3_walkforward.py 共通のカバレッジガード。

2026-07-19、ローカル SQLite (data/keirin.db) が 2026-07-10 で日次収集停止したまま
rebuild_s3_walkforward.py を実行し、build_rows が 7/11〜7/18 分を返せない状態で
wipe_rank_rows だけが VPS 本番の同期間 picks_history（S3(M)の実ライブ稼働記録）を
削除して復元されず消失する事故が発生した。同型の再発を検知して中断する安全弁。

このモジュールはインポート時副作用（環境変数操作等）を持たない。
rebuild_*_walkforward.py 側で `os.environ.pop("KEIRIN_DB_URL")` する前に
import すること（import 自体はどちらの順序でも安全）。
"""
from __future__ import annotations

import sys


def assert_local_covers_pg(date_from: str, date_to: str, pg_url: str | None, force: bool) -> None:
    """ローカル SQLite (wt_races) が VPS PG と同等に対象期間をカバーしているか確認する。"""
    if not pg_url:
        return
    from src.database import get_connection
    import psycopg2

    with get_connection() as c:
        local_counts = dict(c.execute(
            "SELECT race_date, COUNT(*) FROM wt_races "
            "WHERE race_date BETWEEN ? AND ? GROUP BY race_date",
            (date_from, date_to)))
    with psycopg2.connect(pg_url) as pg:
        with pg.cursor() as cur:
            cur.execute(
                "SELECT race_date::text, COUNT(*) FROM keirin.wt_races "
                "WHERE race_date BETWEEN %s AND %s GROUP BY race_date",
                (date_from, date_to))
            pg_counts = dict(cur.fetchall())

    bad_days = sorted(
        (d, local_counts.get(d, 0), pg_n)
        for d, pg_n in pg_counts.items()
        if pg_n > 0 and local_counts.get(d, 0) < pg_n * 0.5
    )
    if not bad_days:
        print(f"[coverage-guard] ローカルDBカバレッジ確認OK（{date_from}〜{date_to}）")
        return

    msg = (
        f"[coverage-guard] ローカル SQLite (data/keirin.db) が VPS PG より "
        f"{len(bad_days)}日分で著しく件数不足です（date, local, pg の先頭5件: "
        f"{bad_days[:5]}）。\n"
        f"[coverage-guard] この状態で wipe を実行すると、VPS本番の該当期間 "
        f"picks_history が削除された上で再構築されず消失します"
        f"（2026-07-19に実際に発生した事故と同型）。"
    )
    if force:
        print(msg + "\n[coverage-guard] --force指定のため続行します。")
    else:
        print(msg + "\n[coverage-guard] 中断します（ローカルDBの日次収集を復旧するか、"
                     "--force で無視して続行できます）。", file=sys.stderr)
        sys.exit(1)
