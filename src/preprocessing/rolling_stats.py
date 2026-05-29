"""
DBの race_results からローリング統計を計算し race_entries に書き戻す。

- recent_win_rate_6m  : 直近6ヶ月（180日）勝率（DBから計算）
- recent_top3_rate_6m : 直近6ヶ月3着内率
- days_since_last_race: 前走からの経過日数

データ収集完了後に一度実行し、その後は collect の都度差分更新する想定。
"""
import logging
from datetime import datetime

import pandas as pd

from ..database import get_connection

log = logging.getLogger(__name__)


def compute_rolling_stats(dry_run: bool = False) -> dict:
    """
    全 race_entries の rolling stats を計算して更新する。
    既に値が入っているエントリはスキップ（force=True で上書き）。
    Returns: {"updated": N, "skipped": N}
    """
    return _run(dry_run=dry_run, force=False)


def recompute_rolling_stats(dry_run: bool = False) -> dict:
    """全エントリを強制再計算（モデル再訓練前に使用）"""
    return _run(dry_run=dry_run, force=True)


def _run(dry_run: bool, force: bool) -> dict:
    log.info("Loading race results from DB...")

    with get_connection() as conn:
        # レース日付付きの全結果
        results_df = pd.read_sql_query("""
            SELECT res.player_id, r.race_date, res.race_key,
                   res.finish_position,
                   r.venue_code
            FROM race_results res
            JOIN races r ON res.race_key = r.race_key
            WHERE res.finish_position IS NOT NULL
            ORDER BY res.player_id, r.race_date
        """, conn)

        # 更新対象エントリ
        where = "" if force else "WHERE e.recent_win_rate_6m IS NULL"
        entries_df = pd.read_sql_query(f"""
            SELECT e.id, e.race_key, e.player_id, r.race_date, r.venue_code
            FROM race_entries e
            JOIN races r ON e.race_key = r.race_key
            {where}
            ORDER BY r.race_date, e.race_key
        """, conn)

    if entries_df.empty:
        log.info("No entries to update.")
        return {"updated": 0, "with_data": 0}

    log.info(f"Computing stats for {len(entries_df):,} entries...")

    results_df["race_date"] = pd.to_datetime(results_df["race_date"])
    entries_df["race_date"] = pd.to_datetime(entries_df["race_date"])

    # player_id ごとに結果をグループ化（高速ルックアップ用）
    player_results = {
        pid: grp.sort_values("race_date")
        for pid, grp in results_df.groupby("player_id")
    }

    updates = []
    for _, row in entries_df.iterrows():
        pid = row["player_id"]
        race_date = row["race_date"]
        venue = row["venue_code"]

        p_res = player_results.get(pid)
        if p_res is None:
            updates.append((None, None, None, None, row["id"]))
            continue

        # 当該レースより前のみ
        past = p_res[p_res["race_date"] < race_date]
        if past.empty:
            updates.append((None, None, None, None, row["id"]))
            continue

        # 直近6ヶ月
        cutoff_6m = race_date - pd.Timedelta(days=180)
        recent_6m = past[past["race_date"] >= cutoff_6m]
        if len(recent_6m) >= 3:
            wr6m = (recent_6m["finish_position"] == 1).mean()
            t3r6m = (recent_6m["finish_position"] <= 3).mean()
        else:
            wr6m, t3r6m = None, None

        # 前走からの経過日数
        last_race_date = past["race_date"].max()
        days_since = (race_date - last_race_date).days

        # 同場勝率（全期間）
        venue_past = past[past["venue_code"] == venue]
        if len(venue_past) >= 3:
            venue_wr = (venue_past["finish_position"] == 1).mean()
        else:
            venue_wr = None

        updates.append((wr6m, t3r6m, days_since, venue_wr, row["id"]))

    if not dry_run:
        with get_connection() as conn:
            conn.executemany("""
                UPDATE race_entries
                SET recent_win_rate_6m  = ?,
                    recent_top3_rate_6m = ?,
                    days_since_last_race= ?,
                    venue_win_rate      = ?
                WHERE id = ?
            """, updates)
        log.info(f"Updated {len(updates):,} entries.")
    else:
        log.info(f"[dry-run] Would update {len(updates):,} entries.")

    non_null = sum(1 for u in updates if u[0] is not None)
    return {"updated": len(updates), "with_data": non_null}
