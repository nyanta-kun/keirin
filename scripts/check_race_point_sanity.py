#!/usr/bin/env python3
"""race_point（WINTICKET競走得点）の日次収集直後の健全性チェック。

2026-07-23、日次08:00収集(daily_picks_wt.sh)の時点でWINTICKET側がその日の
race_pointをまだ確定しておらず、異常に低い暫定値（平均4.3、正常時62-90）を
拾ってしまう事象が発生した（暫定値は相対順序も確定値と入れ替わっており、
モデル特徴量として使うと指数・推奨の質が劣化する）。

本スクリプトは対象日の平均race_pointを直近日の中央値と比較し、
異常に低ければ非ゼロ終了する（daily_picks_wt.shが再収集→再チェックの
リトライに使う・詳細はCLAUDE.md/メモリ参照）。

使い方:
    PYTHONPATH=. .venv/bin/python3 scripts/check_race_point_sanity.py --date 2026-07-23
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

RATIO_THRESHOLD = 0.5   # 直近日中央値のこの割合を下回れば異常とみなす
BASELINE_DAYS = 7        # 直近何日分を基準にするか
MIN_ENTRIES = 10         # 対象日のサンプルがこれ未満なら判定不能としてOK扱い（未収集等）


def check(target_date: str) -> tuple[bool, str]:
    """returns (is_ok, message)"""
    with get_connection() as conn:
        today_row = conn.execute(
            "SELECT AVG(e.race_point) avg_rp, COUNT(*) n FROM wt_entries e "
            "JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.race_date = ? AND e.race_point IS NOT NULL",
            (target_date,)).fetchone()
        today_avg = today_row["avg_rp"]
        today_n = today_row["n"]

        baseline_rows = conn.execute(
            "SELECT r.race_date, AVG(e.race_point) avg_rp FROM wt_entries e "
            "JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.race_date < ? AND e.race_point IS NOT NULL "
            "GROUP BY r.race_date ORDER BY r.race_date DESC LIMIT ?",
            (target_date, BASELINE_DAYS)).fetchall()

    if today_n is None or today_n < MIN_ENTRIES:
        return True, f"{target_date}: サンプル数不足(n={today_n})のため判定スキップ"

    baseline_values = [r["avg_rp"] for r in baseline_rows if r["avg_rp"] is not None]
    if len(baseline_values) < 3:
        return True, f"{target_date}: 基準日データ不足のため判定スキップ"

    baseline_median = statistics.median(baseline_values)
    if baseline_median <= 0:
        return True, f"{target_date}: 基準中央値が0のため判定スキップ"

    ratio = today_avg / baseline_median
    if ratio < RATIO_THRESHOLD:
        return False, (
            f"{target_date}: race_point異常検知 — 平均{today_avg:.2f}"
            f"（直近{len(baseline_values)}日中央値{baseline_median:.2f}の{ratio*100:.0f}%・"
            f"閾値{RATIO_THRESHOLD*100:.0f}%未満）n={today_n}"
        )
    return True, (
        f"{target_date}: race_point正常 — 平均{today_avg:.2f}"
        f"（直近{len(baseline_values)}日中央値{baseline_median:.2f}の{ratio*100:.0f}%）n={today_n}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    is_ok, message = check(args.date)
    print(f"[race_point_sanity] {message}")
    sys.exit(0 if is_ok else 1)


if __name__ == "__main__":
    main()
