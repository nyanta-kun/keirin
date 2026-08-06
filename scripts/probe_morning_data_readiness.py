"""朝の推奨作成を何時まで前倒しできるかを測る（読み取り専用・DB書き込みなし）。

## 背景

朝の `daily_picks_wt.sh` は 8:00 起動。ユーザー要望「7:00 / 7:30 にデータが
集まっているか確認したい」に答えるための実測ツール。

## なぜ実測しか手がないか

`wt_entries` のライン列・印・競走得点は **最新状態で UPSERT される**ため、
「過去のある時刻に何%が公開済みだったか」を遡って調べられない
（`check_line_readiness.py` の docstring にも同じ制約が記録されている）。
したがって実際にその時刻に WINTICKET を叩いて測るしかない。

## 測るもの（推奨生成が実際に依存する3つ）

| 項目 | 欠けると何が起きるか |
|---|---|
| `race_point`（競走得点） | `check_race_point_sanity.py` が異常検知 → 当日の指数算出・推奨を丸ごとスキップ |
| `linePrediction`（並び） | `n_lines=0` になる。7SS の同一ライン条件・ライン特徴量が壊れる |
| `prediction_mark`（◎○△） | overlap 判定が None → **7S/7A/7SS/7B すべてが対象外になり推奨ゼロ** |

⚠️ **DB には一切書き込まない**（scraper の fetch のみ使用）。
⚠️ WINTICKET へのリクエストが発生するので `--max-venues` で加減すること。

使い方:
    .venv/bin/python3 scripts/probe_morning_data_readiness.py
    .venv/bin/python3 scripts/probe_morning_data_readiness.py --date 2026-08-07
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.scraper.winticket import VENUE_SLUGS, WinticketScraper  # noqa: E402

JST = timezone(timedelta(hours=9))


def probe(target_date: str, max_venues: int, max_races: int) -> None:
    sc = WinticketScraper()
    now = datetime.now(JST).strftime("%H:%M:%S")
    print(f"=== 測定時刻 {now} JST / 対象日 {target_date} ===", flush=True)

    found = []
    for vid in VENUE_SLUGS:
        info = sc.find_cup_info(vid, target_date)
        if info:
            found.append((vid, *info))
            if len(found) >= max_venues:
                break
    if not found:
        print("開催会場が見つかりません（当日開催なし or 未公開）")
        return
    print(f"開催会場 {len(found)} 件を検出（--max-venues={max_venues} で打ち切り）\n")

    tot = {"races": 0, "rp": 0, "line": 0, "mark": 0}
    print(f"  {'会場':<6}{'R':>3}  {'頭数':>4}  {'得点':>6}  {'ライン':>7}  {'印':>6}  発走")
    for vid, cup_id, day_index in found:
        n_races = sc.get_race_count(vid, cup_id, day_index) or 0
        for rno in range(1, min(n_races, max_races) + 1):
            d = sc.fetch_race_data(vid, target_date, rno, cup_id, day_index)
            if not d:
                continue
            ents = d.get("entries") or []
            if not ents:
                continue
            tot["races"] += 1
            n = len(ents)
            n_rp = sum(1 for e in ents if e.get("race_point"))
            n_mk = sum(1 for e in ents if e.get("prediction_mark"))
            # ⚠️ fetch_race_data の返り値に "lineup" キーは**無い**（docstring は誤り）。
            # ライン情報は各 entry に line_group / n_lines として統合される。
            # 未公開時は n_lines=0・line_group=0 が既定値で入る
            # （check_line_readiness.py と同じ判定）。
            n_line = sum(1 for e in ents if (e.get("n_lines") or 0) > 0)
            ok_rp = n_rp == n
            ok_line = n_line > 0
            ok_mk = n_mk > 0
            tot["rp"] += ok_rp
            tot["line"] += ok_line
            tot["mark"] += ok_mk
            st = (d.get("race_info") or {}).get("start_at") or "-"
            try:
                st = datetime.fromtimestamp(int(st), JST).strftime("%H:%M")
            except (ValueError, TypeError):
                st = "-"
            f = lambda ok, k, n: ("✓" if ok else "✗") + f"{k}/{n}"
            print(f"  {vid:<6}{rno:>3}  {n:>4}  {f(ok_rp,n_rp,n):>6}"
                  f"  {f(ok_line,n_line,n):>7}  {f(ok_mk,n_mk,n):>6}  {st}")

    r = tot["races"]
    if not r:
        print("\nレースを取得できませんでした")
        return
    print(f"\n=== {now} JST 時点の充足率（{r} レース）===")
    for k, lbl in (("rp", "競走得点"), ("line", "ライン(並び)"), ("mark", "WT印 ◎○△")):
        print(f"  {lbl:<14} {tot[k]:>3}/{r}  ({100*tot[k]/r:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(JST).strftime("%Y-%m-%d"))
    ap.add_argument("--max-venues", type=int, default=4)
    ap.add_argument("--max-races", type=int, default=3)
    a = ap.parse_args()
    probe(a.date, a.max_venues, a.max_races)


if __name__ == "__main__":
    main()
