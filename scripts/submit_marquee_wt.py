#!/usr/bin/env python3
"""看板レースの取りこぼしを埋める（2026-08-09 新設）。

通常の波（`daily_picks_wt.sh` 07:00 / `wave_submit_wt.sh` noon・evening）が
出し終えた**後**に走り、**看板レースとその前後で商品が無いもの**を
`--marquee` 経路で入稿する。

## なぜ必要か

ランクのゲートは的中率・ROI で切っているので、**売れるかどうかは見ていない**。
その結果 2026-08-08 / 08-09 と連続で、当日最大の看板（GIII S級決勝・
GI ガールズ決勝）に商品がゼロだった。08-09 は手作業で11件を埋めた。
本スクリプトはそれを自動化する。詳細は `src/marquee.py` の docstring。

## 判定と軸

- 対象 = 看板レース（決勝/特選/選抜/特秀）とその前後1R（`src/marquee.py`）
- 既にどれかのランクで入稿済み → skip（1レース1商品）
- 発走済み → skip
- 車数が 7/9 以外 → skip（ランクが7車/9車しか無いため構造的に入稿できない）
- 軸 = 当日の指数（allindex JSON）の pred 上位2車

⚠️ **ミッドナイト（第1R 18時以降）は evening の波より前に出さない。**
   三連複の板が朝は63%欠損しており、傾斜配分が均等割りへ落ちる。

使い方:
    python scripts/submit_marquee_wt.py [YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection  # noqa: E402
from src.marquee import marquee_race_nos  # noqa: E402

JST = timezone(timedelta(hours=9))
PICKS = Path(__file__).resolve().parent.parent / "data" / "picks"
# 手動入稿で使うランク。ゲート表示の付かない中立のものを選ぶ
# （看板レースは「必ず出す」ので「自信あり」を意味するランクは使わない）。
RANK_BY_CARS = {7: "7A", 9: "9A"}


def _load_allindex(date: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for suffix in ("", "_night"):
        p = PICKS / f"wave_picks_wt_{date}{suffix}_allindex.json"
        if not p.exists():
            continue
        for x in json.loads(p.read_text(encoding="utf-8")):
            out[str(x["race_key"])] = x
    return out


def _axes(entry: dict) -> tuple[int, int] | None:
    riders = sorted(entry.get("riders") or [], key=lambda r: r.get("ai_rank", 99))
    if len(riders) < 2:
        return None
    return int(riders[0]["frame_no"]), int(riders[1]["frame_no"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=datetime.now(JST).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    date = args.date
    now_ts = int(datetime.now(JST).timestamp())

    with get_connection() as conn:
        races = [dict(r) for r in conn.execute(
            "SELECT race_key, venue_id, race_no, race_type, n_entries, start_at, cup_id "
            "FROM wt_races WHERE race_date = ? ORDER BY venue_id, race_no", (date,))]
        submitted = {str(dict(r)["race_key"]).split("#")[0] for r in conn.execute(
            "SELECT race_key FROM netkeirin_submissions")}

    if not races:
        print(f"[marquee] {date}: レースが無い", flush=True)
        return 0

    # 開催（cup_id）ごとに看板＋前後を割り出す
    by_cup: dict[str, list[dict]] = {}
    for r in races:
        by_cup.setdefault(str(r["cup_id"]), []).append(r)

    allidx = _load_allindex(date)
    targets: list[dict] = []
    for cup, rs in by_cup.items():
        want = marquee_race_nos(rs)
        # 🔴 ミッドナイトは evening の波を待つ（朝に出すと板が育っておらず
        #    傾斜配分が均等割りへ落ちる）。開催の第1R発走で判定する。
        first_start = min(int(x["start_at"]) for x in rs if x.get("start_at"))
        hour = datetime.fromtimestamp(first_start, JST).hour
        is_night = hour >= 18
        for r in rs:
            if int(r["race_no"]) not in want:
                continue
            if r["race_key"] in submitted:
                continue
            if r.get("start_at") and int(r["start_at"]) <= now_ts:
                continue
            if int(r.get("n_entries") or 0) not in RANK_BY_CARS:
                continue
            if is_night and datetime.now(JST).hour < 18:
                continue
            targets.append(r)

    if not targets:
        print(f"[marquee] {date}: 埋める看板レースは無い", flush=True)
        return 0

    ok = ng = 0
    for r in sorted(targets, key=lambda x: int(x["start_at"] or 0)):
        e = allidx.get(r["race_key"])
        if not e:
            print(f"[marquee] {r['race_key']}: 指数が無い（skip）", flush=True)
            ng += 1
            continue
        ax = _axes(e)
        if ax is None:
            print(f"[marquee] {r['race_key']}: 軸を決められない（skip）", flush=True)
            ng += 1
            continue
        rank = RANK_BY_CARS[int(r["n_entries"])]
        # 記録される波ラベルは実行時刻に合わせる（DB の netkeirin_submissions.session）。
        # 固定にすると後から「どの波で埋めたか」が追えない。
        h = datetime.now(JST).hour
        session = "morning" if h < 12 else ("noon" if h < 18 else "evening")
        cmd = [sys.executable, "scripts/netkeirin_submit_wt.py", date, session,
               "--marquee", "--race-key", r["race_key"],
               "--manual-rank-key", rank, "--axis1", str(ax[0]), "--axis2", str(ax[1])]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[marquee] {e.get('venue_name')}{r['race_no']}R "
              f"({r['race_type']}) 軸={ax[0]}-{ax[1]} → {rank}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parent.parent))
        sys.stdout.write(p.stdout)
        if p.returncode == 0 and "入稿失敗" not in p.stdout:
            ok += 1
        else:
            ng += 1
            sys.stderr.write(p.stderr)
    print(f"[marquee] {date}: 完了（成功{ok}件・失敗{ng}件）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
