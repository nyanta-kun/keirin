#!/usr/bin/env python3
"""オッズ帯を「丸ごと買った」ときの素の ROI と高額到達率を測る。

## なぜこれが決定的なのか

[[exp_highpay_ceiling_census]] で「30N倍を超える的中がどれくらいの頻度で起きるか」は
分かったが、それは**当てられた場合**の話。実際に狙うには帯を買う必要があり、
その帯の**素の回収率**が高額イベントの上限を直接決める:

    P(高額) = 帯ROI / (要求倍率) = 帯ROI / 30

つまり帯ROIが控除率どおり0.75なら 2.5%、40%しかないなら 1.33% が上限になる。
競馬・競輪には favorite–longshot bias（人気薄が買われすぎる）があり、
**高オッズ帯ほど素のROIは悪い**のが定説。ここではそれを実測する。

方法: レースをサンプリングし、三連単・三連複の全通りをオッズ帯に分類。
帯ごとに「点数」と「その帯に的中が落ちたときの配当」を積み上げ、

    帯ROI = Σ(帯に的中したときの配当) / Σ(帯の点数)

DB は読み取りのみ。全レース走査は重いのでサンプリングする。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

BANDS = [(0, 15), (15, 30), (30, 60), (60, 90), (90, 150), (150, 300),
         (300, 600), (600, 1200), (1200, 10 ** 9)]


def _band_of(o: float) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= o < hi:
            return i
    return len(BANDS) - 1


def _band_label(i: int) -> str:
    lo, hi = BANDS[i]
    return f"{lo:5}-{hi if hi < 10 ** 9 else '∞':>6}倍"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--from", dest="date_from", default="2024-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-08-04")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with get_connection() as c:
        q = """
        WITH res AS (
          SELECT race_key,
                 MAX(CASE WHEN finish_order = 1 THEN frame_no END) AS f1,
                 MAX(CASE WHEN finish_order = 2 THEN frame_no END) AS f2,
                 MAX(CASE WHEN finish_order = 3 THEN frame_no END) AS f3
          FROM keirin.wt_entries GROUP BY race_key
        )
        SELECT res.race_key, res.f1, res.f2, res.f3
        FROM res JOIN keirin.wt_races r ON r.race_key = res.race_key
        WHERE r.race_date >= ? AND r.race_date <= ? AND r.n_entries = ?
          AND res.f1 IS NOT NULL AND res.f2 IS NOT NULL AND res.f3 IS NOT NULL
        """
        races = [dict(r) for r in
                 c.execute(q, (args.date_from, args.date_to, args.n_car)).fetchall()]
        print(f"母集団 {len(races)} レース ({args.n_car}車)", flush=True)
        random.seed(args.seed)
        if len(races) > args.sample:
            races = random.sample(races, args.sample)
        print(f"サンプル {len(races)} レース", flush=True)

        winner = {}
        for r in races:
            f1, f2, f3 = int(r["f1"]), int(r["f2"]), int(r["f3"])
            winner[r["race_key"]] = {
                "trifecta": f"{f1}-{f2}-{f3}",
                "trio": frozenset((f1, f2, f3)),
            }

        # 点数・回収を帯ごとに積む
        stat = {bt: defaultdict(lambda: {"n_pts": 0, "ret": 0.0, "hits": 0})
                for bt in ("trifecta", "trio")}
        keys = list(winner)
        for i in range(0, len(keys), 300):
            chunk = keys[i:i + 300]
            ph = ",".join("?" * len(chunk))
            rows = c.execute(
                "SELECT race_key, bet_type, combination, odds_value FROM keirin.wt_odds "
                f"WHERE bet_type IN ('trifecta','trio') AND race_key IN ({ph})",
                chunk).fetchall()
            for row in rows:
                od = row["odds_value"]
                if od is None or float(od) <= 0:
                    continue
                od = float(od)
                bt = row["bet_type"]
                b = _band_of(od)
                s = stat[bt][b]
                s["n_pts"] += 1
                w = winner[row["race_key"]][bt]
                comb = row["combination"]
                if bt == "trifecta":
                    is_win = comb == w
                else:
                    is_win = frozenset(
                        int(x) for x in re.split(r"[-=→]", comb)) == w
                if is_win:
                    s["ret"] += od
                    s["hits"] += 1
            if (i // 300) % 5 == 0:
                print(f"  ...{i + len(chunk)}/{len(keys)}", flush=True)

    n_race = len(keys)
    for bt in ("trifecta", "trio"):
        print(f"\n=== {args.n_car}車 {bt}  帯別の素の成績（サンプル {n_race} レース）===")
        print("  帯              点数/R   的中率   帯ROI    30x到達必要N  P(高額)上限")
        tot_pts = tot_ret = 0
        for b in range(len(BANDS)):
            s = stat[bt][b]
            if not s["n_pts"]:
                continue
            tot_pts += s["n_pts"]
            tot_ret += s["ret"]
            pts_per_race = s["n_pts"] / n_race
            hit_rate = s["hits"] / n_race
            roi = s["ret"] / s["n_pts"]
            # 帯の下限オッズから逆算した「30N倍を満たす最大点数」
            lo = BANDS[b][0]
            max_n = lo / 30.0
            print(f"  {_band_label(b)}  {pts_per_race:7.1f} {hit_rate * 100:7.2f}% "
                  f"{roi * 100:7.1f}%   N<={max_n:6.1f}      {roi / 30 * 100:6.3f}%")
        print(f"  {'全通り':<14}  {tot_pts / n_race:7.1f} {100.0:7.2f}% "
              f"{tot_ret / tot_pts * 100:7.1f}%")


if __name__ == "__main__":
    main()
