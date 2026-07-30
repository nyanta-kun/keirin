"""not_both_top3を「exactly_one(◎◯どちらか一方のみ3着内)」と「neither(両方圏外)」
に分解し、S7の真のターゲットであるボリュームゾーンを特定する（2026-07-29・
[[keirin_s7_foundational_rethink_2026_07_29]]）。

ユーザー指摘: S7が真にターゲットとすべきは「WINTICKETの◎◯のいずれかのみ3着以内に
来るケース(exactly_one)」のボリュームゾーンである。「両方圏外(neither)」は
稀だが高配当な別カテゴリで、条件次第ではS7とは別ランクで狙う対象。まずこの
2カテゴリの母数・配当をhonestに切り分ける。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
PCTS = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def load_data():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.n_entries = 7 AND e.pred_top3_pct IS NOT NULL "
            "AND r.race_date >= :from_date",
            {"from_date": TRAIN_FROM}).fetchall()
    return rows


def load_trio_odds(race_keys):
    import re
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
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
                    out.setdefault(rk, {})[parts] = fv
    return out


def pctile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
    return sorted_vals[idx]


def main():
    print("データ読み込み中(2024-01-01〜)...")
    rows = load_data()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)
    race_keys = list(by_race.keys())
    trio_odds = load_trio_odds(race_keys)

    races = []
    for rk, ents in by_race.items():
        if len(ents) != 7:
            continue
        race_date = str(ents[0]["race_date"])
        honmei = next((e for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((e for e in ents if e["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = honmei["frame_no"] in winners
        t_hit = taikou["frame_no"] in winners
        if h_hit and t_hit:
            category = "both"
        elif h_hit or t_hit:
            category = "exactly_one"
        else:
            category = "neither"
        trio = trio_odds.get(rk)
        odds = trio.get(winners) if trio else None
        races.append({
            "race_key": rk, "race_date": race_date, "category": category,
            "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]

    for label, data in (("全期間", races), ("TRAIN", train), ("TEST", test)):
        print(f"\n{'='*70}\n[{label}] カテゴリ別 母数・配当分布 (n={len(data)})\n{'='*70}")
        for cat in ("both", "exactly_one", "neither"):
            sub = [r for r in data if r["category"] == cat and r["trio_odds"] is not None]
            n = len(sub)
            share = n / len(data) * 100
            if n == 0:
                continue
            vals = sorted(r["trio_odds"] for r in sub)
            mean = sum(vals) / n
            over5 = sum(1 for v in vals if v >= 5) / n * 100
            over30 = sum(1 for v in vals if v >= 30) / n * 100
            print(f"  {cat:<14} n={n:>6} ({share:>5.1f}%)  平均={mean:>7.1f}倍  "
                  f"中央値={pctile(vals,50):>6.1f}倍  p90={pctile(vals,90):>6.1f}倍  "
                  f"5倍+率={over5:>5.1f}%  30倍+率={over30:>5.1f}%")


if __name__ == "__main__":
    main()
