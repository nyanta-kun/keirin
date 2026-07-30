"""◎◯が両方は3着内に来ないケース(not_both_top3)の配当分布 + 事前識別精度検証
（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]の続き）。

前段(`exp_both_top3_race_identification.py`)で「◎◯両方3着内」は事前にmark_sum
（◎◯複勝確率合算）+同一ラインの組み合わせでかなりの精度(precision~65%)で識別
できることを確認した。本スクリプトはその鏡像: **「両方は3着内に来ない
(not_both_top3、◎◯の少なくとも一方が3着圏外)」ケース**の配当分布と、
同じ特徴（mark_sum低・別ライン）でこちらを検出できるかを検証する。

not_both_top3はS7の本来のターゲット層（軸が◎◯そのものでなくとも、三連複が
高配当になりやすい母集団）に相当するため、「除外ルールの精度」だけでなく
「積極的に狙う母集団の輪郭」としても重要。

対象: n_entries=7の全レース、pred_top3_pct/pred_win_pct格納済み(2024-01-01〜)。
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
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float("inf"))]


def load_data():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, r.grade, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.pred_win_pct, e.line_group, e.finish_order "
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
    print("データ読み込み中(2024-01-01〜・pred_top3_pct格納済みのみ)...")
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
        grade = ents[0]["grade"]
        honmei = next((e for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((e for e in ents if e["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        both_top3 = (honmei["frame_no"] in winners) and (taikou["frame_no"] in winners)
        not_both = int(not both_top3)
        trio = trio_odds.get(rk)
        odds = trio.get(winners) if trio else None
        mark_sum = float(honmei["pred_top3_pct"]) + float(taikou["pred_top3_pct"])
        win_sum = None
        if honmei["pred_win_pct"] is not None and taikou["pred_win_pct"] is not None:
            win_sum = float(honmei["pred_win_pct"]) + float(taikou["pred_win_pct"])
        same_line = bool(honmei["line_group"] is not None
                          and honmei["line_group"] == taikou["line_group"])
        races.append({
            "race_key": rk, "race_date": race_date, "grade": grade,
            "mark_sum": mark_sum, "win_sum": win_sum, "same_line": same_line,
            "not_both": not_both, "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    # ===== 1. not_both_top3レースの配当分布 =====
    print("\n" + "=" * 70)
    print("1. not_both_top3（◎◯少なくとも一方が3着圏外）の三連複配当分布")
    print("=" * 70)
    for label, data in (("全期間", races), ("TRAIN", train), ("TEST", test)):
        sub = [r for r in data if r["not_both"] and r["trio_odds"] is not None]
        vals = sorted(r["trio_odds"] for r in sub)
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        print(f"\n[{label}] n={n} ({n/len([r for r in data if r['trio_odds'] is not None])*100:.1f}% of全体)")
        print(f"  平均={mean:.2f}倍  中央値={pctile(vals,50):.2f}倍  "
              f"p25={pctile(vals,25):.2f}倍  p75={pctile(vals,75):.2f}倍  "
              f"p90={pctile(vals,90):.2f}倍  p95={pctile(vals,95):.2f}倍  p99={pctile(vals,99):.2f}倍")
        print(f"  区間分布:")
        for lo, hi in BUCKETS:
            c = sum(1 for v in vals if lo <= v < hi)
            lr = f"{lo}-{hi}倍" if hi != float("inf") else f"{lo}倍+"
            print(f"    {lr:<12}{c:>8}件 ({c/n*100:>5.1f}%)")
        breakeven = sum(1 for v in vals if v >= 5)
        print(f"  5倍以上(5点流し500円の分岐点クリア)率: {breakeven/n*100:.1f}%")

    # ===== 2. mark_sum（低い方）+ 別ラインでnot_bothを検出できるか =====
    print("\n" + "=" * 70)
    print("2. mark_sum低値・別ラインでnot_both_top3を検出する識別精度")
    print("=" * 70)

    def summarize_id(data, mark_th, require_diff_line):
        flagged = [r for r in data if r["mark_sum"] < mark_th
                   and (not r["same_line"] if require_diff_line else True)]
        n = len(flagged)
        total_not_both = sum(r["not_both"] for r in data)
        if n == 0 or total_not_both == 0:
            return 0, 0.0, 0.0, 0.0
        prec = sum(r["not_both"] for r in flagged) / n * 100
        recall = sum(r["not_both"] for r in flagged) / total_not_both * 100
        odds = [r["trio_odds"] for r in flagged if r["trio_odds"] is not None]
        mean_odds = sum(odds) / len(odds) if odds else 0.0
        return n, prec, recall, mean_odds

    print(f"\n{'条件':<28}{'標本':>10}{'n':>8}{'precision':>12}{'recall':>10}{'平均配当':>10}")
    for label, data in (("TRAIN", train), ("TEST", test)):
        for th in [110, 120, 130, 140, 150, 160]:
            n, prec, recall, mo = summarize_id(data, th, False)
            print(f"mark_sum<{th:<20}{label:>10}{n:>8}{prec:>11.1f}%{recall:>9.1f}%{mo:>9.1f}倍")
        for th in [130, 150]:
            n, prec, recall, mo = summarize_id(data, th, True)
            print(f"mark_sum<{th}+別ライン{'':<9}{label:>10}{n:>8}{prec:>11.1f}%{recall:>9.1f}%{mo:>9.1f}倍")

    # ===== 3. not_both内で「高配当(30倍以上)」がどこに集中するか =====
    print("\n" + "=" * 70)
    print("3. not_both_top3の中でも高配当(>=30倍)がmark_sumのどの帯に集中するか")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        nb = [r for r in data if r["not_both"] and r["trio_odds"] is not None]
        edges = [(0, 60), (60, 80), (80, 100), (100, 110), (110, 120), (120, 130),
                 (130, 140), (140, 150), (150, 160), (160, 170), (170, 200)]
        print(f"{'mark_sum帯':<14}{'n':>8}{'中央値':>10}{'30倍+率':>10}")
        for lo, hi in edges:
            sub = [r for r in nb if lo <= r["mark_sum"] < hi]
            n = len(sub)
            if n == 0:
                continue
            odds = sorted(r["trio_odds"] for r in sub)
            med = pctile(odds, 50)
            over30 = sum(1 for v in odds if v >= 30) / n * 100
            lr = f"{lo}-{hi}" if hi != 200 else f"{lo}+"
            print(f"{lr:<14}{n:>8}{med:>9.1f}倍{over30:>9.1f}%")


if __name__ == "__main__":
    main()
