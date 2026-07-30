"""「neither」(◎◯が両方3着圏外＝真の波乱)の事前識別精度検証（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]の続き。カテゴリ分解の結果:
  - both(50.1%): 中央値4.2倍・低配当
  - exactly_one(42.2%・S7の真のターゲットであるボリュームゾーン): 中央値16.7倍
  - neither(7.7%・稀だが高配当): 中央値37.5倍・30倍+率58.2%

ユーザー方針: 条件次第でneitherはS7とは別ランクで高配当の的中を狙う候補。
本スクリプトはentropy(低=市場の見方集中)×mark_sum(高=◎◯を強く支持)の組み合わせで
neitherを事前に(発走前情報のみで)識別できるか、precision/recallで検証する。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"


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


def field_entropy(pcts):
    total = sum(pcts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in pcts:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


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
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = honmei["frame_no"] in winners
        t_hit = taikou["frame_no"] in winners
        neither = int(not h_hit and not t_hit)
        trio = trio_odds.get(rk)
        odds = trio.get(winners) if trio else None

        mark_sum = float(honmei["pred_top3_pct"]) + float(taikou["pred_top3_pct"])
        pcts = [float(e["pred_top3_pct"]) for e in ents]
        ent = field_entropy(pcts)

        races.append({
            "race_key": rk, "race_date": race_date, "mark_sum": mark_sum,
            "entropy": ent, "neither": neither, "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def summarize_id(data, ent_th, mark_th):
        flagged = [r for r in data if r["entropy"] <= ent_th and r["mark_sum"] >= mark_th]
        n = len(flagged)
        total_neither = sum(r["neither"] for r in data)
        if n == 0 or total_neither == 0:
            return 0, 0.0, 0.0, 0.0
        prec = sum(r["neither"] for r in flagged) / n * 100
        recall = sum(r["neither"] for r in flagged) / total_neither * 100
        odds = [r["trio_odds"] for r in flagged if r["trio_odds"] is not None]
        mean_odds = sum(odds) / len(odds) if odds else 0.0
        return n, prec, recall, mean_odds

    print("\n" + "=" * 70)
    print("neither(◎◯両方圏外)の識別精度: entropy<=X & mark_sum>=Y")
    print("=" * 70)
    base_rate_train = sum(r["neither"] for r in train) / len(train) * 100
    base_rate_test = sum(r["neither"] for r in test) / len(test) * 100
    print(f"ベースレート(無条件のneither率): TRAIN={base_rate_train:.1f}% TEST={base_rate_test:.1f}%")

    print(f"\n{'条件':<28}{'標本':>8}{'n':>8}{'precision':>12}{'recall':>10}{'平均配当':>10}")
    grid = [(1.9, 100), (1.9, 120), (1.8, 120), (1.8, 130), (1.7, 130), (1.7, 140),
            (1.6, 130), (1.6, 140), (1.5, 130)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        for ent_th, mark_th in grid:
            n, prec, recall, mo = summarize_id(data, ent_th, mark_th)
            print(f"ent<={ent_th},mark>={mark_th:<12}{label:>8}{n:>8}{prec:>11.1f}%{recall:>9.1f}%{mo:>9.1f}倍")

    # TRAINで最良のprecision改善(ベースレート比のリフト)条件を選び、TESTで一度だけ確認
    print("\n--- リフト(precision/ベースレート)最大の条件をTRAINで選定 → TEST検証 ---")
    best = None
    for ent_th, mark_th in grid:
        n, prec, recall, mo = summarize_id(train, ent_th, mark_th)
        if n < 50:
            continue
        lift = prec / base_rate_train if base_rate_train else 0
        if best is None or lift > best[0]:
            best = (lift, ent_th, mark_th, n, prec, recall)
    if best:
        lift, ent_th, mark_th, n, prec, recall = best
        print(f"[選定] entropy<={ent_th} & mark_sum>={mark_th}: "
              f"TRAIN n={n} precision={prec:.1f}% (リフト{lift:.1f}倍) recall={recall:.1f}%")
        n2, prec2, recall2, mo2 = summarize_id(test, ent_th, mark_th)
        lift2 = prec2 / base_rate_test if base_rate_test else 0
        print(f"[TEST検証] n={n2} precision={prec2:.1f}% (リフト{lift2:.1f}倍) "
              f"recall={recall2:.1f}% 平均配当={mo2:.1f}倍")


if __name__ == "__main__":
    main()
