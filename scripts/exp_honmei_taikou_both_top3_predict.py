"""◎◯「両方3着内」の事前予測可能性検証（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]。前段の分析で「◎◯両方3着内」は
中央値4.2倍・分岐(5倍)未達58%という低配当母集団であり、一律除外が妥当と判明した。
本スクリプトはその一歩先: **発走前に「◎◯が両方3着内に来そうか」を予測できるか**
を検証する。wt_entries.pred_top3_pct/pred_win_pct は2024-01-01以降、四半期
walk-forward vintageモデル（S7監査で使用しているものと同一系列）による発走前
予測値が本番格納済みのため、これをそのまま「事前情報」として使う（モデル再学習
不要）。

シグナル定義:
  mark_sum = pred_top3_pct(◎) + pred_top3_pct(◯)
  （◎◯個別の複勝的中確率の合計。値が高いほどモデルも◎◯を強く支持している）

TRAIN(2024-01-01〜2025-12-31)でmark_sumと実際の"both_top3"率の関係
（閾値ごとのboth_top3率・AUC相当のランク相関）を確認し、有望な閾値を選定→
TEST(2026-01-01〜)で一度だけ検証する。
"""
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


def main():
    print("データ読み込み中(2024-01-01〜・pred_top3_pct格納済みのみ)...")
    rows = load_data()
    print(f"  entries行数: {len(rows)}")

    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)

    race_keys = list(by_race.keys())
    trio_odds = load_trio_odds(race_keys)

    races = []
    for rk, ents in by_race.items():
        race_date = str(ents[0]["race_date"])
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
        both_top3 = int(honmei["frame_no"] in winners) and int(taikou["frame_no"] in winners)
        trio = trio_odds.get(rk)
        odds = None
        if trio:
            odds = trio.get(winners)
        mark_sum = float(honmei["pred_top3_pct"]) + float(taikou["pred_top3_pct"])
        races.append({
            "race_key": rk, "race_date": race_date, "mark_sum": mark_sum,
            "both_top3": both_top3, "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")

    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def bucket_report(data, label, edges):
        print(f"\n=== {label}: mark_sumで区間別both_top3率 ===")
        print(f"{'区間':<16}{'n':>8}{'both_top3率':>12}{'trio中央値':>12}{'5倍以上率':>10}")
        for lo, hi in edges:
            sub = [r for r in data if lo <= r["mark_sum"] < hi]
            n = len(sub)
            if n == 0:
                continue
            bt3 = sum(r["both_top3"] for r in sub) / n * 100
            odds_vals = sorted(r["trio_odds"] for r in sub if r["trio_odds"] is not None)
            med = odds_vals[len(odds_vals) // 2] if odds_vals else 0.0
            over5 = (sum(1 for v in odds_vals if v >= 5) / len(odds_vals) * 100
                     if odds_vals else 0.0)
            label_r = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
            print(f"{label_r:<16}{n:>8}{bt3:>11.1f}%{med:>11.1f}倍{over5:>9.1f}%")

    edges = [(0, 60), (60, 80), (80, 100), (100, 110), (110, 120), (120, 130),
             (130, 140), (140, 150), (150, 160), (160, 170), (170, 200)]
    bucket_report(train, "TRAIN", edges)
    bucket_report(test, "TEST", edges)

    # rank correlation（spearman近似: mark_sum順位とboth_top3の相関）
    def spearman(data):
        n = len(data)
        ranks = {r["race_key"]: i for i, r in enumerate(
            sorted(data, key=lambda r: r["mark_sum"]))}
        xs = [ranks[r["race_key"]] for r in data]
        ys = [r["both_top3"] for r in data]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        return cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else 0.0

    print(f"\nmark_sum順位 vs both_top3 の相関(TRAIN): {spearman(train):.3f}")
    print(f"mark_sum順位 vs both_top3 の相関(TEST):  {spearman(test):.3f}")

    # 閾値選定: TRAINでboth_top3率が高い(かつ低配当な)閾値を探し、TESTで確認
    for th in (110, 120, 130, 140, 150):
        tr_sub = [r for r in train if r["mark_sum"] >= th]
        te_sub = [r for r in test if r["mark_sum"] >= th]
        if not tr_sub or not te_sub:
            continue
        tr_bt3 = sum(r["both_top3"] for r in tr_sub) / len(tr_sub) * 100
        te_bt3 = sum(r["both_top3"] for r in te_sub) / len(te_sub) * 100
        print(f"\n閾値 mark_sum>={th}: TRAIN n={len(tr_sub)} both_top3率={tr_bt3:.1f}% / "
              f"TEST n={len(te_sub)} both_top3率={te_bt3:.1f}%")


if __name__ == "__main__":
    main()
