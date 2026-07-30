"""ライン連携「初共演」シグナルの全データ検証（2026-07-29・ユーザー指示で範囲拡大）。

`exp_linecoop_blind_verify.py`はS7本番pick(527件)のみが対象でTEST n=3という
極小サンプルになり判断不能だった。本スクリプトはS7の絞り込み(モデル予測・
axis選定・7車限定)を一切かけず、**全期間(2022-12-01〜)・全レース・全同ラインペア**
を対象に「初共演(過去に同ラインを組んだことがない)」シグナルが実際に効くかを
まず素データで検証する（モデル不要・純粋に実績の集計）。

定義:
  - 同ラインペア: あるレースで line_group が同じ2名（3并走ラインなら3ペア生成）
  - lc_n_prev: そのレース"時点"より前に同ラインを組んだ回数（point-in-time）
  - first_pairing: lc_n_prev == 0（初共演）
  - both_top3: そのペアの2名が両者ともそのレースで3着以内に入ったか
    （S7の的中条件=軸2車が両者3着内、に直接対応する指標）

TRAIN(2024-01-01〜2025-12-31)で first_pairing vs 経験済み の both_top3 率を比較し、
TEST(2026-01-01〜)で同じ比較を一度だけ確認する（データ自体に対する再現性チェック。
まだ「賭けて儲かるか」ではなく「シグナルとして実在するか」の検証）。
"""
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

MIN_DATE = "2022-12-01"
TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"


def load_all_entries():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, r.start_at, e.frame_no, e.player_id, "
            "e.line_group, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.race_date >= :from_date ORDER BY r.race_date, r.start_at, e.race_key",
            {"from_date": MIN_DATE}).fetchall()
    return rows


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    print("全履歴読み込み中(2022-12-01〜)...")
    rows = load_all_entries()
    print(f"  entries行数: {len(rows)}")

    by_race = defaultdict(list)
    race_order = []
    seen = set()
    for r in rows:
        by_race[r["race_key"]].append(r)
        if r["race_key"] not in seen:
            seen.add(r["race_key"])
            race_order.append(r["race_key"])
    print(f"  レース数: {len(race_order)}")

    pair_n: dict = defaultdict(int)
    pair_hit: dict = defaultdict(int)
    pair_events = []  # (race_date, first_pairing, lc_n_prev, both_top3, n_lines_in_group)

    for rk in race_order:
        g = by_race[rk]
        race_date = str(g[0]["race_date"])
        pids = [e["player_id"] for e in g]
        lines = [e["line_group"] for e in g]
        fins = [e["finish_order"] for e in g]
        fin_map = dict(zip(pids, fins))

        # このレースの同ラインペアを列挙し、更新前の状態でイベントを記録
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if lines[i] is None or lines[i] != lines[j]:
                    continue
                a, b = pids[i], pids[j]
                key = (a, b) if a < b else (b, a)
                n_prev = pair_n[key]
                fi, fj = fin_map[a], fin_map[b]
                both_top3 = (fi is not None and fj is not None
                             and 1 <= fi <= 3 and 1 <= fj <= 3)
                pair_events.append({
                    "race_date": race_date, "first_pairing": n_prev == 0,
                    "lc_n_prev": n_prev, "both_top3": int(both_top3),
                })

        # 更新
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if lines[i] is None or lines[i] != lines[j]:
                    continue
                key = (pids[i], pids[j]) if pids[i] < pids[j] else (pids[j], pids[i])
                pair_n[key] += 1
                fi, fj = fin_map[pids[i]], fin_map[pids[j]]
                if fi is not None and fj is not None and 1 <= fi <= 3 and 1 <= fj <= 3:
                    pair_hit[key] += 1

    print(f"\n同ラインペアイベント総数: {len(pair_events)}")

    def summarize(evs):
        n = len(evs)
        hits = sum(e["both_top3"] for e in evs)
        rate = hits / n if n else 0.0
        lo, hi = wilson_ci(hits, n)
        return n, hits, rate, lo, hi

    train = [e for e in pair_events if TRAIN_FROM <= e["race_date"] <= TRAIN_TO]
    test = [e for e in pair_events if TEST_FROM <= e["race_date"] <= TEST_TO]
    print(f"TRAIN({TRAIN_FROM}〜{TRAIN_TO}): {len(train)}件 / TEST({TEST_FROM}〜): {len(test)}件")

    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n===== {label} =====")
        fp = [e for e in data if e["first_pairing"]]
        exp_ = [e for e in data if not e["first_pairing"]]
        n1, h1, r1, lo1, hi1 = summarize(fp)
        n2, h2, r2, lo2, hi2 = summarize(exp_)
        print(f"初共演(lc_n_prev=0)     n={n1:>7} both_top3率={r1:.1%} "
              f"(95%CI {lo1:.1%}-{hi1:.1%})")
        print(f"経験済み(lc_n_prev>=1)  n={n2:>7} both_top3率={r2:.1%} "
              f"(95%CI {lo2:.1%}-{hi2:.1%})")
        print(f"差分(初共演-経験済み): {r1 - r2:+.1%}")

        # lc_n_prev を細分化して単調性も確認
        print("\n  lc_n_prev別内訳:")
        by_np = defaultdict(list)
        for e in data:
            b = e["lc_n_prev"] if e["lc_n_prev"] < 5 else "5+"
            by_np[b].append(e)
        for k in sorted(by_np.keys(), key=lambda x: (isinstance(x, str), x)):
            n, h, rate, lo, hi = summarize(by_np[k])
            print(f"    lc_n_prev={k}: n={n:>7} both_top3率={rate:.1%} (95%CI {lo:.1%}-{hi:.1%})")


if __name__ == "__main__":
    main()
