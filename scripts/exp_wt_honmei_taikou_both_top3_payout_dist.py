"""◎◯が両方3着内に来た場合の三連複配当分布（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]。現行S7は`wt_overlap_n==2`
（軸2車が◎◯と完全一致）を一律除外している。この除外が妥当か判断するため、
「◎(honmei)・◯(taikou)が両方3着内に入ったレース」に限定して、実際の三連複配当
（勝ち組み合わせのオッズ）の分布を見る。分布の大半が低配当（購入対象として
価値が薄い）なら現行の一律除外は妥当。逆に高配当の裾（外れ値的な荒れ）が
無視できない割合であれば、「一律除外」ではなく条件付き除外の余地がある。

対象: n_entries=7 の全レース（S7の他ゲートは適用しない）。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
PCTS = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float("inf"))]


def load_races_7():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE n_entries = 7").fetchall()
    return {r["race_key"]: str(r["race_date"]) for r in rows}


def load_entries(race_keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pm in c.execute(q, chunk):
                out[rk][int(fno)] = {"finish_order": fo, "prediction_mark": pm}
    return out


def load_trio_win_odds(race_keys):
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
    print("データ読み込み中...")
    races = load_races_7()
    race_keys = list(races.keys())
    entries = load_entries(race_keys)
    trio_odds = load_trio_win_odds(race_keys)

    rows = []
    for rk, race_date in races.items():
        ent = entries.get(rk)
        if not ent:
            continue
        fin = [(fo["finish_order"], fno) for fno, fo in ent.items()
               if fo["finish_order"] is not None and fo["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        trio = trio_odds.get(rk)
        if not trio:
            continue
        odds = trio.get(winners)
        if odds is None:
            continue
        honmei = next((fno for fno, v in ent.items() if v["prediction_mark"] == 1), None)
        taikou = next((fno for fno, v in ent.items() if v["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        rows.append({
            "race_key": rk, "race_date": race_date, "trio_odds": odds,
            "both_top3": (honmei in winners) and (taikou in winners),
        })

    both = [r for r in rows if r["both_top3"]]
    not_both = [r for r in rows if not r["both_top3"]]
    print(f"\n対象レース総数: {len(rows)} / ◎◯両方3着内: {len(both)} "
          f"({len(both)/len(rows)*100:.1f}%) / それ以外: {len(not_both)}")

    def show_dist(label, data):
        vals = sorted(r["trio_odds"] for r in data)
        n = len(vals)
        print(f"\n=== {label} (n={n}) ===")
        print("  パーセンタイル: " + " / ".join(
            f"p{p}={pctile(vals, p):.1f}倍" for p in PCTS))
        mean = sum(vals) / n if n else 0.0
        print(f"  平均={mean:.1f}倍  中央値={pctile(vals,50):.1f}倍")
        print(f"  {'区間':<14}{'件数':>8}{'割合':>8}")
        for lo, hi in BUCKETS:
            c = sum(1 for v in vals if lo <= v < hi)
            label_r = f"{lo}-{hi}倍" if hi != float("inf") else f"{lo}倍+"
            print(f"  {label_r:<14}{c:>8}{c/n*100:>7.1f}%")
        # 500円(5点100円)の損益分岐点=5倍以上必要（1点的中でも払戻>=500円）
        breakeven = sum(1 for v in vals if v >= 5)
        print(f"  参考: 5点流し(500円)の単純収支分岐(payout>=5倍)を満たす割合: "
              f"{breakeven}/{n} ({breakeven/n*100:.1f}%)")

    show_dist("◎◯両方3着内", both)
    show_dist("それ以外(◎◯どちらか外れ)", not_both)

    print("\n--- TRAIN/TEST分割（◎◯両方3着内のみ） ---")
    for label, frm, to in (("TRAIN", TRAIN_FROM, TRAIN_TO), ("TEST", TEST_FROM, TEST_TO)):
        sub = [r for r in both if frm <= r["race_date"] <= to]
        vals = sorted(r["trio_odds"] for r in sub)
        n = len(vals)
        if n == 0:
            continue
        breakeven = sum(1 for v in vals if v >= 5)
        p50 = pctile(vals, 50)
        p90 = pctile(vals, 90)
        p95 = pctile(vals, 95)
        print(f"{label}({frm}〜{to}): n={n} 中央値={p50:.1f}倍 p90={p90:.1f}倍 "
              f"p95={p95:.1f}倍 5倍以上率={breakeven/n*100:.1f}%")


if __name__ == "__main__":
    main()
