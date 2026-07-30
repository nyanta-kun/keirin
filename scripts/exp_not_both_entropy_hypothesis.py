"""not_both_top3内での「真の波乱 vs 拮抗レース」仮説検証（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]の続き。ユーザー仮説:
  A) レース全体が実力伯仲・市場人気が分散(entropy高)している場合、◎◯が3着内に
     来なくても、代わりに来た馬も"それなり"に支持されていた馬である可能性が高く、
     配当は低いままになりやすい。
  B) 逆に市場が◎◯を強く(絶対的に)支持していた(entropy低・mark_sumが高い)のに
     その2車が飛んだ場合は「真の波乱」であり、配当は高くなりやすい。

前段(`exp_both_top3_race_identification.py`)で計算したfield_entropy
（出走7車全体のpred_top3_pctから計算したエントロピー・値が低いほど予測確率が
一部の車に集中）をそのまま使い、not_both_top3母集団内でentropy帯別に配当を
比較する。あわせて「◎◯以外の5車の中で最も支持されている馬(others_max)」も
別角度の集中度指標として確認する（others_maxが低い＝◎◯以外は軒並み人気薄
=真の波乱の代理指標になりうる）。

TRAIN(2024-01-01〜2025-12-31)で傾向を確認し、TEST(2026-01-01〜)で再現するかを
honestに検証する。
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
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
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
        pcts = [float(e["pred_top3_pct"]) for e in ents]
        ent = field_entropy(pcts)
        others_pcts = [float(e["pred_top3_pct"]) for e in ents
                        if e["frame_no"] not in (honmei["frame_no"], taikou["frame_no"])]
        others_max = max(others_pcts) if others_pcts else 0.0
        others_sum = sum(others_pcts)

        races.append({
            "race_key": rk, "race_date": race_date, "mark_sum": mark_sum,
            "entropy": ent, "others_max": others_max, "others_sum": others_sum,
            "not_both": not_both, "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def payout_stats(data, key, edges, pool_filter=None):
        pool = [r for r in data if r["not_both"] and r["trio_odds"] is not None]
        if pool_filter:
            pool = [r for r in pool if pool_filter(r)]
        print(f"{'区間':<16}{'n':>8}{'中央値':>10}{'平均':>10}{'30倍+率':>10}{'50倍+率':>10}")
        for lo, hi in edges:
            sub = [r for r in pool if lo <= r[key] < hi]
            n = len(sub)
            if n == 0:
                continue
            odds = sorted(r["trio_odds"] for r in sub)
            med = pctile(odds, 50)
            mean = sum(odds) / n
            over30 = sum(1 for v in odds if v >= 30) / n * 100
            over50 = sum(1 for v in odds if v >= 50) / n * 100
            lr = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
            print(f"{lr:<16}{n:>8}{med:>9.1f}倍{mean:>9.1f}倍{over30:>9.1f}%{over50:>9.1f}%")

    # ===== 仮説A/B: field_entropy帯別のnot_both配当 =====
    print("\n" + "=" * 70)
    print("1. field_entropy帯別のnot_both_top3配当（低いほど市場の見方が集中＝真の波乱候補）")
    print("=" * 70)
    ent_edges = [(0, 1.4), (1.4, 1.6), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.5)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        payout_stats(data, "entropy", ent_edges)

    # ===== 仮説の直接的な形: entropy低 かつ mark_sum高 の交差 =====
    print("\n" + "=" * 70)
    print("2. entropy(低=集中) × mark_sum(高=◎◯強力支持) の交差検証（'真の波乱'候補の絞り込み）")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        pool = [r for r in data if r["not_both"] and r["trio_odds"] is not None]
        print(f"\n[{label}] 全体not_both: n={len(pool)}")
        for ent_th, mark_th in [(1.7, 130), (1.7, 150), (1.6, 130), (1.6, 150), (1.5, 130)]:
            sub = [r for r in pool if r["entropy"] <= ent_th and r["mark_sum"] >= mark_th]
            n = len(sub)
            if n == 0:
                continue
            odds = sorted(r["trio_odds"] for r in sub)
            med = pctile(odds, 50)
            mean = sum(odds) / n
            over30 = sum(1 for v in odds if v >= 30) / n * 100
            print(f"  entropy<={ent_th} & mark_sum>={mark_th}: n={n:>6} "
                  f"中央値={med:.1f}倍 平均={mean:.1f}倍 30倍+率={over30:.1f}%")

    # ===== others_max（◎◯以外の最強馬の複勝確率）帯別 =====
    print("\n" + "=" * 70)
    print("3. others_max（◎◯以外で最も支持された馬の複勝確率）帯別のnot_both配当")
    print("   低いほど「◎◯以外は軒並み人気薄」＝本当の伏兵が来た可能性")
    print("=" * 70)
    om_edges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        payout_stats(data, "others_max", om_edges)

    # ===== 全体との比較のためentropy⇔mark_sum の相関確認 =====
    print("\n" + "=" * 70)
    print("4. 参考: entropyとmark_sumの相関（同じ情報の言い換えでないか確認）")
    print("=" * 70)

    def pearson(data, k1, k2):
        n = len(data)
        xs = [r[k1] for r in data]
        ys = [r[k2] for r in data]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        return cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else 0.0

    print(f"corr(entropy, mark_sum) TRAIN: {pearson(train,'entropy','mark_sum'):.3f}")
    print(f"corr(entropy, mark_sum) TEST:  {pearson(test,'entropy','mark_sum'):.3f}")
    print(f"corr(entropy, others_max) TRAIN: {pearson(train,'entropy','others_max'):.3f}")


if __name__ == "__main__":
    main()
