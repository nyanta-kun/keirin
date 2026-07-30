"""exactly_oneにおける軸1・軸2の同時的中率 + 「突出 vs 拮抗」による精度差の検証
（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

前段で軸1(◎◯のどちらを選ぶか・pred_win_pct/top3_pct高い方)は精度67-70%、
軸2(非マーク5車の最上位1車)は精度61-65%と、それぞれ単独ではランダムを
明確に上回ることを確認した。本スクリプトは2つの検証を行う:

1. 軸1・軸2が「同時に」正しい確率（トリオの3頭中2頭を正しく当てる確率）。
   単純な掛け算(0.67×0.65≈44%)は両者が独立という仮定に基づくため、
   実際の同時的中率と比較して相関の有無を確認する。

2. ユーザー懸念への回答: 「pred_win_pct/top3_pctで1-2車が突出しているレース」と
   「拮抗しているレース」で軸選定精度が異なるか。具体的には、軸1候補(◎◯)間の
   確率差(gap)・軸2候補(非マーク上位2車)間の確率差(gap)を計算し、gapの大小で
   選定精度が変化するかを確認する。gapが精度と強く相関するなら、pred_win_pct/
   top3_pctの「順位」だけでなく「差の大きさ」も情報として活用する価値がある
   （現行のaxis_sum/entropyゲートと同様の追加情報になりうる）。gapが精度と
   ほぼ無相関なら、順位に従うだけで既に情報を使い切っていることになる。
"""
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"


def load_data():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.pred_win_pct, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.n_entries = 7 AND e.pred_top3_pct IS NOT NULL "
            "AND r.race_date >= :from_date",
            {"from_date": TRAIN_FROM}).fetchall()
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
    print("データ読み込み中(2024-01-01〜)...")
    rows = load_data()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)

    races = []
    for rk, ents in by_race.items():
        if len(ents) != 7:
            continue
        race_date = str(ents[0]["race_date"])
        honmei = next((e for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((e for e in ents if e["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        others = [e for e in ents if e["frame_no"] not in (honmei["frame_no"], taikou["frame_no"])]
        if len(others) != 5:
            continue
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
            continue
        if any(e["pred_top3_pct"] is None for e in others):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = honmei["frame_no"] in winners
        t_hit = taikou["frame_no"] in winners
        if h_hit == t_hit:
            continue  # exactly_oneのみ

        h_top3 = float(honmei["pred_top3_pct"])
        t_top3 = float(taikou["pred_top3_pct"])
        axis1_correct_top3 = (h_top3 > t_top3) == h_hit
        gap1 = abs(h_top3 - t_top3)

        other_hits = frozenset(e["frame_no"] for e in others if e["frame_no"] in winners)
        top3_ranked = sorted(others, key=lambda e: -float(e["pred_top3_pct"]))
        axis2_pick = top3_ranked[0]["frame_no"]
        axis2_correct = axis2_pick in other_hits
        gap2 = float(top3_ranked[0]["pred_top3_pct"]) - float(top3_ranked[1]["pred_top3_pct"])

        races.append({
            "race_key": rk, "race_date": race_date,
            "axis1_correct": axis1_correct_top3, "gap1": gap1,
            "axis2_correct": axis2_correct, "gap2": gap2,
            "both_correct": axis1_correct_top3 and axis2_correct,
        })

    print(f"  exactly_one対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    # ===== 1. 同時的中率 =====
    print("\n" + "=" * 70)
    print("1. 軸1・軸2の同時的中率（トリオ3頭中2頭を正しく当てる確率）")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n = len(data)
        a1 = sum(r["axis1_correct"] for r in data) / n * 100
        a2 = sum(r["axis2_correct"] for r in data) / n * 100
        both = sum(r["both_correct"] for r in data) / n * 100
        indep_expect = a1 / 100 * a2 / 100 * 100
        lo, hi = wilson_ci(sum(r["both_correct"] for r in data), n)
        print(f"  [{label}] n={n} 軸1単独={a1:.1f}% 軸2単独={a2:.1f}% "
              f"同時的中(実測)={both:.1f}% (95%CI {lo*100:.1f}%-{hi*100:.1f}%) "
              f"独立仮定の期待値={indep_expect:.1f}%")

    # ===== 2. gap(突出度)と軸1精度の関係 =====
    print("\n" + "=" * 70)
    print("2. 軸1: gap1(◎◯の確率差)と選定精度の関係")
    print("=" * 70)
    gap1_edges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n  [{label}]")
        for lo, hi in gap1_edges:
            sub = [r for r in data if lo <= r["gap1"] < hi]
            n = len(sub)
            if n == 0:
                continue
            acc = sum(r["axis1_correct"] for r in sub) / n * 100
            print(f"    gap1[{lo}-{hi}): n={n:>6} 精度={acc:>5.1f}%")

    # ===== 3. gap(突出度)と軸2精度の関係 =====
    print("\n" + "=" * 70)
    print("3. 軸2: gap2(非マーク上位1位と2位の確率差)と選定精度の関係")
    print("=" * 70)
    gap2_edges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n  [{label}]")
        for lo, hi in gap2_edges:
            sub = [r for r in data if lo <= r["gap2"] < hi]
            n = len(sub)
            if n == 0:
                continue
            acc = sum(r["axis2_correct"] for r in sub) / n * 100
            print(f"    gap2[{lo}-{hi}): n={n:>6} 精度={acc:>5.1f}%")

    # ===== 4. gap1×gap2 交差での同時的中率 =====
    print("\n" + "=" * 70)
    print("4. gap1(高)×gap2(高)＝両軸とも突出しているレースでの同時的中率")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        for g1_th, g2_th in [(0, 0), (5, 5), (10, 10), (20, 10)]:
            sub = [r for r in data if r["gap1"] >= g1_th and r["gap2"] >= g2_th]
            n = len(sub)
            if n == 0:
                continue
            both = sum(r["both_correct"] for r in sub) / n * 100
            print(f"  [{label}] gap1>={g1_th} & gap2>={g2_th}: n={n:>6} 同時的中={both:.1f}%")


if __name__ == "__main__":
    main()
