"""【着順の相関構造の定量化】周辺確率の積では説明できないペア相関を測る
（2026-07-30・ユーザー指摘: 「単純な1車目選定時の単勝率・複勝率ではなく相関関係がある」）。

## 問題意識

現行モデルは各選手の**周辺確率**(pred_win_pct/pred_top3_pct)を独立に予測している。
しかし競輪の着順はライン展開を通じて相関するはず:
  - ラインの先頭が逃げれば同ラインが引かれて共に上位に来る（正の相関）
  - 同じ位置を争う別ライン同士は共倒れしうる（負の相関）
つまり **P(A∧B が3着内) ≠ P(A) × P(B)**。

既に兆候はある: 同ライン軸の的中率50.2% vs 別ライン軸34.9%（15pt差）。
ただしこれには「同ラインの方が強い選手同士だった」という周辺確率の効果も
混ざっているため、**純粋な相関効果を分離**する必要がある。

## 測定方法（lift = 観測同時確率 / 独立仮定の期待値）

全レースの全ペア(i,j) 21通り/レース について:
  expected_independent = p_top3(i) * p_top3(j)     ← 独立仮定
  observed = 実際に両者が3着内だったか
  **lift = 観測率 / 平均expected**
lift > 1 なら正の相関（独立仮定より一緒に来やすい）、< 1 なら負の相関。

これを以下で層別する:
  - 同ライン / 別ライン
  - ライン内の位置関係（先頭+番手 / 先頭+3番手 / 番手+3番手）
  - ライン規模
  - 周辺確率の水準（強い同士 / 強×弱 / 弱同士）← 交絡の確認

## 3列目の条件付き確率も測る
軸2車が3着内だったという条件下で、残り5頭のうちどれが3着に入ったか。
その条件付き確率が周辺確率から予測されるものとどれだけズレるか
（= 3列目選定にも相関構造を使えるか）。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: str(r["race_date"]) for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, "
                 "       line_group, line_pos, line_size, is_line_leader, style, "
                 "       finish_order FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load]   entries races: {len(by_race)}", flush=True)
    return races, by_race


def build(races, entries_by_race):
    out = []
    for rk, rdate in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None or e["pred_win_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        order = [fno for _, fno in fin[:3]]
        by_frame = {int(e["frame_no"]): e for e in ents}
        out.append({"race_key": rk, "race_date": rdate, "by_frame": by_frame,
                    "order": order, "top3": frozenset(order)})
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def pair_bucket_same_line(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None:
        return "ライン不明"
    return "同ライン" if li == lj else "別ライン"


def pair_bucket_linepos(bf, i, j):
    """同ライン内の位置関係。別ラインは None を返す。"""
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None or li != lj:
        return None
    pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
    if pi is None or pj is None:
        return None
    a, b = sorted([int(pi), int(pj)])
    return f"同ライン{a}-{b}番手"


def strength_bucket(pi, pj):
    """周辺確率の水準（交絡確認用）。pi,pj は 0-1 スケール。"""
    hi = max(pi, pj)
    lo = min(pi, pj)
    def lvl(x):
        if x >= 0.75:
            return "強"
        if x >= 0.45:
            return "中"
        return "弱"
    return f"{lvl(hi)}×{lvl(lo)}"


def report_lift(rows, label, keyfn, min_n=200):
    """keyfn(bf,i,j) -> bucket名 or None。バケット別に lift を出す。"""
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        frames = list(bf.keys())
        for i, j in combinations(frames, 2):
            b = keyfn(bf, i, j)
            if b is None:
                continue
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if i in r["top3"] and j in r["top3"]:
                a["obs"] += 1
    print(f"\n  [{label}]")
    print(f"    {'バケット':<20}{'ペア数':>10}{'観測同時率':>12}{'独立期待':>11}{'lift':>9}")
    for b in sorted(agg.keys(), key=lambda x: -agg[x]["n"]):
        a = agg[b]
        if a["n"] < min_n:
            continue
        obs = a["obs"] / a["n"]
        exp = a["exp"] / a["n"]
        lift = obs / exp if exp > 0 else 0
        print(f"    {b:<20}{a['n']:>10}{obs*100:>11.1f}%{exp*100:>10.1f}%{lift:>8.3f}x")


def report_third_conditional(rows, label):
    """軸=複勝上位2頭が両方3着内だったレースで、3着車の条件付き分布を
    周辺確率ベースの期待と比較する。"""
    agg_same = {"n": 0, "obs": 0, "exp": 0.0}
    agg_diff = {"n": 0, "obs": 0, "exp": 0.0}
    for r in rows:
        bf = r["by_frame"]
        frames = list(bf.keys())
        tsorted = sorted(frames, key=lambda f: -float(bf[f]["pred_top3_pct"]))
        a1, a2 = tsorted[0], tsorted[1]
        if not ({a1, a2} <= r["top3"]):
            continue
        third = next(iter(r["top3"] - {a1, a2}), None)
        if third is None:
            continue
        others = [f for f in frames if f not in (a1, a2)]
        # 残り5頭の周辺確率を正規化して「3着になる期待確率」とする
        tot = sum(float(bf[f]["pred_top3_pct"]) for f in others)
        if tot <= 0:
            continue
        axis_lines = {bf[a1]["line_group"], bf[a2]["line_group"]} - {None}
        for f in others:
            p_exp = float(bf[f]["pred_top3_pct"]) / tot
            same = bf[f]["line_group"] in axis_lines if bf[f]["line_group"] is not None else False
            tgt = agg_same if same else agg_diff
            tgt["n"] += 1
            tgt["exp"] += p_exp
            if f == third:
                tgt["obs"] += 1
    print(f"\n  [{label}] 3列目の条件付き分布（軸2車が3着内のレースのみ）")
    print(f"    {'3列目候補':<20}{'候補数':>9}{'実際に3着':>11}{'周辺期待':>10}{'lift':>9}")
    for name, a in (("軸と同ライン", agg_same), ("軸と別ライン", agg_diff)):
        if a["n"] == 0:
            continue
        obs = a["obs"] / a["n"]
        exp = a["exp"] / a["n"]
        lift = obs / exp if exp > 0 else 0
        print(f"    {name:<20}{a['n']:>9}{obs*100:>10.1f}%{exp*100:>9.1f}%{lift:>8.3f}x")


def main():
    races, entries_by_race = load_all()
    rows = build(races, entries_by_race)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    print("\n" + "=" * 90)
    print("1. ライン関係別のペア相関 (lift = 観測同時率 / 独立仮定の期待値)")
    print("   lift > 1 = 独立仮定より一緒に来やすい（正の相関）")
    print("=" * 90)
    for label, data in (("TRAIN", train), ("TEST", test)):
        report_lift(data, label, pair_bucket_same_line)

    print("\n" + "=" * 90)
    print("2. 同ライン内の位置関係別のペア相関")
    print("=" * 90)
    for label, data in (("TRAIN", train), ("TEST", test)):
        report_lift(data, label, pair_bucket_linepos)

    print("\n" + "=" * 90)
    print("3. 交絡確認: 周辺確率の水準 × ライン関係")
    print("   （同ラインのliftが高いのが『強い選手同士だから』ではないことの確認）")
    print("=" * 90)

    def keyfn_cross(bf, i, j):
        sl = pair_bucket_same_line(bf, i, j)
        if sl == "ライン不明":
            return None
        pi = float(bf[i]["pred_top3_pct"]) / 100.0
        pj = float(bf[j]["pred_top3_pct"]) / 100.0
        return f"{sl}/{strength_bucket(pi, pj)}"

    for label, data in (("TRAIN", train), ("TEST", test)):
        report_lift(data, label, keyfn_cross)

    print("\n" + "=" * 90)
    print("4. ライン規模別のペア相関（同ラインのみ）")
    print("=" * 90)

    def keyfn_size(bf, i, j):
        li, lj = bf[i]["line_group"], bf[j]["line_group"]
        if li is None or lj is None or li != lj:
            return None
        sz = bf[i]["line_size"]
        return f"同ライン(規模{int(sz)})" if sz else None

    for label, data in (("TRAIN", train), ("TEST", test)):
        report_lift(data, label, keyfn_size)

    print("\n" + "=" * 90)
    print("5. 3列目の条件付き相関")
    print("=" * 90)
    for label, data in (("TRAIN", train), ("TEST", test)):
        report_third_conditional(data, label)


if __name__ == "__main__":
    main()
