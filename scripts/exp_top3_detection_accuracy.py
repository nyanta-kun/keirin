"""【3着以内に来る選手の検出精度】ROIではなく検出精度そのものを測る
（2026-07-30・ユーザー指摘「ROIの前に、正しくそのレースで3着以内に来る選手を
検出できるかが先」）。

## 問題設定

高配当レースは的中確率が下がるのは当然。しかし**何らかの条件で「高配当かつ
検出できる」領域を絞れれば高ROIになる**。よってまず測るべきは検出精度であり、
ROIではない。

## 測定する検出精度（すべて配当帯で層別する）

1. **recall@k**: 実際の3着内3名のうち、予測上位k名に何名含まれるか（平均）
2. **exact3**: 予測上位3名が実際の3着内3名と完全一致した率
3. **2of3**: 予測上位3名のうち2名が的中した率
4. **recall@4/@5**: カバー範囲を広げた場合の捕捉率
5. **triple_rank**: 実際の3着内3名の組（35通りのうちの1つ）が、
   予測スコア順で何位に位置したか（低いほど良い検出）

配当帯: 0-5 / 5-10 / 10-20 / 20-30 / 30-50 / 50-100 / 100+ 倍

## 相関補正の効果検証（本命の仮説）

現行は各選手の周辺確率の積で組を評価している。ペア相関の測定
（`exp_pair_correlation_broad_scan.py`）で「別ライン lift 0.67 / 同ライン 1.12、
同一強度帯でも1.4-2.0倍の差」という巨大な構造が判明した。

そこで3頭の組(i,j,k)のスコアを
    naive     = p_i · p_j · p_k
    corrected = p_i · p_j · p_k · lift(i,j) · lift(i,k) · lift(j,k)
の2通りで計算し、**実際の3着内の組をどちらが上位にランクできるか**を比較する。
lift はTRAINのみで推定しTESTに固定適用（リーク防止）。

corrected が naive より明確に良ければ、「相関を入れれば検出精度が上がる」＝
現行モデルの構造的欠陥が実証され、改善の方向が確定する。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
"""
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
BANDS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float("inf"))]


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, v.bank_length "
            "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = 7 AND r.cancel = 0 AND r.race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: {"race_date": str(r["race_date"]), "grade": r["grade"],
                              "bank_length": r["bank_length"]} for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, pred_win_pct, "
                 "       line_group, line_pos, line_size, n_lines, style, "
                 "       finish_order FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] trio odds ...", flush=True)
    win_odds = {}
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            boards = defaultdict(dict)
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
                    boards[rk][parts] = fv
            for rk, b in boards.items():
                ents = by_race.get(rk)
                if not ents:
                    continue
                fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
                       if e["finish_order"] is not None and e["finish_order"] >= 1]
                if len(fin) < 3:
                    continue
                fin.sort()
                w = frozenset(fno for _, fno in fin[:3])
                if w in b:
                    win_odds[rk] = b[w]
            if (i // 900) % 20 == 0:
                print(f"[load]   trio progress: {i}/{len(keys)}", flush=True)
    print(f"[load]   win odds: {len(win_odds)}", flush=True)
    return races, by_race, win_odds


def build(races, entries_by_race, win_odds):
    out = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        if rk not in win_odds:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        by_frame = {int(e["frame_no"]): e for e in ents}
        out.append({
            "race_key": rk, "race_date": meta["race_date"], "meta": meta,
            "by_frame": by_frame,
            "top3": frozenset(fno for _, fno in fin[:3]),
            "payout": win_odds[rk],
        })
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def pair_bucket(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None:
        return "unknown"
    if li != lj:
        return "diff"
    pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
    if pi is None or pj is None:
        return "same_other"
    a, b = sorted([int(pi), int(pj)])
    if (a, b) == (1, 2):
        return "same_12"
    if (a, b) == (2, 3):
        return "same_23"
    if (a, b) == (1, 3):
        return "same_13"
    return "same_other"


def estimate_lifts(rows):
    """TRAINからペアバケット別liftを推定する。"""
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        for i, j in combinations(bf.keys(), 2):
            b = pair_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if i in r["top3"] and j in r["top3"]:
                a["obs"] += 1
    lifts = {}
    for b, a in agg.items():
        if a["n"] < 100 or a["exp"] <= 0:
            lifts[b] = 1.0
            continue
        lifts[b] = (a["obs"] / a["n"]) / (a["exp"] / a["n"])
    return lifts


def band_of(p):
    for lo, hi in BANDS:
        if lo <= p < hi:
            return f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
    return "?"


def detection_metrics(rows, lifts=None):
    """配当帯別に検出精度を集計する。lifts指定時は相関補正スコアも算出。"""
    agg = defaultdict(lambda: {"n": 0, "recall3": 0, "exact3": 0, "two_of3": 0,
                               "recall4": 0, "recall5": 0,
                               "rank_naive": 0, "rank_corr": 0, "rank_n": 0})
    for r in rows:
        bf = r["by_frame"]
        frames = list(bf.keys())
        tsorted = sorted(frames, key=lambda f: -float(bf[f]["pred_top3_pct"]))
        band = band_of(r["payout"])
        a = agg[band]
        a["n"] += 1
        hit3 = len(set(tsorted[:3]) & r["top3"])
        a["recall3"] += hit3
        a["exact3"] += 1 if hit3 == 3 else 0
        a["two_of3"] += 1 if hit3 >= 2 else 0
        a["recall4"] += len(set(tsorted[:4]) & r["top3"])
        a["recall5"] += len(set(tsorted[:5]) & r["top3"])

        if lifts is not None:
            p = {f: float(bf[f]["pred_top3_pct"]) / 100.0 for f in frames}
            scored_naive, scored_corr = [], []
            for tri in combinations(frames, 3):
                base = p[tri[0]] * p[tri[1]] * p[tri[2]]
                mult = 1.0
                for x, y in combinations(tri, 2):
                    mult *= lifts.get(pair_bucket(bf, x, y), 1.0)
                scored_naive.append((base, frozenset(tri)))
                scored_corr.append((base * mult, frozenset(tri)))
            scored_naive.sort(key=lambda t: -t[0])
            scored_corr.sort(key=lambda t: -t[0])
            rn = next((idx + 1 for idx, (_, s) in enumerate(scored_naive) if s == r["top3"]), None)
            rc = next((idx + 1 for idx, (_, s) in enumerate(scored_corr) if s == r["top3"]), None)
            if rn and rc:
                a["rank_naive"] += rn
                a["rank_corr"] += rc
                a["rank_n"] += 1
    return agg


def print_detection(agg, label, with_rank=False):
    print(f"\n  [{label}]")
    hdr = (f"    {'配当帯':<10}{'n':>7}{'recall@3':>10}{'exact3':>9}{'2of3':>8}"
           f"{'recall@4':>10}{'recall@5':>10}")
    if with_rank:
        hdr += f"{'組rank素':>10}{'組rank補正':>12}{'改善':>8}"
    print(hdr)
    order = [f"{lo}-{hi}" if hi != float("inf") else f"{lo}+" for lo, hi in BANDS]
    for band in order:
        a = agg.get(band)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        line = (f"    {band:<10}{n:>7}{a['recall3']/n:>9.2f}"
                f"{a['exact3']/n*100:>8.1f}%{a['two_of3']/n*100:>7.1f}%"
                f"{a['recall4']/n:>9.2f}{a['recall5']/n:>9.2f}")
        if with_rank and a["rank_n"]:
            rn = a["rank_naive"] / a["rank_n"]
            rc = a["rank_corr"] / a["rank_n"]
            imp = rn - rc
            line += f"{rn:>10.2f}{rc:>12.2f}{imp:>+8.2f}"
        print(line)
    # 全体
    tot = {k: sum(a[k] for a in agg.values()) for k in
           ("n", "recall3", "exact3", "two_of3", "recall4", "recall5",
            "rank_naive", "rank_corr", "rank_n")}
    if tot["n"]:
        line = (f"    {'全体':<10}{tot['n']:>7}{tot['recall3']/tot['n']:>9.2f}"
                f"{tot['exact3']/tot['n']*100:>8.1f}%{tot['two_of3']/tot['n']*100:>7.1f}%"
                f"{tot['recall4']/tot['n']:>9.2f}{tot['recall5']/tot['n']:>9.2f}")
        if with_rank and tot["rank_n"]:
            rn = tot["rank_naive"] / tot["rank_n"]
            rc = tot["rank_corr"] / tot["rank_n"]
            line += f"{rn:>10.2f}{rc:>12.2f}{rn-rc:>+8.2f}"
        print(line)


def main():
    races, entries_by_race, win_odds = load_all()
    rows = build(races, entries_by_race, win_odds)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    print("\n[lift] TRAINからペアバケット別liftを推定 ...", flush=True)
    lifts = estimate_lifts(train)
    for b in sorted(lifts, key=lambda x: -lifts[x]):
        print(f"    {b:<12} lift={lifts[b]:.4f}")

    print("\n" + "=" * 100)
    print("1. 配当帯別の3着内検出精度（recall@3 = 実際の3名のうち予測上位3名に何名入ったか）")
    print("   相関補正あり/なしで『実際の3着内の組』が35通り中何位にランクされたかも比較")
    print("=" * 100)
    for label, data in (("TRAIN", train), ("TEST", test)):
        agg = detection_metrics(data, lifts=lifts)
        print_detection(agg, label, with_rank=True)

    print("\n" + "=" * 100)
    print("2. 高配当レース(30倍以上)に絞った検出精度の条件別内訳")
    print("   『高配当かつ検出できる』条件を探す（=ユーザーが求める発生条件の絞り込み）")
    print("=" * 100)

    def strat_entropy(r):
        bf = r["by_frame"]
        import math
        vals = [float(bf[f]["pred_top3_pct"]) for f in bf]
        tot = sum(vals)
        ent = 0.0
        for v in vals:
            s = max(v / tot, 1e-9)
            ent -= s * math.log(s)
        if ent < 1.70:
            return "entropy低(<1.70)"
        if ent < 1.80:
            return "entropy中(1.70-1.80)"
        return "entropy高(>=1.80)"

    def strat_nlines(r):
        bf = r["by_frame"]
        nl = next(iter(bf.values()))["n_lines"]
        return f"{int(nl)}分戦" if nl else None

    def strat_maxline(r):
        bf = r["by_frame"]
        sizes = defaultdict(int)
        for f in bf:
            if bf[f]["line_group"] is not None:
                sizes[bf[f]["line_group"]] += 1
        m = max(sizes.values()) if sizes else 0
        return f"最大ライン{m}車"

    def strat_grade(r):
        return r["meta"]["grade"] or None

    STRATS = [("フィールドentropy", strat_entropy), ("分戦数", strat_nlines),
              ("最大ライン規模", strat_maxline), ("グレード", strat_grade)]

    for sname, sfn in STRATS:
        print(f"\n--- 層別: {sname}（30倍以上のレースのみ） ---")
        print(f"  {'バケット':<22}{'TRAIN n':>9}{'recall@3':>10}{'2of3':>8}"
              f"{'TEST n':>9}{'recall@3':>10}{'2of3':>8}")
        buckets = {}
        for label, data in (("TRAIN", train), ("TEST", test)):
            hi = [r for r in data if r["payout"] >= 30]
            agg = defaultdict(lambda: {"n": 0, "recall3": 0, "two_of3": 0})
            for r in hi:
                b = sfn(r)
                if b is None:
                    continue
                bf = r["by_frame"]
                tsorted = sorted(bf.keys(), key=lambda f: -float(bf[f]["pred_top3_pct"]))
                hit3 = len(set(tsorted[:3]) & r["top3"])
                a = agg[b]
                a["n"] += 1
                a["recall3"] += hit3
                a["two_of3"] += 1 if hit3 >= 2 else 0
            buckets[label] = agg
        allb = sorted(set(buckets["TRAIN"]) | set(buckets["TEST"]),
                      key=lambda b: -buckets["TRAIN"].get(b, {"n": 0})["n"])
        for b in allb:
            a1 = buckets["TRAIN"].get(b)
            a2 = buckets["TEST"].get(b)
            if not a1 or not a2 or a1["n"] < 50 or a2["n"] < 20:
                continue
            print(f"  {str(b):<22}{a1['n']:>9}{a1['recall3']/a1['n']:>9.2f}"
                  f"{a1['two_of3']/a1['n']*100:>7.1f}%"
                  f"{a2['n']:>9}{a2['recall3']/a2['n']:>9.2f}"
                  f"{a2['two_of3']/a2['n']*100:>7.1f}%")


if __name__ == "__main__":
    main()
