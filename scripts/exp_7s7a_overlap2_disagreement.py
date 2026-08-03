"""【読み取り専用】overlap==2 の中に「市場との部分的不一致」を作れるかの検証。

ユーザー方針「単純な市場との不一致を避けることに価値がある」を、軸だけでなく
**相手・順序**にも適用する:

  軸2車が ◎◯ と一致していても、
    D1: 順序不一致 — モデルの pred_win 最上位が WT ◎ ではない
    D2: 相手不一致 — 買い目の相手にWT △(ana) を含めない（△を意図的に切る）
    D3: D1 ∧ D2
  なら「完全なコンセンサス」ではなくなり、配当が残るのではないか。

前段（exp_7s7a_overlap2_sweep.py）で判明済み:
  overlap2 は ROI が 72〜76% で完全に平坦・的中中央値 2.3〜4.0倍・20倍超がほぼ出ない。
  「的中率が高い」と「配当が残る」は集中度という同一軸の両端で同時に立たない。
本スクリプトはその制約を相手側の不一致で破れるかを見る最後の試行。

DB書き込みなし。
"""
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_7S_STAKE

CACHE = Path(sys.argv[1])


def load():
    out = {}
    for p in sorted(CACHE.glob("*.pkl")):
        with open(p, "rb") as f:
            out[p.stem] = pickle.load(f)
    return out


def legs(c, k, drop_ana=False):
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    if drop_ana:
        ana = c["wt_marks"].get("ana")
        ranked = [x for x in ranked if x != ana]
    return [x for x in ranked[:k] if c["trio_legs"].get(x) is not None]


def score(cands, k, drop_ana=False):
    bet = ret = hit = gami = 0
    pays = []
    for c in cands:
        lg = legs(c, k, drop_ana)
        if not lg:
            continue
        stake = len(lg) * RANK_7S_STAKE
        bet += stake
        top3 = set(c["actual_top3"])
        rest = top3 - {c["axis1"], c["axis2"]}
        if len(top3 & {c["axis1"], c["axis2"]}) == 2 and len(rest) == 1 and rest.pop() in lg:
            hit += 1
            got = c["trio_pay"] * RANK_7S_STAKE // 100
            ret += got
            pays.append(c["trio_pay"])
            if got < stake:
                gami += 1
    n = len(cands)
    return {"n": n, "bet": bet, "ret": ret, "hit": hit,
            "hit_rate": 100.0 * hit / n if n else 0.0,
            "roi": 100.0 * ret / bet if bet else 0.0,
            "gami": 100.0 * gami / hit if hit else 0.0,
            "med": statistics.median(pays) / 100 if pays else 0.0,
            "n20": sum(1 for p in pays if p >= 2000),
            "max": max(pays) / 100 if pays else 0.0}


def line(label, r, days, k):
    return (f"{label:44} {k:2d} {r['n']/days:6.2f} {r['hit_rate']:6.1f}% "
            f"{r['roi']:6.1f}% {r['gami']:5.1f}% {r['med']:6.1f}倍 {r['n20']:6d} {r['max']:7.1f}倍")


HEAD = (f"{'条件':44} {'K':>2} {'件/日':>6} {'的中率':>7} {'ROI':>7} "
        f"{'ガミ':>6} {'中央値':>7} {'20倍+':>6} {'最高':>8}")


def main():
    months = load()
    allc = [c for cs in months.values() for c in cs]
    days = len({c["race_date"] for c in allc})
    o2 = [c for c in allc if c["wt_overlap_n"] == 2]
    cur = [c for c in allc if c["wt_overlap_n"] in (0, 1)
           and (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX) + (c["entropy"] > RANK_7S_ENTROPY_MAX) <= 1]

    # D1: モデルの pred_win 最上位が WT ◎ ではない
    for c in o2:
        top_win = max(c["win_probs"], key=lambda f: c["win_probs"][f])
        c["order_disagree"] = (top_win != c["wt_marks"].get("honmei"))

    print(f"期間 {min(months)}〜{max(months)}  {days}日  overlap2 {len(o2)}件")
    print(f"順序不一致(D1)の割合: {100*sum(c['order_disagree'] for c in o2)/len(o2):.1f}%\n")

    print("=" * 112)
    print("基準")
    print("=" * 112)
    print(HEAD)
    print(line("現行 7S+7A（overlap 0/1）", score(cur, 5), days, 5))
    print(line("overlap2 全件", score(o2, 5), days, 5))

    print()
    print("=" * 112)
    print("D1: 順序不一致（モデル1位 ≠ WT◎）で分割")
    print("=" * 112)
    print(HEAD)
    for k in (5, 3):
        for lbl, sub in (("D1成立（順序も不一致）", [c for c in o2 if c["order_disagree"]]),
                         ("D1不成立（順序も一致＝完全コンセンサス）",
                          [c for c in o2 if not c["order_disagree"]])):
            print(line(lbl, score(sub, k), days, k))
        print("-" * 112)

    print()
    print("=" * 112)
    print("D2: 相手からWT△(ana)を除外して買う")
    print("=" * 112)
    print(HEAD)
    for k in (3, 2):
        print(line(f"overlap2 相手上位{k}（△込み・従来）", score(o2, k), days, k))
        print(line(f"overlap2 相手上位{k}（△を除外）", score(o2, k, drop_ana=True), days, k))
        print("-" * 112)

    print()
    print("=" * 112)
    print("D3: D1 ∧ D2 の組み合わせ + entropy帯")
    print("=" * 112)
    print(HEAD)
    d1 = [c for c in o2 if c["order_disagree"]]
    for e in (1.8329, 1.87, 9.0):
        for k in (3, 2):
            sub = [c for c in d1 if c["entropy"] <= e]
            print(line(f"D1 ∧ ent<=%.4f ∧ △除外" % e, score(sub, k, drop_ana=True), days, k))

    # 月次安定性
    print()
    print("=" * 112)
    print("有望候補の月次安定性")
    print("=" * 112)
    print(f"{'条件':52} {'K':>2} {'件/日':>6} {'的中率':>7} {'ROI':>7} {'月σ':>6} {'<60%':>5} {'月数':>5}")
    cands = [
        ("D1 ∧ △除外", d1, True),
        ("D1 ∧ ent<=1.8329 ∧ △除外", [c for c in d1 if c["entropy"] <= 1.8329], True),
        ("overlap2 ∧ △除外", o2, True),
    ]
    for name, sub, da in cands:
        for k in (3, 2):
            by_m = defaultdict(list)
            for c in sub:
                by_m[c["race_date"][:7]].append(c)
            rois = [score(v, k, da)["roi"] for v in by_m.values() if score(v, k, da)["bet"]]
            r = score(sub, k, da)
            sd = statistics.pstdev(rois) if len(rois) > 1 else 0.0
            print(f"{name:52} {k:2d} {r['n']/days:6.2f} {r['hit_rate']:6.1f}% "
                  f"{r['roi']:6.1f}% {sd:6.1f} {sum(1 for x in rois if x < 60):5d} {len(rois):5d}")


if __name__ == "__main__":
    main()
