"""【読み取り専用】overlap==2 の中で「配当が残る × 的中率が高い × 相手が絞れる」
3条件を同時に満たす帯を2次元スイープで探す（2026-08-03）。

exp_7s7a_overlap2_conditional_value.py の続き。前段で判明したこと:
  - overlap2 は的中率が現行7S/7Aより +19pt 高いが、ROI はどの切り口でも 68〜76% で
    ほぼ平坦（控除率の壁）。「隠れた高ROI帯」は存在しない。
  - 集中度（低entropy / 高axis_sum / 高share3）を上げると的中率は上がるが
    配当中央値が下がりガミ率が跳ね上がる。3条件は真正面からトレードオフする。
本スクリプトは「ガミ率を現行7S/7A(41.2%)以下に抑えたまま的中率を最大化する」
という制約付き最適化として運用点を選ぶ。

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
GAMI_BUDGET = 41.2   # 現行7S+7A(K=5)のガミ率。これを超えない範囲で的中率を最大化する


def load():
    months = {}
    for p in sorted(CACHE.glob("*.pkl")):
        with open(p, "rb") as f:
            months[p.stem] = pickle.load(f)
    return months


def legs(c, k):
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    return [x for x in ranked[:k] if c["trio_legs"].get(x) is not None]


def score(cands, k):
    bet = ret = hit = gami = 0
    pays = []
    for c in cands:
        lg = legs(c, k)
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
            "n20": sum(1 for p in pays if p >= 2000)}


def monthly_sd(cands, k):
    by_m = defaultdict(list)
    for c in cands:
        by_m[c["race_date"][:7]].append(c)
    rois = [score(v, k)["roi"] for v in by_m.values() if score(v, k)["bet"]]
    return (statistics.pstdev(rois) if len(rois) > 1 else 0.0,
            sum(1 for r in rois if r < 60), len(rois))


def main():
    months = load()
    allc = [c for cs in months.values() for c in cs]
    days = len({c["race_date"] for c in allc})
    for c in allc:
        ps = sorted((c["top3_probs"][x] for x in c["others"]), reverse=True)
        c["share3"] = sum(ps[:3]) / sum(ps) if sum(ps) else 0.0

    o2 = [c for c in allc if c["wt_overlap_n"] == 2]
    cur = [c for c in allc if c["wt_overlap_n"] in (0, 1)
           and (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX) + (c["entropy"] > RANK_7S_ENTROPY_MAX) <= 1]

    print(f"期間 {min(months)}〜{max(months)}  {days}日  overlap2 {len(o2)}件\n")
    cs = score(cur, 5)
    print(f"【基準】現行7S+7A K=5: {cs['n']/days:.2f}件/日 的中{cs['hit_rate']:.1f}% "
          f"ROI{cs['roi']:.1f}% ガミ率{cs['gami']:.1f}% 的中中央値{cs['med']:.1f}倍\n")

    # ── 2次元スイープ: entropy上限 × share3下限（K=3固定）──────────────
    ent_cuts = [1.75, 1.78, 1.80, 1.8329, 1.87, 9.0]
    sh = sorted(c["share3"] for c in o2)
    sh_cuts = [0.0] + [sh[int(len(sh) * f)] for f in (0.3, 0.5, 0.7, 0.85)]

    for k in (3, 4):
        print("=" * 112)
        print(f"overlap2 スイープ: entropy<=行 × share3>=列   [K={k}]  "
              f"セル= 件/日 / 的中% / ROI% / ガミ%")
        print("=" * 112)
        hdr = "  ent<=  |" + "".join(f"  share3>={c:.3f}      " for c in sh_cuts)
        print(hdr)
        for e in ent_cuts:
            line = f"  {e:5.4f} |"
            for s in sh_cuts:
                sub = [c for c in o2 if c["entropy"] <= e and c["share3"] >= s]
                r = score(sub, k)
                line += f" {r['n']/days:5.2f}/{r['hit_rate']:4.1f}/{r['roi']:4.1f}/{r['gami']:4.1f}"
            print(line)
        print()

    # ── ガミ率制約つきの運用点候補 ─────────────────────────────
    print("=" * 112)
    print(f"運用点候補（ガミ率 <= {GAMI_BUDGET}% ＝現行と同等以下 の制約つき）")
    print("=" * 112)
    print(f"{'条件':46} {'K':>2} {'件/日':>6} {'的中率':>7} {'ROI':>7} "
          f"{'ガミ':>6} {'中央値':>7} {'20倍+':>6} {'月σ':>6} {'<60%':>5}")
    rows = []
    for k in (2, 3, 4, 5):
        for e in ent_cuts:
            for s in sh_cuts:
                sub = [c for c in o2 if c["entropy"] <= e and c["share3"] >= s]
                r = score(sub, k)
                if r["n"] / days < 2.0 or r["gami"] > GAMI_BUDGET:
                    continue
                rows.append((r["hit_rate"], k, e, s, sub, r))
    rows.sort(reverse=True, key=lambda x: x[0])
    seen = set()
    for hr, k, e, s, sub, r in rows[:14]:
        key = (round(r["n"] / days, 1), k)
        if key in seen:
            continue
        seen.add(key)
        sd, low, nm = monthly_sd(sub, k)
        print(f"{'ent<=%.4f ∧ share3>=%.3f' % (e, s):46} {k:2d} {r['n']/days:6.2f} "
              f"{r['hit_rate']:6.1f}% {r['roi']:6.1f}% {r['gami']:5.1f}% {r['med']:6.1f}倍 "
              f"{r['n20']:6d} {sd:6.1f} {low:5d}")

    # ── 現行7S/7A に上乗せしたときの合算像 ────────────────────────
    print()
    print("=" * 112)
    print("現行7S/7A(K=5) に overlap2枠を上乗せした合算（1日あたり・投資額つき）")
    print("=" * 112)
    print(f"{'上乗せ条件':46} {'K':>2} {'合計件/日':>9} {'合計的中率':>9} "
          f"{'合算ROI':>8} {'投資/日':>8}")
    base_bet, base_ret, base_hit, base_n = cs["bet"], cs["ret"], cs["hit"], cs["n"]
    for hr, k, e, s, sub, r in rows[:14]:
        key = (round(r["n"] / days, 1), k)
        n = base_n + r["n"]
        bet = base_bet + r["bet"]
        ret = base_ret + r["ret"]
        hit = base_hit + r["hit"]
        print(f"{'ent<=%.4f ∧ share3>=%.3f' % (e, s):46} {k:2d} {n/days:9.2f} "
              f"{100.0*hit/n:8.1f}% {100.0*ret/bet:7.1f}% {bet/days:7.0f}円")


if __name__ == "__main__":
    main()
