"""【読み取り専用】overlap==2（軸2車がWT公式印◎◯と完全一致）を条件付きで
7S/7A の対象へ戻せるかの honest 検証（2026-08-03）。

ユーザー方針:
  「単純な市場との不一致を避けることに価値がある（＝7S/7Aの思想は正しい）。
    ただし ◎◯ が一致しても
      ① ある程度の配当が見込める
      ② 的中率が他より高い
      ③ 相手を絞ることができる
    ときは価値がある。」

本スクリプトはこの3条件を overlap==2 母集団の中で切り分けられるかを測る。

相手絞り: 三連複 軸2車 + 残り5車のうち、モデルの top3 予測確率（pred_prob）
上位K車のみを買う。K=5 が現行（総流し）。
  → K を絞ると 1レースあたり投資が K*100円 になるため、
    「的中しても賭け金を割る（ガミ）」の発生率が構造的に下がる。

配当見込み: 買い目5点の三連複オッズを使う。
  ⚠️ キャッシュの trio オッズは wt_odds＝**最終オッズ**であり、本番 judge が
  見る発走15分前オッズではない。オッズを使う判定はこの stale-odds バイアスを
  含むため、モデルのみで完結する指標（entropy / axis_sum / pred_prob）による
  絞り込みを主、オッズ条件は参考として扱う。

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
SINCE = sys.argv[2] if len(sys.argv) > 2 else "0000-00"


def load():
    months = {}
    for p in sorted(CACHE.glob("*.pkl")):
        if p.stem < SINCE:
            continue
        with open(p, "rb") as f:
            months[p.stem] = pickle.load(f)
    return months


def legs_by_model(c, k):
    """相手5車を pred_prob 降順に並べ、上位k車を買い目とする（オッズ非依存）。"""
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    return [x for x in ranked[:k] if c["trio_legs"].get(x) is not None]


def score(cands, k=5):
    bet = ret = hit = gami = 0
    payouts = []
    for c in cands:
        legs = legs_by_model(c, k)
        if not legs:
            continue
        stake = len(legs) * RANK_7S_STAKE
        bet += stake
        top3 = set(c["actual_top3"])
        rest = top3 - {c["axis1"], c["axis2"]}
        if len(top3 & {c["axis1"], c["axis2"]}) == 2 and len(rest) == 1 and rest.pop() in legs:
            hit += 1
            got = c["trio_pay"] * RANK_7S_STAKE // 100
            ret += got
            payouts.append(c["trio_pay"])
            if got < stake:
                gami += 1
    n = len(cands)
    return {
        "n": n, "bet": bet, "ret": ret, "hit": hit,
        "hit_rate": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "gami_rate": 100.0 * gami / hit if hit else 0.0,
        "med_pay": statistics.median(payouts) / 100 if payouts else 0.0,
        "n20x": sum(1 for p in payouts if p >= 2000),
        "max_pay": max(payouts) / 100 if payouts else 0.0,
    }


def row(label, s, days):
    return (f"{label:30} {s['n']:6d} {s['n']/days:6.2f} {s['hit_rate']:6.1f}% "
            f"{s['roi']:6.1f}% {s['gami_rate']:6.1f}% {s['med_pay']:7.1f}倍 "
            f"{s['n20x']:6d} {s['max_pay']:7.1f}倍")


HEAD = (f"{'条件':30} {'n':>6} {'件/日':>6} {'的中率':>7} {'ROI':>7} "
        f"{'ガミ率':>7} {'的中中央値':>8} {'20倍+':>6} {'最高':>8}")


def ok_axis(c):
    return c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX


def ok_ent(c):
    return c["entropy"] <= RANK_7S_ENTROPY_MAX


def main():
    months = load()
    allc = [c for cs in months.values() for c in cs]
    days = len({c["race_date"] for c in allc})
    print(f"期間 {min(months)}〜{max(months)}  {days}日  生候補 {len(allc)}件"
          f"（{len(allc)/days:.1f}件/日）\n")

    o01 = [c for c in allc if c["wt_overlap_n"] in (0, 1)]
    o2 = [c for c in allc if c["wt_overlap_n"] == 2]
    gate01 = [c for c in o01 if (not ok_axis(c)) + (not ok_ent(c)) <= 1]

    # ── ① 現行との素の比較 ────────────────────────────────
    print("=" * 104)
    print("① 現行(overlap01・2ゲート不合格<=1) vs overlap2 素の比較 [K=5 総流し]")
    print("=" * 104)
    print(HEAD)
    print(row("現行 7S+7A", score(gate01), days))
    print(row("overlap2 全件", score(o2), days))
    print(row("overlap2 (2ゲート<=1)", score([c for c in o2 if (not ok_axis(c)) + (not ok_ent(c)) <= 1]), days))

    # ── ② 相手絞り K の効果 ───────────────────────────────
    print()
    print("=" * 104)
    print("② 相手絞り（pred_prob上位K車のみ購入・オッズ非依存）")
    print("=" * 104)
    print(HEAD)
    for k in (5, 4, 3, 2):
        print(row(f"現行7S+7A  K={k}", score(gate01, k), days))
    print("-" * 104)
    for k in (5, 4, 3, 2):
        print(row(f"overlap2   K={k}", score(o2, k), days))

    # ── ③ overlap2 の中で「配当が見込める・的中率が高い」帯を探す ──────
    print()
    print("=" * 104)
    print("③ overlap2 を axis_sum / entropy で分解（K=3）")
    print("=" * 104)
    print(HEAD)
    for lo, hi in ((0, 1.1), (1.1, 1.3), (1.3, 1.5), (1.5, 1.7), (1.7, 9)):
        sub = [c for c in o2 if lo <= c["axis_sum"] < hi]
        print(row(f"  axis_sum [{lo},{hi})", score(sub, 3), days))
    print("-" * 104)
    for lo, hi in ((0, 1.75), (1.75, 1.80), (1.80, 1.8329), (1.8329, 1.87), (1.87, 9)):
        sub = [c for c in o2 if lo <= c["entropy"] < hi]
        print(row(f"  entropy [{lo},{hi})", score(sub, 3), days))

    # ── ④ 「相手が絞れている」レースの定義を探す ────────────────
    # 相手5車の pred_prob 分布の集中度: 上位1車 / 上位3車の占有率
    print()
    print("=" * 104)
    print("④ overlap2 を「相手の絞りやすさ」で分解（K=3）")
    print("   share3 = 相手5車のうち pred_prob 上位3車が占める割合（高い=絞れている）")
    print("=" * 104)
    print(HEAD)
    for c in allc:
        ps = sorted((c["top3_probs"][x] for x in c["others"]), reverse=True)
        c["share3"] = sum(ps[:3]) / sum(ps) if sum(ps) else 0.0
    qs = sorted(c["share3"] for c in o2)
    cuts = [qs[int(len(qs) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
    bands = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]),
             (cuts[2], cuts[3]), (cuts[3], 1.01)]
    for lo, hi in bands:
        sub = [c for c in o2 if lo <= c["share3"] < hi]
        print(row(f"  share3 [{lo:.3f},{hi:.3f})", score(sub, 3), days))

    # ── ⑤ 有望条件の組み合わせ（月次安定性つき）─────────────────
    print()
    print("=" * 104)
    print("⑤ 組み合わせ候補の月次安定性（K=3）")
    print("=" * 104)
    combos = [
        ("overlap2 全件", lambda c: True),
        ("overlap2 ∧ ent<=1.8329", ok_ent),
        ("overlap2 ∧ axis<=1.5", ok_axis),
        ("overlap2 ∧ 2ゲート両方通過", lambda c: ok_axis(c) and ok_ent(c)),
        ("overlap2 ∧ share3上位40%", lambda c: c["share3"] >= cuts[2]),
        ("overlap2 ∧ ent<=1.8329 ∧ share3上位40%",
         lambda c: ok_ent(c) and c["share3"] >= cuts[2]),
    ]
    print(f"{'条件':40} {'n':>6} {'件/日':>6} {'的中率':>7} {'ROI':>7} "
          f"{'月ROIσ':>7} {'月ROI<60%':>9}")
    for name, pred in combos:
        sub = [c for c in o2 if pred(c)]
        by_m = defaultdict(list)
        for c in sub:
            by_m[c["race_date"][:7]].append(c)
        mrois = [score(v, 3)["roi"] for v in by_m.values() if score(v, 3)["bet"]]
        s = score(sub, 3)
        sd = statistics.pstdev(mrois) if len(mrois) > 1 else 0.0
        print(f"{name:40} {s['n']:6d} {s['n']/days:6.2f} {s['hit_rate']:6.1f}% "
              f"{s['roi']:6.1f}% {sd:6.1f} {sum(1 for r in mrois if r < 60):9d}")


if __name__ == "__main__":
    main()
