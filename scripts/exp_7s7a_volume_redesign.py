"""【読み取り専用】7S/7A を「1日10本前後」へ増やす選出規則の honest 比較。

exp_7s7a_volume_cache.py が吐いた月次キャッシュ（月次凍結vintageモデル）を読み、
以下を出す:

  A) 診断: 月別に「生候補数 / wt_overlap_n別内訳 / 各ゲート通過率 / 現行選出数」
  B) 規則比較: 現行 vs 候補案の 件数/日・的中率・ROI（月次ROI標準偏差つき）

DB書き込みなし。
"""
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_DAILY_CAP, RANK_7S_ENTROPY_MAX, RANK_7S_STAKE,
)

CACHE = Path(sys.argv[1])
SINCE = sys.argv[2] if len(sys.argv) > 2 else "2024-01"


def load():
    months = {}
    for p in sorted(CACHE.glob("*.pkl")):
        if p.stem < SINCE:
            continue
        with open(p, "rb") as f:
            months[p.stem] = pickle.load(f)
    return months


def score(selected):
    """selected: 候補dictのリスト。5点流し uniform 100円で採点する。"""
    bet = ret = hit = 0
    payouts = []
    for c in selected:
        legs = [x for x, o in c["trio_legs"].items() if o is not None]
        if not legs:
            continue
        bet += len(legs) * RANK_7S_STAKE
        top3 = set(c["actual_top3"])
        won = top3 - {c["axis1"], c["axis2"]}
        if len(top3 & {c["axis1"], c["axis2"]}) == 2 and len(won) == 1 and won.pop() in legs:
            hit += 1
            ret += c["trio_pay"] * RANK_7S_STAKE // 100
            payouts.append(c["trio_pay"])
    n = len(selected)
    return {
        "n": n, "bet": bet, "ret": ret, "hit": hit,
        "hit_rate": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "max_pay": max(payouts) if payouts else 0,
        "n_over20x": sum(1 for p in payouts if p >= 2000),
    }


# ── 選出規則（いずれも「その日の候補リスト」→「採用リスト」）────────────────
def ok_axis(c):
    return c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX


def ok_ent(c):
    return c["entropy"] <= RANK_7S_ENTROPY_MAX


def cur_pool(day):
    """現行本番: overlap∈{0,1} かつ 2ゲートの不合格が0個(7S)または1個(7A)。"""
    out = []
    for c in day:
        if c["wt_overlap_n"] not in (0, 1):
            continue
        if (not ok_axis(c)) + (not ok_ent(c)) <= 1:
            out.append(c)
    return out


def rule_current(day):
    o = [c for c in cur_pool(day) if ok_axis(c) and ok_ent(c)]
    a = [c for c in cur_pool(day) if not (ok_axis(c) and ok_ent(c))]
    return sorted(o, key=lambda c: c["entropy"])[:RANK_7S_DAILY_CAP] + a


def rule_overlap2_included(day):
    """案1: overlapゲートを撤廃（2も許可）し、他は現行のまま。"""
    out = []
    for c in day:
        if (not ok_axis(c)) + (not ok_ent(c)) <= 1:
            out.append(c)
    return out


def make_topn(n, key, pred=None):
    """案2: 日次で pred を満たす候補を key 昇順に上位n本だけ採る（件数固定）。"""
    def _r(day):
        pool = [c for c in day if (pred is None or pred(c))]
        return sorted(pool, key=key)[:n]
    return _r


def overlap01(c):
    return c["wt_overlap_n"] in (0, 1)


def gate_le1(c):
    return (not ok_axis(c)) + (not ok_ent(c)) <= 1


RULES = [
    ("現行(7S+7A)", rule_current),
    ("案1: overlap2も許可", rule_overlap2_included),
    ("案2a: 全体からent昇順10", make_topn(10, lambda c: c["entropy"])),
    ("案2b: 全体からaxis昇順10", make_topn(10, lambda c: c["axis_sum"])),
    ("案2c: 2ゲート通過からent昇順10", make_topn(10, lambda c: c["entropy"], gate_le1)),
    ("案3: overlap01優先+不足をoverlap2で補充10",
     lambda day: (lambda a: a + sorted(
         [c for c in day if not overlap01(c) and gate_le1(c)],
         key=lambda c: c["entropy"])[:max(0, 10 - len(a))]
      )(rule_current(day)[:10])),
]


def main():
    months = load()
    print(f"キャッシュ月数: {len(months)}  ({min(months)}〜{max(months)})\n")

    # ── A) 診断 ────────────────────────────────────────────────
    print("=" * 108)
    print("A) 月別診断: 生候補・overlap内訳・ゲート通過率")
    print("=" * 108)
    print(f"{'月':8} {'日数':>4} {'生候補':>6} {'/日':>6} "
          f"{'ovl0':>5} {'ovl1':>6} {'ovl2':>6} {'None':>5} {'ovl01率':>7} "
          f"{'axis通過':>8} {'ent通過':>7} {'現行選出':>8} {'/日':>6}")
    for ym, cands in months.items():
        days = len({c["race_date"] for c in cands})
        by_day = defaultdict(list)
        for c in cands:
            by_day[c["race_date"]].append(c)
        sel = sum(len(rule_current(d)) for d in by_day.values())
        cnt = defaultdict(int)
        for c in cands:
            cnt[c["wt_overlap_n"]] += 1
        n = len(cands)
        o01 = cnt[0] + cnt[1]
        print(f"{ym:8} {days:4d} {n:6d} {n/days:6.1f} "
              f"{cnt[0]:5d} {cnt[1]:6d} {cnt[2]:6d} {cnt[None]:5d} {100*o01/n:6.1f}% "
              f"{100*sum(map(ok_axis, cands))/n:7.1f}% {100*sum(map(ok_ent, cands))/n:6.1f}% "
              f"{sel:8d} {sel/days:6.1f}")

    # ── B) 規則比較 ─────────────────────────────────────────────
    print()
    print("=" * 108)
    print("B) 選出規則の honest 比較（全期間合算 + 月次ROI標準偏差）")
    print("=" * 108)
    print(f"{'規則':34} {'n':>6} {'件/日':>6} {'的中率':>7} {'ROI':>7} "
          f"{'月ROIσ':>7} {'月ROI<50%':>9} {'20倍+的中':>9} {'最高配当':>8}")
    total_days = sum(len({c["race_date"] for c in cs}) for cs in months.values())
    for name, rule in RULES:
        allsel, monthly = [], []
        for cands in months.values():
            by_day = defaultdict(list)
            for c in cands:
                by_day[c["race_date"]].append(c)
            msel = []
            for d in by_day.values():
                msel.extend(rule(d))
            allsel.extend(msel)
            ms = score(msel)
            if ms["bet"]:
                monthly.append(ms["roi"])
        s = score(allsel)
        sd = statistics.pstdev(monthly) if len(monthly) > 1 else 0.0
        low = sum(1 for r in monthly if r < 50)
        print(f"{name:34} {s['n']:6d} {s['n']/total_days:6.2f} {s['hit_rate']:6.1f}% "
              f"{s['roi']:6.1f}% {sd:6.1f} {low:9d} {s['n_over20x']:9d} {s['max_pay']/100:7.1f}倍")

    # ── C) overlap2 単独の実力（直近12ヶ月）────────────────────────
    print()
    print("=" * 108)
    print("C) overlap別の実力（2ゲート不合格<=1 の候補のみ・月別ROI）")
    print("=" * 108)
    print(f"{'月':8} {'ovl01 n':>8} {'的中':>6} {'ROI':>7} | {'ovl2 n':>7} {'的中':>6} {'ROI':>7}")
    agg = {"a": [], "b": []}
    for ym, cands in months.items():
        a = [c for c in cands if overlap01(c) and gate_le1(c)]
        b = [c for c in cands if c["wt_overlap_n"] == 2 and gate_le1(c)]
        sa, sb = score(a), score(b)
        agg["a"].extend(a)
        agg["b"].extend(b)
        print(f"{ym:8} {sa['n']:8d} {sa['hit_rate']:5.1f}% {sa['roi']:6.1f}% | "
              f"{sb['n']:7d} {sb['hit_rate']:5.1f}% {sb['roi']:6.1f}%")
    sa, sb = score(agg["a"]), score(agg["b"])
    print(f"{'合計':8} {sa['n']:8d} {sa['hit_rate']:5.1f}% {sa['roi']:6.1f}% | "
          f"{sb['n']:7d} {sb['hit_rate']:5.1f}% {sb['roi']:6.1f}%")


if __name__ == "__main__":
    main()
