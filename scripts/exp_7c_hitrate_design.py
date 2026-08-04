"""【読み取り専用】的中重視ランクの設計掃引（2026-08-04）。

背景（ユーザー要望）:
  「8/1後半以降 当たりが少ない。予想家として当たりがないのが厳しいので
    的中率を上げたい」。診断の結果、的中率(32〜38%)は安定しており、原因は
    推奨件数の長期的な枯渇（48件/日 → 12件/日）だった。
  wt_overlap_n==2（軸2車がWT公式印◎◯と完全一致）を的中重視で採用する方針
  （A-2: 相手3点）をユーザーが選択。入稿上限は1日20本前後。
  続いて「ガミは厳しいので点数を絞るか、三連単とする必要がある」との追加方針。

本スクリプトは overlap2 母集団を
  ・7B（順序不一致 ∧ △除外3点。2026-08-03 導入済み・配当重視）
  ・新ランク候補（順序一致側。7Bと論理的に排他）
に分け、新ランク候補について
  ・三連複（相手K点・△込み/除外）
  ・三連単（軸2車の着順を片方向/両方向 × 相手K点）
の件数・的中率・ROI・**ガミ率**・月次安定性を測る。

⚠️ オッズは wt_odds＝最終オッズ（stale）。選出条件はオッズ非依存
  （axis_sum/entropy/pred_prob）に保っているため選択バイアスは入らないが、
  払戻額そのものは発走15分前オッズとは一致しない。

DB書き込みなし。キャッシュは scripts/exp_7c_cache.py が生成したもの。

使い方:
    python scripts/exp_7c_hitrate_design.py data/exp_7c_cache
"""
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX

STAKE = 100  # 円/点


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


# ---------------------------------------------------------------- 補助
def legs(c: dict, k: int, drop_ana: bool) -> list[int]:
    """相手を pred_prob 降順に並べ、（任意で△を除いて）上位k車を返す。"""
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    if drop_ana and c.get("wt_ana") is not None:
        ranked = [x for x in ranked if x != c["wt_ana"]]
    return ranked[:k]


def order_disagree(c: dict) -> bool | None:
    """モデルの pred_win 最上位が WT◎ でないか（7B の順序不一致判定と同一）。"""
    if c.get("wt_honmei") is None or not c["win_probs"]:
        return None
    return max(c["win_probs"], key=lambda f: c["win_probs"][f]) != c["wt_honmei"]


def axes_by_win(c: dict) -> tuple[int, int]:
    """軸2車を pred_win 降順で (強い方, 弱い方) にして返す。"""
    a, b = c["axis1"], c["axis2"]
    return (a, b) if c["win_probs"].get(a, 0) >= c["win_probs"].get(b, 0) else (b, a)


def _pay(odds: float | None) -> int:
    """最終オッズ→100円あたり払戻（公式に合わせ10円単位切り捨て）。"""
    if odds is None:
        return 0
    return round(odds * 100) // 10 * 10


# ---------------------------------------------------------------- 券面
def bets_trio(c: dict, k: int, drop_ana: bool) -> list[tuple[frozenset, int]]:
    """三連複 軸2車 + 相手上位k車。"""
    out = []
    for x in legs(c, k, drop_ana):
        od = c["trio_legs"].get(x)
        if od is not None:
            out.append((frozenset({c["axis1"], c["axis2"], x}), _pay(od)))
    return out


def bets_tri(c: dict, k: int, drop_ana: bool, both_ways: bool) -> list[tuple[tuple, int]]:
    """三連単 軸2車を1-2着（片方向 or 両方向）+ 相手上位k車を3着。"""
    a, b = axes_by_win(c)
    orders = [(a, b)] if not both_ways else [(a, b), (b, a)]
    out = []
    for x in legs(c, k, drop_ana):
        for p, q in orders:
            od = c["tri_perm"].get((p, q, x))
            if od is not None:
                out.append(((p, q, x), _pay(od)))
    return out


def bets_tri_3rd_axis(c: dict, k: int, drop_ana: bool) -> list[tuple[tuple, int]]:
    """三連単 1着=pred_win上位の軸、2着=相手上位k車、3着=もう一方の軸。

    「軸2車が1-2着」に固定せず、相手が2着に割り込む形も拾う変形。
    """
    a, b = axes_by_win(c)
    out = []
    for x in legs(c, k, drop_ana):
        od = c["tri_perm"].get((a, x, b))
        if od is not None:
            out.append(((a, x, b), _pay(od)))
    return out


# ---------------------------------------------------------------- 採点
def evaluate(cands: list[dict], bet_fn, ordered: bool) -> dict:
    """bet_fn(c) -> [(key, payout)] を評価する。ordered=True なら三連単判定。"""
    bet = ret = hit = gami = n_used = 0
    payouts: list[int] = []
    by_month: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    days = set()
    for c in cands:
        bs = bet_fn(c)
        if not bs:
            continue
        n_used += 1
        days.add(c["race_date"])
        ym = str(c["race_date"])[:7]
        stake = len(bs) * STAKE
        bet += stake
        by_month[ym][0] += stake
        win_key = c["order3"] if ordered else frozenset(c["order3"])
        got = next((p for key, p in bs if key == win_key), 0)
        if got:
            hit += 1
            ret += got
            by_month[ym][1] += got
            payouts.append(got)
            if got < stake:
                gami += 1
    mrois = [100.0 * r / b for b, r in by_month.values() if b > 0]
    return {
        "n": n_used,
        "per_day": n_used / len(days) if days else 0.0,
        "pts": bet / n_used / STAKE if n_used else 0.0,
        "hit_rate": 100.0 * hit / n_used if n_used else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "gami_rate": 100.0 * gami / hit if hit else 0.0,
        "med_pay": statistics.median(payouts) / 100 if payouts else 0.0,
        "n20x": sum(1 for p in payouts if p >= 2000),
        "m_sd": statistics.pstdev(mrois) if len(mrois) > 1 else 0.0,
        "m_lt60": sum(1 for r in mrois if r < 60),
        "n_months": len(mrois),
    }


HDR = (f"{'案':34} {'n':>6} {'件/日':>6} {'点数':>5} {'的中':>6} {'ROI':>7} "
       f"{'ガミ':>6} {'中央値':>7} {'20倍超':>6} {'月σ':>6} {'<60%':>5}")


def row(label: str, s: dict) -> str:
    return (f"{label:34} {s['n']:6d} {s['per_day']:6.2f} {s['pts']:5.1f} "
            f"{s['hit_rate']:5.1f}% {s['roi']:6.1f}% {s['gami_rate']:5.1f}% "
            f"{s['med_pay']:6.1f}倍 {s['n20x']:6d} {s['m_sd']:6.1f} "
            f"{s['m_lt60']:2d}/{s['n_months']:2d}")


def apply_cap(cands: list[dict], cap: int, key, reverse: bool) -> list[dict]:
    """日ごとに key でソートして上位 cap 件のみ残す。"""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_day[c["race_date"]].append(c)
    out = []
    for lst in by_day.values():
        out.extend(sorted(lst, key=key, reverse=reverse)[:cap])
    return out


def main() -> None:
    rows = load(Path(sys.argv[1]))
    days = {c["race_date"] for c in rows}
    print(f"候補総数 {len(rows)}件 / {len(days)}日 ({min(days)}〜{max(days)})\n")

    ov01 = [c for c in rows if c["wt_overlap_n"] in (0, 1)]
    ov2 = [c for c in rows if c["wt_overlap_n"] == 2]
    print(f"overlap∈{{0,1}}: {len(ov01):6d}件 ({100*len(ov01)/len(rows):.1f}%)")
    print(f"overlap==2    : {len(ov2):6d}件 ({100*len(ov2)/len(rows):.1f}%)\n")

    cur_7s = [c for c in ov01
              if c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX and c["entropy"] <= RANK_7S_ENTROPY_MAX]
    cur_7a = [c for c in ov01
              if (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX) != (c["entropy"] > RANK_7S_ENTROPY_MAX)]
    cur_7b = [c for c in ov2 if order_disagree(c) is True]
    pool = [c for c in ov2 if order_disagree(c) is False]   # 新ランク候補（7Bと排他）

    print("【現行ランクの honest 再現】")
    print(HDR)
    print(row("7S 三連複5点", evaluate(cur_7s, lambda c: bets_trio(c, 5, False), False)))
    print(row("7A 三連複5点", evaluate(cur_7a, lambda c: bets_trio(c, 5, False), False)))
    print(row("7B 三連複3点(△除外)", evaluate(cur_7b, lambda c: bets_trio(c, 3, True), False)))
    print(row("  7S+7A 合算", evaluate(cur_7s + cur_7a, lambda c: bets_trio(c, 5, False), False)))
    print()

    by_axis_desc = (lambda c: c["axis_sum"], True)

    print("【新ランク候補（順序一致・axis_sum降順cap）× 三連複の点数】")
    print(HDR)
    for cap in (8, 10, 12):
        capped = apply_cap(pool, cap, *by_axis_desc)
        for k in (3, 2, 1):
            for drop in (False, True):
                tag = "△除外" if drop else "△込み"
                print(row(f"cap={cap} 三連複{k}点 {tag}",
                          evaluate(capped, lambda c, k=k, d=drop: bets_trio(c, k, d), False)))
        print()

    print("【新ランク候補 × 三連単（軸2車1-2着・相手K点）】")
    print(HDR)
    for cap in (8, 10, 12):
        capped = apply_cap(pool, cap, *by_axis_desc)
        for k in (3, 2, 1):
            for both in (False, True):
                tag = "両順序" if both else "片順序"
                print(row(f"cap={cap} 三連単{tag}×相手{k}",
                          evaluate(capped, lambda c, k=k, b=both: bets_tri(c, k, False, b), True)))
        print(row(f"cap={cap} 三連単 軸→相手→軸 K=3",
                  evaluate(capped, lambda c: bets_tri_3rd_axis(c, 3, False), True)))
        print()

    print("【1点/2点に絞ったときの cap 拡大余地（件数をどこまで伸ばせるか）】")
    print(HDR)
    for k in (1, 2):
        for cap in (8, 10, 12, 15, 20, 999):
            capped = apply_cap(pool, cap, *by_axis_desc)
            lbl = "cap無" if cap == 999 else f"cap={cap}"
            print(row(f"三連複{k}点 △込み {lbl}",
                      evaluate(capped, lambda c, k=k: bets_trio(c, k, False), False)))
        print()

    print("【相手1車の選び方（cap=10・三連複1点）】")
    print(HDR)
    capped10 = apply_cap(pool, 10, *by_axis_desc)

    def _one(c: dict, pick) -> list[tuple[frozenset, int]]:
        x = pick(c)
        if x is None:
            return []
        od = c["trio_legs"].get(x)
        return [(frozenset({c["axis1"], c["axis2"], x}), _pay(od))] if od is not None else []

    def _nth_by_p(c: dict, n: int) -> int | None:
        r = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
        return r[n] if len(r) > n else None

    variants = [
        ("pred_prob 1位", lambda c: _nth_by_p(c, 0)),
        ("pred_prob 2位", lambda c: _nth_by_p(c, 1)),
        ("WT△(ana)", lambda c: c.get("wt_ana") if c.get("wt_ana") in c["others"] else None),
        ("△除外の pred_prob 1位", lambda c: next(
            (x for x in sorted(c["others"], key=lambda y: -c["top3_probs"][y])
             if x != c.get("wt_ana")), None)),
    ]
    for name, pick in variants:
        print(row(f"相手= {name}", evaluate(capped10, lambda c, p=pick: _one(c, p), False)))
    print()

    print("【参考: cap無し（母集団全体）での券面比較】")
    print(HDR)
    print(row("三連複3点 △込み", evaluate(pool, lambda c: bets_trio(c, 3, False), False)))
    print(row("三連複2点 △込み", evaluate(pool, lambda c: bets_trio(c, 2, False), False)))
    print(row("三連単 片順序×相手3", evaluate(pool, lambda c: bets_tri(c, 3, False, False), True)))
    print(row("三連単 両順序×相手3", evaluate(pool, lambda c: bets_tri(c, 3, False, True), True)))
    print()

    # ---- 有力案の月次推移 --------------------------------------------------
    finalists = [
        ("三連複1点 cap=10", apply_cap(pool, 10, *by_axis_desc),
         lambda c: bets_trio(c, 1, False), False),
        ("三連複2点 cap=10", apply_cap(pool, 10, *by_axis_desc),
         lambda c: bets_trio(c, 2, False), False),
        ("三連単 軸→相手→軸 3点 cap=10", apply_cap(pool, 10, *by_axis_desc),
         lambda c: bets_tri_3rd_axis(c, 3, False), True),
    ]
    for name, cands, fn, ordered in finalists:
        print(f"【月次推移: {name}】")
        by_ym: dict[str, list[dict]] = defaultdict(list)
        for c in cands:
            by_ym[str(c["race_date"])[:7]].append(c)
        print(f"  {'月':9} {'n':>5} {'的中':>7} {'ROI':>8} {'ガミ':>7}")
        for ym in sorted(by_ym):
            s = evaluate(by_ym[ym], fn, ordered)
            print(f"  {ym:9} {s['n']:5d} {s['hit_rate']:6.1f}% "
                  f"{s['roi']:7.1f}% {s['gami_rate']:6.1f}%")
        print()


if __name__ == "__main__":
    main()
