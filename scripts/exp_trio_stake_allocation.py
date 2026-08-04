"""【読み取り専用】三連複2軸流しの「配分」と「相手の削り」でROIを確保できるかの検証
（2026-08-04）。

ユーザー提案:
  「ワイドの的中と三連複総流しはどちらも的中になるはず。均等でなく人気の目への
    傾斜配分、または3着以内に来なそうな相手の目を削ることでのROI確保が考えられる。」

前提の確認（実測で一致済み）:
  ワイド（軸2車）の的中 = 軸2車がともに3着内 = 三連複総流し(K=5)の的中。
  したがって K=5 は的中率でワイドと等価で、**削る**と的中率はワイドより下がる。
  問題は「削って落とす的中率」と「上がるROI」の交換レートである。

測定内容:
  ① 相手の削り（K=5→1・pred_prob 降順）× 母集団別の 的中率／ROI
  ② 配分方式の比較（均等／pred_prob比例／オッズ逆比例＝人気傾斜／EV比例／EV足切り）
  ③ ①②の組み合わせ最良と、ワイド1点との直接比較

配分方式の区別（重要）:
  - オッズ非依存（pred_prob のみ）… 朝の時点で実行可能。**実運用可**
  - オッズ依存（EV・人気傾斜）    … 本検証のオッズは wt_odds＝**最終オッズ**のため
                                    stale-odds バイアスを含む。実運用では朝オッズで
                                    再検証しない限り数値をそのまま信じてはいけない

DB書き込みなし。

使い方:
    python scripts/exp_trio_stake_allocation.py data/exp_7c_cache
"""
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX

TOTAL_STAKE = 500.0  # 1レースあたり総投資額を固定して配分だけを比較する


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def p_of(c: dict, f: int) -> float:
    return c["top3_probs"].get(f, 0.0)


def legs_of(c: dict, k: int) -> list[int]:
    """相手を pred_prob 降順に並べ、上位k車（オッズが存在するもの）を返す。"""
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    return [x for x in ranked[:k] if c["trio_legs"].get(x)]


# ---------------------------------------------------------------- 配分方式
def w_flat(c: dict, xs: list[int]) -> list[float]:
    return [1.0] * len(xs)


def w_prob(c: dict, xs: list[int]) -> list[float]:
    """3着目に来る確率（pred_prob）に比例。オッズ非依存。"""
    return [max(p_of(c, x), 1e-9) for x in xs]


def w_prob2(c: dict, xs: list[int]) -> list[float]:
    """pred_prob の2乗に比例（より急な人気傾斜）。オッズ非依存。"""
    return [max(p_of(c, x), 1e-9) ** 2 for x in xs]


def w_inv_odds(c: dict, xs: list[int]) -> list[float]:
    """オッズ逆比例＝市場の人気度に比例（ユーザー提案の『人気の目に傾斜』）。"""
    return [1.0 / max(c["trio_legs"][x], 1e-9) for x in xs]


def w_ev(c: dict, xs: list[int]) -> list[float]:
    """EV（our_prob × odds）に比例。our_prob は3車の確率積を正規化した近似。"""
    out = []
    for x in xs:
        pr = p_of(c, c["axis1"]) * p_of(c, c["axis2"]) * p_of(c, x)
        out.append(max(pr * c["trio_legs"][x], 1e-9))
    return out


def w_ev_gate(c: dict, xs: list[int]) -> list[float]:
    """EV>=1.0 の目だけ均等（既存 exp_s7_ev_threshold_staking_validation と同型）。"""
    out = []
    for x in xs:
        pr = p_of(c, c["axis1"]) * p_of(c, c["axis2"]) * p_of(c, x)
        out.append(1.0 if pr * c["trio_legs"][x] >= 1.0 else 0.0)
    return out


ALLOCATIONS = [
    ("均等（現行）", w_flat, False),
    ("pred_prob比例", w_prob, False),
    ("pred_prob^2比例", w_prob2, False),
    ("オッズ逆比例(人気傾斜)", w_inv_odds, True),
    ("EV比例", w_ev, True),
    ("EV>=1.0のみ均等", w_ev_gate, True),
]


def evaluate(cands: list[dict], k: int, wfn) -> dict:
    """総投資額を TOTAL_STAKE に固定して配分方式を比較する。"""
    bet = ret = hit = gami = n = 0
    pays: list[float] = []
    for c in cands:
        xs = legs_of(c, k)
        if not xs:
            continue
        ws = wfn(c, xs)
        tot_w = sum(ws)
        if tot_w <= 0:
            continue
        n += 1
        bet += TOTAL_STAKE
        win = frozenset(c["order3"])
        got = 0.0
        for x, w in zip(xs, ws):
            if frozenset({c["axis1"], c["axis2"], x}) == win:
                got = (w / tot_w) * TOTAL_STAKE * c["trio_legs"][x]
        if got:
            hit += 1
            ret += got
            pays.append(got)
            if got < TOTAL_STAKE:
                gami += 1
    return {
        "n": n,
        "hit": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "gami": 100.0 * gami / hit if hit else 0.0,
        "med": statistics.median(pays) / TOTAL_STAKE if pays else 0.0,
    }


def wide_stats(cands: list[dict]) -> dict:
    """比較用: ワイド1点（軸2車）。"""
    bet = ret = hit = gami = n = 0
    pays = []
    for c in cands:
        od = c.get("wide_axis")
        if not od:
            continue
        n += 1
        bet += 100
        if {c["axis1"], c["axis2"]} <= set(c["order3"]):
            hit += 1
            ret += od
            pays.append(od)
            if od < 100:
                gami += 1
    return {"n": n, "hit": 100.0 * hit / n if n else 0.0,
            "roi": 100.0 * ret / bet if bet else 0.0,
            "gami": 100.0 * gami / hit if hit else 0.0,
            "med": statistics.median(pays) / 100 if pays else 0.0}


def segments(rows: list[dict]) -> dict:
    ov01 = [c for c in rows if c["wt_overlap_n"] in (0, 1)]
    return {
        "全体": rows,
        "7S(市場と不一致・2ゲート合格)": [
            c for c in ov01 if c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
            and c["entropy"] <= RANK_7S_ENTROPY_MAX],
        "7A(市場と不一致・1ゲート不合格)": [
            c for c in ov01
            if (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX) != (c["entropy"] > RANK_7S_ENTROPY_MAX)],
        "overlap2(◎◯一致)": [c for c in rows if c["wt_overlap_n"] == 2],
        "p2>=0.80(高精度帯)": [c for c in rows if p_of(c, c["axis2"]) >= 0.80],
    }


def round_to_units(ws: list[float], total_units: int) -> list[int]:
    """重みを「100円 × total_units」に最大剰余法で丸める（実購入は100円単位）。

    合計が total_units になるよう配分し、0単位になった目は買わない扱いになる。
    """
    tot = sum(ws)
    if tot <= 0:
        return [0] * len(ws)
    raw = [w / tot * total_units for w in ws]
    base = [int(r) for r in raw]
    rem = total_units - sum(base)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - base[i]))
    for i in order[:rem]:
        base[i] += 1
    return base


def evaluate_units(cands: list[dict], k: int, wfn, total_units: int) -> dict:
    """100円単位に丸めた実購入で評価する。"""
    bet = ret = hit = gami = n = 0
    pays: list[float] = []
    n_pts = 0
    for c in cands:
        xs = legs_of(c, k)
        if not xs:
            continue
        units = round_to_units(wfn(c, xs), total_units)
        if sum(units) <= 0:
            continue
        n += 1
        stake = sum(units) * 100
        bet += stake
        n_pts += sum(1 for u in units if u > 0)
        win = frozenset(c["order3"])
        got = 0.0
        for x, u in zip(xs, units):
            if u > 0 and frozenset({c["axis1"], c["axis2"], x}) == win:
                got = u * 100 * c["trio_legs"][x]
        if got:
            hit += 1
            ret += got
            pays.append(got / stake)
            if got < stake:
                gami += 1
    return {
        "n": n,
        "pts": n_pts / n if n else 0.0,
        "hit": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "gami": 100.0 * gami / hit if hit else 0.0,
        "med": statistics.median(pays) if pays else 0.0,
    }


def main() -> None:
    rows = load(Path(sys.argv[1]))
    days = sorted({c["race_date"] for c in rows})
    print(f"母集団: 7車立て・軸選定成功 {len(rows)}件 / {len(days)}日 "
          f"({days[0]}〜{days[-1]})")
    print(f"※ 全案とも1レースあたり総投資額 {TOTAL_STAKE:.0f}円 固定。"
          f"配分だけを比較する\n")

    segs = segments(rows)

    # ---------------------------------------------------------------- ①削り
    print("【① 相手を削る（pred_prob 降順に上位K車のみ購入・均等配分）】")
    print("  ※ K=5 が総流し＝ワイドと同じ的中率。削るほど的中率は落ちる")
    print(f"  {'母集団':30} {'K':>2} {'n':>6} {'的中':>7} {'ROI':>8} "
          f"{'ガミ':>7} {'中央値':>8}")
    for name, sub in segs.items():
        if not sub:
            continue
        for k in (5, 4, 3, 2, 1):
            s = evaluate(sub, k, w_flat)
            print(f"  {name:30} {k:2d} {s['n']:6d} {s['hit']:6.1f}% {s['roi']:7.1f}% "
                  f"{s['gami']:6.1f}% {s['med']:7.2f}倍")
        w = wide_stats(sub)
        print(f"  {'  └ 参考: ワイド1点':30} {'-':>2} {w['n']:6d} {w['hit']:6.1f}% "
              f"{w['roi']:7.1f}% {w['gami']:6.1f}% {w['med']:7.2f}倍")
        print()

    # ---------------------------------------------------------------- ②配分
    print("【② 配分方式の比較（K=5 総流し・的中率は配分に依存せず一定）】")
    for name, sub in segs.items():
        if not sub:
            continue
        print(f"  -- {name} --")
        print(f"  {'配分':26} {'n':>6} {'的中':>7} {'ROI':>8} {'ガミ':>7} "
              f"{'中央値':>8} {'オッズ依存':>10}")
        for aname, wfn, uses_odds in ALLOCATIONS:
            s = evaluate(sub, 5, wfn)
            mark = "⚠️stale" if uses_odds else "非依存"
            print(f"  {aname:26} {s['n']:6d} {s['hit']:6.1f}% {s['roi']:7.1f}% "
                  f"{s['gami']:6.1f}% {s['med']:7.2f}倍 {mark:>10}")
        print()

    # ---------------------------------------------------------------- ④実購入
    print("【④ 実購入の制約（100円単位に丸め）での傾斜配分・K=5】")
    print("  ※ 総額は『100円×単位数』。単位数が少ないと傾斜が丸めで潰れる")
    for name, sub in segs.items():
        if not sub:
            continue
        print(f"  -- {name} --")
        print(f"  {'案':34} {'総額':>6} {'点数':>5} {'的中':>7} {'ROI':>8} "
              f"{'ガミ':>7} {'中央値':>8}")
        for units in (5, 8, 10, 20):
            for aname, wfn, uses_odds in ALLOCATIONS:
                if uses_odds:
                    continue
                s = evaluate_units(sub, 5, wfn, units)
                print(f"  {units:2d}単位 {aname:28} {units*100:5d}円 {s['pts']:5.1f} "
                      f"{s['hit']:6.1f}% {s['roi']:7.1f}% {s['gami']:6.1f}% "
                      f"{s['med']:7.2f}倍")
            print()

    # ---------------------------------------------------------------- ③組合せ
    print("【③ 削り × 配分 の組み合わせ（オッズ非依存の配分のみ）】")
    for name, sub in segs.items():
        if not sub:
            continue
        print(f"  -- {name} --")
        print(f"  {'案':34} {'的中':>7} {'ROI':>8} {'ガミ':>7} {'中央値':>8}")
        for k in (5, 3, 2):
            for aname, wfn, uses_odds in ALLOCATIONS:
                if uses_odds:
                    continue
                s = evaluate(sub, k, wfn)
                print(f"  K={k} {aname:29} {s['hit']:6.1f}% {s['roi']:7.1f}% "
                      f"{s['gami']:6.1f}% {s['med']:7.2f}倍")
        print()


if __name__ == "__main__":
    main()
