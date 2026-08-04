"""【読み取り専用】「軸2を3着内80%まで絞れるか」の到達可能性を実測する（2026-08-04）。

ユーザー方針:
  「3着内に来る選手を当てる精度の向上を継続する。ROIは精度を上げた結果として
    レースを絞り、ROIを確保できない低配当レースを除外し、再現頻度が低い
    高配当帯も除いて、的中率とROIの両立を目指す。
    軸1の70%超えは良い精度なので、残り6車から1車を80%程度で絞れれば
    的中率50%になる見込み。そこを目指す方法を検討したい。」

測定内容:
  ① 3着内枠の配分制約（7車で3枠＝確率の総和は3.0）の実測
  ② モデル評価順位（pred_prob 1〜7位）別の実測3着内率＝取りこぼしの所在
  ③ 軸2の予測確率 p2 の帯別: 件数・実測3着内率・両方3着内率・配当・ROI
     → 「軸2を80%で絞る」を実行したとき何件残り配当がどうなるかを直接見る
  ④ 精度×配当のフロンティア: 両方3着内率を上げると三連複配当がどう動くか
  ⑤ 低配当除外・高配当依存除去を適用した後のROI（ユーザー方針の直接評価）

honest: 月次凍結vintageモデルのキャッシュ（scripts/exp_7c_cache.py）を使用。
⚠️ オッズは wt_odds＝最終オッズ（stale）。層別条件はオッズ非依存
   （pred_prob のみ）に保っているため選択バイアスは入らない。

DB書き込みなし。

使い方:
    python scripts/exp_axis2_precision_frontier.py data/exp_7c_cache
"""
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STAKE = 100


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def p_of(c: dict, f: int) -> float:
    return c["top3_probs"].get(f, 0.0)


def trio_flow(c: dict, k: int = 5) -> list[tuple[frozenset, int]]:
    """三連複 軸2車 + 相手 pred_prob 上位k車（k=5 は総流し）。"""
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])[:k]
    out = []
    for x in ranked:
        od = c["trio_legs"].get(x)
        if od is not None:
            out.append((frozenset({c["axis1"], c["axis2"], x}), round(od * 100) // 10 * 10))
    return out


def seg_stats(cands: list[dict], k: int = 5) -> dict:
    """セグメントの精度と、三連複k点流しの成績をまとめて返す。"""
    n = both3 = 0
    a1_3 = a2_3 = 0
    bet = ret = hit = gami = 0
    payouts: list[int] = []
    days = set()
    for c in cands:
        n += 1
        days.add(c["race_date"])
        o = set(c["order3"])
        if c["axis1"] in o:
            a1_3 += 1
        if c["axis2"] in o:
            a2_3 += 1
        if {c["axis1"], c["axis2"]} <= o:
            both3 += 1
        bs = trio_flow(c, k)
        if not bs:
            continue
        stake = len(bs) * STAKE
        bet += stake
        got = next((p for key, p in bs if key == frozenset(c["order3"])), 0)
        if got:
            hit += 1
            ret += got
            payouts.append(got)
            if got < stake:
                gami += 1
    return {
        "n": n,
        "per_day": n / len(days) if days else 0.0,
        "a1": 100.0 * a1_3 / n if n else 0.0,
        "a2": 100.0 * a2_3 / n if n else 0.0,
        "both": 100.0 * both3 / n if n else 0.0,
        "hit": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "gami": 100.0 * gami / hit if hit else 0.0,
        "med": statistics.median(payouts) / 100 if payouts else 0.0,
        "p90": (sorted(payouts)[int(len(payouts) * 0.9)] / 100) if payouts else 0.0,
    }


HDR = (f"{'区分':22} {'n':>6} {'件/日':>6} {'軸1':>6} {'軸2':>6} {'両方':>6} "
       f"{'的中':>6} {'ROI':>7} {'ガミ':>6} {'中央値':>7} {'p90':>7}")


def row(label: str, s: dict) -> str:
    return (f"{label:22} {s['n']:6d} {s['per_day']:6.2f} {s['a1']:5.1f}% {s['a2']:5.1f}% "
            f"{s['both']:5.1f}% {s['hit']:5.1f}% {s['roi']:6.1f}% {s['gami']:5.1f}% "
            f"{s['med']:6.1f}倍 {s['p90']:6.1f}倍")


def main() -> None:
    rows = load(Path(sys.argv[1]))
    days = sorted({c["race_date"] for c in rows})
    print(f"母集団: 7車立て・軸選定成功 {len(rows)}件 / {len(days)}日 "
          f"({days[0]}〜{days[-1]})\n")

    # ---------------------------------------------------------------- ①制約
    print("【① 3着内確率の配分制約（7車で3枠なので総和は必ず3.0）】")
    tot = [sum(c["top3_probs"].values()) for c in rows]
    p1s = [p_of(c, c["axis1"]) for c in rows]
    p2s = [p_of(c, c["axis2"]) for c in rows]
    rest = [sum(c["top3_probs"][x] for x in c["others"]) for c in rows]
    print(f"  レース内 pred_prob 合計   平均 {statistics.mean(tot):.3f}"
          f"（理論値 3.000）")
    print(f"  軸1 の p1               平均 {statistics.mean(p1s):.3f}")
    print(f"  軸2 の p2               平均 {statistics.mean(p2s):.3f}")
    print(f"  残り5車の合計            平均 {statistics.mean(rest):.3f}")
    print(f"  → p2 を 0.80 にするには残り5車の合計を "
          f"{statistics.mean(rest):.3f} → "
          f"{3.0 - statistics.mean(p1s) - 0.80:.3f} まで削る必要がある")
    print(f"  実測 p2>=0.80 のレース: {sum(1 for p in p2s if p >= 0.80)}件 "
          f"({100*sum(1 for p in p2s if p >= 0.80)/len(p2s):.1f}%)")
    print()

    # ---------------------------------------------------------------- ②順位別
    print("【② モデル評価順位（pred_prob 1〜7位）別の実測3着内率＝取りこぼしの所在】")
    rank_hit = defaultdict(lambda: [0, 0])
    for c in rows:
        o = set(c["order3"])
        for i, f in enumerate(sorted(c["top3_probs"], key=lambda x: -c["top3_probs"][x])):
            rank_hit[i][0] += 1
            if f in o:
                rank_hit[i][1] += 1
    print(f"  {'評価順位':10} {'n':>7} {'実測3着内率':>11} {'3着内の内訳':>12}")
    total_in = sum(v[1] for v in rank_hit.values())
    for i in sorted(rank_hit):
        n, h = rank_hit[i]
        print(f"  {i+1}位{'':7} {n:7d} {100.0*h/n:10.1f}% {100.0*h/total_in:11.1f}%")
    print(f"  → 3着内に来る選手の {100.0*sum(rank_hit[i][1] for i in (3,4,5,6))/total_in:.1f}% は"
          f" モデル評価4位以下（ここが取りこぼしの本体）")
    print()

    # ---------------------------------------------------------------- ③p2帯別
    print("【③ 軸2の予測確率 p2 の帯別（＝「軸2を80%で絞る」の実行結果）】")
    print(HDR)
    edges = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 0.8),
             (0.8, 0.85), (0.85, 1.01)]
    for lo, hi in edges:
        sub = [c for c in rows if lo <= p_of(c, c["axis2"]) < hi]
        if sub:
            print(row(f"p2 {lo:.2f}〜{hi:.2f}", seg_stats(sub)))
    print()
    for thr in (0.70, 0.75, 0.80):
        sub = [c for c in rows if p_of(c, c["axis2"]) >= thr]
        if sub:
            print(row(f"p2 >= {thr:.2f} 累積", seg_stats(sub)))
    print()

    # ---------------------------------------------------------------- ④フロンティア
    print("【④ 精度×配当フロンティア: p1*p2（両方3着内の予測確率）帯別】")
    print(HDR)
    prod_edges = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.62),
                  (0.62, 0.68), (0.68, 1.01)]
    for lo, hi in prod_edges:
        sub = [c for c in rows if lo <= p_of(c, c["axis1"]) * p_of(c, c["axis2"]) < hi]
        if sub:
            print(row(f"p1*p2 {lo:.2f}〜{hi:.2f}", seg_stats(sub)))
    print()

    # ---------------------------------------------------------------- ⑤方針評価
    print("【⑤ ユーザー方針の直接評価: 高精度セグメント × 低配当除外】")
    print("  対象 = 両方3着内の予測確率 p1*p2 >= 0.55（＝実測でも高精度な帯）")
    print("  そこから『三連複の買い目オッズが安すぎる目』を除外したときのROI")
    print(f"  {'買い目オッズ下限':16} {'n':>6} {'点数':>5} {'的中':>7} {'ROI':>8} "
          f"{'ガミ':>7} {'中央値':>8}")
    hi_seg = [c for c in rows if p_of(c, c["axis1"]) * p_of(c, c["axis2"]) >= 0.55]
    for floor in (0, 3, 5, 8, 12, 20):
        bet = ret = hit = gami = n_used = 0
        payouts = []
        for c in hi_seg:
            bs = [(k, p) for k, p in trio_flow(c, 5) if p >= floor * 100]
            if not bs:
                continue
            n_used += 1
            stake = len(bs) * STAKE
            bet += stake
            got = next((p for k, p in bs if k == frozenset(c["order3"])), 0)
            if got:
                hit += 1
                ret += got
                payouts.append(got)
                if got < stake:
                    gami += 1
        if n_used:
            print(f"  {floor:2d}倍以上{'':8} {n_used:6d} {bet/n_used/STAKE:5.1f} "
                  f"{100.0*hit/n_used:6.1f}% {100.0*ret/bet:7.1f}% "
                  f"{100.0*gami/hit if hit else 0:6.1f}% "
                  f"{statistics.median(payouts)/100 if payouts else 0:7.1f}倍")
    print()
    print("  ※ 併せて『再現頻度が低い高配当』の寄与も確認する")
    _high_payout_check(hi_seg)

    # ---------------------------------------------------------------- ⑥ワイド
    print()
    print("【⑥ ワイド1点（軸2車）— 的中＝両方3着内。ユーザー方針に最も近い券面】")
    print(f"  {'区分':20} {'n':>6} {'件/日':>6} {'的中':>7} {'ROI':>8} {'ガミ':>7} "
          f"{'中央値':>8}")

    def wide_stats(cands: list[dict]) -> tuple:
        bet = ret = hit = gami = n = 0
        pays = []
        days = set()
        for c in cands:
            od = c.get("wide_axis")
            if not od:
                continue
            n += 1
            days.add(c["race_date"])
            bet += STAKE
            if {c["axis1"], c["axis2"]} <= set(c["order3"]):
                hit += 1
                ret += od
                pays.append(od)
                if od < STAKE:
                    gami += 1
        return (n, n / len(days) if days else 0.0,
                100.0 * hit / n if n else 0.0, 100.0 * ret / bet if bet else 0.0,
                100.0 * gami / hit if hit else 0.0,
                statistics.median(pays) / 100 if pays else 0.0)

    for lo, hi in edges:
        sub = [c for c in rows if lo <= p_of(c, c["axis2"]) < hi]
        if sub:
            s = wide_stats(sub)
            print(f"  p2 {lo:.2f}〜{hi:.2f}{'':6} {s[0]:6d} {s[1]:6.2f} {s[2]:6.1f}% "
                  f"{s[3]:7.1f}% {s[4]:6.1f}% {s[5]:7.1f}倍")
    print()
    print("  ワイドオッズ下限で足切りした場合（⚠️ 最終オッズでの足切り＝stale bias）")
    print(f"  {'下限':20} {'n':>6} {'件/日':>6} {'的中':>7} {'ROI':>8} {'ガミ':>7} "
          f"{'中央値':>8}")
    for floor in (0, 1.5, 2.0, 2.5, 3.0, 4.0):
        sub = [c for c in rows if (c.get("wide_axis") or 0) >= floor * 100]
        if sub:
            s = wide_stats(sub)
            print(f"  {floor:.1f}倍以上{'':11} {s[0]:6d} {s[1]:6.2f} {s[2]:6.1f}% "
                  f"{s[3]:7.1f}% {s[4]:6.1f}% {s[5]:7.1f}倍")


def _high_payout_check(hi_seg: list[dict]) -> None:
    """『再現頻度が低い高配当』にROIが依存していないかの確認。"""
    for floor in (0, 5):
        bs_all = []
        for c in hi_seg:
            bs = [(k, p) for k, p in trio_flow(c, 5) if p >= floor * 100]
            if not bs:
                continue
            got = next((p for k, p in bs if k == frozenset(c["order3"])), 0)
            bs_all.append((len(bs) * STAKE, got))
        if not bs_all:
            continue
        bet = sum(b for b, _ in bs_all)
        ret = sum(g for _, g in bs_all)
        top = sorted((g for _, g in bs_all), reverse=True)[:5]
        print(f"  下限{floor:2d}倍: ROI {100.0*ret/bet:.1f}%  "
              f"→ 最高配当5本を除くと {100.0*(ret-sum(top))/bet:.1f}%  "
              f"（除外分 {sum(top)/100:.0f}倍相当）")


if __name__ == "__main__":
    main()
