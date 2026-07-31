"""【賭け金配分の最適化】S7型（軸2車＋5点流し）の配分方式を比較する（2026-07-30）。

`inputs/netkeirin予想家_ROI100超の構造分析.md` 次元③・タスク4に対応。
メモの指摘: 「新しいedgeを生むものではなく、既存のedgeを効率的に取り出す操作。
比1.056の信号を1.333にはできない。しかし79%→85%程度の改善は配分だけで
起こり得る」。これを実測する。

## 買い目構造

S7型: 軸2車 {a,b}（`exp_pair_market_mispricing.py` の S_jointpair 方式で選定・
`pi×pj×lift(pair)` 最大のペア）＋ 残り5車それぞれとの三連複＝5点流し。

各点の的中確率は、35通り全体で正規化した our_prob（積×ライン相関lift）を使う
（`exp_market_edge_diagnosis.py` と同じ手法）。各点のオッズは実際の三連複配当。

## 4つの配分方式

1. **uniform**: 5点均等（現行の暗黙の前提）
2. **ev_proportional**: 期待値（our_prob×odds）に比例した配分
3. **kelly_fractional**: Kelly基準の1/4（分散抑制）。負のKellyは0にクリップ
4. **ev_threshold_filter**: 期待値1.0未満の点を除外し、残りに均等配分
   （メモが「これだけでROIは改善する」と最も重視した方式）

いずれも1レースあたりの総投資額を同一（5ユニット）に正規化し、配分だけの
効果を比較する（点数を減らす場合は総額を残った点に再配分＝メモのMr.T実例と
同じ発想）。

## 測定対象

- 全レース（ベースライン）
- レース選択後（top2_prob_sum 上位20%・[[keirin_netkeirin_race_selection_verification_2026_07_30]]
  で確認済みの「堅いレース」）
- 上記の掛け合わせでどこまで伸びるか

## 注意（既知のモデル癖）

[[keirin_clean_baseline_market_efficiency_2026_07_30]]で「高オッズ帯でモデルは
実確率を1.3〜2.2倍過大評価する」ことが判明済み。EVベースの配分（②③）は
高オッズ点に多く賭ける方向に働くため、この癖の影響を受けやすい。
TEST realized ROI（バックテストの実測値）で判定することで、EV予測の
バイアス自体も含めて公平に評価する。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_segment_market_edge import (  # noqa: E402
    TAKEOUT_RETURN, TRAIN_TO, build_rows, load_entries, load_races, load_trio_odds,
    pair_bucket,
)

MIN_BOARD = 33
TOTAL_STAKE_PER_RACE = 5.0     # 5点均等時に1点=1ユニットとなるよう正規化
KELLY_FRACTION = 0.25
MIN_N = 300


def estimate_lifts(rows):
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        tm = r["top3_mask"]
        for i, j in combinations(sorted(bf), 2):
            b = pair_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if (tm >> i) & 1 and (tm >> j) & 1:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


def race_our_probs(r, lifts):
    """35通り全体で正規化した our_prob を返す（exp_market_edge_diagnosis.py と同一手法）。"""
    bf = r["by_frame"]
    frames = sorted(bf)
    p = {f: float(bf[f]["pred_top3_pct"]) / 100.0 for f in frames}
    raw = {}
    for tri in combinations(frames, 3):
        s = p[tri[0]] * p[tri[1]] * p[tri[2]]
        for x, y in combinations(tri, 2):
            s *= lifts.get(pair_bucket(bf, x, y), 1.0)
        raw[frozenset(tri)] = s
    tot = sum(raw.values())
    if tot <= 0:
        return None
    return {k: v / tot for k, v in raw.items()}


def pick_axis_pair(r, lifts):
    bf = r["by_frame"]
    frames = sorted(bf)
    best_pair, best_score = None, -1.0
    for i, j in combinations(frames, 2):
        pi = float(bf[i]["pred_top3_pct"]) / 100.0
        pj = float(bf[j]["pred_top3_pct"]) / 100.0
        s = pi * pj * lifts.get(pair_bucket(bf, i, j), 1.0)
        if s > best_score:
            best_score, best_pair = s, (i, j)
    return best_pair


def kelly_stake(p, odds):
    """odds は倍率（配当/投資額）。b = odds - 1（純利益倍率）。負なら0にクリップ。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0) * KELLY_FRACTION


class Sim:
    __slots__ = ("n_races", "n_bet_races", "stake", "ret", "n_points")

    def __init__(self):
        self.n_races = 0
        self.n_bet_races = 0
        self.stake = 0.0
        self.ret = 0.0
        self.n_points = 0

    def add_race(self, points):
        """points: [(our_prob, odds, hit)] の5点。配分方式ごとに呼び分ける。"""
        self.n_races += 1

    def roi(self):
        return (self.ret / self.stake * 100.0) if self.stake > 0 else 0.0


def simulate(points_list, scheme):
    """points_list: 各レースの [(our_prob, odds, hit), ...]（5点）のリスト。"""
    stake_total = ret_total = 0.0
    n_bet = 0
    n_points_total = 0
    for points in points_list:
        evs = [p * o for p, o, _h in points]
        if scheme == "uniform":
            stakes = [1.0] * len(points)
        elif scheme == "ev_proportional":
            pos = [max(e, 0.0) for e in evs]
            s = sum(pos)
            stakes = [x / s for x in pos] if s > 0 else [1.0 / len(points)] * len(points)
        elif scheme == "kelly_fractional":
            ks = [kelly_stake(p, o) for p, o, _h in points]
            s = sum(ks)
            stakes = [x / s for x in ks] if s > 0 else [0.0] * len(points)
        elif scheme == "ev_threshold_filter":
            qualify = [1.0 if e >= 1.0 else 0.0 for e in evs]
            s = sum(qualify)
            stakes = [x / s for x in qualify] if s > 0 else [0.0] * len(points)
        else:
            raise ValueError(scheme)

        s_sum = sum(stakes)
        if s_sum <= 0:
            continue    # 賭けない（ev_threshold_filterで0点の場合等）
        stakes = [x / s_sum * TOTAL_STAKE_PER_RACE for x in stakes]
        n_bet += 1
        n_points_total += sum(1 for x in stakes if x > 0)
        for (p, o, hit), st in zip(points, stakes):
            stake_total += st
            if hit:
                ret_total += st * o
    roi = (ret_total / stake_total * 100.0) if stake_total > 0 else 0.0
    avg_pts = n_points_total / n_bet if n_bet else 0.0
    return {"n_races": len(points_list), "n_bet": n_bet, "stake": stake_total,
            "ret": ret_total, "roi": roi, "avg_points": avg_pts}


def main():
    races = load_races()
    entries = load_entries(races.keys())
    rows = build_rows(races, entries)
    del entries

    train_rows = [r for r in rows if r["race_date"] <= TRAIN_TO]
    print(f"[split] TRAIN={len(train_rows)}R TEST={len(rows)-len(train_rows)}R")

    lifts = estimate_lifts(train_rows)
    print("[lift]", {k: round(v, 3) for k, v in lifts.items()})

    # レース選択カット（TRAINのtop3_sum_top2で上位20%閾値を決定）
    tr_sums = sorted(r["top3_sum_top2"] for r in train_rows)
    top20_cut = tr_sums[int(len(tr_sums) * 0.8)]
    print(f"[cut] top2_prob_sum 上位20%閾値: {top20_cut:.2f}")

    by_month = defaultdict(list)
    for r in rows:
        by_month[r["race_date"][:7]].append(r)

    race_points = {"TRAIN": {"ALL": [], "TOP20": []}, "TEST": {"ALL": [], "TOP20": []}}

    for ym in sorted(by_month):
        chunk = by_month[ym]
        boards = load_trio_odds([r["race_key"] for r in chunk])
        for r in chunk:
            board = boards.get(r["race_key"])
            if not board or len(board) < MIN_BOARD:
                continue
            probs = race_our_probs(r, lifts)
            if probs is None:
                continue
            a, b = pick_axis_pair(r, lifts)
            frames = sorted(r["by_frame"])
            others = [f for f in frames if f not in (a, b)]
            if len(others) != 5:
                continue

            tm = r["top3_mask"]
            points = []
            ok = True
            for x in others:
                mask = 0
                for f in (a, b, x):
                    mask |= 1 << f
                odds = board.get(mask)
                if odds is None or odds <= 0:
                    ok = False
                    break
                key = frozenset((a, b, x))
                p = probs.get(key)
                if p is None:
                    ok = False
                    break
                hit = 1 if mask == tm else 0
                points.append((p, odds, hit))
            if not ok or len(points) != 5:
                continue

            w = "TRAIN" if r["race_date"] <= TRAIN_TO else "TEST"
            race_points[w]["ALL"].append(points)
            if r["top3_sum_top2"] >= top20_cut:
                race_points[w]["TOP20"].append(points)
        print(f"  {ym}: {len(chunk)}R", flush=True)

    print("\n" + "=" * 108)
    print("賭け金配分方式の比較（S7型・軸2車+5点流し・総投資額は方式間で統一）")
    print("=" * 108)
    print(f"{'選択':<10}{'窓':<6}{'方式':<22}{'対象R':>8}{'賭けR':>8}{'平均点数':>9}"
          f"{'総投資':>10}{'総回収':>10}{'ROI%':>9}")
    schemes = ["uniform", "ev_proportional", "kelly_fractional", "ev_threshold_filter"]
    summary = {}
    for sel in ("ALL", "TOP20"):
        for w in ("TRAIN", "TEST"):
            pts = race_points[w][sel]
            if len(pts) < MIN_N:
                continue
            for sc in schemes:
                res = simulate(pts, sc)
                summary[(sel, w, sc)] = res
                print(f"{sel:<10}{w:<6}{sc:<22}{res['n_races']:>8}{res['n_bet']:>8}"
                      f"{res['avg_points']:>9.2f}{res['stake']:>10.1f}{res['ret']:>10.1f}"
                      f"{res['roi']:>9.1f}")
        print()

    print("\n" + "=" * 108)
    print("【結論】ベースライン(uniform)からの改善幅（TEST）")
    print("=" * 108)
    for sel in ("ALL", "TOP20"):
        base = summary.get((sel, "TEST", "uniform"))
        if not base:
            continue
        print(f"\n  [{sel}] uniform ROI={base['roi']:.1f}%（基準）")
        for sc in schemes[1:]:
            r_ = summary.get((sel, "TEST", sc))
            if not r_:
                continue
            diff = r_["roi"] - base["roi"]
            print(f"    {sc:<22} ROI={r_['roi']:.1f}%  差分={diff:+.1f}pt")


if __name__ == "__main__":
    main()
