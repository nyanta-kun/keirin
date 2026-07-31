"""【1車軸の選定で市場に勝てるか】車単位の周辺確率で市場エッジを診断する
（2026-07-30・ユーザー方針「1車予想の軸となる選手を特定し、相手を選定する」の前提検証）。

## 位置づけ（なぜ組単位の検証だけでは足りないか）

`exp_segment_market_edge.py` で、発走前情報で定義できる全45セグメントにおいて
**組単位（三連複35通り）の予測精度で市場に負ける**ことが確定した（Δll -0.16〜-0.29・
全セグメントでt<-4）。

しかしあの検証の our_prob は「pred_top3_pct の積 × ライン相関lift」という
**joint（同時確率）の近似**である。負けた原因が
  (a) 車単位の3着内確率（周辺確率）自体が市場より劣る
  (b) 周辺確率は良いが、3頭同時確率への合成（積×lift）が粗い
のどちらなのかは切り分けられていない。

(b) なら「1車軸の選定」は市場に勝てている可能性があり、ユーザー方針
「1車軸を特定し相手を選ぶ」には道が残る。(a) なら軸選定の時点で負けており道はない。
**本スクリプトはこれを切り分ける。**

## 市場の周辺確率の作り方

三連複オッズから各車の「3着内確率」を厳密に周辺化できる:
  market_prob(組) = 正規化(0.75 / trio_odds)
  market_P(車i が3着内) = Σ_{iを含む組} market_prob(組)
各車は35通り中15通りに現れ、正しい同時分布なら Σ_i P_i = 3 が厳密に成立する
（3着以内に必ず3車入るため）。この整合性も検算する。

## 測る3つの問い

### 問1: 車単位の周辺確率で市場に勝てるか（全車・paired）
our_p_i = pred_top3_pct/100 vs market_P_i を、二値結果（3着内か）に対して
Brier / logloss で比較する。

### 問2【本題】軸選定の直接対決
我々の軸 = pred_win_pct 最上位（w1）
市場の軸 = market_P(3着内) 最上位
**両者が食い違ったレースだけに絞り**、どちらの選んだ車が実際に3着内に入ったかを比べる。
  我々の勝率 > 市場の勝率 → 軸選定にエッジあり（しかも我々の軸は市場評価が低い＝
                             オッズが高い＝ROIの源泉になる）
  我々の勝率 < 市場の勝率 → 軸選定でも負けており、この方向は閉じる
これはユーザー方針そのものを最短で検証する形になっている。

### 問3: 軸候補(w1)の3着内確率の較正
w1 に絞って our_p vs market_P vs 実測 を比較。我々が過大評価しているなら
「軸として信頼できる」という判断自体が誤りになる。

すべてセグメント別（突出パターン・硬さ四分位・rp_std四分位）にも分解する。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
（本スクリプトは学習を行わないが、既存検証との比較のため同じ窓で区切って両方報告する）
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

# 同一ロジックの重複を避けるため既存スクリプトのローダを再利用する
from exp_segment_market_edge import (  # noqa: E402
    MIN_BOARD, TAKEOUT_RETURN, TEST_FROM, TEST_TO, TRAIN_FROM, TRAIN_TO,
    build_rows, load_entries, load_races, load_trio_odds, pattern_of, q_label,
    quartile_cuts,
)

MIN_SEG = 150


def marginals(r, board):
    """三連複オッズから各車の市場3着内確率を周辺化して返す。"""
    frames = sorted(r["by_frame"].keys())
    mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
    if len(mk_raw) < MIN_BOARD:
        return None
    tot = sum(mk_raw.values())
    if tot <= 0:
        return None
    out = {f: 0.0 for f in frames}
    for m, v in mk_raw.items():
        p = v / tot
        for f in frames:
            if (m >> f) & 1:
                out[f] += p
    return out


def main():
    races = load_races()
    entries = load_entries(races.keys())
    rows = build_rows(races, entries)
    del entries

    train_rows = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    cuts = {
        "chalk": quartile_cuts([r["top3_sum_top2"] for r in train_rows]),
        "rp_std": quartile_cuts([r["rp_std"] for r in train_rows]),
    }
    tie_thr = quartile_cuts([max(r["g12"], r["g23"], r["g34"]) for r in train_rows])[0]
    print(f"[thr] 全体拮抗閾値={tie_thr:.2f}  硬さカット={cuts['chalk']}")

    # 集計器
    #   問1: 全車 paired
    m1 = defaultdict(lambda: {"n": 0, "br_our": 0.0, "br_mkt": 0.0,
                              "ll_our": 0.0, "ll_mkt": 0.0, "d": 0.0, "d2": 0.0})
    #   問2: 軸選定の直接対決（食い違ったレースのみ）
    m2 = defaultdict(lambda: {"n": 0, "our_hit": 0, "mkt_hit": 0,
                              "our_mp": 0.0, "mkt_mp": 0.0, "d": 0, "d2": 0})
    #   問3: w1 の較正
    m3 = defaultdict(lambda: {"n": 0, "our_p": 0.0, "mkt_p": 0.0, "act": 0})
    #   整合性検算
    sum_check = []
    agree_n = total_n = 0

    by_month = defaultdict(list)
    for r in rows:
        by_month[r["race_date"][:7]].append(r)

    for ym in sorted(by_month):
        chunk = by_month[ym]
        boards = load_trio_odds([r["race_key"] for r in chunk])
        for r in chunk:
            board = boards.get(r["race_key"])
            if not board:
                continue
            mk = marginals(r, board)
            if mk is None:
                continue
            window = "TRAIN" if r["race_date"] <= TRAIN_TO else "TEST"
            if len(sum_check) < 3000:
                sum_check.append(sum(mk.values()))

            pat = pattern_of(r, tie_thr)
            chalk = q_label(r["top3_sum_top2"], cuts["chalk"], "硬さ")
            rpq = q_label(r["rp_std"], cuts["rp_std"], "rp_std")
            segs = [("ALL", "全体"), ("pattern", pat), ("chalk", chalk), ("rp_std", rpq)]

            tm = r["top3_mask"]
            bf = r["by_frame"]

            # ---- 問1: 全車 paired ----
            for f in sorted(bf.keys()):
                y = 1.0 if (tm >> f) & 1 else 0.0
                po = min(max(float(bf[f]["pred_top3_pct"]) / 100.0, 1e-6), 1 - 1e-6)
                pm = min(max(mk[f], 1e-6), 1 - 1e-6)
                bo, bm = (po - y) ** 2, (pm - y) ** 2
                lo = -(y * math.log(po) + (1 - y) * math.log(1 - po))
                lm = -(y * math.log(pm) + (1 - y) * math.log(1 - pm))
                for dim, seg in segs:
                    a = m1[(window, dim, seg)]
                    a["n"] += 1
                    a["br_our"] += bo
                    a["br_mkt"] += bm
                    a["ll_our"] += lo
                    a["ll_mkt"] += lm
                    d = lm - lo          # 正なら我々の勝ち
                    a["d"] += d
                    a["d2"] += d * d

            # ---- 問2: 軸選定の直接対決 ----
            our_axis = r["win_order"][0]                      # pred_win_pct 最上位
            mkt_axis = max(mk, key=lambda f: mk[f])           # 市場3着内確率 最上位
            total_n += 1
            if our_axis == mkt_axis:
                agree_n += 1
            else:
                oh = 1 if (tm >> our_axis) & 1 else 0
                mh = 1 if (tm >> mkt_axis) & 1 else 0
                for dim, seg in segs:
                    a = m2[(window, dim, seg)]
                    a["n"] += 1
                    a["our_hit"] += oh
                    a["mkt_hit"] += mh
                    a["our_mp"] += mk[our_axis]
                    a["mkt_mp"] += mk[mkt_axis]
                    a["d"] += oh - mh
                    a["d2"] += (oh - mh) ** 2

            # ---- 問3: w1 の較正 ----
            f = our_axis
            for dim, seg in segs:
                a = m3[(window, dim, seg)]
                a["n"] += 1
                a["our_p"] += float(bf[f]["pred_top3_pct"]) / 100.0
                a["mkt_p"] += mk[f]
                a["act"] += 1 if (tm >> f) & 1 else 0
        print(f"  {ym}: {len(chunk)}R", flush=True)

    print("\n" + "=" * 112)
    print("[検算] Σ_i market_P(3着内) は 3.0 になるべき（正しい同時分布の整合性）")
    print(f"        平均 {statistics.mean(sum_check):.4f} / 中央値 "
          f"{statistics.median(sum_check):.4f}  (n={len(sum_check)})")
    print(f"[参考] 我々の軸(w1)と市場の軸が一致した率: {agree_n/total_n*100:.1f}% "
          f"({agree_n}/{total_n})  → 食い違い {total_n-agree_n} レースが問2の母集団")

    dims = [("ALL", "全体"), ("pattern", "突出パターン"),
            ("chalk", "硬さ四分位"), ("rp_std", "rp_std四分位")]

    # ---------------- 問1 ----------------
    print("\n" + "=" * 112)
    print("問1【車単位の周辺確率】我々の pred_top3_pct は市場の周辺確率に勝てるか")
    print("     Δll = ll_market - ll_our  →  正なら我々が正確")
    print("=" * 112)
    print(f"{'セグメント':<20}{'窓':<6}{'車数':>8}{'ll_our':>9}{'ll_mkt':>9}{'Δll':>9}"
          f"{'t値':>8}{'br_our':>9}{'br_mkt':>9}")
    for dim, _ in dims:
        segs = sorted({k[2] for k in m1 if k[1] == dim})
        for seg in segs:
            for w in ("TRAIN", "TEST"):
                a = m1.get((w, dim, seg))
                if not a or a["n"] < MIN_SEG * 7:
                    continue
                n = a["n"]
                md = a["d"] / n
                var = max(a["d2"] / n - md * md, 0.0)
                t = md / math.sqrt(var / n) if var > 0 else 0.0
                mark = "  ★我々優位" if md > 0 and t > 3 else ""
                print(f"{seg if w == 'TRAIN' else '':<20}{w:<6}{n:>8}"
                      f"{a['ll_our']/n:>9.4f}{a['ll_mkt']/n:>9.4f}{md:>+9.4f}{t:>+8.2f}"
                      f"{a['br_our']/n:>9.4f}{a['br_mkt']/n:>9.4f}{mark}")
        print()

    # ---------------- 問2 ----------------
    print("\n" + "=" * 112)
    print("問2【本題・軸選定の直接対決】我々の軸と市場の軸が食い違ったレースのみ")
    print("     我々の軸3着内率 > 市場の軸3着内率 なら軸選定にエッジあり")
    print("=" * 112)
    print(f"{'セグメント':<20}{'窓':<6}{'n':>7}{'我々軸3着内%':>13}{'市場軸3着内%':>13}"
          f"{'差pt':>8}{'t値':>8}{'我々軸の市場評価':>16}")
    for dim, _ in dims:
        segs = sorted({k[2] for k in m2 if k[1] == dim})
        for seg in segs:
            for w in ("TRAIN", "TEST"):
                a = m2.get((w, dim, seg))
                if not a or a["n"] < MIN_SEG:
                    continue
                n = a["n"]
                oh, mh = a["our_hit"] / n * 100, a["mkt_hit"] / n * 100
                md = a["d"] / n
                var = max(a["d2"] / n - md * md, 0.0)
                t = md / math.sqrt(var / n) if var > 0 else 0.0
                mark = "  ★我々優位" if md > 0 and t > 3 else ""
                print(f"{seg if w == 'TRAIN' else '':<20}{w:<6}{n:>7}{oh:>13.1f}{mh:>13.1f}"
                      f"{(oh-mh):>+8.1f}{t:>+8.2f}{a['our_mp']/n*100:>15.1f}%{mark}")
        print()

    # ---------------- 問3 ----------------
    print("\n" + "=" * 112)
    print("問3【軸候補w1の較正】我々・市場・実測の3着内率")
    print("     our > 実測 なら我々が軸を過大評価している（軸信頼の判断自体が誤り）")
    print("=" * 112)
    print(f"{'セグメント':<20}{'窓':<6}{'n':>7}{'our予測%':>10}{'市場%':>10}{'実測%':>10}"
          f"{'our誤差pt':>11}{'市場誤差pt':>11}")
    for dim, _ in dims:
        segs = sorted({k[2] for k in m3 if k[1] == dim})
        for seg in segs:
            for w in ("TRAIN", "TEST"):
                a = m3.get((w, dim, seg))
                if not a or a["n"] < MIN_SEG:
                    continue
                n = a["n"]
                op, mp, ac = a["our_p"] / n * 100, a["mkt_p"] / n * 100, a["act"] / n * 100
                print(f"{seg if w == 'TRAIN' else '':<20}{w:<6}{n:>7}{op:>10.1f}{mp:>10.1f}"
                      f"{ac:>10.1f}{(op-ac):>+11.1f}{(mp-ac):>+11.1f}")
        print()


if __name__ == "__main__":
    main()
