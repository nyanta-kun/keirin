"""【軸選定の公平な直接対決】our軸の選び方を3通り試して市場軸と比較する（2026-07-30）。

`exp_axis_marginal_market_edge.py` の問2では
  our軸  = pred_win_pct 最上位（単勝確率）
  市場軸 = 市場の3着内周辺確率 最上位
を「3着内に入ったか」で判定していた。判定指標が3着内なのに我々だけ単勝確率で
選んでいるため、我々に不利な非対称性がある。本スクリプトは基準を揃えて再検証する。

our軸の候補（3通り）:
  A: pred_win_pct 最上位     （元の定義・参考）
  B: pred_top3_pct 最上位    （★市場と同一基準＝公平な比較）
  C: 記者◎                   （人間の印を軸にした場合）

市場軸 = 三連複オッズから周辺化した3着内確率の最上位（すべて共通）。

各定義について「我々の軸 ≠ 市場の軸」のレースのみに絞り、実際に3着内に
入った率を比較する。差が正なら軸選定にエッジあり（かつ我々の軸は市場評価が
低いのでオッズが高い＝ROIの源泉になる）。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection  # noqa: F401  (loaders が使用)

from exp_axis_marginal_market_edge import marginals  # noqa: E402
from exp_segment_market_edge import (  # noqa: E402
    TRAIN_TO, build_rows, load_entries, load_races, load_trio_odds,
    pattern_of, q_label, quartile_cuts,
)

MIN_SEG = 120


def main():
    races = load_races()
    entries = load_entries(races.keys())
    rows = build_rows(races, entries)
    del entries

    train_rows = [r for r in rows if r["race_date"] <= TRAIN_TO]
    tie_thr = quartile_cuts([max(r["g12"], r["g23"], r["g34"]) for r in train_rows])[0]
    chalk_cuts = quartile_cuts([r["top3_sum_top2"] for r in train_rows])

    # (defn, window, dim, seg) -> acc
    acc = defaultdict(lambda: {"n": 0, "our": 0, "mkt": 0, "d": 0, "d2": 0,
                               "our_mp": 0.0, "mkt_mp": 0.0})
    agree = defaultdict(lambda: [0, 0])   # defn -> [agree, total]

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
            w = "TRAIN" if r["race_date"] <= TRAIN_TO else "TEST"
            tm = r["top3_mask"]
            mkt_axis = max(mk, key=lambda f: mk[f])

            # 記者◎
            honmei = None
            for f, e in r["by_frame"].items():
                pm = e["prediction_mark"]
                if pm is not None and str(pm).strip() == "1":
                    honmei = f
                    break

            cands = {"A:pred_win 1位": r["win_order"][0],
                     "B:pred_top3 1位": r["top3_order"][0]}
            if honmei is not None:
                cands["C:記者◎"] = honmei

            pat = pattern_of(r, tie_thr)
            chalk = q_label(r["top3_sum_top2"], chalk_cuts, "硬さ")

            for defn, ax in cands.items():
                agree[(defn, w)][1] += 1
                if ax == mkt_axis:
                    agree[(defn, w)][0] += 1
                    continue
                oh = 1 if (tm >> ax) & 1 else 0
                mh = 1 if (tm >> mkt_axis) & 1 else 0
                for dim, seg in (("ALL", "全体"), ("pattern", pat), ("chalk", chalk)):
                    a = acc[(defn, w, dim, seg)]
                    a["n"] += 1
                    a["our"] += oh
                    a["mkt"] += mh
                    a["d"] += oh - mh
                    a["d2"] += (oh - mh) ** 2
                    a["our_mp"] += mk[ax]
                    a["mkt_mp"] += mk[mkt_axis]
        print(f"  {ym}: {len(chunk)}R", flush=True)

    print("\n" + "=" * 116)
    print("軸選定の公平な直接対決（我々の軸 ≠ 市場の軸 のレースのみ）")
    print("  差pt = 我々軸3着内% - 市場軸3着内%  →  正なら軸選定にエッジあり")
    print("  『我々軸の市場評価%』と『我々軸の実測%』が一致していれば、")
    print("  市場は我々の選択を正しく値付けしている（＝我々の乖離に価値がない）")
    print("=" * 116)

    for defn in ("A:pred_win 1位", "B:pred_top3 1位", "C:記者◎"):
        print("\n" + "-" * 116)
        ag = {w: agree.get((defn, w), [0, 0]) for w in ("TRAIN", "TEST")}
        print(f"【{defn}】 市場軸との一致率: "
              + " / ".join(f"{w} {(v[0]/v[1]*100 if v[1] else 0):.1f}% ({v[0]}/{v[1]})"
                           for w, v in ag.items()))
        print("-" * 116)
        print(f"{'セグメント':<18}{'窓':<6}{'n':>7}{'我々軸3着内%':>13}{'市場軸3着内%':>13}"
              f"{'差pt':>8}{'t値':>8}{'我々軸の市場評価%':>18}{'較正差pt':>10}")
        for dim in ("ALL", "pattern", "chalk"):
            segs = sorted({k[3] for k in acc if k[0] == defn and k[2] == dim})
            for seg in segs:
                for w in ("TRAIN", "TEST"):
                    a = acc.get((defn, w, dim, seg))
                    if not a or a["n"] < MIN_SEG:
                        continue
                    n = a["n"]
                    oh, mh = a["our"] / n * 100, a["mkt"] / n * 100
                    md = a["d"] / n
                    var = max(a["d2"] / n - md * md, 0.0)
                    t = md / math.sqrt(var / n) if var > 0 else 0.0
                    mp = a["our_mp"] / n * 100
                    flag = "  ★我々優位" if md > 0 and t > 3 else ""
                    print(f"{seg if w == 'TRAIN' else '':<18}{w:<6}{n:>7}{oh:>13.1f}"
                          f"{mh:>13.1f}{md*100:>+8.1f}{t:>+8.2f}{mp:>18.1f}"
                          f"{(mp-oh):>+10.1f}{flag}")
            print()


if __name__ == "__main__":
    main()
