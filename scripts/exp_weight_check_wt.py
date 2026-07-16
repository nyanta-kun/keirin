"""傾斜ベット可否の検証: ◎不一致レースで +EV(オッズ×確率>1) の集中先があるか（2026-07-15）

傾斜ベットのROI上限 = 集合中の最高EV買い目のEV。全買い目がEV<1なら傾斜しても<1。
→ ◎不一致OOSレースの二車単 our(モデル1位)→相手 を、
   ①相手別（◎/モデル2..6位）② (our→j)のオッズ帯別
   に分解し、EV(=ROI)>1 の的が存在するか確認する。

窓: OOS 2026-03-01〜2026-07-10（全て学習外）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import CACHE_DIR  # noqa: E402
from exp_overlap_order_wt import load_ordered, load_marks  # noqa: E402

PH = CACHE_DIR / "perhorse_n7.pkl"
OOS = ("2026-03-01", "2026-07-10")


def main():
    ph = pd.read_pickle(PH)
    ph = ph[(ph["race_date"] >= OOS[0]) & (ph["race_date"] <= OOS[1])]
    mark = load_marks(ph["race_key"].unique().tolist())
    exa, tri = load_ordered(ph["race_key"].unique().tolist())

    # 相手別ROI集計器: key=相手区分 -> [n, hit, bet, pay]
    by_partner = defaultdict(lambda: [0, 0, 0, 0])
    # (our->j) オッズ帯別
    band_edges = [0, 3, 5, 8, 12, 20, 40, 9999]
    by_band = defaultdict(lambda: [0, 0, 0, 0])
    n_races = 0
    our_win = 0

    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        frames = g["frame_no"].astype(int).tolist()
        fo = {int(f): (int(o) if pd.notna(o) else 99)
              for f, o in zip(frames, g["finish_order"])}
        pos = {v: k for k, v in fo.items()}
        if not all(p in pos for p in (1, 2, 3)):
            continue
        mk = mark.get(rk, {})
        honmei = next((f for f in frames if mk.get(f) == 1), None)
        if honmei is None:
            continue
        our = frames[0]
        if our == honmei:   # ◎不一致のみ
            continue
        er = exa.get(rk, {})
        if not er:
            continue
        n_races += 1
        if pos[1] == our:
            our_win += 1
        actual = (pos[1], pos[2])
        # 相手区分ラベル
        rank_of = {f: i for i, f in enumerate(frames)}  # モデル順位0始
        for j in frames:
            if j == our:
                continue
            od = er.get((our, j))
            if od is None:
                continue
            hit = 1 if actual == (our, j) else 0
            if j == honmei:
                lab = "相手=◎"
            else:
                lab = f"相手=モデル{rank_of[j]+1}位"
            a = by_partner[lab]
            a[0] += 1; a[2] += 100; a[1] += hit; a[3] += int(od * 100) * hit
            # オッズ帯
            for lo, hi in zip(band_edges[:-1], band_edges[1:]):
                if lo <= od < hi:
                    b = by_band[(lo, hi)]
                    b[0] += 1; b[2] += 100; b[1] += hit; b[3] += int(od * 100) * hit
                    break

    print(f"◎不一致 OOSレース: {n_races:,}  我々の頭の1着率={our_win/n_races:.1%}")
    print("\n=== 二車単 our→相手 の相手別 EV(=ROI) ===")
    print("（傾斜の集中先候補。EV>100%なら傾斜で黒字化可能）")
    order = ["相手=◎"] + [f"相手=モデル{i}位" for i in range(2, 8)]
    for lab in order:
        a = by_partner.get(lab)
        if a and a[0]:
            print(f"  {lab:<16} n={a[0]:>4} 的中率={a[1]/a[0]:5.1%} 平均配当={(a[3]/a[1]/100 if a[1] else 0):6.1f}倍 "
                  f"EV(ROI)={a[3]/a[2]:6.1%}")

    print("\n=== (our→相手) オッズ帯別 EV(=ROI) ===")
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        b = by_band.get((lo, hi))
        if b and b[0]:
            print(f"  [{lo:>4}-{hi:>4}倍) n={b[0]:>5} 的中率={b[1]/b[0]:5.1%} EV(ROI)={b[3]/b[2]:6.1%}")

    # 傾斜の理論上限 = 最高EVの相手/帯
    best_p = max((a[3]/a[2], lab) for lab, a in by_partner.items() if a[2])
    best_b = max((b[3]/b[2], k) for k, b in by_band.items() if b[2])
    print(f"\n傾斜ROI上限（最高EVの単一集中先）: 相手別={best_p[0]:.1%}({best_p[1]}) / "
          f"オッズ帯別={best_b[0]:.1%}({best_b[1]}倍)")


if __name__ == "__main__":
    main()
