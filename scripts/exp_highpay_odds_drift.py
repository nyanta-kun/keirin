#!/usr/bin/env python3
"""朝オッズ→最終オッズのドリフトが「30N倍ちょうど狙い」を壊さないか監査する。

## なぜ致命的になりうるか

`exp_highpay_trifecta_design.py` の結論は「要求ライン 30N 倍に**張り付ける**のが唯一の
レバー」だった。実際に選ばれる目は最終オッズ 31〜32倍。ところが本番の入稿は朝で、
朝 32 倍の目が最終 28 倍まで下がれば払い戻しは 28万円で **30万円に届かない**。

つまりこの戦略の成否は**ドリフトの下振れ分布**が決める。安全マージン m を掛けて
`o_morning >= 30N * m` で選べば取りこぼしは減るが、要求オッズが上がるので
頻度（= Σ 0.75/o）はその分だけ下がる。**m の最適点を実測で決める**のが本スクリプト。

対象: `wt_odds_snapshot` に morning の三連単がある期間（2026-06-08〜）。

DB は読み取りのみ。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

STAKE = 10_000
HIGHPAY = 300_000


def main() -> None:
    with get_connection() as c:
        q = """
        WITH res AS (
          SELECT race_key,
                 MAX(CASE WHEN finish_order = 1 THEN frame_no END) AS f1,
                 MAX(CASE WHEN finish_order = 2 THEN frame_no END) AS f2,
                 MAX(CASE WHEN finish_order = 3 THEN frame_no END) AS f3
          FROM keirin.wt_entries GROUP BY race_key
        )
        SELECT r.race_key, r.race_date, res.f1, res.f2, res.f3
        FROM res JOIN keirin.wt_races r ON r.race_key = res.race_key
        WHERE r.n_entries = 7 AND r.race_date >= '2026-06-08'
          AND res.f1 IS NOT NULL AND res.f2 IS NOT NULL AND res.f3 IS NOT NULL
        """
        races = {r["race_key"]: f"{int(r['f1'])}-{int(r['f2'])}-{int(r['f3'])}"
                 for r in c.execute(q).fetchall()}
        keys = sorted(races)
        print(f"対象 {len(keys)} レース（7車・2026-06-08〜）", flush=True)

        morning: dict[str, dict[str, float]] = defaultdict(dict)
        final: dict[str, dict[str, float]] = defaultdict(dict)
        for i in range(0, len(keys), 200):
            ch = keys[i:i + 200]
            ph = ",".join("?" * len(ch))
            for row in c.execute(
                    "SELECT race_key, combination, odds_value FROM keirin.wt_odds_snapshot "
                    f"WHERE bet_type='trifecta' AND snapshot_type='morning' "
                    f"AND race_key IN ({ph}) AND odds_value > 0", ch).fetchall():
                morning[row["race_key"]][row["combination"]] = float(row["odds_value"])
            for row in c.execute(
                    "SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                    f"WHERE bet_type='trifecta' AND race_key IN ({ph}) "
                    "AND odds_value > 0", ch).fetchall():
                final[row["race_key"]][row["combination"]] = float(row["odds_value"])

    common = [k for k in keys if morning.get(k) and final.get(k)]
    print(f"朝・最終ともオッズあり: {len(common)} レース\n")

    # --- ドリフト全体像（朝 30〜60倍の目） ---
    ratios = []
    for k in common:
        for comb, mo in morning[k].items():
            if 30 <= mo <= 60 and comb in final[k]:
                ratios.append(final[k][comb] / mo)
    ratios = np.array(ratios)
    print(f"=== 朝30〜60倍の目のドリフト（最終/朝）  n={len(ratios):,} ===")
    for p in (1, 5, 10, 25, 50, 75, 90, 99):
        print(f"  p{p:<3} {np.percentile(ratios, p):.3f}")
    print(f"  平均 {ratios.mean():.3f} / 下振れ(<1.0) {np.mean(ratios < 1.0) * 100:.1f}% "
          f"/ 0.90未満 {np.mean(ratios < 0.90) * 100:.1f}% "
          f"/ 0.80未満 {np.mean(ratios < 0.80) * 100:.1f}%\n")

    # --- 安全マージン m の掃引（N=1・朝オッズで選び、最終オッズで精算） ---
    print("=== 安全マージン掃引: 朝オッズ >= 30*m の最低オッズ目を1点1万円 ===")
    print("   m     選定閾値  レース   的中%  高額%  高額数  取りこぼし%  ROI%   平均最終倍率")
    for m in (1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00):
        thr = 30.0 * m
        n_race = n_hit = n_big = 0
        ret = 0.0
        finals = []
        for k in common:
            elig = [(comb, mo) for comb, mo in morning[k].items() if mo >= thr]
            if not elig:
                continue
            comb, mo = min(elig, key=lambda x: x[1])
            fo = final[k].get(comb)
            if fo is None:
                continue
            n_race += 1
            finals.append(fo)
            if comb == races[k]:
                n_hit += 1
                pay = STAKE * fo
                ret += pay
                if pay >= HIGHPAY:
                    n_big += 1
        if not n_race:
            continue
        miss = (n_hit - n_big) / n_hit * 100 if n_hit else 0.0
        print(f"  {m:.2f}  {thr:7.1f}倍  {n_race:6}  {n_hit / n_race * 100:6.2f} "
              f"{n_big / n_race * 100:6.2f}  {n_big:5}   {miss:8.1f}   "
              f"{ret / (n_race * STAKE) * 100:6.1f}   {np.mean(finals):8.1f}")

    print("\n※ 取りこぼし% = 的中したのに最終オッズが30倍を割って30万円に届かなかった割合")


if __name__ == "__main__":
    main()
