#!/usr/bin/env python3
"""高額払い戻し（1万円投資 → 30万円以上）の到達可能性を測る基礎センサス。

## 何を測るのか

ユーザー要望は「1レース1万円の購入に対し **30万円以上**の払い戻し」＝ **30倍以上**の
リターンを、券種・買い目構成を問わず狙うこと。ROI ではなく**裾の頻度**が目的関数。

N点を等分（1点あたり 10000/N 円）で買うとき、的中目のオッズを o とすると

    払い戻し = (10000 / N) * o        →  30万円以上 ⇔ **o >= 30 * N**

したがって「何点買うか」が要求オッズを線形に決める（5点なら150倍、12点なら360倍）。

## 理論的上限（先に押さえる）

1円あたりの期待回収は控除率により **最大0.75**（競輪の実測到達点でも0.85）。
高額イベントは1回あたり30倍以上を返すので

    P(高額) <= 0.85 / 30 = **2.83%**  ＝ 35レースに1回が絶対上限

しかもこれは「回収の100%が30倍超の的中だけから来る」場合。小さい的中を1円でも
拾えばその分だけ上限から遠ざかる。**狙うべきは最大配当ではなく 30N 倍をわずかに
超える帯**（オッズが要求ラインより高すぎると同じ回収でイベント数が減る）。

本スクリプトはこの理屈を実データで裏付ける:

1. 的中目（三連単・三連複）オッズの分布と P(o >= T)
2. オッズ帯ごとの「市場が織り込む確率質量」と「実測の的中率」の突合
   （＝帯を丸ごと買ったときの的中率と、その帯の点数）
3. 各構成 N 点における 30N 倍ラインの到達可能性

DB は読み取りのみ。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

DATE_FROM = "2024-01-01"
DATE_TO = "2026-08-04"


def _fetch_winning_odds() -> list[dict]:
    """各レースの的中目（三連単／三連複）の最終オッズを返す。"""
    q = """
    WITH res AS (
      SELECT race_key,
             MAX(CASE WHEN finish_order = 1 THEN frame_no END) AS f1,
             MAX(CASE WHEN finish_order = 2 THEN frame_no END) AS f2,
             MAX(CASE WHEN finish_order = 3 THEN frame_no END) AS f3
      FROM keirin.wt_entries
      GROUP BY race_key
    )
    SELECT r.race_key, r.race_date, r.n_entries, r.race_type,
           tf.odds_value AS tf_odds, tr.odds_value AS tr_odds
    FROM res
    JOIN keirin.wt_races r ON r.race_key = res.race_key
    LEFT JOIN keirin.wt_odds tf
      ON tf.race_key = res.race_key AND tf.bet_type = 'trifecta'
     AND tf.combination = res.f1 || '-' || res.f2 || '-' || res.f3
    LEFT JOIN keirin.wt_odds tr
      ON tr.race_key = res.race_key AND tr.bet_type = 'trio'
     AND tr.combination = LEAST(res.f1, res.f2, res.f3) || '='
                       || (res.f1 + res.f2 + res.f3
                           - LEAST(res.f1, res.f2, res.f3)
                           - GREATEST(res.f1, res.f2, res.f3)) || '='
                       || GREATEST(res.f1, res.f2, res.f3)
    WHERE r.race_date >= ? AND r.race_date <= ?
      AND res.f1 IS NOT NULL AND res.f2 IS NOT NULL AND res.f3 IS NOT NULL
      AND r.n_entries IN (7, 9)
    """
    with get_connection() as c:
        return [dict(r) for r in c.execute(q, (DATE_FROM, DATE_TO)).fetchall()]


def _pct(x: float) -> str:
    return f"{x * 100:6.3f}%"


def main() -> None:
    rows = _fetch_winning_odds()
    print(f"取得: {len(rows)} レース ({DATE_FROM}〜{DATE_TO})\n")

    for n_car in (7, 9):
        sub = [r for r in rows if r["n_entries"] == n_car]
        tf = sorted(float(r["tf_odds"]) for r in sub if r["tf_odds"])
        tr = sorted(float(r["tr_odds"]) for r in sub if r["tr_odds"])
        print(f"=== {n_car}車立て  レース {len(sub)} / 三連単オッズ有 {len(tf)} / "
              f"三連複オッズ有 {len(tr)} ===")

        for label, arr in (("三連単", tf), ("三連複", tr)):
            if not arr:
                continue
            def q(p: float) -> float:
                return arr[min(len(arr) - 1, int(len(arr) * p))]
            print(f"  {label} 的中目オッズ: "
                  f"中央 {q(0.5):8.1f} / p75 {q(0.75):8.1f} / p90 {q(0.90):9.1f} / "
                  f"p95 {q(0.95):9.1f} / p99 {q(0.99):9.1f}")
            # P(o >= 30N) を N=1..12 について
            line = "    30N倍 到達率: "
            for n_pt in (1, 2, 3, 5, 6, 8, 10, 12, 18, 24):
                thr = 30 * n_pt
                p = sum(1 for o in arr if o >= thr) / len(arr)
                line += f"N={n_pt}({thr:4.0f}x):{_pct(p)}  "
                if n_pt in (5, 10):
                    line += "\n                  "
            print(line)
        print()

    # --- オッズ帯ごとの実測的中率（＝その帯を丸ごと買ったときの命中率） ---
    print("=== 的中目オッズの帯別シェア（＝帯を全部買ったときの的中率の上限） ===")
    bands = [(0, 30), (30, 60), (60, 90), (90, 150), (150, 300), (300, 600),
             (600, 1200), (1200, 10 ** 9)]
    for n_car in (7, 9):
        for label, key in (("三連単", "tf_odds"), ("三連複", "tr_odds")):
            arr = [float(r[key]) for r in rows
                   if r["n_entries"] == n_car and r[key]]
            if not arr:
                continue
            print(f"  {n_car}車 {label} (n={len(arr)})")
            for lo, hi in bands:
                cnt = sum(1 for o in arr if lo <= o < hi)
                # その帯に的中がある確率 × その帯の平均オッズ ＝ 帯の回収寄与
                if cnt:
                    mean_o = sum(o for o in arr if lo <= o < hi) / cnt
                else:
                    mean_o = 0.0
                share = cnt / len(arr)
                print(f"    {lo:5}-{hi if hi < 10**9 else '∞':>6}倍: "
                      f"{share * 100:6.2f}%  平均{mean_o:8.1f}倍  "
                      f"回収寄与 {share * mean_o:6.3f}")
            print()


if __name__ == "__main__":
    main()
