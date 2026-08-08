"""9H1 候補を「その年に実際に運用していたら」の形で月次集計する。

## 何を再現しているか

[[keirin_9car_upset_bets_2026_08_08]] ③ の**オッズ非依存**の構成（朝の入稿で完結する側）。

    母集団 : 9車ちょうど・三連単の板がそろっているレース
    選別   : 6/7/9車の統合学習で作った波乱スコアの上位 SEL_Q
             （スコアは半年ごとの walk-forward = そのレースより前のデータだけで学習）
    買い目 : 1着 = モデル3着内率 k位 の1車固定
             2着 = モデル上位 m2 車 / 3着 = モデル上位 m3 車   → m2×(m3-1) 点
    予算   : 1レース 10,000円 を点数で等分（100円単位・切り捨て）

## 🔴 選別の閾値は「対象年より前」のスコア分布で固定する

全期間の分位で切ると、**その年の結果を見てから閾値を決めた**ことになる。
本スクリプトは `--year` の**前年末までのスコア分布**から閾値を取り、
対象年へはそのまま当てる。＝「年初にその閾値で運用を始めていたら」の集計。

⚠️ 最終オッズで採点している。朝の板で買うと下振れするため、実運用の払戻は
これより低く出る（[[keirin_highpay_tail_mispricing_2026_08_08]]）。
また落車・失格絡みの買い目も購入のまま外れ計上する実精算方式に合わせてある
（欠車があったレースは母集団から外れる＝9行そろわないため）。

## 使い方

    .venv/bin/python scripts/exp_9h1_monthly.py --year 2025
    .venv/bin/python scripts/exp_9h1_monthly.py --year 2025 --k 7 --m2 3 --m3 4

DB へは書き込まない。
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_9car_upset_bets import _boards, _rank_orders  # noqa: E402
from exp_pooled_upset_screen import (  # noqa: E402
    MODEL_COLS, _load, _make_target, _walk_forward,
)

RACE_BUDGET = 10_000
UNIT = 100
BIG = 300_000


def _legs(order: list[int], k: int, m2: int, m3: int) -> list[str]:
    if not order or k > len(order):
        return []
    lead = order[k - 1]
    rest = [f for f in order if f != lead]
    return [f"{lead}-{a}-{b}" for a in rest[:m2] for b in rest[:m3] if b != a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2025",
                    help="集計する年（カンマ区切り）。データを1回だけ読んで使い回す")
    ap.add_argument("--pool", default="6,7,9")
    ap.add_argument("--eval-ne", type=int, default=9)
    ap.add_argument("--sel-q", type=float, default=0.2)
    ap.add_argument("--target-q", type=float, default=0.25)
    ap.add_argument("--a-thr", type=float, default=300.0)
    ap.add_argument("--k", type=int, default=5, help="1着固定に使うモデル3着内率の順位")
    ap.add_argument("--m2", type=int, default=2, help="2着に使う上位車数")
    ap.add_argument("--m3", type=int, default=4, help="3着に使う上位車数")
    args = ap.parse_args()

    rows = _load([int(x) for x in args.pool.split(",")], args.a_thr, 50.0)
    drop = ("race_key", "date", "win_odds", "impA", "impB")
    cols = [c for c in rows[0] if c not in drop and c not in MODEL_COLS]
    s = _walk_forward(rows, cols, _make_target(rows, "quantile", args.a_thr, args.target_q))

    ne = np.array([r["n_entries"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    ok = (~np.isnan(s)) & (ne == args.eval_ne)

    boards = _boards(args.eval_ne)
    orders = _rank_orders(args.eval_ne)
    for y in [int(x) for x in args.years.split(",")]:
        _year_report(y, rows, s, ok, dates, boards, orders, args)


def _year_report(year, rows, s, ok, dates, boards, orders, args) -> None:
    """1年ぶんの月次集計。

    🔴 選別の閾値は**対象年より前**のスコア分布から取り、対象年へはそのまま当てる。
    ＝「年初にその閾値で運用を始めていたら」の集計になる。

    ⚠️ 2024年だけは前がない（walk-forward の最初の fold が 2024H1 で、それ以前は
       スコアが存在しない）。その場合のみ**対象年自身のスコア分布**から閾値を取り、
       見出しに明記する。閾値は「モデルスコアの分位」であって結果を見ていないので
       着順の先読みにはならないが、当時知り得なかった情報ではある。
    """
    prior = ok & (dates < f"{year}-01-01")
    if prior.sum() >= 200:
        thr = np.nanquantile(s[prior], 1 - args.sel_q)
        src = f"{year}年より前 {prior.sum()}R"
    else:
        own = ok & (dates >= f"{year}-01-01") & (dates < f"{year + 1}-01-01")
        if own.sum() < 100:
            print(f"\n===== {year}年: データ不足 =====")
            return
        thr = np.nanquantile(s[own], 1 - args.sel_q)
        src = f"⚠️ {year}年自身 {own.sum()}R（前年のスコアが無いため）"

    n_legs_nominal = args.m2 * (args.m3 - 1)
    print(f"\n===== {year}年 / 1着=モデル3着内率{args.k}位固定 × 2着上位{args.m2}車 "
          f"× 3着上位{args.m3}車 = {n_legs_nominal}点 "
          f"({RACE_BUDGET // n_legs_nominal // UNIT * UNIT:,}円/点) =====")
    print(f"選別閾値 = {src} のスコア上位{args.sel_q:.0%}点（{thr:.4f}）")

    months: dict[str, dict] = collections.defaultdict(
        lambda: {"pop": 0, "n": 0, "bet": 0, "hit": 0, "pay": 0, "max": 0,
                 "big": 0, "o500": 0})
    detail: list[tuple] = []
    for i, r in enumerate(rows):
        if not ok[i] or not r["date"].startswith(str(year)):
            continue
        months[r["date"][:7]]["pop"] += 1        # その月の母集団（9車・板あり）
        if s[i] < thr:
            continue
        b = boards.get(r["race_key"])
        if not b:
            continue
        legs = [x for x in _legs(orders.get(r["race_key"], []), args.k, args.m2, args.m3)
                if x in b["board"]]
        if not legs:
            continue
        stake = RACE_BUDGET // len(legs) // UNIT * UNIT
        if stake < UNIT:
            continue
        m = months[r["date"][:7]]
        m["n"] += 1
        m["bet"] += stake * len(legs)
        if b["win"] in legs:
            pay = int(stake * b["board"][b["win"]])
            m["hit"] += 1
            m["pay"] += pay
            m["max"] = max(m["max"], pay)
            m["big"] += int(pay >= BIG)
            m["o500"] += int(b["board"][b["win"]] >= 500)
            detail.append((r["date"], r["race_key"], b["win"],
                           b["board"][b["win"]], pay))

    print(f"{'月':<8}{'母集団':>6}{'対象R':>6}{'購入額':>11}{'的中':>5}{'的中率':>8}"
          f"{'払戻合計':>12}{'回収率':>8}{'最高払戻':>11}{'30万+':>6}{'500倍+':>7}")
    tot = {"pop": 0, "n": 0, "bet": 0, "hit": 0, "pay": 0, "max": 0, "big": 0, "o500": 0}
    for mo in sorted(months):
        v = months[mo]
        for kk in tot:
            tot[kk] = max(tot[kk], v[kk]) if kk == "max" else tot[kk] + v[kk]
        print(f"{mo:<8}{v['pop']:>6}{v['n']:>6}{v['bet']:>11,}{v['hit']:>5}"
              f"{v['hit']/v['n']*100 if v['n'] else 0:>7.1f}%"
              f"{v['pay']:>12,}{v['pay']/v['bet']*100 if v['bet'] else 0:>7.1f}%"
              f"{v['max']:>11,}{v['big']:>6}{v['o500']:>7}")
    print("-" * 82)
    print(f"{'合計':<8}{tot['pop']:>6}{tot['n']:>6}{tot['bet']:>11,}{tot['hit']:>5}"
          f"{tot['hit']/tot['n']*100 if tot['n'] else 0:>7.1f}%"
          f"{tot['pay']:>12,}{tot['pay']/tot['bet']*100 if tot['bet'] else 0:>7.1f}%"
          f"{tot['max']:>11,}{tot['big']:>6}{tot['o500']:>7}")
    n_mo = len(months) or 1
    print(f"収支 = {tot['pay'] - tot['bet']:+,}円（{tot['n']}レース・"
          f"{tot['n']/n_mo:.1f}件/月・月次100%超 "
          f"{sum(1 for v in months.values() if v['bet'] and v['pay'] >= v['bet'])}/{n_mo}）")

    # 平均は1本の万車券に支配されるので、上位k本を除いた回収率を必ず併記する
    pays = sorted((d[4] for d in detail), reverse=True)
    for k in (1, 2, 3):
        if len(pays) > k and tot["bet"]:
            rest = sum(pays[k:])
            print(f"  除・上{k}本: 回収率 {rest/tot['bet']*100:5.1f}%  "
                  f"収支 {rest - tot['bet']:+,}円")

    if detail:
        print("  --- 的中したレース ---")
        for d, rk, combo, odds, pay in sorted(detail):
            print(f"    {d}  {rk:<18} {combo:<8} {odds:>8.1f}倍  {pay:>9,}円")


if __name__ == "__main__":
    main()
