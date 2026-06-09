"""7車立て 払戻分布 日毎集計（三連複/三連単・配当4区分）

実着順の払戻(オッズ×100)を <1000 / 1000-5000 / 5000-10000 / >=10000 に分類し、
指定期間の日毎レース数を集計する。7車戦の配当構造の把握用。

使い方:
  PYTHONPATH=. .venv/bin/python3 scripts/analyze_7car_payout_daily_wt.py [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--cars 7]
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt

BK = ["<1000", "1000-5000", "5000-10000", ">=10000"]


def bucket(p):
    return "<1000" if p < 1000 else "1000-5000" if p < 5000 else "5000-10000" if p < 10000 else ">=10000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="f", default="2026-06-01")
    ap.add_argument("--to", dest="t", default="2026-06-30")
    ap.add_argument("--cars", type=int, default=7, help="車立て(エントリ数)")
    args = ap.parse_args()

    with get_connection() as c:
        df = pd.read_sql(
            "SELECT e.race_key, r.race_date, e.frame_no, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key "
            "WHERE r.race_date>=? AND r.race_date<=?", c, params=(args.f, args.t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz == args.cars].index)]
    pm = _load_payouts_wt(list(df["race_key"].unique()))

    data = {"三連複": [], "三連単": []}
    for rk, g in df.groupby("race_key"):
        fin = g[g["finish_order"].between(1, 3)].sort_values("finish_order")
        if len(fin) < 3:
            continue
        order = tuple(int(x) for x in fin["frame_no"])
        d = g["race_date"].iloc[0]
        trio = pm.get(rk, {}).get(("trio", frozenset(order)))
        tri = pm.get(rk, {}).get(("trifecta", order))
        if trio:
            data["三連複"].append({"date": d, "bucket": bucket(trio), "pay": trio})
        if tri:
            data["三連単"].append({"date": d, "bucket": bucket(tri), "pay": tri})

    for name, rows in data.items():
        dd = pd.DataFrame(rows)
        ct = pd.crosstab(dd["date"], dd["bucket"]).reindex(columns=BK, fill_value=0)
        ct["計"] = ct.sum(axis=1)
        print(f"\n=== {args.cars}車立て {name} 払戻 日毎集計（{args.f}〜{args.t}・{len(dd)}R）===")
        print(ct.to_string())
        tot = ct[BK].sum()
        print(f"  合計: " + " / ".join(f"{b} {tot[b]}({tot[b]/tot.sum()*100:.0f}%)" for b in BK))
        print(f"  中央値 {dd['pay'].median():.0f}円 / 平均 {dd['pay'].mean():.0f}円")


if __name__ == "__main__":
    main()
