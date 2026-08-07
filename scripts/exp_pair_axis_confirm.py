"""二軸ペアモデルの確認: 差の信頼区間・ROI・配当・セグメント安定性（2026-08-07）。

exp_pair_axis_model_v2.py が出した「現行3ヘッド軸 53.48% → ペアモデル 54.84%」の
+1.35pt が、
  ① レース単位 paired bootstrap で 0 を跨がないか
  ② 的中率を上げた分だけ配当が落ちて ROI が悪化していないか（＝ただの堅い方への
     移動になっていないか）
  ③ 裾依存でないか（上位k本の的中を除いても差が残るか）
  ④ セグメント（種別・グレード・分戦数）で符号が揃うか
を確認する。⚠️ 読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import RANK_AXIS2_BAD_WEIGHT, _race_zscore  # noqa: E402

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
SCORED = REPO / "data" / "exp_cache" / "pair_axis_scored_v2.pkl"
TEST_START = "2025-07-01"
STAKE = 100


def main() -> None:
    races = {r["rk"]: r for r in pickle.load(open(DETAIL, "rb"))}
    te = pd.read_pickle(SCORED)
    pick = te.loc[te.groupby("rk").s_bin.idxmax()][["rk", "date", "hi", "lo", "label"]]

    rows = []
    for _, p in pick.iterrows():
        r = races[p.rk]
        p3, pw, pb, board = r["p3"], r["pw"], r["pb"], r["board"]
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        cars = sorted(p3)
        # 現行3ヘッド軸
        a1 = max(pw, key=lambda k: pw[k])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in p3}
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        odds = board[top3]
        ret = round(odds * 100) // 10 * 10

        def legs(x, y):
            return [frozenset({x, y, z}) for z in cars if z not in (x, y)
                    and frozenset({x, y, z}) in board]

        lb, lp = legs(a1, a2), legs(int(p.hi), int(p.lo))
        if not lb or not lp:
            continue
        hb, hp = int(top3 in lb), int(top3 in lp)
        rows.append(dict(
            rk=p.rk, date=p.date,
            hit_base=hb, bet_base=len(lb) * STAKE, ret_base=ret if hb else 0,
            hit_pair=hp, bet_pair=len(lp) * STAKE, ret_pair=ret if hp else 0,
            odds=odds, same_pick=int({a1, a2} == {int(p.hi), int(p.lo)}),
        ))
    d = pd.DataFrame(rows)
    n = len(d)
    print(f"評価対象 {n:,}R（{d.date.nunique()}日 = {n/d.date.nunique():.1f}件/日）")
    print(f"軸2車が現行と一致したレース: {100*d.same_pick.mean():.1f}%")

    hb, hp = d.hit_base.mean(), d.hit_pair.mean()
    rb = d.ret_base.sum() / d.bet_base.sum()
    rp_ = d.ret_pair.sum() / d.bet_pair.sum()
    print(f"\n{'':16s} {'的中%':>7s} {'ROI%':>7s} {'的中時平均配当':>12s} {'的中/日':>8s}")
    for tag, h, roi, m in (("現行3ヘッド軸", hb, rb, d[d.hit_base == 1].odds.mean()),
                           ("ペアモデル", hp, rp_, d[d.hit_pair == 1].odds.mean())):
        print(f"{tag:16s} {100*h:7.2f} {100*roi:7.2f} {m:12.2f} "
              f"{h*n/d.date.nunique():8.1f}")

    # ① paired bootstrap（レース単位・差）
    diff = (d.hit_pair - d.hit_base).values
    rng = np.random.default_rng(42)
    bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n① 的中率の差 {100*diff.mean():+.2f}pt  95%CI [{100*lo:+.2f}, {100*hi:+.2f}]  "
          f"P(差>0)={100*(bs > 0).mean():.1f}%")

    # ROI 差も
    dr = (d.ret_pair - d.ret_base).values
    dbet = (d.bet_pair - d.bet_base).values
    print(f"   ROI 差 {100*(rp_-rb):+.2f}pt "
          f"（購入額の差は1レースあたり平均 {dbet.mean():+.1f}円）")

    # ③ 裾依存: 高配当的中を除いても差が残るか
    print("\n③ 高配当的中の除外耐性（配当上位k本を両方式から除く）")
    for k in (0, 5, 20, 50):
        dd = d.sort_values("odds", ascending=False).iloc[k:]
        print(f"   上位{k:3d}本除外: 現行 {100*dd.hit_base.mean():.2f}% → "
              f"ペア {100*dd.hit_pair.mean():.2f}% "
              f"({100*(dd.hit_pair.mean()-dd.hit_base.mean()):+.2f}pt) / "
              f"ROI {100*dd.ret_base.sum()/dd.bet_base.sum():.1f}% → "
              f"{100*dd.ret_pair.sum()/dd.bet_pair.sum():.1f}%")

    # ④ セグメント
    meta = pd.DataFrame([
        dict(rk=k, month=v["date"][:7]) for k, v in races.items()])
    d2 = d.merge(meta, on="rk", how="left")
    print("\n④ 四半期別")
    d2["q"] = pd.PeriodIndex(pd.to_datetime(d2.date), freq="Q").astype(str)
    for q, g in d2.groupby("q"):
        print(f"   {q}  n={len(g):5,d}  {100*g.hit_base.mean():5.2f}% → "
              f"{100*g.hit_pair.mean():5.2f}%  "
              f"({100*(g.hit_pair.mean()-g.hit_base.mean()):+.2f}pt)")


if __name__ == "__main__":
    main()
