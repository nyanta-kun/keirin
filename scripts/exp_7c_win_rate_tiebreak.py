"""7C: 複勝率が僅差のとき単勝率で軸2を決めるべきか（2026-08-07）。

ユーザー指摘:
  「3着内率のみでは差がないが、**単勝率の差**が的中率に影響あればそちらでの
    判断として下さい」

確定仕様（軸=p3上位2車 / 相手=p3>=15% / 相手4点以上 / 予算10,000円）の上で:

  【A】複勝率が僅差の帯で、2位と3位のどちらが実際に3着内へ入るか
       ＝ 単勝率が高い方を選ぶと当たりやすいのか（純粋な弁別力）
  【B】軸2の選び方を変える:
       B0 現行     : p3 上位2車
       B1 tiebreak : p3 の2位3位が僅差なら **pw の高い方**を軸2にする（閾値掃引）
       B2 blend    : レース内 z(p3) + w·z(pw) の上位2車（w 掃引）
       B3 pw優先   : 軸1=p3最上位、軸2=残りの pw 最上位
  【C】選別条件として **pw の差**（軸2と3位、または軸1と軸2）を足す価値があるか
  【D】3ヘッド軸（本番 7S 系: 軸1=pw最上位 / 軸2=z(p3)−0.3z(pb)）との比較

評価は常に二軸的中率（＝三連複2軸流しの的中率）と ROI。
⚠️ オッズは精算にのみ使う。読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN, RANK_AXIS2_BAD_WEIGHT, _race_zscore,
    rank_7c_select_legs, rank_7c_unit_stake,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
ALL_PER_DAY = 76.3


def settle(a1, a2, p3, board, top3):
    """軸2車を決めた後の 7C 買い目で精算する（相手足切り・予算枠は共通）。"""
    others = sorted(set(p3) - {a1, a2})
    legs = rank_7c_select_legs(others, p3)
    combos = [frozenset({a1, a2, t}) for t in legs
              if frozenset({a1, a2, t}) in board]
    if len(combos) < RANK_7C_LEGS_MIN:
        return None
    k = len(combos)
    s = rank_7c_unit_stake(k)
    hit = top3 in combos
    od = board[top3]
    return dict(k=k, bet=k * s, ret=(round(od * s) // 10 * 10) if hit else 0,
                hit=int(hit), low=int(hit and od <= 2.0))


def build(races) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, pw, pb, board = r["p3"], r["pw"], r["pb"], r["board"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        c1, c2, c3 = ranked[0], ranked[1], ranked[2]
        if p3[c1] + p3[c2] < RANK_7C_P3_SUM_MIN:
            continue
        base = settle(c1, c2, p3, board, top3)
        if base is None:
            continue

        rec = dict(rk=r["rk"], date=r["date"],
                   win="確認" if r["date"] <= CONFIRM_END else "掃引",
                   sum2=p3[c1] + p3[c2], gap23=p3[c2] - p3[c3],
                   pw2=pw[c2], pw3=pw[c3], pw_gap23=pw[c2] - pw[c3],
                   pw_gap12=pw[c1] - pw[c2],
                   c2_in=int(c2 in top3), c3_in=int(c3 in top3))
        for k, v in base.items():
            rec[f"B0_{k}"] = v

        # B1 tiebreak: p3 が僅差なら pw の高い方を軸2にする
        for th in (0.02, 0.05, 0.10, 0.15):
            a2 = c3 if (p3[c2] - p3[c3] < th and pw[c3] > pw[c2]) else c2
            g = settle(c1, a2, p3, board, top3)
            if g:
                for k, v in g.items():
                    rec[f"B1t{int(th*100):02d}_{k}"] = v
                rec[f"B1t{int(th*100):02d}_swap"] = int(a2 != c2)

        # B2 blend: z(p3) + w·z(pw) の上位2車
        zp, zw = _race_zscore(p3), _race_zscore(pw)
        for w in (0.1, 0.2, 0.3, 0.5):
            sc = {f: zp[f] + w * zw[f] for f in p3}
            o = sorted(sc, key=lambda f: (-sc[f], f))
            g = settle(o[0], o[1], p3, board, top3)
            if g:
                for k, v in g.items():
                    rec[f"B2w{int(w*10)}_{k}"] = v
                rec[f"B2w{int(w*10)}_swap"] = int({o[0], o[1]} != {c1, c2})

        # B3 軸1=p3最上位 / 軸2=残りの pw 最上位
        a2 = max((f for f in p3 if f != c1), key=lambda f: (pw[f], -f))
        g = settle(c1, a2, p3, board, top3)
        if g:
            for k, v in g.items():
                rec[f"B3_{k}"] = v
            rec["B3_swap"] = int(a2 != c2)

        # D 3ヘッド軸（本番 7S 系）
        h1 = max(pw, key=lambda f: pw[f])
        zb = _race_zscore(pb)
        sc = {f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f] for f in p3}
        h2 = max((f for f in sc if f != h1), key=lambda f: sc[f])
        g = settle(h1, h2, p3, board, top3)
        if g:
            for k, v in g.items():
                rec[f"D_{k}"] = v
            rec["D_swap"] = int({h1, h2} != {c1, c2})
        rows.append(rec)
    return pd.DataFrame(rows)


VARIANTS = [
    ("B0 現行（複勝率上位2車）", "B0"),
    ("B1 僅差2pt→単勝で入替", "B1t02"),
    ("B1 僅差5pt→単勝で入替", "B1t05"),
    ("B1 僅差10pt→単勝で入替", "B1t10"),
    ("B1 僅差15pt→単勝で入替", "B1t15"),
    ("B2 z(複)+0.1z(単)", "B2w1"),
    ("B2 z(複)+0.2z(単)", "B2w2"),
    ("B2 z(複)+0.3z(単)", "B2w3"),
    ("B2 z(複)+0.5z(単)", "B2w5"),
    ("B3 軸2=単勝最上位", "B3"),
    ("D  3ヘッド軸(7S系)", "D"),
]


def report(d: pd.DataFrame, wname: str) -> None:
    days = d.date.nunique()
    print(f"\n=== {wname}  n={len(d):,}R / {days}日 ===")
    print(f"{'軸の決め方':26s} {'件/日':>6s} {'入替%':>6s} {'的中%':>6s} "
          f"{'的中/日':>6s} {'低配当/日':>7s} {'ROI%':>6s}")
    for lab, pre in VARIANTS:
        h = f"{pre}_hit"
        if h not in d.columns:
            continue
        g = d.dropna(subset=[h])
        sw = f"{pre}_swap"
        swp = f"{100*g[sw].mean():6.1f}" if sw in g.columns else "     —"
        print(f"{lab:26s} {len(g)/days:6.2f} {swp} {100*g[h].mean():6.2f} "
              f"{g[h].sum()/days:6.2f} {g[f'{pre}_low'].sum()/days:7.2f} "
              f"{100*g[f'{pre}_ret'].sum()/g[f'{pre}_bet'].sum():6.1f}")


def main() -> None:
    d = build(pickle.load(open(DETAIL, "rb")))
    ev, cf = d[d.win == "掃引"], d[d.win == "確認"]

    print("=== 【A】複勝率が僅差の帯で、単勝率は2位/3位を弁別できるか（評価窓）===")
    print("   対象: 複勝率2位と3位の差が小さいレース。単勝率の大小で層別し、")
    print("   実際に3着内へ入った率を比べる（弁別できるなら差が出るはず）")
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10)):
        g = ev[(ev.gap23 >= lo) & (ev.gap23 < hi)]
        if len(g) < 100:
            continue
        w2 = g[g.pw_gap23 > 0]     # 単勝率も2位の方が高い
        w3 = g[g.pw_gap23 <= 0]    # 単勝率は3位の方が高い（＝逆転している）
        print(f"\n  複勝率差 {lo:.0%}〜{hi:.0%}  n={len(g):,}")
        print(f"    単勝率も2位が上 (n={len(w2):5,d}): "
              f"複勝2位が3着内 {100*w2.c2_in.mean():5.2f}% / "
              f"3位が3着内 {100*w2.c3_in.mean():5.2f}%  差 "
              f"{100*(w2.c2_in.mean()-w2.c3_in.mean()):+5.2f}pt")
        if len(w3) >= 50:
            print(f"    単勝率は3位が上 (n={len(w3):5,d}): "
                  f"複勝2位が3着内 {100*w3.c2_in.mean():5.2f}% / "
                  f"3位が3着内 {100*w3.c3_in.mean():5.2f}%  差 "
                  f"{100*(w3.c2_in.mean()-w3.c3_in.mean()):+5.2f}pt")

    report(ev, "【B】【D】評価窓 2025-07〜2026-08")
    report(cf, "【B】【D】確認窓 2024-07〜2025-06")

    print("\n=== 【C】選別に単勝率の差を足す価値があるか（評価窓・件数は揃えない）===")
    days = ev.date.nunique()
    print(f"{'条件':26s} {'件/日':>6s} {'的中%':>6s} {'的中/日':>6s} "
          f"{'低配当/日':>7s} {'ROI%':>6s}")
    for lab, mask in (
        ("現行（条件なし）", ev.index == ev.index),
        ("軸1単勝−軸2単勝 >= 10pt", ev.pw_gap12 >= 0.10),
        ("軸1単勝−軸2単勝 >= 20pt", ev.pw_gap12 >= 0.20),
        ("軸2単勝−3位単勝 >= 0", ev.pw_gap23 >= 0.0),
        ("軸2単勝−3位単勝 >= 5pt", ev.pw_gap23 >= 0.05),
        ("軸2単勝−3位単勝 >= 10pt", ev.pw_gap23 >= 0.10),
    ):
        g = ev[mask]
        if len(g) < 200:
            continue
        print(f"{lab:26s} {len(g)/days:6.2f} {100*g.B0_hit.mean():6.2f} "
              f"{g.B0_hit.sum()/days:6.2f} {g.B0_low.sum()/days:7.2f} "
              f"{100*g.B0_ret.sum()/g.B0_bet.sum():6.1f}")


if __name__ == "__main__":
    main()
