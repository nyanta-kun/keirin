"""7C のガミ対策: 上限で見送るか / 相手を絞るか（2026-08-07）。

ユーザー指摘（岐阜2R 2026-08-07・複勝率 88.4+82.1=170.5% で三連複 ¥110）:
  「力差がはっきりしていると的中しても総流しでは大きくガミになるだけの場合がある。
    **相手を絞る**か**レースを見送る**のどちらが良いか。見送るなら母数が減るので
    当初25%に制限したものを**30%に広げる**のが良いかも併せて確認して」

ガミの定義: 三連複5点×100円=500円に対し払戻 < 500円（＝的中オッズ < 5.0倍）。
  「実質的中」= 的中 ∧ 払戻 >= 投資額（元返し以上）とする。

比較する設計:
  【見送り】選別を下限だけでなく **上限つきバンド** にする（sum2 ∈ [lo, hi]）
  【相手絞り】総流し5点をやめ、3列目を **p3 の弱い方から k 車** に絞る
             （オッズ非依存でなければ朝に決められない。強い3列目＝低配当なので
               弱い側を残すのがガミ回避の向き。7B の △除外と同じ発想）
  網羅率は 25% と 30% の両方で揃えて比較する。

⚠️ オッズは精算にのみ使う（選別・相手選びには一切使わない）。読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100
ALL_PER_DAY = 76.3


def build(races) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, board = r["p3"], r["board"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2 = ranked[0], ranked[1]
        others = ranked[2:]                      # p3 の強い順（3列目候補5車）
        rec = dict(rk=r["rk"], date=r["date"], sum2=p3[a1] + p3[a2],
                   win="確認" if r["date"] <= CONFIRM_END else "掃引")
        # 3列目の絞り方。weak_k = 弱い方から k 車（＝高配当側を残す）
        variants = {
            "総流し5点": others,
            "弱い4車": others[1:],
            "弱い3車": others[2:],
            "弱い2車": others[3:],
            "強い3車": others[:3],
        }
        for name, legs in variants.items():
            combos = [frozenset({a1, a2, t}) for t in legs
                      if frozenset({a1, a2, t}) in board]
            if not combos:
                continue
            hit = top3 in combos
            bet = len(combos) * STAKE
            odds = board[top3]
            ret = (round(odds * 100) // 10 * 10) if hit else 0
            rec[f"hit__{name}"] = int(hit)
            rec[f"bet__{name}"] = bet
            rec[f"ret__{name}"] = ret
            rec[f"net__{name}"] = int(hit and ret >= bet)   # 実質的中（元返し以上）
            rec[f"odds__{name}"] = odds if hit else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


VARIANTS = ["総流し5点", "弱い4車", "弱い3車", "弱い2車", "強い3車"]


def stats(g: pd.DataFrame, v: str, days: int) -> dict:
    h, bet, ret = f"hit__{v}", f"bet__{v}", f"ret__{v}"
    net, od = f"net__{v}", f"odds__{v}"
    hits = g[g[h] == 1]
    return dict(
        n_day=len(g) / days,
        hit=100 * g[h].mean(),
        net_hit=100 * g[net].mean(),
        gami=100 * (g[h].sum() - g[net].sum()) / max(g[h].sum(), 1),
        roi=100 * g[ret].sum() / g[bet].sum(),
        avg=float(hits[od].mean()) if len(hits) else 0.0,
        pts=g[bet].mean() / STAKE,
    )


def show(tag, g, days, v):
    s = stats(g, v, days)
    print(f"{tag:24s} {s['pts']:4.1f}点 {s['n_day']:6.2f}件/日 "
          f"的中 {s['hit']:5.2f}%  実質的中 {s['net_hit']:5.2f}%  "
          f"ガミ率 {s['gami']:5.1f}%  ROI {s['roi']:5.1f}%  平均配当 {s['avg']:5.2f}")


def main() -> None:
    d = build(pickle.load(open(DETAIL, "rb")))
    ev = d[d.win == "掃引"].copy()
    cf = d[d.win == "確認"].copy()
    days_e, days_c = ev.date.nunique(), cf.date.nunique()
    print(f"評価窓 {len(ev):,}R/{days_e}日  確認窓 {len(cf):,}R/{days_c}日\n")

    # ── 【0】そもそもガミはどこで起きているか（sum2 帯別）────────────────
    print("=== 【0】上位2車の複勝率合計 × ガミ（総流し5点・評価窓）===")
    ev["band"] = pd.cut(ev.sum2, [0, 1.40, 1.5734, 1.65, 1.70, 1.75, 1.80, 2.0],
                        labels=["~140", "140-157", "157-165", "165-170",
                                "170-175", "175-180", "180~"])
    print(f"{'合計帯':>10s} {'n':>7s} {'的中%':>6s} {'実質的中%':>8s} {'ガミ率%':>7s} "
          f"{'ROI%':>6s} {'的中時中央値':>10s}")
    for b, g in ev.groupby("band", observed=True):
        hits = g[g["hit__総流し5点"] == 1]
        s = stats(g, "総流し5点", days_e)
        med = float(hits["odds__総流し5点"].median()) if len(hits) else 0.0
        print(f"{str(b):>10s} {len(g):7,d} {s['hit']:6.2f} {s['net_hit']:8.2f} "
              f"{s['gami']:7.1f} {s['roi']:6.1f} {med:10.2f}")

    # ── 【1】見送り（上限バンド）: 網羅率を 25% / 30% に揃えて上限を動かす ──
    print("\n=== 【1】見送り案: 上限つきバンド（下限は網羅率が合うよう自動調整）===")
    for cover, per_day in ((0.25, 19.1), (0.30, 22.9)):
        print(f"\n-- 網羅 {cover:.0%}（{per_day}件/日）--")
        print(f"{'上限':>10s} {'下限':>8s} | "
              f"{'評価窓 的中%':>11s} {'実質的中%':>9s} {'ROI%':>6s} | "
              f"{'確認窓 的中%':>11s} {'実質的中%':>9s} {'ROI%':>6s}")
        for hi in (None, 1.85, 1.80, 1.75, 1.70):
            # 上限を先に適用し、残りから上位 per_day*days 件を取る下限を決める
            pool_e = ev if hi is None else ev[ev.sum2 < hi]
            k = int(per_day * days_e)
            if k >= len(pool_e):
                continue
            lo = pool_e.sort_values("sum2", ascending=False).sum2.iloc[k]
            ge = pool_e[pool_e.sum2 >= lo]
            gc = cf[(cf.sum2 >= lo) & (cf.sum2 < hi if hi else cf.sum2 >= lo)]
            se, sc = stats(ge, "総流し5点", days_e), stats(gc, "総流し5点", days_c)
            print(f"{(f'{hi:.2f}' if hi else 'なし'):>10s} {lo:8.4f} | "
                  f"{se['hit']:11.2f} {se['net_hit']:9.2f} {se['roi']:6.1f} | "
                  f"{sc['hit']:11.2f} {sc['net_hit']:9.2f} {sc['roi']:6.1f}")

    # ── 【2】相手絞り: 同じレース集合で3列目の絞り方だけ変える ────────────
    print("\n=== 【2】相手絞り案: 選別は下限のみ（157.34%）で 3列目の絞り方を変える ===")
    for wname, g0, days in (("評価窓", ev, days_e), ("確認窓", cf, days_c)):
        g = g0[g0.sum2 >= 1.5734]
        print(f"\n-- {wname} n={len(g):,} --")
        for v in VARIANTS:
            show(f"  {v}", g, days, v)

    # ── 【3】高配当側だけ絞る（力差が大きいレースに限って相手を削る）────────
    print("\n=== 【3】折衷案: 合計が非常に高いレース(>=1.70)だけ3列目を絞る ===")
    for wname, g0, days in (("評価窓", ev, days_e), ("確認窓", cf, days_c)):
        base = g0[g0.sum2 >= 1.5734]
        hi_g, lo_g = base[base.sum2 >= 1.70], base[base.sum2 < 1.70]
        print(f"\n-- {wname} 合計>=170%: {len(hi_g):,}R / 157-170%: {len(lo_g):,}R --")
        for v in ("総流し5点", "弱い4車", "弱い3車"):
            bet = lo_g["bet__総流し5点"].sum() + hi_g[f"bet__{v}"].sum()
            ret = lo_g["ret__総流し5点"].sum() + hi_g[f"ret__{v}"].sum()
            hit = lo_g["hit__総流し5点"].sum() + hi_g[f"hit__{v}"].sum()
            net = lo_g["net__総流し5点"].sum() + hi_g[f"net__{v}"].sum()
            n = len(base)
            print(f"  高合計帯を{v:8s}: 的中 {100*hit/n:5.2f}%  実質的中 {100*net/n:5.2f}%  "
                  f"ROI {100*ret/bet:5.1f}%  投資/日 {bet/days:7,.0f}円  "
                  f"収支/日 {(ret-bet)/days:+8,.0f}円")


if __name__ == "__main__":
    main()
