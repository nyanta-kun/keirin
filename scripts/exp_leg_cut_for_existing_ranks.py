"""7C で効いた「相手の足切り＋点数ゲート」を 7SS/7S/7A にも適用できるか（2026-08-07）。

ユーザー質問:
  「現在の点数絞り込みについて、7SS/7S/7A についても同様のことが言えますか？
    言えるのであれば適用が必要と考えます」

7C で確認できたのは2つ:
  (a) 相手を **3着内率15%以上**に足切りすると、的中率をわずかに落として ROI が上がる
  (b) 足切り後が **4点未満のレースは買わない** と、的中率とROIを保ったまま
      低配当的中（2.0倍以下）を7割落とせる
      ＝「相手が絞れる＝実力差が大きい＝配当が付かない」

7SS/7S/7A は **7C とは逆側**（軸2車の信頼が低い＝配当が高い側）を取るランクなので、
同じ現象が起きるとは限らない。各ランクの母集団で同じ検証を行う。

賭け金は netkeirin と揃えて **1レース10,000円の予算枠**（現行は 2,000円×5点＝10,000円
なので、総流しのままなら現行と完全に一致する）。

⚠️ 軸は各ランクの本番と同じ **3ヘッド軸**（軸1=pw最上位 / 軸2=z(p3)−0.3z(pb)最上位）。
⚠️ オッズは精算にのみ使う。読み取りのみ。
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEG_P3_MIN, RANK_7C_LEGS_MIN, RANK_7S_AXIS_SUM_MAX,
    RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT, _race_zscore,
    rank_7c_unit_stake, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
LOWPAY = 2.0


def load_race_type(keys):
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute("SELECT race_key, race_type FROM keirin.wt_races "
                    "WHERE race_key = ANY(%s)", (keys,))
        return dict(cur.fetchall())


def rank_of(ov, a_ok, e_ok, same, rtype, a1, hon):
    if ov == 2:
        return "7B" if (rtype == "準決勝" and a1 == hon) else None
    if ov in (0, 1):
        if a_ok and e_ok:
            return "7S"
        if not a_ok and e_ok:
            return "7A"
        if a_ok and not e_ok:
            return "7SS" if same else None
    return None


def build(races, rtypes) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, pw, pb, board, line = r["p3"], r["pw"], r["pb"], r["board"], r["line"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        a1 = max(pw, key=lambda f: pw[f])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        sc = {f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f] for f in p3}
        a2 = max((f for f in sc if f != a1), key=lambda f: sc[f])
        hon = next((f for f, m in r["mk"].items() if m == 1), None)
        tai = next((f for f, m in r["mk"].items() if m == 2), None)
        ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
        ov = -1 if ov is None else ov
        rk_name = rank_of(
            ov, p3[a1] + p3[a2] <= RANK_7S_AXIS_SUM_MAX,
            rank_7s_field_entropy(p3) <= RANK_7S_ENTROPY_MAX,
            line.get(a1) == line.get(a2), rtypes.get(r["rk"]), a1, hon)
        if rk_name is None:
            continue

        others = sorted(set(p3) - {a1, a2})
        variants = {
            "現行(総流し5点)": others,
            "足切り15%": [x for x in sorted(others, key=lambda c: -p3[c])
                          if p3[x] >= RANK_7C_LEG_P3_MIN],
        }
        rec = dict(rk=r["rk"], date=r["date"], rank=rk_name,
                   win="確認" if r["date"] <= CONFIRM_END else "掃引")
        for name, legs in variants.items():
            combos = [frozenset({a1, a2, t}) for t in legs
                      if frozenset({a1, a2, t}) in board]
            if not combos:
                continue
            k = len(combos)
            s = rank_7c_unit_stake(k)
            od = board[top3]
            hit = top3 in combos
            rec[f"k__{name}"] = k
            rec[f"h__{name}"] = int(hit)
            rec[f"b__{name}"] = k * s
            rec[f"r__{name}"] = (round(od * s) // 10 * 10) if hit else 0
            rec[f"l__{name}"] = int(hit and od <= LOWPAY)
            rec[f"o__{name}"] = od if hit else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(g, name, days):
    h, b, r_, l_, o, k = (f"h__{name}", f"b__{name}", f"r__{name}",
                          f"l__{name}", f"o__{name}", f"k__{name}")
    if h not in g.columns:
        return None
    g = g.dropna(subset=[h])
    if len(g) < 100:
        return None
    hits = g[g[h] == 1]
    return dict(
        n_day=len(g) / days, pts=g[k].mean(), hit=100 * g[h].mean(),
        roi=100 * g[r_].sum() / g[b].sum(), low=g[l_].sum() / days,
        low_rate=100 * g[l_].sum() / max(g[h].sum(), 1),
        avg=float(hits[o].mean()) if len(hits) else 0.0)


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    rtypes = load_race_type([r["rk"] for r in races])
    d = build(races, rtypes)

    for rank in ("7SS", "7S", "7A", "7B"):
        print(f"\n{'='*96}\n=== {rank} ===")
        for wname, w in (("評価窓 2025-07〜2026-08", "掃引"),
                         ("確認窓 2024-07〜2025-06", "確認")):
            g0 = d[(d["rank"] == rank) & (d.win == w)]
            if g0.empty:
                continue
            days = g0.date.nunique()
            print(f"-- {wname}  n={len(g0):,}R --")
            print(f"   {'買い目':22s} {'平均点':>6s} {'件/日':>6s} {'的中%':>6s} "
                  f"{'ROI%':>6s} {'低配当/日':>7s} {'的中の低配当率%':>13s} {'的中時平均':>8s}")
            for name in ("現行(総流し5点)", "足切り15%"):
                s = stats(g0, name, days)
                if s:
                    print(f"   {name:22s} {s['pts']:6.2f} {s['n_day']:6.2f} "
                          f"{s['hit']:6.2f} {s['roi']:6.1f} {s['low']:7.2f} "
                          f"{s['low_rate']:13.1f} {s['avg']:8.2f}")
            # 点数ゲート（足切り後の点数で層別 → 4点以上のみ買う）
            gg = g0.dropna(subset=["k__足切り15%"])
            if len(gg) >= 200:
                print(f"   {'（足切り後の点数帯別）':22s}")
                for k, x in gg.groupby("k__足切り15%"):
                    if len(x) < 60:
                        continue
                    hit = x["h__足切り15%"]
                    low = x["l__足切り15%"]
                    print(f"     {int(k)}点: n={len(x):5,d} ({100*len(x)/len(gg):4.1f}%) "
                          f"的中 {100*hit.mean():5.2f}%  "
                          f"的中の低配当率 {100*low.sum()/max(hit.sum(),1):5.1f}%  "
                          f"ROI {100*x['r__足切り15%'].sum()/x['b__足切り15%'].sum():5.1f}%")
                sel = gg[gg["k__足切り15%"] >= RANK_7C_LEGS_MIN]
                if len(sel) >= 100:
                    s = stats(sel, "足切り15%", days)
                    print(f"   {'足切り15% ∧ 4点以上':22s} {s['pts']:6.2f} "
                          f"{s['n_day']:6.2f} {s['hit']:6.2f} {s['roi']:6.1f} "
                          f"{s['low']:7.2f} {s['low_rate']:13.1f} {s['avg']:8.2f}")
                else:
                    print(f"   {'足切り15% ∧ 4点以上':22s} （n={len(sel)} で不足）")


if __name__ == "__main__":
    main()
