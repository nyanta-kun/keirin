"""7車立ての未購入領域と、季節×レース場の傾向（2026-08-06・探索）。

ユーザー依頼:
  「現在買っていないレースで狙える条件がないか検討して。
    また季節とレース場の掛け合わせで傾向がないかも見てみて」

前提: `exp_7s7a_axis_and_band.py` が作った `axis_detail_7car.pkl`（48,541R・
2024-07〜2026-08・honest walk-forward 予測）を使う。

  【A】被覆マップ — 現行6ランクがどこを買っていて、どこが空いているか
  【B】未購入セルの実力（三連複 二軸総流し5点で統一評価）
  【C】季節（月）× レース場（個別場・バンク長・屋内外）の傾向

⚠️ オッズは wt_odds（最終）。DB 書き込みなし。
⚠️ 掃引窓 2025-07-01〜2026-08-04 / 確認窓 2024-07-01〜2025-06-30 で必ず分ける。
⚠️ 【C】は**セル数が多い＝多重比較**。掃引窓で光ったものが確認窓で残るかだけを見る。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100


def load_meta(keys):
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute("""
          SELECT r.race_key, r.race_date, r.venue_id, r.grade, r.race_type,
                 v.bank_length, v.is_indoor
          FROM keirin.wt_races r LEFT JOIN keirin.venue_info v ON v.venue_code = r.venue_id
          WHERE r.race_key = ANY(%s)""", (keys,))
        return {r[0]: r[1:] for r in cur.fetchall()}


def build(races, meta):
    rows = []
    for r in races:
        a1 = max(r["pw"], key=lambda k: r["pw"][k])
        zp, zb = _race_zscore(r["p3"]), _race_zscore(r["pb"])
        sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in r["p3"]}
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        hon = next((k for k, m in r["mk"].items() if m == 1), None)
        tai = next((k for k, m in r["mk"].items() if m == 2), None)
        ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
        legs = [x for x in r["p3"] if x not in (a1, a2)
                and frozenset({a1, a2, x}) in r["board"]]
        if not legs:
            continue
        rest = r["top3"] - {a1, a2}
        hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
        odds = r["board"][frozenset(r["top3"])]
        m = meta.get(r["rk"])
        if m is None:
            continue
        rows.append(dict(
            rk=r["rk"], date=r["date"], month=int(r["date"][5:7]),
            venue=m[1], grade=m[2], rtype=m[3],
            bank=m[4], indoor=m[5],
            ov=(-1 if ov is None else ov),
            asum=r["p3"][a1] + r["p3"][a2], ent=rank_7s_field_entropy(r["p3"]),
            same=int(r["line"][a1] == r["line"][a2]),
            # 7B の order 一致判定（軸1 = pred_win 最上位 なので a1 と◎の一致）
            order_agree=int(hon is not None and a1 == hon),
            hit=int(hit), bet=len(legs) * STAKE,
            ret=(round(odds * 100) // 10 * 10) if hit else 0, odds=odds))
    df = pd.DataFrame(rows)
    df["win"] = np.where(df.date <= CONFIRM_END, "確認", "掃引")
    df["a_ok"] = df.asum <= RANK_7S_AXIS_SUM_MAX
    df["e_ok"] = df.ent <= RANK_7S_ENTROPY_MAX
    return df


def rank_label(r):
    if r.ov == 2:
        return "7B" if (r.rtype == "準決勝" and r.order_agree) else "◎○一致(未購入)"
    if r.ov in (0, 1):
        if r.a_ok and r.e_ok:
            return "7S"
        if (not r.a_ok) and r.e_ok:
            return "7A"
        if r.a_ok and (not r.e_ok):
            return "7SS" if r.same else "空白E(未購入)"
        return "空白AE(未購入)"
    return "印欠損(未購入)"


def agg(s):
    if len(s) < 30:
        return None
    h = s[s.hit == 1]
    return dict(n=len(s), hit=100 * s.hit.mean(), roi=100 * s.ret.sum() / s.bet.sum(),
                avg=float(h.odds.mean()) if len(h) else 0.0,
                ge10=100 * float((h.odds >= 10).mean()) if len(h) else 0.0)


HDR = f"  {'':<26}" + "".join([f"{'n':>7}{'件/日':>7}{'的中':>7}{'ROI':>8}{'平均':>7}" for _ in range(2)])


def show(lbl, s, days, w=26):
    txt = f"  {lbl:<{w}}"
    for win in ("掃引", "確認"):
        a = agg(s[s.win == win])
        txt += (f"{'—':>7}{'—':>7}{'—':>7}{'—':>8}{'—':>7}" if a is None else
                f"{a['n']:>7}{a['n']/days[win]:>7.2f}{a['hit']:>6.1f}%"
                f"{a['roi']:>7.1f}%{a['avg']:>7.1f}")
    print(txt)


def main():
    races = pd.read_pickle(DETAIL)
    meta = load_meta(sorted({r["rk"] for r in races}))
    df = build(races, meta)
    days = {w: df[df.win == w].date.nunique() for w in ("掃引", "確認")}
    print(f"7車立て {len(df):,}R  掃引{days['掃引']}日 / 確認{days['確認']}日")
    df["lbl"] = [rank_label(r) for r in df.itertuples(index=False)]

    print("\n" + "=" * 100)
    print("【A】被覆マップ（すべて三連複 二軸総流し5点で統一評価）")
    print("     ※ 7B は実際は3点買いなので、ここでの7B行は『同じ5点で買ったら』の仮想値")
    print(HDR)
    order = ["7S", "7A", "7SS", "7B", "空白E(未購入)", "空白AE(未購入)",
             "◎○一致(未購入)", "印欠損(未購入)"]
    for lbl in order:
        show(lbl, df[df.lbl == lbl], days)
    print(f"\n     全 7車立て: {len(df):,}R / 現行購入対象は "
          f"{100*df.lbl.isin(['7S','7A','7SS','7B']).mean():.1f}%")

    print("\n" + "=" * 100)
    print("【B】最大の未購入領域『◎○一致』を分解する（母集団の約6割）")
    ov2 = df[df.ov == 2]
    print(HDR)
    show("◎○一致 全体", ov2, days)
    show("  ∧ 準決勝 ∧ order一致(=7B)", ov2[(ov2.rtype == "準決勝") & (ov2.order_agree == 1)], days)
    show("  ∧ 準決勝 ∧ order不一致", ov2[(ov2.rtype == "準決勝") & (ov2.order_agree == 0)], days)
    show("  ∧ 準決勝以外 ∧ order一致", ov2[(ov2.rtype != "準決勝") & (ov2.order_agree == 1)], days)
    show("  ∧ 準決勝以外 ∧ order不一致", ov2[(ov2.rtype != "準決勝") & (ov2.order_agree == 0)], days)
    print("     ── さらに 2ゲートで切る")
    for a_ok in (True, False):
        for e_ok in (True, False):
            s = ov2[(ov2.a_ok == a_ok) & (ov2.e_ok == e_ok)]
            show(f"  asum{'合' if a_ok else '否'}/ent{'合' if e_ok else '否'}", s, days)
    print("     ── 同一ライン軸で切る")
    show("  ∧ 軸2車が同一ライン", ov2[ov2.same == 1], days)
    show("  ∧ 軸2車が別ライン", ov2[ov2.same == 0], days)

    print("\n" + "=" * 100)
    print("【C-1】季節（月）— 母集団全体（7車立て全レース・二軸総流し5点）")
    print(f"  {'月':<6}" + "".join(f"{w:>26}" for w in ("掃引窓", "確認窓")))
    print(f"  {'':<6}" + "".join(f"{'n':>8}{'的中':>8}{'ROI':>9}" for _ in range(2)))
    for m in range(1, 13):
        s = df[df.month == m]
        line = f"  {m:>2}月  "
        for win in ("掃引", "確認"):
            a = agg(s[s.win == win])
            line += (f"{'—':>8}{'—':>8}{'—':>9}" if a is None else
                     f"{a['n']:>8}{a['hit']:>7.1f}%{a['roi']:>8.1f}%")
        print(line)

    print("\n【C-2】レース場（個別・掃引窓 n>=200 の場のみ・ROI 降順）")
    sw = df[df.win == "掃引"]
    cf = df[df.win == "確認"]
    stat = []
    for v, g in sw.groupby("venue"):
        if len(g) < 200:
            continue
        c = cf[cf.venue == v]
        if len(c) < 100:
            continue
        stat.append((v, len(g), 100 * g.ret.sum() / g.bet.sum(),
                     len(c), 100 * c.ret.sum() / c.bet.sum()))
    stat.sort(key=lambda x: -x[2])
    print(f"  {'場':<6}{'掃引n':>7}{'掃引ROI':>9}{'確認n':>7}{'確認ROI':>9}   判定")
    for v, n1, r1, n2, r2 in stat:
        flag = "◎両窓とも85%超" if r1 >= 85 and r2 >= 85 else (
            "×両窓とも75%割れ" if r1 < 75 and r2 < 75 else "")
        print(f"  {str(v):<6}{n1:>7}{r1:>8.1f}%{n2:>7}{r2:>8.1f}%   {flag}")

    print("\n【C-3】季節 × 場の属性（セルを大きくして交互作用を見る）")
    df["bank_c"] = pd.cut(pd.to_numeric(df.bank, errors="coerce"),
                          [0, 335, 400, 1000], labels=["333m", "400m", "500m"])
    df["season"] = pd.cut(df.month, [0, 3, 6, 9, 12],
                          labels=["冬(1-3)", "春(4-6)", "夏(7-9)", "秋冬(10-12)"])
    for key, lbl in (("bank_c", "バンク長"), ("indoor", "屋内(1)/屋外(0)")):
        print(f"\n  ── 季節 × {lbl}（各セル: 掃引ROI / 確認ROI・n>=150 のみ）")
        cats = [c for c in df[key].dropna().unique()]
        print(f"     {'季節':<12}" + "".join(f"{str(c):>22}" for c in cats))
        for s in df.season.cat.categories:
            line = f"     {s:<12}"
            for c in cats:
                t = df[(df.season == s) & (df[key] == c)]
                a1_, a2_ = agg(t[t.win == "掃引"]), agg(t[t.win == "確認"])
                if a1_ is None or a2_ is None or a1_["n"] < 150 or a2_["n"] < 150:
                    line += f"{'—':>22}"
                else:
                    line += f"{a1_['roi']:>9.1f}% /{a2_['roi']:>7.1f}%  "
            print(line)


if __name__ == "__main__":
    main()
