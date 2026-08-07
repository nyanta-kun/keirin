"""ベースモデルの網羅率 × 二軸的中率（2026-08-07）。

ユーザー要件（追補）:
  「1日単位で見たレース全体の **30%程度** を網羅できる条件で検討。
    レースごとにバリエーションが多いので条件分け（選手の組合せ・ライン等）も
    必要なら含める。ただし**汎用的な条件で確保できるなら条件分けしない方が良い**」

ペアモデルは「その2車がともに3着内に入る確率」を直接出す。したがって
**そのスコアの高い順にレースを取る**のが、条件分けを一切しない最も汎用的な選別。
ここで測るのは:

  【A】網羅率（件/日）と二軸的中率の関係（＝どこまで絞れば何%当たるか）
  【B】絶対閾値 と 日次相対順位 のどちらで切るべきか
      （7H1 で相対順位にすると件数が半減した前例があるため必ず両方見る）
  【C】条件分け（ライン構成・グレード・種別・選手構成）に**追加の**情報があるか
      ＝ 同じスコア帯の中でセグメント間に差が残るか。残らなければ条件分けは不要
  【D】既存ランク（7S/7A/7B/7SS）との重なりと、その領域を除いた場合の実力

⚠️ オッズ不使用（選別・軸選定とも）。ROI は参考表示。読み取りのみ。
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
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
SCORED = REPO / "data" / "exp_cache" / "pair_axis_scored_v2.pkl"
DATASET = REPO / "data" / "exp_cache" / "pair_axis_dataset_v2.pkl"
STAKE = 100
# 1日の総レース数（全車立て）。2025-08〜2026-08 実測 76.3件/日、うち7車 64.9件/日。
ALL_RACES_PER_DAY = 76.3


def build(races: dict, pick: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in pick.iterrows():
        r = races[p.rk]
        p3, pw, pb, board = r["p3"], r["pw"], r["pb"], r["board"]
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        cars = sorted(p3)
        hi, lo = int(p.hi), int(p.lo)
        legs = [frozenset({hi, lo, z}) for z in cars if z not in (hi, lo)
                and frozenset({hi, lo, z}) in board]
        if not legs:
            continue
        hit = int(top3 in legs)
        # 既存ランクの帰属（被覆マップと同じ判定）
        a1 = max(pw, key=lambda k: pw[k])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in p3}
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        hon = next((k for k, m in r["mk"].items() if m == 1), None)
        tai = next((k for k, m in r["mk"].items() if m == 2), None)
        ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
        asum, ent = p3[a1] + p3[a2], rank_7s_field_entropy(p3)
        rows.append(dict(
            rk=p.rk, date=p.date, score=p.s_bin, hit=hit,
            bet=len(legs) * STAKE,
            ret=(round(board[top3] * 100) // 10 * 10) if hit else 0,
            odds=board[top3],
            same_line=int(r["line"].get(hi) == r["line"].get(lo)),
            n_lines=len(set(r["line"].values())),
            ov=(-1 if ov is None else ov),
            a_ok=int(asum <= RANK_7S_AXIS_SUM_MAX),
            e_ok=int(ent <= RANK_7S_ENTROPY_MAX),
            base_hit_3head=int(a1 in r["top3"] and a2 in r["top3"]),
        ))
    return pd.DataFrame(rows)


def existing_rank(r) -> str:
    if r.ov == 2:
        return "◎○一致(7B含む)"
    if r.ov in (0, 1):
        if r.a_ok and r.e_ok:
            return "7S"
        if not r.a_ok and r.e_ok:
            return "7A"
        if r.a_ok and not r.e_ok:
            return "7SS/空白E"
        return "空白AE"
    return "印欠損"


def show(tag, g, days):
    if len(g) < 100:
        return
    print(f"{tag:26s} n={len(g):6,d} {len(g)/days:6.2f}件/日 "
          f"({100*len(g)/days/ALL_RACES_PER_DAY:5.1f}%) "
          f"二軸的中 {100*g.hit.mean():5.2f}%  "
          f"的中/日 {g.hit.sum()/days:5.2f}  "
          f"ROI {100*g.ret.sum()/g.bet.sum():5.1f}%  "
          f"平均配当 {g[g.hit==1].odds.mean():5.2f}")


def main() -> None:
    races = {r["rk"]: r for r in pickle.load(open(DETAIL, "rb"))}
    te = pd.read_pickle(SCORED)
    pick = te.loc[te.groupby("rk").s_bin.idxmax()][["rk", "date", "hi", "lo", "s_bin"]]
    d = build(races, pick)
    days = d.date.nunique()
    print(f"評価窓 2025-07-01〜  {len(d):,}R / {days}日 = {len(d)/days:.1f}件/日 "
          f"(7車は全レースの85.1%)\n")

    print("=== 【A】スコア上位から取った場合の網羅率 × 二軸的中率 ===")
    print(f"{'選別':26s} {'n':>8s} {'件/日':>8s} {'全体比':>7s} {'的中%':>7s} "
          f"{'的中/日':>7s} {'ROI%':>7s} {'平均配当':>7s}")
    show("全7車（無選別）", d, days)
    ds = d.sort_values("score", ascending=False)
    for frac in (0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
        k = int(len(ds) * frac)
        show(f"スコア上位{frac:.0%}", ds.iloc[:k], days)

    print("\n=== 【B】絶対閾値 vs 日次相対順位（目標: 全体の約30% = 22.9件/日）===")
    # 絶対閾値: 上位30%相当のスコアで切る
    thr = ds.score.iloc[int(len(ds) * 0.30)]
    abs_sel = d[d.score >= thr]
    print(f"絶対閾値 score >= {thr:.4f}")
    show("  絶対閾値", abs_sel, days)
    # 日次相対: 各日の上位 n 件
    for per_day in (20, 23, 26):
        rel = (d.sort_values(["date", "score"], ascending=[True, False])
                 .groupby("date").head(per_day))
        show(f"  日次上位{per_day}件", rel, days)
    dd = d.assign(day_n=d.groupby("date").rk.transform("size"))
    print(f"  1日あたり7車レース数の分布: 中央値 {dd.day_n.median():.0f} / "
          f"最小 {dd.day_n.min()} / 最大 {dd.day_n.max()}")
    ab = abs_sel.groupby("date").size()
    print(f"  絶対閾値で選ばれる件数の分布: 中央値 {ab.median():.0f} / "
          f"最小 {ab.min()} / 最大 {ab.max()} / 0件の日 {days - len(ab)}日")

    print("\n=== 【B2】全体の30%（22.9件/日）を狙う絶対閾値 ===")
    target = 22.9 * days
    thr30 = ds.score.iloc[min(int(target), len(ds) - 1)]
    sel30 = d[d.score >= thr30]
    print(f"score >= {thr30:.4f}")
    show("  絶対閾値(30%狙い)", sel30, days)
    ab30 = sel30.groupby("date").size()
    print(f"  日次件数: 中央値 {ab30.median():.0f} / 最小 {ab30.min()} / "
          f"最大 {ab30.max()} / 0件の日 {days - len(ab30)}日")

    print("\n=== 【C】同じスコア帯の中にセグメント差が残るか（＝条件分けの要否）===")
    # レース単位の属性を dataset から引く（ペア行で不変の列のみ）
    ds2 = pd.read_pickle(DATASET)
    rmeta = (ds2.groupby("rk")[["grade_c", "rtype_c", "bank", "rp_sd", "p3_sd",
                                "p3_gap12"]].first().reset_index())
    d = d.merge(rmeta, on="rk", how="left")
    d["bank_c"] = pd.cut(d.bank, [0, 350, 450, 900], labels=["333m級", "400m級", "500m級"])
    d["rpsd_c"] = pd.qcut(d.rp_sd, 3, labels=["得点接近", "中", "得点開き"])
    d5 = d.assign(band=pd.qcut(d.score, 5, labels=[f"Q{i+1}" for i in range(5)]))
    for col, name in (("same_line", "軸2車が同一ライン"), ("n_lines", "分戦数"),
                      ("ov", "公式◎○との重なり"), ("grade_c", "グレード"),
                      ("rtype_c", "レース種別"), ("bank_c", "バンク長"),
                      ("rpsd_c", "競走得点のばらつき")):
        print(f"-- {name} --")
        piv = d5.pivot_table(index="band", columns=col, values="hit",
                             aggfunc=["mean", "size"], observed=True)
        for band in piv.index:
            parts = []
            for c in sorted(d5[col].unique()):
                try:
                    m, n = piv.loc[band, ("mean", c)], piv.loc[band, ("size", c)]
                except KeyError:
                    continue
                if pd.isna(n) or n < 150:
                    continue
                parts.append(f"{c}:{100*m:5.1f}%(n={int(n):5,d})")
            print(f"   {band}  " + "  ".join(parts))

    print("\n=== 【D】既存ランクとの重なり（スコア上位30%の内訳）===")
    top30 = ds.iloc[:int(len(ds) * 0.30)].copy()
    top30["er"] = top30.apply(existing_rank, axis=1)
    for er, g in top30.groupby("er"):
        show(f"  {er}", g, days)
    print("\n（参考）既存ランク領域を除いた上位30%＝新規に増える分:")
    show("  ◎○一致以外", top30[top30.er != "◎○一致(7B含む)"], days)


if __name__ == "__main__":
    main()
