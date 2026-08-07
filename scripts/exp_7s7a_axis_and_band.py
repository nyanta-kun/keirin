"""7S/7A: ①高分散帯の頑健性 ②軸2をライン相関考慮に変える案（2026-08-06・探索）。

## 動機

`exp_7s7a_payout_frontier.py` で 2 つの手掛かりが出た。

1. **7S+7A のうち指数のばらつきが大きい側 25% は、掃引・確認の両窓で ROI が
   基準を上回り、平均配当と ≥10倍率がほぼ倍**（ユーザー要望の方向と一致）。
   ただし entropy と p3_std は r=-0.93 でほぼ同じ量。既存 entropy ゲートの
   「通過側の中での位置」を使うことになるので、帯として妥当か検証する。
2. **軸2車の3着内は独立でない**。別ライン同士だと負の相関（7A −1.7pt /
   空白E −7.0pt）、同一ラインだと正の相関（7SS +5.0pt）。
   現行の軸2選定はラインを一切見ない独立選定なので、ここに未使用の構造がある。

## やること

- 詳細データセット（1レース = 各車の pp3/ppw/pbad・ライン・印・着順・三連複盤面）を作る
- ①: entropy 十分位 × ROI・裾依存・四半期一貫性
- ②: 軸2の代替規則を同一母集団で総当たり比較（的中・ROI・平均配当）

⚠️ オッズは wt_odds（最終）。DB 書き込みなし。掃引窓/確認窓を分けて出す。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)
from scripts.exp_7s7a_payout_structure import (  # noqa: E402
    PRED_FILES, load_entries, load_trio, CACHE_DIR,
)

DETAIL = CACHE_DIR / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100


def build_detail() -> list[dict]:
    print("予測キャッシュ ...", flush=True)
    pred = pd.concat([pd.read_pickle(CACHE_DIR / fn) for _, _, fn in PRED_FILES],
                     ignore_index=True).drop_duplicates(["race_key", "frame_no"])
    print("出走表 ...", flush=True)
    ent = load_entries("2024-07-01", "2026-08-04")
    ent["frame_no"] = ent["frame_no"].astype(int)
    pred["frame_no"] = pred["frame_no"].astype(int)
    df = ent.merge(pred, on=["race_key", "frame_no"], how="inner")
    print("三連複オッズ ...", flush=True)
    trio = load_trio(sorted(df["race_key"].unique().tolist()))

    races = []
    for rk, g in df.groupby("race_key", sort=False):
        if len(g) != 7:
            continue
        board = trio.get(rk)
        if not board:
            continue
        fo = {}
        for t in g.itertuples(index=False):
            v = t.finish_order
            try:
                fo[int(t.frame_no)] = int(v) if v is not None and v == v else 0
            except (TypeError, ValueError):
                fo[int(t.frame_no)] = 0
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3 or frozenset(top3) not in board:
            continue
        lg = {}
        for t in g.itertuples(index=False):
            v = t.line_group
            lg[int(t.frame_no)] = f"L{v}" if v is not None and v == v else f"S{int(t.frame_no)}"
        races.append(dict(
            rk=rk, date=str(g["race_date"].iloc[0]), top3=top3, board=board, line=lg,
            p3={int(t.frame_no): float(t.pp3) for t in g.itertuples(index=False)},
            pw={int(t.frame_no): float(t.ppw) for t in g.itertuples(index=False)},
            pb={int(t.frame_no): float(t.pbad) for t in g.itertuples(index=False)},
            mk={int(t.frame_no): t.prediction_mark for t in g.itertuples(index=False)},
        ))
    print(f"詳細データセット {len(races):,} レース", flush=True)
    return races


# ---------------------------------------------------------------------------
# 軸2の代替規則
# ---------------------------------------------------------------------------
def axis1(r):
    return max(r["pw"], key=lambda k: r["pw"][k])


def _score(r):
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["pb"])
    return {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in r["p3"]}


def a2_current(r, a1):
    """現行: z(pp3) − 0.3·z(pbad) の最上位（ラインを見ない）。"""
    sc = _score(r)
    return max((k for k in sc if k != a1), key=lambda k: sc[k])


def a2_same_line_first(r, a1):
    """軸1と同一ラインの中から選ぶ。単騎なら現行にフォールバック。"""
    sc = _score(r)
    same = [k for k in sc if k != a1 and r["line"][k] == r["line"][a1]]
    return max(same, key=lambda k: sc[k]) if same else a2_current(r, a1)


def _mk_a2_lift(bonus: float):
    """同一ラインへ一定のボーナスを与える（連続版）。bonus は z 単位。"""
    def f(r, a1):
        sc = _score(r)
        cand = [k for k in sc if k != a1]
        return max(cand, key=lambda k: sc[k] + (bonus if r["line"][k] == r["line"][a1] else 0.0))
    return f


def a2_line_only_strict(r, a1):
    """同一ラインに相棒がいなければ『買わない』（None を返す）。"""
    sc = _score(r)
    same = [k for k in sc if k != a1 and r["line"][k] == r["line"][a1]]
    return max(same, key=lambda k: sc[k]) if same else None


RULES = {
    "現行（ライン非考慮）": a2_current,
    "同一ライン優先(単騎は現行)": a2_same_line_first,
    "同一ライン+0.3z": _mk_a2_lift(0.3),
    "同一ライン+0.6z": _mk_a2_lift(0.6),
    "同一ライン必須(無ければ見送り)": a2_line_only_strict,
}


def evaluate(races, a2fn, gate):
    """gate(r, a1, a2, asum, ent, ov) -> bool。戻りは行のリスト。"""
    rows = []
    for r in races:
        a1 = axis1(r)
        a2 = a2fn(r, a1)
        if a2 is None:
            continue
        ov = rank_7s_wt_overlap_n(
            a1, a2,
            next((k for k, m in r["mk"].items() if m == 1), None),
            next((k for k, m in r["mk"].items() if m == 2), None))
        asum = r["p3"][a1] + r["p3"][a2]
        ent = rank_7s_field_entropy(r["p3"])
        if not gate(r, a1, a2, asum, ent, ov):
            continue
        legs = [x for x in r["p3"] if x not in (a1, a2)
                and frozenset({a1, a2, x}) in r["board"]]
        if not legs:
            continue
        rest = r["top3"] - {a1, a2}
        hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
        odds = r["board"][frozenset(r["top3"])]
        rows.append((r["date"], len(legs) * STAKE,
                     (round(odds * 100) // 10 * 10) if hit else 0, int(hit), odds,
                     int(r["line"][a1] == r["line"][a2])))
    return rows


def summarize(rows, win):
    t = [x for x in rows if (x[0] <= CONFIRM_END) == (win == "確認")]
    if len(t) < 20:
        return None
    bet = sum(x[1] for x in t)
    ret = sum(x[2] for x in t)
    h = [x for x in t if x[3]]
    return dict(n=len(t), hit=100 * len(h) / len(t), roi=100 * ret / bet,
                avg=float(np.mean([x[4] for x in h])) if h else 0.0,
                ge10=100 * float(np.mean([x[4] >= 10 for x in h])) if h else 0.0,
                same=100 * float(np.mean([x[5] for x in t])))


HDR = (f"  {'':<32}" + "".join(
    [f"{'n':>7}{'的中':>7}{'ROI':>8}{'平均':>7}{'≥10':>6}{'同L':>6}" for _ in range(2)]))


def show(label, rows, w=32):
    txt = f"  {label:<{w}}"
    for win in ("掃引", "確認"):
        s = summarize(rows, win)
        txt += (f"{'—':>7}{'—':>7}{'—':>8}{'—':>7}{'—':>6}{'—':>6}" if s is None else
                f"{s['n']:>7}{s['hit']:>6.1f}%{s['roi']:>7.1f}%{s['avg']:>7.1f}"
                f"{s['ge10']:>5.0f}%{s['same']:>5.0f}%")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if DETAIL.exists() and not args.rebuild:
        races = pd.read_pickle(DETAIL)
        print(f"[cache] {DETAIL.name} {len(races):,} レース")
    else:
        races = build_detail()
        pd.to_pickle(races, DETAIL)

    is7s = lambda a, e: a <= RANK_7S_AXIS_SUM_MAX and e <= RANK_7S_ENTROPY_MAX
    is7a = lambda a, e: a > RANK_7S_AXIS_SUM_MAX and e <= RANK_7S_ENTROPY_MAX
    g_7s7a = lambda r, a1, a2, a, e, ov: ov in (0, 1) and (is7s(a, e) or is7a(a, e))

    print("\n" + "=" * 110)
    print("【8】軸2の選定規則を入れ替える（母集団は 7S+7A・左=掃引窓 右=確認窓）")
    print("     同L = 軸2車が同一ラインだった割合")
    print(HDR)
    base_rows = None
    for lab, fn in RULES.items():
        rows = evaluate(races, fn, g_7s7a)
        if base_rows is None:
            base_rows = rows
        show(lab, rows)

    print("\n  ── 現行規則のまま『軸2車が同一ラインか』で分解する")
    rows = base_rows
    print(HDR)
    show("7S+7A 全体", rows)
    show("うち 軸2車が同一ライン", [x for x in rows if x[5] == 1])
    show("うち 軸2車が別ライン", [x for x in rows if x[5] == 0])

    # ---- ① 高分散帯の頑健性 -------------------------------------------------
    print("\n" + "=" * 110)
    print("【9】entropy（=指数のばらつき）の十分位別 — 通過帯の中で ROI はどう動くか")
    ents = []
    for r in races:
        a1 = axis1(r)
        a2 = a2_current(r, a1)
        ov = rank_7s_wt_overlap_n(
            a1, a2, next((k for k, m in r["mk"].items() if m == 1), None),
            next((k for k, m in r["mk"].items() if m == 2), None))
        a = r["p3"][a1] + r["p3"][a2]
        e = rank_7s_field_entropy(r["p3"])
        if ov in (0, 1) and (is7s(a, e) or is7a(a, e)):
            ents.append((r, a1, a2, a, e))
    sweep_e = [x[4] for x in ents if x[0]["date"] > CONFIRM_END]
    qs = np.quantile(sweep_e, np.arange(0.1, 1.0, 0.1))
    print(f"  掃引窓 entropy 十分位境界: {' '.join(f'{q:.4f}' for q in qs)}")
    print(HDR)
    edges = [-np.inf, *qs, np.inf]
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        rows = evaluate(races, a2_current,
                        lambda r, a1, a2, a, e, ov, lo=lo, hi=hi:
                        ov in (0, 1) and (is7s(a, e) or is7a(a, e)) and lo < e <= hi)
        show(f"D{i+1} entropy ({lo:.4f},{hi:.4f}]", rows)

    print("\n  ── 上位25%帯（＝最もばらつく側）の裾依存と四半期一貫性")
    thr = float(np.quantile(sweep_e, 0.75))
    hi_rows = evaluate(races, a2_current,
                       lambda r, a1, a2, a, e, ov: ov in (0, 1)
                       and (is7s(a, e) or is7a(a, e)) and e >= thr)
    all_rows = base_rows
    for lab, rws in (("7S+7A 基準", all_rows), (f"entropy>={thr:.4f}", hi_rows)):
        for win in ("掃引", "確認"):
            t = [x for x in rws if (x[0] <= CONFIRM_END) == (win == "確認")]
            r = np.sort(np.array([x[2] for x in t]))[::-1]
            bet = sum(x[1] for x in t)
            print(f"     {lab:<22}{win}  ROI {100*r.sum()/bet:5.1f}% → 上位5件除 "
                  f"{100*r[5:].sum()/bet:5.1f}% → 上位10件除 {100*r[10:].sum()/bet:5.1f}%"
                  f"（上位5件が回収の {100*r[:5].sum()/r.sum():4.1f}%）")
    print(f"\n     {'四半期':<10}{'n':>6}{'的中':>8}{'ROI':>8}   基準(同四半期の7S+7A)")
    q_of = lambda d: str(pd.Period(d, freq="Q"))
    for q in sorted({q_of(x[0]) for x in hi_rows}):
        t = [x for x in hi_rows if q_of(x[0]) == q]
        b = [x for x in all_rows if q_of(x[0]) == q]
        if len(t) < 15:
            continue
        print(f"     {q:<10}{len(t):>6}{100*sum(x[3] for x in t)/len(t):>7.1f}%"
              f"{100*sum(x[2] for x in t)/sum(x[1] for x in t):>7.1f}%"
              f"        {100*sum(x[2] for x in b)/sum(x[1] for x in b):>6.1f}%")


if __name__ == "__main__":
    main()
