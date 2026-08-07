"""7S/7A: 買い方（相手の削り方）と、差の有意性（2026-08-06・探索）。

前提: `exp_7s7a_axis_and_band.py` が作った詳細データセットを使う。

  【10】軸1の代替規則（現行 = pred_win 最上位）
  【11】相手の削り方 — 5点総流し / 強い相手を外す / 弱い相手を外す
  【12】主要比較のブートストラップCI（レース単位リサンプル）

⚠️ オッズは wt_odds（最終）。DB 書き込みなし。掃引窓/確認窓を分けて出す。
"""
from __future__ import annotations

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

CACHE_DIR = REPO / "data" / "exp_cache"
DETAIL = CACHE_DIR / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100
RNG = np.random.default_rng(42)


def _score(r):
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["pb"])
    return {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in r["p3"]}


A1_RULES = {
    "現行 pred_win 最上位": lambda r: max(r["pw"], key=lambda k: r["pw"][k]),
    "pred_top3 最上位": lambda r: max(r["p3"], key=lambda k: r["p3"][k]),
    "3ヘッドスコア最上位": lambda r: max(_score(r), key=lambda k: _score(r)[k]),
    "最強ラインの先頭": None,  # 下で特別扱い
}


def a1_strongest_line(r):
    tot = {}
    for k, v in r["line"].items():
        tot.setdefault(v, []).append(k)
    best = max(tot.values(), key=lambda mem: sum(r["p3"][x] for x in mem))
    return max(best, key=lambda k: r["pw"][k])


def build_rows(races, a1fn, legs_rule, gate_extra=None):
    """legs_rule(r, a1, a2, others) -> 実際に買う相手のリスト。"""
    rows = []
    for r in races:
        a1 = a1fn(r)
        sc = _score(r)
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        ov = rank_7s_wt_overlap_n(
            a1, a2, next((k for k, m in r["mk"].items() if m == 1), None),
            next((k for k, m in r["mk"].items() if m == 2), None))
        if ov not in (0, 1):
            continue
        asum = r["p3"][a1] + r["p3"][a2]
        ent = rank_7s_field_entropy(r["p3"])
        if ent > RANK_7S_ENTROPY_MAX:      # 7S+7A は entropy 合格が共通条件
            continue
        if gate_extra is not None and not gate_extra(r, a1, a2, asum, ent):
            continue
        others = [x for x in r["p3"] if x not in (a1, a2)
                  and frozenset({a1, a2, x}) in r["board"]]
        legs = legs_rule(r, a1, a2, others)
        if not legs:
            continue
        rest = r["top3"] - {a1, a2}
        hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
        odds = r["board"][frozenset(r["top3"])]
        rows.append((r["date"], len(legs) * STAKE,
                     (round(odds * 100) // 10 * 10) if hit else 0, int(hit), odds))
    return rows


def all_legs(r, a1, a2, others):
    return others


def _mk_drop_strong(n):
    def f(r, a1, a2, others):
        s = sorted(others, key=lambda k: r["p3"][k], reverse=True)
        return s[n:]
    return f


def _mk_keep_strong(n):
    def f(r, a1, a2, others):
        s = sorted(others, key=lambda k: r["p3"][k], reverse=True)
        return s[:n]
    return f


LEG_RULES = {
    "5点 総流し（現行）": all_legs,
    "4点 最強の相手を外す": _mk_drop_strong(1),
    "3点 上位2の相手を外す": _mk_drop_strong(2),
    "3点 上位3の相手だけ": _mk_keep_strong(3),
    "2点 上位2の相手だけ": _mk_keep_strong(2),
}


def summ(rows, win):
    t = [x for x in rows if (x[0] <= CONFIRM_END) == (win == "確認")]
    if len(t) < 20:
        return None
    bet = sum(x[1] for x in t); ret = sum(x[2] for x in t)
    h = [x for x in t if x[3]]
    return dict(n=len(t), hit=100 * len(h) / len(t), roi=100 * ret / bet,
                avg=float(np.mean([x[4] for x in h])) if h else 0.0,
                ge10=100 * float(np.mean([x[4] >= 10 for x in h])) if h else 0.0,
                bet=bet / len(t))


HDR = f"  {'':<30}" + "".join(
    [f"{'n':>7}{'的中':>7}{'ROI':>8}{'平均':>7}{'≥10':>6}{'円/R':>7}" for _ in range(2)])


def show(label, rows, w=30):
    txt = f"  {label:<{w}}"
    for win in ("掃引", "確認"):
        s = summ(rows, win)
        txt += (f"{'—':>7}{'—':>7}{'—':>8}{'—':>7}{'—':>6}{'—':>7}" if s is None else
                f"{s['n']:>7}{s['hit']:>6.1f}%{s['roi']:>7.1f}%{s['avg']:>7.1f}"
                f"{s['ge10']:>5.0f}%{s['bet']:>7.0f}")
    print(txt)


def boot_roi(rows, n=2000):
    """レース単位リサンプルの ROI 95%CI。"""
    if len(rows) < 30:
        return (np.nan, np.nan)
    bet = np.array([x[1] for x in rows], float)
    ret = np.array([x[2] for x in rows], float)
    idx = RNG.integers(0, len(rows), size=(n, len(rows)))
    r = 100 * ret[idx].sum(1) / bet[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    races = pd.read_pickle(DETAIL)
    print(f"[cache] {DETAIL.name} {len(races):,} レース")
    a1_cur = A1_RULES["現行 pred_win 最上位"]

    print("\n" + "=" * 108)
    print("【10】軸1の選定規則（母集団 7S+7A・左=掃引窓 右=確認窓）")
    print(HDR)
    for lab, fn in A1_RULES.items():
        f = a1_strongest_line if fn is None else fn
        show(lab, build_rows(races, f, all_legs))

    print("\n" + "=" * 108)
    print("【11】買い方 — 相手の削り方（軸は現行のまま・母集団 7S+7A）")
    print("     ※ 市場が効率的なら点数を削っても ROI は動かず、的中と配当だけが動くはず")
    print(HDR)
    for lab, rule in LEG_RULES.items():
        show(lab, build_rows(races, a1_cur, rule))

    print("\n  ── 同じ削り方を『高配当が出やすい側』へ適用（entropy 上位25% ∧ 軸2車が別ライン）")
    thr = None
    tmp = build_rows(races, a1_cur, all_legs)
    ents = []
    for r in races:
        a1 = a1_cur(r); sc = _score(r)
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        e = rank_7s_field_entropy(r["p3"])
        if e <= RANK_7S_ENTROPY_MAX and r["date"] > CONFIRM_END:
            ents.append(e)
    thr = float(np.quantile(ents, 0.75))

    def gate_hi(r, a1, a2, asum, ent):
        return ent >= thr and r["line"][a1] != r["line"][a2]
    print(f"     entropy>={thr:.4f} ∧ 別ライン")
    print(HDR)
    for lab, rule in LEG_RULES.items():
        show(lab, build_rows(races, a1_cur, rule, gate_hi))

    print("\n" + "=" * 108)
    print("【12】主要な比較のブートストラップ95%CI（レース単位・2000回）")
    cases = {
        "7S+7A 基準（5点）": (all_legs, None),
        "entropy上位25%（5点）": (all_legs, lambda r, a1, a2, a, e: e >= thr),
        "軸2が別ライン（5点）": (all_legs, lambda r, a1, a2, a, e: r["line"][a1] != r["line"][a2]),
        "entropy上位25%∧別ライン": (all_legs, gate_hi),
        "同上 ∧ 4点（最強外し）": (_mk_drop_strong(1), gate_hi),
    }
    print(f"  {'条件':<26}{'窓':<5}{'n':>6}{'ROI':>8}{'95%CI':>18}{'平均配当':>10}{'≥10倍率':>9}")
    for lab, (rule, g) in cases.items():
        rows = build_rows(races, a1_cur, rule, g)
        for win in ("掃引", "確認"):
            t = [x for x in rows if (x[0] <= CONFIRM_END) == (win == "確認")]
            if len(t) < 30:
                continue
            lo, hi = boot_roi(t)
            s = summ(rows, win)
            print(f"  {lab:<26}{win:<5}{s['n']:>6}{s['roi']:>7.1f}%"
                  f"  [{lo:5.1f}, {hi:5.1f}]{s['avg']:>9.1f}倍{s['ge10']:>8.0f}%")


if __name__ == "__main__":
    main()
