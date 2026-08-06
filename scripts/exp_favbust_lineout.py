#!/usr/bin/env python3
"""P-A: 本命ラインを落として「別ライン＋単騎」で組む買い目の実測（オッズ非使用）。

## 根拠（`exp_favbust_roles.py` の実測）

本命が実際に飛んだ621レースでの役割別 1着率 / 3着内率:

| 役割 | 1着率 | 3着内率 |
|---|---|---|
| 別ライン先頭(最強) | **28.05%** | **64.85%** |
| 別ライン番手 | 17.52% | 58.72% |
| 単騎 | 15.70% | 43.93% |
| 本命ライン3番手以降 | 8.21% | 31.61% |
| **本命ライン番手** | **7.79%** | **33.27%** |

**本命が飛ぶときは、その番手も一緒に飛ぶ**（ライン共倒れ）。
1着の内訳も別ライン勢だけで75.5%、本命ライン勢は10.9%しかない。

⚠️ 本命を除外している以上、**本命が3着内に残る71%のレースでは構造上絶対に当たらない**。
よって設計は「本命が飛んだケース」の分布だけで行うのが正しい。

## 測る買い目

| 記号 | 内容 |
|---|---|
| A1 | 【対照】6車のモデル上位3車BOX 三連単（Phase6 の現行最良） |
| A2 | 【対照】6車BOX 三連複（20点） |
| B1 | 別ライン＋単騎 の BOX 三連複（点数可変） |
| B2 | 別ライン＋単騎 の上位3車BOX 三連単（6点） |
| B3 | 1着=別ライン先頭(最強)固定 × 2-3着=別ライン＋単騎の残り順列 |
| B4 | 1着=別ライン＋単騎の r1,r2（2車） × 2-3着=同プール残り順列 |
| B5 | 1着=別ライン先頭(最強)固定 × 2着=同プールr1,r2 × **3着=全5車総流し** |
| B6 | 1着=別ライン＋単騎の r1,r2 × 2着=同プール上位3 × **3着=全5車総流し** |
| C1 | 本命ライン**番手のみ**除外した5車の上位3車BOX 三連単（6点） |

DB は読み取りのみ（すべてキャッシュ利用）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_favbust_roles import load_ents, role_of  # noqa: E402
from scripts.exp_highpay_fav_bust import load_preds3  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
SCORED = CACHE_DIR / "favbust_scored.pkl"
PAYCACHE = CACHE_DIR / "favbust_payouts.pkl"
STAKE = 10_000


def build(others: list[int], roles: dict[int, str]) -> dict[str, tuple]:
    """others: 本命を除く6車（モデル3着内率順）。"""
    out: dict[str, tuple] = {}
    if len(others) < 6:
        return out

    # --- 対照 ---
    out["A1 6車の上位3BOX三連単"] = ("tf", [f"{a}-{b}-{c}"
                                     for a, b, c in permutations(others[:3], 3)])
    out["A2 6車BOX三連複"] = ("trio", [frozenset(c) for c in combinations(others, 3)])

    line_roles = {"本命ライン番手", "本命ライン3番手以降"}
    pool = [f for f in others if roles.get(f) not in line_roles]   # 別ライン＋単騎
    lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)

    if len(pool) >= 3:
        out["B1 別ライン+単騎BOX三連複"] = ("trio",
                                    [frozenset(c) for c in combinations(pool, 3)])
        out["B2 別ライン+単騎 上位3BOX三連単"] = ("tf",
                                        [f"{a}-{b}-{c}" for a, b, c
                                         in permutations(pool[:3], 3)])
    if lead is not None and len(pool) >= 3:
        rest = [f for f in pool if f != lead]
        out["B3 別ライン先頭固定×同プール流し"] = ("tf", [f"{lead}-{a}-{b}"
                                          for a, b in permutations(rest, 2)])
        out["B5 別ライン先頭固定×2着r1r2×3着総流し"] = ("tf", [
            f"{lead}-{a}-{c}" for a in rest[:2]
            for c in others if c not in (lead, a)])
    if len(pool) >= 4:
        heads = pool[:2]
        out["B4 別ライン+単騎r1r2頭×同プール流し"] = ("tf", [
            f"{h}-{a}-{b}" for h in heads
            for a, b in permutations([f for f in pool if f != h], 2)])
        out["B6 同プールr1r2頭×2着上位3×3着総流し"] = ("tf", [
            f"{h}-{a}-{c}" for h in heads
            for a in [f for f in pool[:3] if f != h]
            for c in others if c not in (h, a)])

    pool2 = [f for f in others if roles.get(f) != "本命ライン番手"]
    if len(pool2) >= 3:
        out["C1 番手のみ除外5車の上位3BOX三連単"] = ("tf", [
            f"{a}-{b}-{c}" for a, b, c in permutations(pool2[:3], 3)])
    return out


def summarize(items: list[tuple], n_all: int) -> dict:
    pays = np.array([p for _, p in items], dtype=float)
    hit = pays > 0
    return {"点数": float(np.mean([k for k, _ in items])), "n": len(items),
            "成立%": len(items) / n_all * 100,
            "的中%": hit.mean() * 100,
            "ROI%": pays.sum() / (len(items) * STAKE) * 100,
            "平均払戻": float(pays[hit].mean()) if hit.any() else 0.0,
            "中央払戻": float(np.median(pays[hit])) if hit.any() else 0.0,
            "最大": float(pays.max()),
            "10万+": int((pays >= 100_000).sum()),
            "30万+": int((pays >= 300_000).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()

    with SCORED.open("rb") as f:
        data = pickle.load(f)
    with PAYCACHE.open("rb") as f:
        pay = pickle.load(f)
    ents_all = load_ents()
    pr_all = load_preds3()

    data = [d for d in data if d["race_key"] in pay
            and pay[d["race_key"]]["trio_odds"] and pay[d["race_key"]]["tf_odds"]]
    strat = [d for d in data if d["fav_ppw_gap12"] >= args.gap]
    thr = np.quantile([d["score"] for d in strat], 1 - args.top_frac)
    sel = [d for d in strat if d["score"] >= thr]
    n_days = len({d["race_date"] for d in data})
    print(f"選別レース {len(sel):,} ({len(sel) / n_days:.2f}件/日) / "
          f"本命バスト率 {np.mean([d['bust'] for d in sel]) * 100:.2f}%")

    res: dict[str, list] = defaultdict(list)
    by_mo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for d in sel:
        rk = d["race_key"]
        pr, ents = pr_all.get(rk), ents_all.get(rk)
        if not pr or not ents:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        P = pay[rk]
        for nm, (bt, combos) in build(others, roles).items():
            if not combos:
                continue
            win = P["trio"] if bt == "trio" else P["tf"]
            odds = P["trio_odds"] if bt == "trio" else P["tf_odds"]
            v = (len(combos), (STAKE / len(combos) * odds) if win in combos else 0.0)
            res[nm].append(v)
            by_mo[d["race_date"][:7]][nm].append(v)

    order = ["A1 6車の上位3BOX三連単", "A2 6車BOX三連複",
             "B1 別ライン+単騎BOX三連複", "B2 別ライン+単騎 上位3BOX三連単",
             "B3 別ライン先頭固定×同プール流し", "B4 別ライン+単騎r1r2頭×同プール流し",
             "B5 別ライン先頭固定×2着r1r2×3着総流し",
             "B6 同プールr1r2頭×2着上位3×3着総流し",
             "C1 番手のみ除外5車の上位3BOX三連単"]
    print(f"\n{'=' * 122}\n=== 買い目別（1レース1万円・均等配分・選別レースのみ）===")
    print("  買い目                              点数 成立%   n   的中%   ROI%   "
          "平均払戻  中央払戻     最大  10万+ 30万+")
    for nm in order:
        if not res.get(nm):
            continue
        s = summarize(res[nm], len(sel))
        print(f"  {nm:<34} {s['点数']:4.0f} {s['成立%']:5.1f} {s['n']:5} "
              f"{s['的中%']:6.2f} {s['ROI%']:6.1f} {s['平均払戻']:9.0f} "
              f"{s['中央払戻']:9.0f} {s['最大']:9.0f} {s['10万+']:5} {s['30万+']:5}")

    # --- 裾依存 ---
    print(f"\n{'=' * 122}\n=== 裾依存 ===")
    print("  買い目                              ROI%  除・上1 除・上3 除・上5 除・上10  上3シェア%")
    for nm in order:
        if not res.get(nm):
            continue
        pays = np.sort(np.array([p for _, p in res[nm]]))[::-1]
        cost = len(res[nm]) * STAKE
        tot = pays.sum()
        print(f"  {nm:<34} {tot / cost * 100:5.1f}"
              + "".join(f"{(tot - pays[:k].sum()) / cost * 100:8.1f}"
                        for k in (1, 3, 5, 10))
              + f"{pays[:3].sum() / tot * 100:11.1f}")

    # --- 月次一貫性（A1 との比較）---
    print(f"\n{'=' * 122}\n=== 月次 ROI（主要4本）===")
    picks = ["A1 6車の上位3BOX三連単", "B2 別ライン+単騎 上位3BOX三連単",
             "B5 別ライン先頭固定×2着r1r2×3着総流し", "B1 別ライン+単騎BOX三連複"]
    print("   月    " + "".join(f"{p[:22]:>24}" for p in picks))
    keep = defaultdict(list)
    for mo in sorted(by_mo):
        cells = []
        for p in picks:
            it = by_mo[mo].get(p)
            if it and len(it) >= 15:
                v = sum(x for _, x in it) / (len(it) * STAKE) * 100
                keep[p].append(v)
                cells.append(f"{v:24.1f}")
            else:
                cells.append(" " * 24)
        print(f"  {mo}" + "".join(cells))
    print()
    for p in picks:
        v = np.array(keep[p])
        if len(v) == 0:
            continue
        print(f"  {p:<34} 平均{v.mean():6.1f}% 中央{np.median(v):6.1f}% "
              f"100%超 {int((v > 100).sum()):2}/{len(v)} 最低{v.min():6.1f}%")

    # --- 日ブロック bootstrap: B系 vs A1 ---
    print(f"\n{'=' * 122}\n=== 日ブロック bootstrap: 各案 − A1 の ΔROI（2,000回）===")
    days = sorted({d["race_date"] for d in sel})
    idx = {}
    for d in sel:
        idx.setdefault(d["race_date"], []).append(d["race_key"])
    per_day: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    ptr: dict[str, int] = defaultdict(int)
    for d in sel:
        for nm in order:
            pass
    # 再走査して日別に払戻を積む
    per_day.clear()
    for d in sel:
        rk = d["race_key"]
        pr, ents = pr_all.get(rk), ents_all.get(rk)
        if not pr or not ents:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        P = pay[rk]
        for nm, (bt, combos) in build(others, roles).items():
            if not combos:
                continue
            win = P["trio"] if bt == "trio" else P["tf"]
            odds = P["trio_odds"] if bt == "trio" else P["tf_odds"]
            per_day[d["race_date"]][nm].append(
                (STAKE / len(combos) * odds) if win in combos else 0.0)
    rng = np.random.default_rng(42)
    base = "A1 6車の上位3BOX三連単"
    for nm in order:
        if nm == base or not res.get(nm):
            continue
        diffs = []
        for _ in range(2000):
            pk = rng.choice(len(days), len(days), replace=True)
            ca = ra = cb = rb = 0.0
            for i in pk:
                dd = days[i]
                for v in per_day[dd].get(nm, []):
                    ca += STAKE
                    ra += v
                for v in per_day[dd].get(base, []):
                    cb += STAKE
                    rb += v
            if ca > 0 and cb > 0:
                diffs.append(ra / ca * 100 - rb / cb * 100)
        a = np.array(diffs)
        lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
        mark = "有意(＋)" if lo > 0 else ("有意(−)" if hi < 0 else "有意差なし")
        print(f"  {nm:<34} ΔROI {a.mean():+7.1f}pt  95%CI [{lo:+7.1f}, {hi:+7.1f}]  {mark}")


if __name__ == "__main__":
    main()
