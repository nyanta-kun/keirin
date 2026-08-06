#!/usr/bin/env python3
"""P-A: 残り6車の力関係でセグメントを切り、セグメント別に買い目を出し分ける。

## ユーザー仮説（2026-08-06）

> 「B5がベースになると思うが、これも最初の1車を落とした後と同様に**残った6車の力関係**で
>  条件が絞れるように思う。残った全車の競走得点・3着内率が同程度だった、単騎に強い選手が
>  いた、など。6車にした後の残りの組み合わせによりいくつかのパターンで買い目を絞り、
>  それぞれのパターンのROIを最上とし、結果6車に絞った場合のROI平均を上げられる」

## ⚠️ この検証の最大の罠（先に宣言する）

「セグメントごとに最良の構成を選ぶ」は**多重比較そのもの**。
9構成 × 4セグメント = 36セルから argmax を拾えば、同じデータ上では必ず改善して見える。
そこで本スクリプトは次の手順を強制する:

1. **掃引窓**（2025-07-01〜2026-08-04）でセグメント別の argmax を決める（＝方針の決定）
2. その方針を**確認窓**（2024-10-01〜2025-06-30）へそのまま適用し、
   **常に B5 を使う場合**と比較する（一度きり）
3. 日ブロック bootstrap で ΔROI の CI を出す

**確認窓で B5 を上回らなければ不採用。** 掃引窓の数字は方針決定にしか使わない。

## セグメント（すべて発走前・オッズ非使用・結果非依存）

| 次元 | 定義 |
|---|---|
| `拮抗度` | 残り6車のモデル3着内率(pp3)の標準偏差を3分位（拮抗 / 中間 / 序列明確） |
| `強い単騎` | 6車の中の単騎が、6車のモデル順で2位以内にいるか |
| `別ライン先頭の規模` | 最強別ラインの車数（3車以上 / 2車 / 別ライン先頭なし） |
| `本命ライン残存` | 本命ラインから残った車数（0 / 1 / 2以上） |

DB は読み取りのみ（キャッシュ利用）。
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
SWEEP = ("2025-07-01", "2026-08-04")
CONFIRM = ("2024-10-01", "2025-06-30")
LINE_ROLES = {"本命ライン番手", "本命ライン3番手以降"}


def build(others: list[int], roles: dict[int, str]) -> dict[str, tuple]:
    """本命を除く6車（モデル3着内率順）から買い目候補を作る。"""
    out: dict[str, tuple] = {}
    if len(others) < 6:
        return out
    pool = [f for f in others if roles.get(f) not in LINE_ROLES]     # 別ライン＋単騎
    lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)
    solos = [f for f in others if roles.get(f) == "単騎"]

    out["A1 6車上位3BOX三単"] = ("tf", [f"{a}-{b}-{c}"
                                for a, b, c in permutations(others[:3], 3)])
    out["A2 6車BOX三複"] = ("trio", [frozenset(c) for c in combinations(others, 3)])
    out["A3 6車上位4BOX三複"] = ("trio",
                            [frozenset(c) for c in combinations(others[:4], 3)])
    if len(pool) >= 3:
        out["B1 別+単BOX三複"] = ("trio",
                              [frozenset(c) for c in combinations(pool, 3)])
        out["B2 別+単上位3BOX三単"] = ("tf", [f"{a}-{b}-{c}" for a, b, c
                                     in permutations(pool[:3], 3)])
    if lead is not None and len(pool) >= 3:
        rest = [f for f in pool if f != lead]
        out["B5 別先頭固定×2着r1r2×3着総流し"] = ("tf", [
            f"{lead}-{a}-{c}" for a in rest[:2]
            for c in others if c not in (lead, a)])
        out["B7 別先頭固定×2着r1r2r3×3着総流し"] = ("tf", [
            f"{lead}-{a}-{c}" for a in rest[:3]
            for c in others if c not in (lead, a)])
        out["B8 別先頭軸三複総流し"] = ("trio", [frozenset((lead, a, c))
                                    for a, c in combinations(
                                        [f for f in others if f != lead], 2)])
    if len(pool) >= 4:
        h = pool[:2]
        out["B6 別+単r1r2頭×2着上位3×3着総流し"] = ("tf", [
            f"{x}-{a}-{c}" for x in h for a in [f for f in pool[:3] if f != x]
            for c in others if c not in (x, a)])
    if solos:
        s0 = solos[0]
        out["D1 単騎頭×3着総流し"] = ("tf", [
            f"{s0}-{a}-{c}" for a in [f for f in others if f != s0][:2]
            for c in others if c not in (s0, a)])
    return out


def segments(others: list[int], roles: dict[int, str],
             pr: dict, sd_edges: tuple) -> dict[str, str]:
    pp3 = np.array([pr[f][0] for f in others])
    sd = float(pp3.std())
    if sd < sd_edges[0]:
        kin = "拮抗"
    elif sd < sd_edges[1]:
        kin = "中間"
    else:
        kin = "序列明確"
    solos = [f for f in others if roles.get(f) == "単騎"]
    strong_solo = "強単騎あり" if any(others.index(f) < 2 for f in solos) else "強単騎なし"
    lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)
    n_favline = sum(1 for f in others if roles.get(f) in LINE_ROLES)
    return {
        "拮抗度": kin,
        "強い単騎": strong_solo,
        "別先頭": ("別先頭なし" if lead is None else "別先頭あり"),
        "本命ライン残": ("残0" if n_favline == 0 else
                    "残1" if n_favline == 1 else "残2以上"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()

    with SCORED.open("rb") as f:
        data = pickle.load(f)
    with PAYCACHE.open("rb") as f:
        pay = pickle.load(f)
    ents_all, pr_all = load_ents(), load_preds3()
    data = [d for d in data if d["race_key"] in pay
            and pay[d["race_key"]]["trio_odds"] and pay[d["race_key"]]["tf_odds"]]
    strat = [d for d in data if d["fav_ppw_gap12"] >= args.gap]
    thr = np.quantile([d["score"] for d in strat], 1 - args.top_frac)
    sel = [d for d in strat if d["score"] >= thr]

    # --- 全レースを1回だけ展開（構成ごとの払戻とセグメントを確定）---
    sds = []
    for d in sel:
        pr = pr_all.get(d["race_key"])
        if pr:
            fav = max(pr, key=lambda f: pr[f][1])
            sds.append(float(np.std([pr[f][0] for f in pr if f != fav])))
    sd_edges = (float(np.quantile(sds, 1 / 3)), float(np.quantile(sds, 2 / 3)))

    rows = []
    for d in sel:
        rk = d["race_key"]
        pr, ents = pr_all.get(rk), ents_all.get(rk)
        if not pr or not ents:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        P = pay[rk]
        pays = {}
        for nm, (bt, combos) in build(others, roles).items():
            if not combos:
                continue
            win = P["trio"] if bt == "trio" else P["tf"]
            odds = P["trio_odds"] if bt == "trio" else P["tf_odds"]
            pays[nm] = (STAKE / len(combos) * odds) if win in combos else 0.0
        rows.append({"date": d["race_date"], "pays": pays,
                     "seg": segments(others, roles, pr, sd_edges)})
    print(f"選別レース {len(rows):,} / 拮抗度の分位点 sd={sd_edges[0]:.4f}, {sd_edges[1]:.4f}")

    sw = [r for r in rows if SWEEP[0] <= r["date"] <= SWEEP[1]]
    cf = [r for r in rows if CONFIRM[0] <= r["date"] <= CONFIRM[1]]
    print(f"掃引窓 {len(sw):,}R / 確認窓 {len(cf):,}R")

    def roi(items: list[float]) -> float:
        return sum(items) / (len(items) * STAKE) * 100 if items else float("nan")

    ALL = sorted({k for r in rows for k in r["pays"]})
    BASE = "B5 別先頭固定×2着r1r2×3着総流し"

    # --- 掃引窓: セグメント×構成 の表 ---
    for dim in ("拮抗度", "強い単騎", "別先頭", "本命ライン残"):
        print(f"\n{'=' * 118}\n=== 掃引窓: セグメント [{dim}] × 構成の ROI% "
              f"（括弧内は的中%・n）===")
        vals = sorted({r["seg"][dim] for r in sw})
        print(f"  {'構成':<32}" + "".join(f"{v:>26}" for v in vals))
        for nm in ALL:
            cells = []
            for v in vals:
                it = [r["pays"][nm] for r in sw
                      if r["seg"][dim] == v and nm in r["pays"]]
                if len(it) < 40:
                    cells.append(f"{'—':>26}")
                    continue
                hit = sum(1 for x in it if x > 0) / len(it) * 100
                cells.append(f"{roi(it):11.1f} ({hit:4.1f}% n={len(it):4})")
            print(f"  {nm:<32}" + "".join(cells))

    # --- 方針の決定と確認窓での検証 ---
    print(f"\n{'=' * 118}\n=== 混合方針の一度きり検証（掃引窓で決めて確認窓で適用）===")
    days_cf = sorted({r["date"] for r in cf})
    rng = np.random.default_rng(42)
    for dim in ("拮抗度", "強い単騎", "別先頭", "本命ライン残"):
        policy = {}
        for v in sorted({r["seg"][dim] for r in sw}):
            best, bv = BASE, -1e9
            for nm in ALL:
                it = [r["pays"][nm] for r in sw
                      if r["seg"][dim] == v and nm in r["pays"]]
                if len(it) < 60:
                    continue
                if roi(it) > bv:
                    best, bv = nm, roi(it)
            policy[v] = best
        # 確認窓へ適用
        mix, base = [], []
        for r in cf:
            p = policy.get(r["seg"][dim], BASE)
            mix.append(r["pays"].get(p, r["pays"].get(BASE, 0.0)))
            base.append(r["pays"].get(BASE, 0.0))
        # bootstrap
        by_day = defaultdict(lambda: ([], []))
        for r in cf:
            p = policy.get(r["seg"][dim], BASE)
            by_day[r["date"]][0].append(r["pays"].get(p, r["pays"].get(BASE, 0.0)))
            by_day[r["date"]][1].append(r["pays"].get(BASE, 0.0))
        diffs = []
        for _ in range(2000):
            pk = rng.choice(len(days_cf), len(days_cf), replace=True)
            cm = rm = cb = rb = 0.0
            for i in pk:
                m, b = by_day[days_cf[i]]
                cm += STAKE * len(m)
                rm += sum(m)
                cb += STAKE * len(b)
                rb += sum(b)
            if cm > 0 and cb > 0:
                diffs.append(rm / cm * 100 - rb / cb * 100)
        a = np.array(diffs)
        lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
        print(f"\n  [{dim}] 方針: " + " / ".join(f"{k}→{v[:14]}" for k, v in policy.items()))
        print(f"    確認窓 混合 ROI {roi(mix):6.1f}%  vs  常にB5 {roi(base):6.1f}%  "
              f"ΔROI {a.mean():+6.1f}pt 95%CI [{lo:+6.1f}, {hi:+6.1f}]  "
              f"{'✅有意' if lo > 0 else '❌有意差なし'}")

    # --- 参考: 確認窓での各構成の素の成績 ---
    print(f"\n{'=' * 118}\n=== 参考: 確認窓での各構成（全セグメント込み）===")
    print(f"  {'構成':<34}   n   的中%   ROI%")
    for nm in ALL:
        it = [r["pays"][nm] for r in cf if nm in r["pays"]]
        if len(it) < 100:
            continue
        hit = sum(1 for x in it if x > 0) / len(it) * 100
        print(f"  {nm:<34} {len(it):4} {hit:6.2f} {roi(it):6.1f}")


if __name__ == "__main__":
    main()
