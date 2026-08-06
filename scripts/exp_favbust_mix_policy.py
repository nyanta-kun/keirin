#!/usr/bin/env python3
"""P-A: 選別を緩めて標本を増やし、1次元2分割 × 候補3つで出し分けを再検証する。

## 経緯（Phase 8 の反省）

セグメント別の出し分けは4次元すべてで確認窓が悪化した。原因は標本と自由度:
選別後2,167Rで的中100件強、そこへ **9構成 × 3分割 = 27セル**の argmax を当てた。

ユーザー指示（2026-08-06）:
> 「1と3の組み合わせで検証して。1レース購入10000円のため、**三連複・三連単両方を購入**
>  として、**三連複だけでも当てるという推奨**もありかと思います」

したがって本スクリプトは3点を変える:

1. **選別を上位10% → 30%** に緩める（3.22件/日 → 約9.7件/日・標本3倍）
2. **セグメントは1次元・2分割**。候補構成も **3つだけ**（自由度 27セル → 6セル）
3. **三連複と三連単の併買**を candidates に加える
   （1万円を分割。三連複で当てて、三連単で伸ばす）

## 候補構成（3つだけ）

| 記号 | 内容 | 点数 |
|---|---|---|
| `TRIO` | 別ライン＋単騎 BOX 三連複（本命ラインを落とす） | 可変(4〜10) |
| `TANSHO` | 別ライン先頭固定 × 2着=同プールr1r2 × 3着=全5車総流し（三連単） | 8 |
| `MIX` | 上記2つを 5,000円 / 5,000円 で併買 | 合算 |

## 手順（Phase 8 と同じく先に固定）

掃引窓 2025-07-01〜2026-08-04 で方針決定 → 確認窓 2024-10-01〜2025-06-30 で一度きり。
比較対象は「常に MIX」「常に TANSHO」「常に TRIO」。

DB は読み取りのみ（キャッシュ利用）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from itertools import combinations
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


def legs(others: list[int], roles: dict[int, str]) -> tuple[list, list]:
    """(三連複の目, 三連単の目) を返す。"""
    pool = [f for f in others if roles.get(f) not in LINE_ROLES]
    lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)
    trio = [frozenset(c) for c in combinations(pool, 3)] if len(pool) >= 3 else []
    tf = []
    if lead is not None and len(pool) >= 3:
        rest = [f for f in pool if f != lead]
        tf = [f"{lead}-{a}-{c}" for a in rest[:2]
              for c in others if c not in (lead, a)]
    return trio, tf


def payout(trio_legs, tf_legs, P, budget_trio: float, budget_tf: float) -> dict:
    """賭け金配分に対する払戻と、券種別の的中フラグ。"""
    r = {"pay": 0.0, "trio_hit": 0, "tf_hit": 0, "cost": 0.0}
    if trio_legs and budget_trio > 0:
        r["cost"] += budget_trio
        if P["trio"] in trio_legs:
            r["trio_hit"] = 1
            r["pay"] += budget_trio / len(trio_legs) * P["trio_odds"]
    if tf_legs and budget_tf > 0:
        r["cost"] += budget_tf
        if P["tf"] in tf_legs:
            r["tf_hit"] = 1
            r["pay"] += budget_tf / len(tf_legs) * P["tf_odds"]
    return r


def agg(items: list[dict]) -> dict:
    if not items:
        return {}
    pays = np.array([x["pay"] for x in items])
    cost = sum(x["cost"] for x in items)
    return {"n": len(items),
            "的中%": float(np.mean(pays > 0) * 100),
            "三複的中%": float(np.mean([x["trio_hit"] for x in items]) * 100),
            "三単的中%": float(np.mean([x["tf_hit"] for x in items]) * 100),
            "ROI%": pays.sum() / cost * 100 if cost else float("nan"),
            "平均払戻": float(pays[pays > 0].mean()) if (pays > 0).any() else 0.0,
            "10万+": int((pays >= 100_000).sum()),
            "30万+": int((pays >= 300_000).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.30)
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
    n_days = len({d["race_date"] for d in data})
    print(f"選別: 抜け度>={args.gap * 100:.0f}pt × バスト上位{args.top_frac:.0%} … "
          f"{len(sel):,}R ({len(sel) / n_days:.2f}件/日) / "
          f"バスト率 {np.mean([d['bust'] for d in sel]) * 100:.2f}%")

    # 全レースを1回展開
    sds = []
    for d in sel:
        pr = pr_all.get(d["race_key"])
        if pr:
            fav = max(pr, key=lambda f: pr[f][1])
            sds.append(float(np.std([pr[f][0] for f in pr if f != fav])))
    sd_med = float(np.median(sds))

    rows = []
    for d in sel:
        rk = d["race_key"]
        pr, ents = pr_all.get(rk), ents_all.get(rk)
        if not pr or not ents:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        tl, fl = legs(others, roles)
        if not tl or not fl:
            continue
        P = pay[rk]
        solos = [f for f in others if roles.get(f) == "単騎"]
        lead = next((f for f in others if roles.get(f) == "別ライン先頭(最強)"), None)
        lead_size = (sum(1 for e in ents
                         if e["line_group"] is not None and lead is not None
                         and e["line_group"] == next(
                             x["line_group"] for x in ents
                             if int(x["frame_no"]) == lead))
                     if lead is not None else 0)
        rows.append({
            "date": d["race_date"],
            "TRIO": payout(tl, fl, P, STAKE, 0),
            "TANSHO": payout(tl, fl, P, 0, STAKE),
            "MIX": payout(tl, fl, P, STAKE / 2, STAKE / 2),
            "n_trio": len(tl), "n_tf": len(fl),
            "seg": {
                "拮抗度": "拮抗" if float(np.std([pr[f][0] for f in others]))
                        < sd_med else "序列明確",
                "強い単騎": "強単騎あり" if any(others.index(f) < 2 for f in solos)
                         else "強単騎なし",
                "本命ライン残": "残0-1" if sum(
                    1 for f in others if roles.get(f) in LINE_ROLES) <= 1 else "残2以上",
                "別先頭ライン規模": "3車以上" if lead_size >= 3 else "2車",
            }})
    print(f"成立 {len(rows):,}R / 三連複 平均{np.mean([r['n_trio'] for r in rows]):.1f}点 "
          f"/ 三連単 平均{np.mean([r['n_tf'] for r in rows]):.1f}点 "
          f"/ 拮抗度の中央値 sd={sd_med:.4f}")

    sw = [r for r in rows if SWEEP[0] <= r["date"] <= SWEEP[1]]
    cf = [r for r in rows if CONFIRM[0] <= r["date"] <= CONFIRM[1]]
    print(f"掃引窓 {len(sw):,}R / 確認窓 {len(cf):,}R")

    CAND = ["TRIO", "TANSHO", "MIX"]

    def show(items, title):
        print(f"\n  ── {title}（{len(items):,}R）──")
        print("    構成      n   的中%  三複的中% 三単的中%  ROI%   平均払戻  10万+ 30万+")
        for c in CAND:
            a = agg([r[c] for r in items])
            if not a:
                continue
            print(f"    {c:<8} {a['n']:5} {a['的中%']:6.2f} {a['三複的中%']:8.2f} "
                  f"{a['三単的中%']:8.2f} {a['ROI%']:6.1f} {a['平均払戻']:9.0f} "
                  f"{a['10万+']:5} {a['30万+']:5}")

    print(f"\n{'=' * 104}\n=== 1. 3候補の素の成績 ===")
    show(sw, "掃引窓")
    show(cf, "確認窓")

    # --- 賭け金配分の掃引（参考・方針決定には使わない）---
    print(f"\n{'=' * 104}\n=== 2. 三連複／三連単の配分掃引（参考）===")
    print("    三複:三単      掃引窓 ROI%  的中%  30万+  |  確認窓 ROI%  的中%  30万+")
    for bt in (10000, 7500, 5000, 2500, 0):
        line = f"    {bt:5}:{STAKE - bt:<5} "
        for items in (sw, cf):
            vals = []
            for r in items:
                # 再計算は不要: 線形結合で求まる
                t, f = r["TRIO"], r["TANSHO"]
                pay_ = t["pay"] * bt / STAKE + f["pay"] * (STAKE - bt) / STAKE
                vals.append(pay_)
            v = np.array(vals)
            line += (f"  {v.sum() / (len(v) * STAKE) * 100:9.1f} "
                     f"{np.mean(v > 0) * 100:6.2f} {int((v >= 300_000).sum()):5}  |")
        print(line)

    # --- 3. 1次元2分割 × 候補3つ の出し分け ---
    print(f"\n{'=' * 104}\n=== 3. セグメント別 出し分けの一度きり検証 ===")
    days_cf = sorted({r["date"] for r in cf})
    rng = np.random.default_rng(42)
    for dim in ("拮抗度", "強い単騎", "本命ライン残", "別先頭ライン規模"):
        vals = sorted({r["seg"][dim] for r in sw})
        print(f"\n  [{dim}] 掃引窓 ROI%:")
        policy = {}
        for v in vals:
            cells = []
            best, bv = "MIX", -1e9
            for c in CAND:
                it = [r[c] for r in sw if r["seg"][dim] == v]
                a = agg(it)
                cells.append(f"{c}={a['ROI%']:.1f}(n={a['n']})")
                if a["ROI%"] > bv:
                    best, bv = c, a["ROI%"]
            policy[v] = best
            print(f"    {v:<12} " + "  ".join(cells) + f"  → {best}")
        by_day = defaultdict(lambda: ([], [], []))
        for r in cf:
            p = policy.get(r["seg"][dim], "MIX")
            by_day[r["date"]][0].append(r[p]["pay"])
            by_day[r["date"]][1].append(r["MIX"]["pay"])
            by_day[r["date"]][2].append(r["TANSHO"]["pay"])
        mix = [r[policy.get(r["seg"][dim], "MIX")]["pay"] for r in cf]
        b_mix = [r["MIX"]["pay"] for r in cf]
        b_tan = [r["TANSHO"]["pay"] for r in cf]
        diffs = []
        for _ in range(2000):
            pk = rng.choice(len(days_cf), len(days_cf), replace=True)
            s = m = 0.0
            cnt = 0
            for i in pk:
                a, b, _c = by_day[days_cf[i]]
                s += sum(a)
                m += sum(b)
                cnt += len(a)
            if cnt:
                diffs.append((s - m) / (cnt * STAKE) * 100)
        a = np.array(diffs)
        lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
        print(f"    確認窓: 出し分け {np.sum(mix) / (len(mix) * STAKE) * 100:6.1f}%  "
              f"常にMIX {np.sum(b_mix) / (len(b_mix) * STAKE) * 100:6.1f}%  "
              f"常にTANSHO {np.sum(b_tan) / (len(b_tan) * STAKE) * 100:6.1f}%  "
              f"Δ(出し分け−MIX) {a.mean():+6.1f}pt [{lo:+6.1f}, {hi:+6.1f}] "
              f"{'✅有意' if lo > 0 else '❌有意差なし'}")

    # --- 4. 月次（MIX）---
    print(f"\n{'=' * 104}\n=== 4. 月次 ROI（全期間・MIX と TANSHO と TRIO）===")
    by_mo = defaultdict(list)
    for r in rows:
        by_mo[r["date"][:7]].append(r)
    keep = defaultdict(list)
    print("   月      n   " + "".join(f"{c:>12}" for c in CAND))
    for mo in sorted(by_mo):
        g = by_mo[mo]
        if len(g) < 30:
            continue
        cells = []
        for c in CAND:
            v = np.array([x[c]["pay"] for x in g])
            roi = v.sum() / (len(v) * STAKE) * 100
            keep[c].append(roi)
            cells.append(f"{roi:12.1f}")
        print(f"  {mo} {len(g):4}   " + "".join(cells))
    print()
    for c in CAND:
        v = np.array(keep[c])
        print(f"  {c:<8} 平均{v.mean():6.1f}% 中央{np.median(v):6.1f}% "
              f"100%超 {int((v > 100).sum()):2}/{len(v)} 最低{v.min():6.1f}% "
              f"最高{v.max():6.1f}%")


if __name__ == "__main__":
    main()
