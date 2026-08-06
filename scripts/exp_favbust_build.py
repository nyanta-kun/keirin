#!/usr/bin/env python3
"""P-A 本命バスト型: 選別を固定した上で「残り6車」の買い目を組む（オッズ非使用）。

## 固定した母集団（ユーザー承認・2026-08-06）

    7車 ∧ 軸1(モデル1着率最上位) == WINTICKET◎
      ∧ 抜け度（1着率の1位−2位差） >= 20pt
      ∧ バスト確率 上位10%

実測: 1日3.22件 / 実バスト率 28.66%（層内基準13.69%・lift 2.09）/
月次 23/23ヶ月で基準超え。

## 組み立ての前提

**本命が4着以下**という前提なので、3着以内は必ず**残り6車**で埋まる。
したがって本命を除いた6車を、モデル3着内率順に r1..r6 と並べて買い目を作る。

⚠️ **重要な恒等式**: 「6車ボックス三連複（20点）」の的中率は、
   定義上 **本命バスト率そのもの**に一致する（本命が4着以下なら上位3着は必ず6車の中）。
   したがって選別精度がそのまま的中率の上限になり、
   **ROI = バスト率 × E[三連複配当 | バスト] / 点数** で決まる。
   20点なら ROI 0.75 に到達するには E[配当] が 52倍必要。ここを実測で確かめる。

## 出す指標

点数 / 的中率 / **ROI** / **平均払い戻し** / 中央払い戻し / 最大 / 10万+・30万+件数。
比較対象として「選別なし（抜け度>=20pt 全件）」と「非選別レース」も併記する。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_highpay_fav_bust import load_preds3  # noqa: E402
from src.database import get_connection  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
SCORED = CACHE_DIR / "favbust_scored.pkl"
PAYCACHE = CACHE_DIR / "favbust_payouts.pkl"
STAKE = 10_000


def load_payouts(race_keys: list[str]) -> dict:
    """各レースの着順と的中目の三連複/三連単オッズ。"""
    if PAYCACHE.exists():
        with PAYCACHE.open("rb") as f:
            print(f"[cache] {PAYCACHE.name}", flush=True)
            return pickle.load(f)
    out: dict[str, dict] = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 500):
            ch = race_keys[i:i + 500]
            ph = ",".join("?" * len(ch))
            fin = defaultdict(dict)
            for r in c.execute(
                    "SELECT race_key, frame_no, finish_order FROM keirin.wt_entries "
                    f"WHERE race_key IN ({ph})", ch):
                fin[r["race_key"]][int(r["frame_no"])] = r["finish_order"]
            tb, fb = defaultdict(dict), defaultdict(dict)
            for r in c.execute(
                    "SELECT race_key, bet_type, combination, odds_value "
                    f"FROM keirin.wt_odds WHERE bet_type IN ('trio','trifecta') "
                    f"AND race_key IN ({ph}) AND odds_value > 0", ch):
                if r["bet_type"] == "trifecta":
                    fb[r["race_key"]][r["combination"]] = float(r["odds_value"])
                else:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", r["combination"]))
                    if len(p) == 3:
                        tb[r["race_key"]][p] = float(r["odds_value"])
            for rk in ch:
                f = fin.get(rk) or {}
                top = sorted((v, k) for k, v in f.items() if v and v >= 1)[:3]
                if len(top) < 3:
                    continue
                order = [k for _, k in top]
                out[rk] = {
                    "order": order,
                    "trio": frozenset(order),
                    "tf": "-".join(map(str, order)),
                    "trio_odds": tb.get(rk, {}).get(frozenset(order)),
                    "tf_odds": fb.get(rk, {}).get("-".join(map(str, order))),
                }
            if (i // 500) % 10 == 0:
                print(f"  payout {i}/{len(race_keys)}", flush=True)
    tmp = PAYCACHE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(PAYCACHE)
    return out


def build_bets(r: list[int]) -> dict[str, tuple]:
    """本命を除いた6車 r1..r6（モデル3着内率順）から買い目を作る。"""
    if len(r) < 6:
        return {}
    b: dict[str, tuple] = {}
    # --- 三連複 ---
    b["三連複 6車BOX"] = ("trio", [frozenset(c) for c in combinations(r, 3)])
    b["三連複 上位5車BOX"] = ("trio", [frozenset(c) for c in combinations(r[:5], 3)])
    b["三連複 上位4車BOX"] = ("trio", [frozenset(c) for c in combinations(r[:4], 3)])
    b["三連複 r1軸-総流し"] = ("trio",
                          [frozenset((r[0], *c)) for c in combinations(r[1:], 2)])
    b["三連複 r1r2軸-総流し"] = ("trio",
                           [frozenset((r[0], r[1], x)) for x in r[2:]])
    # --- 三連単 ---
    b["三連単 上位3車BOX"] = ("tf", [f"{a}-{x}-{y}"
                                for a, x, y in permutations(r[:3], 3)])
    b["三連単 r1頭-総流し"] = ("tf", [f"{r[0]}-{a}-{c}"
                               for a, c in permutations(r[1:], 2)])
    b["三連単 r1頭×2着r2r3×3着総流し"] = ("tf", [f"{r[0]}-{a}-{c}" for a in r[1:3]
                                       for c in r if c not in (r[0], a)])
    b["三連単 r1r2マルチ×3着総流し"] = ("tf", [f"{a}-{x}-{c}"
                                    for a, x in ((r[0], r[1]), (r[1], r[0]))
                                    for c in r[2:]])
    b["三連単 r2頭×2着r1r3×3着総流し"] = ("tf", [f"{r[1]}-{a}-{c}"
                                       for a in (r[0], r[2])
                                       for c in r if c not in (r[1], a)])
    return b


def summarize(items: list[tuple]) -> dict:
    """items: (n_pt, payout) のリスト。payout=0 は不的中。"""
    n = len(items)
    pays = np.array([p for _, p in items], dtype=float)
    hit = pays > 0
    return {
        "点数": float(np.mean([k for k, _ in items])),
        "n": n, "的中%": hit.mean() * 100,
        "ROI%": pays.sum() / (n * STAKE) * 100,
        "平均払戻": float(pays[hit].mean()) if hit.any() else 0.0,
        "中央払戻": float(np.median(pays[hit])) if hit.any() else 0.0,
        "最大払戻": float(pays.max()),
        ">=10万": int((pays >= 100_000).sum()),
        ">=30万": int((pays >= 300_000).sum()),
    }


def report(title: str, rows: dict[str, list]) -> None:
    print(f"\n  ── {title} ──")
    print("   買い目                        点数   n    的中%   ROI%   平均払戻  "
          "中央払戻   最大払戻  10万+ 30万+")
    for nm, items in rows.items():
        if not items:
            continue
        s = summarize(items)
        print(f"   {nm:<28} {s['点数']:4.0f} {s['n']:5} {s['的中%']:6.2f} "
              f"{s['ROI%']:6.1f} {s['平均払戻']:9.0f} {s['中央払戻']:9.0f} "
              f"{s['最大払戻']:10.0f} {s['>=10万']:5} {s['>=30万']:5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()

    with SCORED.open("rb") as f:
        data = pickle.load(f)
    pay = load_payouts(sorted({d["race_key"] for d in data}))
    pr_all = load_preds3()

    data = [d for d in data if d["race_key"] in pay
            and pay[d["race_key"]]["trio_odds"] and pay[d["race_key"]]["tf_odds"]]
    n_days = len({d["race_date"] for d in data})

    strat = [d for d in data if d["fav_ppw_gap12"] >= args.gap]
    thr = np.quantile([d["score"] for d in strat], 1 - args.top_frac)
    sel = [d for d in strat if d["score"] >= thr]
    rest = [d for d in strat if d["score"] < thr]
    print(f"\n母集団: 一致 ∧ 抜け度>={args.gap * 100:.0f}pt … {len(strat):,}レース "
          f"(バスト率 {np.mean([d['bust'] for d in strat]) * 100:.2f}%)")
    print(f"選別  : + バスト確率上位{args.top_frac:.0%} … {len(sel):,}レース "
          f"({len(sel) / n_days:.2f}件/日・バスト率 "
          f"{np.mean([d['bust'] for d in sel]) * 100:.2f}%)")

    def run(group: list[dict]) -> dict[str, list]:
        out: dict[str, list] = defaultdict(list)
        for d in group:
            rk = d["race_key"]
            pr = pr_all.get(rk)
            if not pr:
                continue
            fav = max(pr, key=lambda f: pr[f][1])
            others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
            bets = build_bets(others)
            P = pay[rk]
            for nm, (bt, combos) in bets.items():
                if not combos:
                    continue
                win = P["trio"] if bt == "trio" else P["tf"]
                odds = P["trio_odds"] if bt == "trio" else P["tf_odds"]
                hit = win in combos
                out[nm].append((len(combos),
                                (STAKE / len(combos) * odds) if hit else 0.0))
        return out

    print(f"\n{'=' * 118}\n=== 本命を除いた6車での組み立て（1レース1万円・均等配分）===")
    report(f"【選別あり】抜け度>={args.gap * 100:.0f}pt × バスト上位{args.top_frac:.0%}"
           f"（{len(sel):,}R・{len(sel) / n_days:.2f}件/日）", run(sel))
    report(f"【選別なし】抜け度>={args.gap * 100:.0f}pt 全件（{len(strat):,}R）", run(strat))
    report(f"【非選別】上記から選別分を除く（{len(rest):,}R）", run(rest))

    # --- 月次一貫性（最良候補2本）---
    print(f"\n{'=' * 118}\n=== 月次の一貫性（選別あり）===")
    picks = ["三連単 上位3車BOX", "三連複 6車BOX"]
    by_mo = defaultdict(list)
    for d in sel:
        by_mo[d["race_date"][:7]].append(d)
    print("   月       n   " + "  ".join(f"{p:<28}" for p in picks))
    print("            " + "  ".join(f"{'的中%   ROI%   平均払戻':<28}" for _ in picks))
    keep = defaultdict(list)
    for mo in sorted(by_mo):
        g = by_mo[mo]
        if len(g) < 20:
            continue
        r = run(g)
        cells = []
        for p in picks:
            s = summarize(r[p]) if r[p] else None
            if s:
                keep[p].append(s["ROI%"])
                cells.append(f"{s['的中%']:6.2f} {s['ROI%']:6.1f} {s['平均払戻']:9.0f}   ")
            else:
                cells.append(" " * 28)
        print(f"  {mo} {len(g):4}   " + "".join(cells))
    for p in picks:
        v = np.array(keep[p])
        print(f"\n  {p}: 月次ROI 平均 {v.mean():.1f}% / 中央 {np.median(v):.1f}% / "
              f"**100%超えた月 {int((v > 100).sum())}/{len(v)}** / 最低 {v.min():.1f}%")

    # --- 裾依存 & 日ブロック bootstrap ---
    print(f"\n{'=' * 118}\n=== 裾依存（高額配当を除いたときの ROI）===")
    sel_run, rest_run = run(sel), run(rest)
    print("   買い目                        ROI%   除・上1  除・上3  除・上5  除・上10  上3が回収に占める%")
    for nm in picks + ["三連単 r2頭×2着r1r3×3着総流し", "三連複 上位4車BOX"]:
        items = sel_run.get(nm)
        if not items:
            continue
        pays = np.sort(np.array([p for _, p in items]))[::-1]
        cost = len(items) * STAKE
        tot = pays.sum()
        print(f"   {nm:<28} {tot / cost * 100:5.1f} "
              + "".join(f"{(tot - pays[:k].sum()) / cost * 100:8.1f}"
                        for k in (1, 3, 5, 10))
              + f"{pays[:3].sum() / tot * 100:16.1f}")

    print(f"\n{'=' * 118}\n=== 日ブロック bootstrap: 選別あり vs 非選別 の ΔROI（2,000回）===")
    days = sorted({d["race_date"] for d in strat})
    di = {d: i for i, d in enumerate(days)}
    for nm in picks + ["三連単 r2頭×2着r1r3×3着総流し"]:
        sd = defaultdict(list); rd = defaultdict(list)
        for grp, store in ((sel, sd), (rest, rd)):
            r = run(grp)
            pass
        # レースごとに払戻を引き直す（run を2回呼ばないよう手元で持つ）
        sd.clear(); rd.clear()
        for d, it in zip(sel, sel_run[nm]):
            sd[d["race_date"]].append(it[1])
        for d, it in zip(rest, rest_run[nm]):
            rd[d["race_date"]].append(it[1])
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(2000):
            pick = rng.choice(len(days), len(days), replace=True)
            cs = rs = cr = rr = 0.0
            for i in pick:
                dd = days[i]
                for v in sd.get(dd, []):
                    cs += STAKE; rs += v
                for v in rd.get(dd, []):
                    cr += STAKE; rr += v
            if cs > 0 and cr > 0:
                diffs.append(rs / cs * 100 - rr / cr * 100)
        a = np.array(diffs)
        lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
        print(f"   {nm:<28} ΔROI {a.mean():+6.1f}pt  95%CI [{lo:+6.1f}, {hi:+6.1f}]  "
              f"{'有意' if lo > 0 else '**有意差なし**'}")


if __name__ == "__main__":
    main()
