#!/usr/bin/env python3
"""P-B / 7H2「3着だけ荒れる型」の検証（オッズ非使用・朝で完結）。

## パターン定義（台帳 keirin_ana_pattern_register の P-B）

    レース選別: 1着が抜けて強い（1着率 gap 大） ∧ 2着候補が同値で2車
    買い目    : 1着=本命固定 × 2着=2番手グループ2車 × 3着=総流し
                → 7車なら 2 × 5 = **10点**・1点1,000円（1レース1万円）

実例（ユーザー提示）: 和歌山8R 2026-08-06 → 2-1-6・479.8倍・14点714円で34.2万円。
7車10点なら **30万円到達には300倍**が必要。

## 7H1（P-A 本命バスト型）との関係

7H1 は「本命が飛ぶ」に賭ける。7H2 は**その裏返しで「本命は来る／荒れるのは3着だけ」**に賭ける。
母集団は同じ `7車 ∧ 軸1(ppw最上位) == WINTICKET◎` を使う
（市場も本命を支持している＝1着固定の前提が最も強い層）。

## 手順（[[keirin_7s7a_threshold_review]] の掃引窓／確認窓の分離を厳守）

1. 掃引窓 2025-07-01〜2026-07-15 で選別条件を掃引して候補を決める
2. 確認窓 2024-10-01〜2025-06-30 へ**そのまま適用**して一度きり確認
3. 月次一貫性・裾依存・日ブロック bootstrap を必ず見る（平均は符号反転を隠す）

DB は読み取りのみ（キャッシュ済み pkl のみ使用）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000
SWEEP = ("2025-07-01", "2026-07-15")
CONFIRM = ("2024-10-01", "2025-06-30")
FRESH = ("2026-07-16", "2026-08-04")


# ---------------------------------------------------------------- データ読み込み
def load_all(n_car: int) -> tuple[list[dict], dict, dict]:
    with (CACHE_DIR / f"favbust_scored_n{n_car}.pkl").open("rb") as f:
        data = pickle.load(f)
    with (CACHE_DIR / f"favbust_entries_n{n_car}.pkl").open("rb") as f:
        ents = pickle.load(f)
    # ⚠️ 払戻キャッシュは車数ごとに分ける（7車用を9車で使い回すと全件落ちる）
    pp = CACHE_DIR / f"favbust_payouts_n{n_car}.pkl"
    if not pp.exists() and n_car == 7:
        pp = CACHE_DIR / "favbust_payouts.pkl"
    with pp.open("rb") as f:
        pay = pickle.load(f)
    data = [d for d in data if d["race_key"] in pay
            and pay[d["race_key"]]["tf_odds"] and pay[d["race_key"]]["trio_odds"]]
    return data, ents, pay


def load_preds(n_car: int) -> dict:
    """wf_preds{,9}_*.pkl（honest walk-forward の pp3 / ppw / pbad）。"""
    import glob

    import pandas as pd
    pat = "wf_preds_*.pkl" if n_car == 7 else "wf_preds9_*.pkl"
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / pat)))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_key", "frame_no"], keep="last")
    out: dict[str, dict[int, tuple]] = defaultdict(dict)
    for rk, fr, a, b, c in zip(df["race_key"], df["frame_no"],
                               df["pp3"], df["ppw"], df["pbad"]):
        out[rk][int(fr)] = (float(a), float(b), float(c))
    return dict(out)


# ---------------------------------------------------------------- レース単位の素材
def build(data: list[dict], ents: dict, pay: dict, preds: dict,
          rank_by: str, n_car: int = 7) -> list[dict]:
    """1レース1行。選別に使う量と、買い目の当否・払戻を先に確定させておく。

    rank_by: "pp3"(3着内率) / "ppw"(1着率) — 2着候補の並べ方
    """
    idx = {"pp3": 0, "ppw": 1}[rank_by]
    rows = []
    for d in data:
        rk = d["race_key"]
        pr, e = preds.get(rk), ents.get(rk)
        if not pr or not e or len(pr) < n_car:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][idx])
        if len(others) < 5:
            continue
        s = [pr[f][idx] for f in others]

        P = pay[rk]
        order = P["order"]
        fav_win = order[0] == fav
        by_frame = {int(x["frame_no"]): x for x in e}
        fav_line = (by_frame[fav]["line_group"]
                    if by_frame.get(fav) and by_frame[fav]["line_group"] is not None
                    else f"s{fav}")

        rows.append({
            "rk": rk, "date": d["race_date"], "fav": fav,
            # 7H1 のバスト確率。7H2 では**逆向き**（低いほど本命が堅い）に使う
            "bust_score": d["score"], "bust": d["bust"],
            # --- 選別に使う量 ---
            "gap12": d["fav_ppw_gap12"],           # 本命の抜け度（1着率 1位-2位差）
            "fav_ppw": d["fav_ppw"], "fav_pp3": d["fav_pp3"],
            "sec_g12": s[0] - s[1],                # 2着候補2車の拮抗度（小さいほど同値）
            "sec_g23": s[1] - s[2],                # 2番手グループと3番手の分離（大きいほど2車で確定）
            "sec_ratio": (s[1] - s[2]) / (s[0] - s[1] + 1e-9),
            "sec_sum2": s[0] + s[1],
            "others": others, "s": s,
            # --- 結果 ---
            "fav_win": fav_win,
            "second": order[1] if fav_win else None,
            "third": order[2] if fav_win else None,
            "tf_odds": P["tf_odds"], "order": order,
            "fav_line_size": len([1 for x in e
                                  if (x["line_group"] if x["line_group"] is not None
                                      else f"s{x['frame_no']}") == fav_line]),
        })
    return rows


# ---------------------------------------------------------------- 買い目の決済
def settle(r: dict, n2: int, n3: int | None, unit100: bool = True,
           third_from: int = 0) -> dict:
    """1着=本命固定 × 2着=上位n2車 × 3着=others[third_from:n3]（n3=None で末尾まで）。

    third_from>0 にすると **3着から人気上位を外す**（＝3着を荒れ側に寄せる）。
    """
    others = r["others"]
    seconds = others[:n2]
    thirds = others[third_from:] if n3 is None else others[third_from:n3]
    legs = [(a, c) for a in seconds for c in thirds if c != a]
    npt = len(legs)
    if npt == 0:
        return {}
    unit = (STAKE // npt) // 100 * 100 if unit100 else STAKE / npt
    if unit < 100:
        return {}
    cost = unit * npt
    hit = (r["fav_win"] and r["second"] in seconds and r["third"] in thirds
           and r["third"] != r["second"])
    pay = unit * r["tf_odds"] if hit else 0.0
    return {"npt": npt, "cost": cost, "pay": pay, "hit": int(bool(hit))}


def agg(items: list[dict]) -> dict:
    if not items:
        return {}
    pays = np.array([x["pay"] for x in items])
    cost = sum(x["cost"] for x in items)
    hit = pays > 0
    return {"n": len(items), "点数": float(np.mean([x["npt"] for x in items])),
            "購入": cost / len(items),
            "的中%": hit.mean() * 100, "ROI%": pays.sum() / cost * 100,
            "平均払戻": float(pays[hit].mean()) if hit.any() else 0.0,
            "中央払戻": float(np.median(pays[hit])) if hit.any() else 0.0,
            "最大": float(pays.max()),
            "10万+": int((pays >= 100_000).sum()),
            "30万+": int((pays >= HIGHPAY).sum()),
            "30万+%": float((pays >= HIGHPAY).mean() * 100)}


def line(tag: str, a: dict, width: int = 34) -> str:
    if not a:
        return f"  {tag:<{width}} （該当なし）"
    return (f"  {tag:<{width}} n={a['n']:5} {a['点数']:4.1f}点 購入{a['購入']:6.0f} "
            f"的中{a['的中%']:6.2f}% ROI{a['ROI%']:7.1f}% 平均払戻{a['平均払戻']:8.0f} "
            f"中央{a['中央払戻']:7.0f} 10万+{a['10万+']:4} 30万+{a['30万+']:3}"
            f"({a['30万+%']:4.2f}%)")


def win(rows: list[dict], w: tuple[str, str]) -> list[dict]:
    return [r for r in rows if w[0] <= r["date"] <= w[1]]


def ndays(rows: list[dict]) -> int:
    return max(1, len({r["date"] for r in rows}))


# ---------------------------------------------------------------- 各段の分析
def census(rows: list[dict]) -> None:
    """前提の成立度: 本命1着率と、そのとき2着が上位n車に入る率。"""
    print(f"\n{'=' * 118}\n=== 前提の素の成立度（母集団 {len(rows):,}R・7車 ∧ 軸1==WT◎）===")
    fw = np.array([r["fav_win"] for r in rows])
    print(f"  本命1着率           {fw.mean() * 100:6.2f}%")
    for n2 in (1, 2, 3):
        c = np.mean([r["fav_win"] and r["second"] in r["others"][:n2] for r in rows])
        print(f"  本命1着 ∧ 2着が上位{n2}車  {c * 100:6.2f}%   "
              f"（本命1着に条件付け {c / fw.mean() * 100:5.2f}%）")

    print("\n  --- 抜け度（fav_ppw_gap12）別 ---")
    qs = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    for g in qs:
        sub = [r for r in rows if r["gap12"] >= g]
        if len(sub) < 200:
            continue
        f2 = np.mean([r["fav_win"] and r["second"] in r["others"][:2] for r in sub])
        print(f"   gap12>={g:.2f}  n={len(sub):6,}  本命1着 "
              f"{np.mean([r['fav_win'] for r in sub]) * 100:6.2f}%  "
              f"∧2着上位2 {f2 * 100:6.2f}%")


def sweep(rows: list[dict], n2: int, n3: int | None) -> None:
    """掃引窓で選別条件を掃引。確認窓の数字は**この段では見ない**。"""
    sw = win(rows, SWEEP)
    print(f"\n{'=' * 118}\n=== 掃引窓 {SWEEP[0]}〜{SWEEP[1]}（{len(sw):,}R・"
          f"{ndays(sw)}日）で選別条件を掃引 ===")
    print(f"    買い目: 1着=本命固定 × 2着=上位{n2}車 × "
          f"3着={'総流し' if n3 is None else f'上位{n3}車'}")

    base = [x for x in (settle(r, n2, n3) for r in sw) if x]
    print(line("【対照】選別なし（全件）", agg(base)))

    print("\n  --- 1次元: 抜け度 gap12 ---")
    for g in (0.10, 0.15, 0.20, 0.25, 0.30):
        sub = [r for r in sw if r["gap12"] >= g]
        it = [x for x in (settle(r, n2, n3) for r in sub) if x]
        print(line(f"gap12>={g:.2f}  ({len(sub) / ndays(sw):.2f}件/日)", agg(it)))

    print("\n  --- 1次元: 2着候補の拮抗度 sec_g12（小さいほど2車が同値）---")
    v = np.array([r["sec_g12"] for r in sw])
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        t = float(np.quantile(v, q))
        sub = [r for r in sw if r["sec_g12"] <= t]
        it = [x for x in (settle(r, n2, n3) for r in sub) if x]
        print(line(f"sec_g12<=q{q:.2f}({t:.4f}) ({len(sub) / ndays(sw):.1f}件/日)",
                   agg(it)))

    print("\n  --- 1次元: 2番手グループの分離 sec_g23（大きいほど2車で確定）---")
    v = np.array([r["sec_g23"] for r in sw])
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        t = float(np.quantile(v, q))
        sub = [r for r in sw if r["sec_g23"] >= t]
        it = [x for x in (settle(r, n2, n3) for r in sub) if x]
        print(line(f"sec_g23>=q{q:.2f}({t:.4f}) ({len(sub) / ndays(sw):.1f}件/日)",
                   agg(it)))

    print("\n  --- 2次元: gap12 × sec_g23（P-B の定義そのもの）---")
    for g in (0.15, 0.20, 0.25):
        sub0 = [r for r in sw if r["gap12"] >= g]
        v = np.array([r["sec_g23"] for r in sub0])
        for q in (0.0, 0.3, 0.5, 0.7):
            t = float(np.quantile(v, q)) if q > 0 else -1e9
            sub = [r for r in sub0 if r["sec_g23"] >= t]
            if len(sub) < 100:
                continue
            it = [x for x in (settle(r, n2, n3) for r in sub) if x]
            print(line(f"gap12>={g:.2f} ∧ g23>=q{q:.1f} "
                       f"({len(sub) / ndays(sw):.2f}件/日)", agg(it)))


def shape_sweep(rows: list[dict]) -> None:
    """買い目の形（2着何車 × 3着どの帯）の掃引。選別は P-B 素の定義で固定。"""
    sw = win(rows, SWEEP)
    sub = [r for r in sw if r["gap12"] >= 0.20]
    print(f"\n{'=' * 118}\n=== 買い目の形の掃引（掃引窓・gap12>=0.20 の {len(sub):,}R）===")
    for n2 in (1, 2, 3):
        for n3 in (3, 4, 5, None):
            it = [x for x in (settle(r, n2, n3) for r in sub) if x]
            if not it:
                continue
            tag = f"2着{n2}車 × 3着{'総流し' if n3 is None else f'上位{n3}車'}"
            print(line(tag, agg(it), width=26))

    print("\n  --- 3着から人気側を外す（＝3着を荒れ側に寄せる）---")
    for n2 in (1, 2, 3):
        for tf in (2, 3, 4):
            if tf < n2:
                continue
            it = [x for x in (settle(r, n2, None, third_from=tf) for r in sub) if x]
            if not it:
                continue
            print(line(f"2着{n2}車 × 3着=r{tf + 1}以下", agg(it), width=26))

    print(f"\n  --- 的中時の三連単オッズ分布（2着2車×3着総流し・的中 {'':s}）---")
    hits = [r["tf_odds"] for r in sub
            if r["fav_win"] and r["second"] in r["others"][:2]]
    h = np.array(hits)
    print(f"    n={len(h):,}  中央{np.median(h):7.1f}倍  p75 {np.percentile(h, 75):7.1f} "
          f"p90 {np.percentile(h, 90):7.1f} p95 {np.percentile(h, 95):7.1f} "
          f"p99 {np.percentile(h, 99):7.1f} 最大{h.max():.1f}")
    for t in (100, 200, 300, 500):
        print(f"    的中のうち {t}倍以上: {(h >= t).mean() * 100:5.2f}% "
              f"（母集団比 {(h >= t).sum() / len(sub) * 100:5.3f}%）")


def antibust(rows: list[dict]) -> None:
    """7H1 のバスト予測モデルを**逆向き**に使う（＝本命が最も堅いレースを選ぶ）。

    7H2 の弱点は「市場と同じ向きの情報しか使っていない」こと。バスト確率だけは
    honest AUC 0.6848 の**市場に無い情報**なので、これが ROI を動かすかどうかが
    7H2 に残された唯一の可能性。掃引窓で分位を決め、確認窓へそのまま流す。
    """
    sw = win(rows, SWEEP)
    print(f"\n{'=' * 118}\n=== バスト確率を逆向きに使う（低いほど本命が堅い）===")
    print(f"  掃引窓 {len(sw):,}R でバスト確率の分位を確定 → 確認窓・未使用へ適用")
    for shape, tag in (((2, None, 0), "2着2車×3着総流し(10点)"),
                       ((1, None, 4), "2着1車×3着r5以下(2点)")):
        n2, n3, tfrom = shape
        print(f"\n  ── 買い目: {tag} ──")
        for q in (0.10, 0.20, 0.30, 0.50, 1.00):
            thr = float(np.quantile([r["bust_score"] for r in sw], q))
            sel = [r for r in rows if r["bust_score"] <= thr] if q < 1 else rows
            for wnm, w in (("掃引", SWEEP), ("確認", CONFIRM), ("未使用", FRESH)):
                g = win(sel, w)
                it = [x for x in (settle(r, n2, n3, third_from=tfrom) for r in g) if x]
                a = agg(it)
                if not a:
                    continue
                bust = np.mean([r["bust"] for r in g]) * 100
                print(f"   下位{q:.0%} {wnm:<4} n={a['n']:6} バスト率{bust:5.2f}% "
                      f"的中{a['的中%']:6.2f}% ROI{a['ROI%']:6.1f}% "
                      f"平均払戻{a['平均払戻']:8.0f} 30万+{a['30万+']:3}"
                      f"({a['30万+%']:4.2f}%)")


def confirm(rows: list[dict], cells: list[tuple]) -> None:
    """確認窓・未使用期間・月次・裾依存。

    cells = [(名前, filter関数, n2, n3, third_from)]
    ⚠️ **掃引窓で決めた条件をそのまま流す**。ここで条件をいじり直してはいけない。
    """
    for nm, fn, n2, n3, tfrom in cells:
        print(f"\n{'=' * 118}\n=== 【{nm}】 ===")
        sel = [r for r in rows if fn(r)]
        for wnm, w in (("掃引窓", SWEEP), ("確認窓", CONFIRM), ("未使用", FRESH)):
            g = win(sel, w)
            it = [x for x in (settle(r, n2, n3, third_from=tfrom) for r in g) if x]
            print(line(f"{wnm} {w[0]}〜{w[1]} ({len(g) / ndays(win(rows, w)):.2f}件/日)",
                       agg(it), width=42))
        it_all = [x for x in (settle(r, n2, n3, third_from=tfrom) for r in sel) if x]
        print(line("全期間", agg(it_all), width=42))

        # --- 月次 ---
        by_mo = defaultdict(list)
        for r in sel:
            x = settle(r, n2, n3, third_from=tfrom)
            if x:
                by_mo[r["date"][:7]].append(x)
        vals = []
        print("\n  月次:")
        for mo in sorted(by_mo):
            g = by_mo[mo]
            if len(g) < 10:
                continue
            a = agg(g)
            vals.append(a["ROI%"])
            print(f"    {mo} n={a['n']:4} 的中{a['的中%']:6.2f}% ROI{a['ROI%']:7.1f}% "
                  f"30万+{a['30万+']:3}")
        if vals:
            v = np.array(vals)
            print(f"    月次ROI 平均{v.mean():6.1f}% 中央{np.median(v):6.1f}% "
                  f"100%超 {int((v > 100).sum())}/{len(v)} "
                  f"最低{v.min():.1f}% 最高{v.max():.1f}%")

        # --- 裾依存 ---
        pays = np.sort(np.array([x["pay"] for x in it_all]))[::-1]
        cost = sum(x["cost"] for x in it_all)
        tot = pays.sum()
        if tot > 0:
            print(f"\n  裾依存: ROI {tot / cost * 100:.1f}% → "
                  + " ".join(f"除・上{k}={((tot - pays[:k].sum()) / cost * 100):.1f}%"
                             for k in (1, 3, 5, 10))
                  + f"  / 上3が回収の {pays[:3].sum() / tot * 100:.1f}%")


def bootstrap(rows: list[dict], fn, n2: int, n3: int | None,
              tfrom: int = 0, iters: int = 2000) -> None:
    """選別あり vs 選別なし（全件）の ΔROI を日ブロック bootstrap で。"""
    sel_d, all_d = defaultdict(list), defaultdict(list)
    for r in rows:
        x = settle(r, n2, n3, third_from=tfrom)
        if not x:
            continue
        all_d[r["date"]].append((x["cost"], x["pay"]))
        if fn(r):
            sel_d[r["date"]].append((x["cost"], x["pay"]))
    days = sorted(all_d)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(iters):
        pick = rng.choice(len(days), len(days), replace=True)
        cs = rs = ca = ra = 0.0
        for i in pick:
            dd = days[i]
            for c, p in sel_d.get(dd, []):
                cs += c
                rs += p
            for c, p in all_d.get(dd, []):
                ca += c
                ra += p
        if cs > 0 and ca > 0:
            diffs.append(rs / cs * 100 - ra / ca * 100)
    a = np.array(diffs)
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    print(f"\n  日ブロック bootstrap ΔROI(選別 − 全件) {a.mean():+6.1f}pt "
          f"95%CI [{lo:+6.1f}, {hi:+6.1f}] "
          f"{'✅有意' if lo > 0 else '❌有意差なし'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-by", default="pp3", choices=("pp3", "ppw"))
    ap.add_argument("--n2", type=int, default=2)
    ap.add_argument("--n3", type=int, default=0, help="0=総流し")
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--stage", default="all",
                    choices=("all", "census", "sweep", "shape", "confirm",
                             "antibust"))
    args = ap.parse_args()
    n3 = None if args.n3 == 0 else args.n3

    data, ents, pay = load_all(args.n_car)
    preds = load_preds(args.n_car)
    rows = build(data, ents, pay, preds, args.rank_by, args.n_car)
    print(f"[data] {args.n_car}車 {len(rows):,}R / {ndays(rows)}日 / "
          f"rank_by={args.rank_by}")

    if args.stage in ("all", "census"):
        census(rows)
    if args.stage in ("all", "shape"):
        shape_sweep(rows)
    if args.stage in ("all", "sweep"):
        sweep(rows, args.n2, n3)
    if args.stage in ("all", "antibust"):
        antibust(rows)
    if args.stage in ("all", "confirm"):
        # 🔴 掃引窓の結果だけで事前に確定させた候補（確認窓を見て選び直さない）
        sw = win(rows, SWEEP)
        q50 = float(np.quantile([r["sec_g23"] for r in sw
                                 if r["gap12"] >= 0.25], 0.5))
        cells = [
            ("C1 P-B素の定義: gap12>=0.20 × 2着2車×3着総流し(10点)",
             lambda r: r["gap12"] >= 0.20, 2, None, 0),
            (f"C2 掃引窓ROI最良: gap12>=0.25 ∧ sec_g23>={q50:.4f} × 10点",
             lambda r: r["gap12"] >= 0.25 and r["sec_g23"] >= q50, 2, None, 0),
            ("C3 掃引窓30万+最良: gap12>=0.20 × 2着1車×3着r5以下(2点)",
             lambda r: r["gap12"] >= 0.20, 1, None, 4),
            ("C4 中庸: gap12>=0.20 × 2着1車×3着r4以下(3点)",
             lambda r: r["gap12"] >= 0.20, 1, None, 3),
        ]
        confirm(rows, cells)
        print(f"\n{'=' * 118}\n=== 選別 vs 全件 の ΔROI（日ブロック bootstrap）===")
        for nm, fn, n2_, n3_, tf_ in cells:
            print(f"  {nm}")
            bootstrap(rows, fn, n2_, n3_, tf_)


if __name__ == "__main__":
    main()
