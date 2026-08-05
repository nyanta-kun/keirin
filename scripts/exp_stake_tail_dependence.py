"""傾斜配分の「穴寄せ」改善が高額配当の裾に依存していないかを検査する（2026-08-05）。

## なぜ必要か

`exp_stake_allocation.py` で、オッズ穴寄せ（`w ∝ o^k`, k>0）が 7SS/7S/空白3×準決勝 で
ROI を上げるという結果が出た。だがこれは**ごく少数の高額配当が平均を作っているだけ**
の可能性がある。9B を棄却したときと同じ型（ROI82.1%だが最高配当3本を除くと73.5%）。

さらに理論的にも疑わしい: 競馬の古典的な人気-穴バイアスでは**穴は過剰に賭けられて
EVが下がる**。ここで出た「穴側のEVが高い」はその**逆**であり、素直に信じてはいけない。

## 検査すること

1. **上位k本の配当を除いたROI**（k=1,3,5,10）。等分と穴寄せで落ち方を比較する
2. **上位5本の払戻が総回収に占める割合**（裾集中度）
3. **的中率**（1点集中は的中率が1/3に落ちるので実運用の性格が変わる）
4. ブートストラップで「穴寄せ − 等分」の差のCI（同一レース集合の paired）

## ⚠️ 実運用上の別問題（バックテストでは絶対に見えない）

三連複はパリミュチュエル。**1点に10,000円を投じるとオッズ自体が下がる**。
100円/点なら無視できるが、小規模開催の1点に1万円は払戻を実測で押し下げうる。
「最高オッズ1点に全額」系の数字は**この影響を含んでいない上限値**として読むこと。

掃引窓のみ。DB書き込みなし。構築結果を data/exp_cache/stake_races.pkl に保存して再利用する。

使い方:
    python scripts/exp_stake_tail_dependence.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7b_order_disagree, rank_7b_select_legs,
    rank_7s_field_entropy, rank_7s_wt_overlap_n, rank_7ss_same_line,
)

BUDGET, UNIT = 10_000, 100
SWEEP = {"w1": ("2026-04-13", "2026-07-15", 94), "w2": ("2026-01-01", "2026-04-12", 102),
         "w3": ("2025-10-01", "2025-12-31", 92), "w4": ("2025-07-01", "2025-09-30", 92)}
SWEEP_TRAIN_FROM = "2024-04-01"
CACHE_DIR = REPO / "data" / "exp_cache"
BUILD_CACHE = CACHE_DIR / "stake_races.pkl"


def cached_preds(tf, tt, train_from):
    p = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{train_from}.pkl"
    if not p.exists():
        raise SystemExit(f"[FATAL] 予測キャッシュがありません: {p}")
    return pd.read_pickle(p)


def load_trio(keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            ch = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(p) == 3:
                        out[rk][p] = v
    return out


def build():
    """必要な情報だけを軽量な dict へ落として保存する（再利用で15分節約）。"""
    if BUILD_CACHE.exists():
        print(f"  [cache] {BUILD_CACHE.name} を利用", flush=True)
        with open(BUILD_CACHE, "rb") as f:
            return pickle.load(f)
    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t, _ in SWEEP.values())
    df = build_features_wt(load_raw_data_wt(min_date=SWEEP_TRAIN_FROM, max_date=max_to))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    raw = []
    for w, (tf, tt, _d) in SWEEP.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, SWEEP_TRAIN_FROM),
            on=["race_key", "frame_no"], how="inner")
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            rows = list(g.itertuples(index=False))
            fo = {int(x.frame_no): (int(x.finish_order)
                                    if x.finish_order == x.finish_order
                                    and x.finish_order is not None else 0) for x in rows}
            top3 = {f for f, v in fo.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            mk = {int(x.frame_no): x.prediction_mark for x in rows}
            lg = {int(x.frame_no): x.line_group for x in rows}
            p3 = {int(x.frame_no): float(x.pp3) for x in rows}
            pw = {int(x.frame_no): float(x.ppw) for x in rows}
            bad = {int(x.frame_no): float(x.pbad) for x in rows}
            hon = next((f for f, m in mk.items() if m == 1), None)
            tai = next((f for f, m in mk.items() if m == 2), None)
            ana = next((f for f, m in mk.items() if m == 3), None)
            a1 = max(pw, key=lambda f: pw[f])
            zp, zb = _race_zscore(p3), _race_zscore(bad)
            cand = [f for f in p3 if f != a1]
            if not cand:
                continue
            a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
            others = sorted(set(p3) - {a1, a2})
            asum = p3[a1] + p3[a2]
            ent = rank_7s_field_entropy(p3)
            raw.append({
                "rk": rk, "w": w, "a1": a1, "a2": a2,
                "ov": rank_7s_wt_overlap_n(a1, a2, hon, tai),
                "axis_ok": asum <= RANK_7S_AXIS_SUM_MAX,
                "ent_ok": ent <= RANK_7S_ENTROPY_MAX,
                "same_line": rank_7ss_same_line(a1, a2, lg),
                "order_dis": rank_7b_order_disagree(pw, hon),
                "race_type": str(rows[0].race_type),
                "others": others,
                "legs3": rank_7b_select_legs(others, p3, ana),
                "p3legs": {f: p3[f] for f in others},
                "win": (next(iter(top3 - {a1, a2}))
                        if len({a1, a2} & top3) == 2 and len(top3 - {a1, a2}) == 1
                        else None),
            })
    trio = load_trio(sorted({r["rk"] for r in raw}))
    out = []
    for r in raw:
        board = trio.get(r["rk"])
        if not board:
            continue
        r["odds"] = {f: board[frozenset({r["a1"], r["a2"], f})]
                     for f in r["others"]
                     if frozenset({r["a1"], r["a2"], f}) in board}
        if r["odds"]:
            out.append(r)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BUILD_CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"  [cache] {BUILD_CACHE.name} を保存（{len(out)} レース）", flush=True)
    return out


RANKS = {
    "7SS": (lambda r: r["ov"] in (0, 1) and r["axis_ok"] and not r["ent_ok"]
            and r["same_line"], "full"),
    "7S":  (lambda r: r["ov"] in (0, 1) and r["axis_ok"] and r["ent_ok"], "full"),
    "7A":  (lambda r: r["ov"] in (0, 1) and not r["axis_ok"] and r["ent_ok"], "full"),
    "空白3×準決勝": (lambda r: r["ov"] == 2 and r["order_dis"] is not True
                and r["race_type"] == "準決勝", "three"),
}


def allocate(weights):
    tot = sum(weights)
    if tot <= 0:
        n = len(weights)
        out = [(BUDGET // n // UNIT) * UNIT] * n
    else:
        out = [int((BUDGET * w / tot) // UNIT) * UNIT for w in weights]
    rest = BUDGET - sum(out)
    if rest > 0:
        out[int(np.argmax(weights)) if sum(weights) > 0 else 0] += rest
    return out


SCHEMES = {
    "等分（現行）":        lambda o, p: [1.0] * len(o),
    "A k=+0.5 やや穴寄せ": lambda o, p: [max(v, 1e-9) ** 0.5 for v in o],
    "A k=+1.0 穴寄せ":    lambda o, p: [max(v, 1e-9) ** 1.0 for v in o],
    "A 最高オッズ1点集中":   lambda o, p: [1.0 if j == int(np.argmax(o)) else 0.0
                                  for j in range(len(o))],
}


def returns_for(races, sel, mode, wfn):
    """レースごとの払戻額リストを返す（賭け金は常に BUDGET）。"""
    rets = []
    for r in races:
        if not sel(r):
            continue
        legs = [x for x in (r["others"] if mode == "full" else r["legs3"])
                if x in r["odds"]]
        if not legs:
            continue
        o = [r["odds"][x] for x in legs]
        p = [r["p3legs"][x] for x in legs]
        st = allocate(wfn(o, p))
        ret = 0.0
        if r["win"] is not None and r["win"] in legs:
            i = legs.index(r["win"])
            ret = (round(o[i] * st[i]) // 10) * 10
        rets.append(ret)
    return np.array(rets, dtype=float)


def main():
    races = build()
    print(f"  対象 {len(races)} レース\n")

    for rank, (sel, mode) in RANKS.items():
        base = returns_for(races, sel, mode, SCHEMES["等分（現行）"])
        if len(base) == 0:
            continue
        print("=" * 104)
        print(f"■ {rank}   n={len(base)}  買い目={'5点' if mode == 'full' else '3点'}")
        print(f"  {'配分':<22}{'ROI':>8}{'的中':>8}"
              f"{'-上1':>8}{'-上3':>8}{'-上5':>8}{'-上10':>8}"
              f"{'上5が回収に占める%':>18}")
        for slbl, sfn in SCHEMES.items():
            rets = returns_for(races, sel, mode, sfn)
            n = len(rets)
            roi = 100 * rets.sum() / (BUDGET * n)
            hit = 100 * (rets > 0).sum() / n
            srt = np.sort(rets)[::-1]
            cuts = []
            for k in (1, 3, 5, 10):
                if n > k:
                    cuts.append(100 * srt[k:].sum() / (BUDGET * (n - k)))
                else:
                    cuts.append(float("nan"))
            share = 100 * srt[:5].sum() / rets.sum() if rets.sum() > 0 else 0
            print(f"  {slbl:<22}{roi:>7.1f}%{hit:>7.1f}%"
                  f"{cuts[0]:>7.1f}%{cuts[1]:>7.1f}%{cuts[2]:>7.1f}%{cuts[3]:>7.1f}%"
                  f"{share:>17.1f}%")

        # ---- 「穴寄せ − 等分」の差の paired ブートストラップ ----
        print(f"  {'':<22}── 等分との差（同一レース集合の paired bootstrap 2,000回）")
        rng = np.random.default_rng(20260805)
        idx = np.arange(len(base))
        for slbl in ("A k=+0.5 やや穴寄せ", "A k=+1.0 穴寄せ", "A 最高オッズ1点集中"):
            alt = returns_for(races, sel, mode, SCHEMES[slbl])
            diffs = []
            for _ in range(2000):
                s = rng.choice(idx, len(idx), replace=True)
                diffs.append(100 * (alt[s].sum() - base[s].sum()) / (BUDGET * len(s)))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            sign = "有意" if lo > 0 else ("負に有意" if hi < 0 else "有意差なし")
            print(f"  {slbl:<22}{np.mean(diffs):>+7.1f}pt "
                  f"[{lo:+6.1f}, {hi:+6.1f}]  {sign}")
        print()

    print("=" * 104)
    print("  -上k = 払戻が大きい順に k 本を除いたROI。等分より大きく落ちるなら**裾依存**。")
    print("  ⚠️ 三連複はパリミュチュエル。1点に10,000円を投じるとオッズ自体が下がる。")
    print("     「1点集中」の数字はその影響を含まない**上限値**。")
    print("  ⚠️ 掃引窓。採否は確認窓で一度きり。")


if __name__ == "__main__":
    main()
