"""空白3×準決勝 を「掃引窓にも確認窓にも使っていない期間」で追試する（2026-08-05）。

## なぜ必要か

`exp_7car_gap_confirm.py` の確認窓（2024-07〜2025-06）で、空白3×準決勝は
ROI 83.0%・4窓すべて✓・土台との差 +8.7pt [+1.2, +16.3] と**有意**になった。
ただし ①同一土台で4通り比較しており多重性がある ②CI下限 +1.2pt は薄い
③同じ `race_type` 軸の「決勝」は掃引窓 87.1% → 確認窓 66.7% と**完全に反転**した。
軸そのものの安定性が確立していないため、独立な期間で一度だけ確かめる。

## 期間

**2026-07-16 〜 2026-08-04（20日）**。掃引窓は 2026-07-15 で終わっており、
確認窓は 2025-06-30 までなので、**どちらにも一度も使われていない**。
学習は 2024-04-01 〜 2026-07-15（＝窓の開始前のみ）。掃引窓と同じ TRAIN_FROM。

## ⚠️ 検出力の限界（結果を読む前に必ず理解すること）

確認窓の実績 4.73件/日 から、20日では **n≈95** にしかならない。
2026-08-05 の 7B 検証では n=190 で ROI の CI が **±20pt** だった。
したがって本追試の CI は **±25pt 前後**になる見込みで、

- **粗い反転（ROI 60%台など）が出れば候補を棄却できる**
- **反転が出なくても +8.7pt を追認したことにはならない**（弱い陽性証拠）

という**非対称な使い方しかできない**。「4窓✓だから確定」と読まないこと。

DB書き込みなし。予測キャッシュが無ければ学習して作る（15分程度）。

使い方:
    python scripts/exp_7car_gap_fresh.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from scripts.exp_axis_rule_decomposition import fit_predict  # noqa: E402
from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7b_order_disagree, rank_7b_select_legs,
    rank_7s_field_entropy, rank_7s_wt_overlap_n, rank_7ss_same_line,
)

STAKE = 100
FRESH_FROM, FRESH_TO, FRESH_DAYS = "2026-07-16", "2026-08-04", 20
TRAIN_FROM = "2024-04-01"        # 掃引窓と同じ。窓の開始前のみで学習
CACHE_DIR = REPO / "data" / "exp_cache"


def preds_for_fresh(df):
    cache = CACHE_DIR / (f"wf_preds_{FRESH_FROM}_{FRESH_TO}"
                         f"_f{len(FEATURE_COLS_WT)}_{TRAIN_FROM}.pkl")
    if cache.exists():
        print(f"  [cache] {cache.name} を利用（再学習なし）", flush=True)
        return pd.read_pickle(cache)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < FRESH_FROM)]
    test = df[(df["race_date"] >= FRESH_FROM) & (df["race_date"] <= FRESH_TO)]
    print(f"  学習中 train {len(train):,} / test {len(test):,} ...", flush=True)
    out = test[["race_key", "frame_no"]].copy()
    out["pp3"] = fit_predict(train, test, TARGET_COL_WT)
    out["ppw"] = fit_predict(train, test, "win_flag")
    out["pbad"] = fit_predict(train, test, "bad6")
    out.to_pickle(cache)
    print(f"  [cache] {cache.name} を保存", flush=True)
    return out


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


def build(df, preds):
    races = []
    t = df[(df["race_date"] >= FRESH_FROM) & (df["race_date"] <= FRESH_TO)].merge(
        preds, on=["race_key", "frame_no"], how="inner")
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
        r = {"rk": rk, "top3": top3,
             "hon": next((f for f, m in mk.items() if m == 1), None),
             "tai": next((f for f, m in mk.items() if m == 2), None),
             "ana": next((f for f, m in mk.items() if m == 3), None),
             "p3": {int(x.frame_no): float(x.pp3) for x in rows},
             "pw": {int(x.frame_no): float(x.ppw) for x in rows},
             "bad": {int(x.frame_no): float(x.pbad) for x in rows},
             "race_type": str(rows[0].race_type)}
        a1 = max(r["pw"], key=lambda f: r["pw"][f])
        zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
        cand = [f for f in r["p3"] if f != a1]
        if not cand:
            continue
        a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
        r["a1"], r["a2"] = a1, a2
        r["ov"] = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
        r["asum"] = r["p3"][a1] + r["p3"][a2]
        r["ent"] = rank_7s_field_entropy(r["p3"])
        r["axis_ok"] = r["asum"] <= RANK_7S_AXIS_SUM_MAX
        r["ent_ok"] = r["ent"] <= RANK_7S_ENTROPY_MAX
        r["same_line"] = rank_7ss_same_line(a1, a2, lg)
        r["order_dis"] = rank_7b_order_disagree(r["pw"], r["hon"])
        r["others"] = sorted(set(r["p3"]) - {a1, a2})
        r["legs3"] = rank_7b_select_legs(r["others"], r["p3"], r["ana"])
        races.append(r)
    trio = load_trio(sorted({r["rk"] for r in races}))
    return [r for r in races if trio.get(r["rk"])], trio


def settle(r, board, legs):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in legs if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def is_gap3(r):
    return r["ov"] == 2 and r["order_dis"] is not True


def main():
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=FRESH_TO))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    print(f"未使用期間 {FRESH_FROM}〜{FRESH_TO} の予測を用意 ...", flush=True)
    preds = preds_for_fresh(df)
    races, trio = build(df, preds)
    print(f"  7車・オッズ有り {len(races)} レース\n")

    CANDS = [
        ("★空白3 × 準決勝（追試対象）",
         lambda r: is_gap3(r) and r["race_type"] == "準決勝", "three"),
        ("  土台: 空白3 全件", is_gap3, "three"),
        ("  対照: 空白3 × 準決勝以外",
         lambda r: is_gap3(r) and r["race_type"] != "準決勝", "three"),
        ("  対照: 空白3 × 決勝（確認窓で反転した軸）",
         lambda r: is_gap3(r) and r["race_type"] == "決勝", "three"),
        ("参考: 7SS", lambda r: r["ov"] in (0, 1) and r["axis_ok"]
         and not r["ent_ok"] and r["same_line"], "full"),
        ("参考: 7S", lambda r: r["ov"] in (0, 1) and r["axis_ok"] and r["ent_ok"], "full"),
        ("参考: 7A", lambda r: r["ov"] in (0, 1) and not r["axis_ok"]
         and r["ent_ok"], "full"),
        ("参考: 7B（畳む）", lambda r: r["ov"] == 2 and r["order_dis"] is True, "three"),
    ]

    print("=" * 100)
    print(f"【未使用期間 {FRESH_FROM}〜{FRESH_TO}（{FRESH_DAYS}日）・一度きり】")
    print(f"  {'候補':<34}{'n':>6}{'件/日':>8}{'的中':>9}{'ROI':>9}")
    store = {}
    for lbl, fn, mode in CANDS:
        rows = []
        for r in races:
            if not fn(r):
                continue
            s = settle(r, trio[r["rk"]], r["others"] if mode == "full" else r["legs3"])
            if s:
                rows.append((s, r))
        if not rows:
            print(f"  {lbl:<34} 該当なし")
            continue
        store[lbl] = rows
        bet = sum(x[0][0] for x in rows); ret = sum(x[0][1] for x in rows)
        h = [x for x in rows if x[0][2]]
        print(f"  {lbl:<34}{len(rows):>6}{len(rows)/FRESH_DAYS:>8.2f}"
              f"{100*len(h)/len(rows):>8.1f}%{100*ret/bet if bet else 0:>8.1f}%")

    # ---- ブートストラップ（検出力が低いことを数字で示す） ----
    print("\n" + "=" * 100)
    print("【ブートストラップ】レース単位 復元抽出 2,000回")
    rng = np.random.default_rng(20260805)
    for lbl in ("★空白3 × 準決勝（追試対象）", "  土台: 空白3 全件"):
        rows = store.get(lbl)
        if not rows:
            continue
        idx = np.arange(len(rows))

        def roi_of(sel):
            b = sum(x[0][0] for x in sel); rt = sum(x[0][1] for x in sel)
            return 100 * rt / b if b else float("nan")

        boot = [roi_of([rows[i] for i in rng.choice(idx, len(idx), replace=True)])
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {lbl:<34} n={len(rows):>4}  ROI {np.mean(boot):6.1f}% "
              f"[{lo:5.1f}, {hi:5.1f}]  幅 {hi-lo:.1f}pt")

    print("\n  ⚠️ n が小さく CI が広い。**粗い反転が出れば棄却**できるが、"
          "反転が無くても確認窓の +8.7pt を追認したことにはならない。")


if __name__ == "__main__":
    main()
