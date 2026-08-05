"""3ヘッド軸選定の「印なし側の軸」を検証する（2026-08-05）。

ユーザー仮説:
  「一軸のWINTICKET◎◯との重複精度は良いが、もう一軸を外しているように見える」

現行ゲート（7S/7A）は `wt_overlap_n ∈ {0,1}` なので、軸2車のうち公式印(◎◯)と
一致するのは**最大1車**。残りは必ずモデル独自の選択になる。そこを測る。

## 測ること

1. **印付き軸 vs 印なし軸の3着内率**（同一レース内の対比なので条件が揃う）
2. **使われなかった公式印の3着内率**。overlap==1 のとき ◎◯ の片方は軸に入って
   いない。**その「余った印」が、我々の印なし軸より当たっているか**が核心。
   当たっているなら「印なし軸を余った印に差し替える」という具体策になる
3. 差し替え案の 両方3着内率・ROI（実オッズ）

## 差し替え案

| 案 | 軸2車 |
|---|---|
| C 現行 | 軸1 = argmax pw / 軸2 = argmax z(p3) − 0.3·z(pbad)（軸1を除く） |
| M1 | 印付き軸を残し、印なし軸を**余った公式印**へ差し替え（＝◎◯ペア） |
| M2 | 軸1(pw最上位)を残し、軸2を**余った公式印**へ差し替え |

⚠️ M1/M2 は wt_overlap_n が 2 になるため、**現行の 7S/7A の定義からは外れる**
（＝7Bが扱う「市場と一致」の領域へ移る）。的中率は上がるが配当が消えることが
[[keirin_7b_rank_2026_08_03]] で測定済み（overlap2 は的中56.3%・ROI72.4%）。
本スクリプトはその構造を現行母集団の上で再確認するもの。

⚠️ オッズは wt_odds＝最終オッズ（stale）。選出条件は確率のみでオッズ非依存。
DB書き込みなし。honest: 窓ごとに学習しなおす walk-forward。

使い方:
    python scripts/exp_axis2_mark_accuracy.py [--windows w1,w2,w3,w4]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
    "w3": ("2025-10-01", "2025-12-31"),
    "w4": ("2025-07-01", "2025-09-30"),
}
SEEDS = [42, 101, 202, 303, 404]
STAKE = 100


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(train[FEATURE_COLS_WT], train[target])
        preds.append(m.predict_proba(test[FEATURE_COLS_WT])[:, 1])
    return np.mean(preds, axis=0)


def load_trio(keys: list[str]) -> dict[str, dict[frozenset, float]]:
    out: dict[str, dict[frozenset, float]] = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(parts) == 3:
                        out[rk][parts] = v
    return out


def select_axes(r: dict) -> tuple[int, int]:
    """本番と同じ3ヘッド選定（w1=0 ＝ 軸1は pred_win 最上位）。"""
    a1 = max(r["pw"], key=lambda f: r["pw"][f])
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
    a2 = max((f for f in r["p3"] if f != a1),
             key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
    return a1, a2


def passes_gate(r: dict, a1: int, a2: int) -> bool:
    """7S/7A のゲート（overlap∈{0,1} かつ 不合格条件が1つ以下）。"""
    ov = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
    if ov not in (0, 1):
        return False
    asum = r["p3"][a1] + r["p3"][a2]
    ent = rank_7s_field_entropy(r["p3"])
    return (asum > RANK_7S_AXIS_SUM_MAX) + (ent > RANK_7S_ENTROPY_MAX) <= 1


def run_window(df: pd.DataFrame, tf: str, tt: str, acc: dict) -> None:
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
    test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
    print(f"\n######## 窓 test={tf}〜{tt}  train {len(train):,} / test {len(test):,} ########",
          flush=True)
    t = test.copy()
    t["pp3"] = fit_predict(train, test, TARGET_COL_WT)
    t["ppw"] = fit_predict(train, test, "win_flag")
    t["pbad"] = fit_predict(train, test, "bad6")

    races = []
    for rk, g in t.groupby("race_key"):
        if len(g) != 7:
            continue
        fo = {int(x.frame_no): (int(x.finish_order)
                                if x.finish_order is not None and x.finish_order == x.finish_order
                                else 0) for x in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue
        mark = {int(x.frame_no): x.prediction_mark for x in g.itertuples(index=False)}
        races.append({
            "rk": rk, "fo": fo, "top3": top3, "mark": mark,
            "hon": next((f for f, m in mark.items() if m == 1), None),
            "tai": next((f for f, m in mark.items() if m == 2), None),
            "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
            "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
            "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)},
        })
    trio = load_trio(sorted({r["rk"] for r in races}))
    sel = []
    for r in races:
        a1, a2 = select_axes(r)
        if passes_gate(r, a1, a2) and trio.get(r["rk"]):
            sel.append((r, a1, a2))
    print(f"  ゲート通過 {len(sel)} レース（母集団 {len(races)}）")

    # ---- 1) 印付き軸 vs 印なし軸 の3着内率 -------------------------------
    def bump(key, ok):
        acc.setdefault(key, [0, 0])
        acc[key][0] += 1
        acc[key][1] += int(ok)

    for r, a1, a2 in sel:
        marks = {r["hon"], r["tai"]} - {None}
        for lbl, a in (("軸1", a1), ("軸2", a2)):
            bump(f"{lbl}/{'印あり' if a in marks else '印なし'}", a in r["top3"])
        # 余った公式印（軸に採用されなかった ◎ or ◯）
        spare = [m for m in (r["hon"], r["tai"]) if m is not None and m not in (a1, a2)]
        for m in spare:
            bump("余った公式印", m in r["top3"])
        # 印なし軸（対照: 同じレースで比べる）
        for a in (a1, a2):
            if a not in marks and spare:
                bump("印なし軸(余り印と同レース)", a in r["top3"])

    # ---- 2) 差し替え案の比較 --------------------------------------------
    def eval_variant(pick):
        n = both = hit = bet = ret = 0
        for r, a1, a2 in sel:
            ax = pick(r, a1, a2)
            if ax is None or len(set(ax)) != 2:
                continue
            b1, b2 = ax
            board = trio[r["rk"]]
            legs = [x for x in r["p3"] if x not in (b1, b2)
                    and frozenset({b1, b2, x}) in board]
            if not legs:
                continue
            n += 1
            if {b1, b2} <= r["top3"]:
                both += 1
            bet += len(legs) * STAKE
            rest = r["top3"] - {b1, b2}
            if len(r["top3"] & {b1, b2}) == 2 and len(rest) == 1 and rest.pop() in legs:
                hit += 1
                ret += round(board[frozenset(r["top3"])] * 100) // 10 * 10
        return {"n": n, "both": 100 * both / n if n else 0,
                "hit": 100 * hit / n if n else 0,
                "roi": 100 * ret / bet if bet else 0}

    def spare_mark(r, a1, a2):
        s = [m for m in (r["hon"], r["tai"]) if m is not None and m not in (a1, a2)]
        return s[0] if s else None

    variants = {
        "C 現行(3ヘッド)": lambda r, a1, a2: (a1, a2),
        "M1 印なし軸→余り印": lambda r, a1, a2: (
            (lambda sp: None if sp is None else
             ((a1, sp) if a1 in {r["hon"], r["tai"]} else
              (a2, sp) if a2 in {r["hon"], r["tai"]} else None))(spare_mark(r, a1, a2))),
        "M2 軸2→余り印": lambda r, a1, a2: (
            (lambda sp: None if sp is None else (a1, sp))(spare_mark(r, a1, a2))),
        "参考 ◎◯ペア": lambda r, a1, a2: (
            (r["hon"], r["tai"]) if r["hon"] and r["tai"] else None),
    }
    for lbl, fn in variants.items():
        s = eval_variant(fn)
        acc.setdefault("V:" + lbl, []).append(s)
        print(f"  {lbl:22} n={s['n']:4d} 両方3着内 {s['both']:5.1f}%  "
              f"的中 {s['hit']:5.1f}%  ROI {s['roi']:6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    args = ap.parse_args()
    max_to = max(t for _, t in WINDOWS.values())
    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    print(f"7車立てに限定: {len(df):,}行")

    acc: dict = {}
    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt, acc)

    print("\n" + "=" * 78)
    print("【1】軸の3着内率（4窓合算）")
    print(f"  {'区分':<28}{'n':>6}{'3着内率':>9}")
    for k in ("軸1/印あり", "軸1/印なし", "軸2/印あり", "軸2/印なし",
              "印なし軸(余り印と同レース)", "余った公式印"):
        if k in acc:
            n, ok = acc[k]
            print(f"  {k:<28}{n:>6}{100*ok/n:>8.1f}%")

    print("\n【2】差し替え案（窓ごとの単純平均）")
    print(f"  {'案':<22}{'n':>6}{'両方3着内':>10}{'的中':>8}{'ROI':>8}")
    for k, v in acc.items():
        if not k.startswith("V:"):
            continue
        m = {x: float(np.mean([s[x] for s in v])) for x in ("n", "both", "hit", "roi")}
        print(f"  {k[2:]:<22}{m['n']:>6.0f}{m['both']:>9.1f}%{m['hit']:>7.1f}%{m['roi']:>7.1f}%")
    print("\n  窓別 ROI（符号反転の確認用）")
    for k, v in acc.items():
        if k.startswith("V:"):
            print(f"  {k[2:]:<22}" + "  ".join(f"{s['roi']:6.1f}%" for s in v))


if __name__ == "__main__":
    main()
