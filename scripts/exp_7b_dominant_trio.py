"""7B「3車が飛び抜けている」ときは △ を切らず1点で買う案の検証（2026-08-05）。

ユーザー発案（[[keirin_7b_dominant_trio_candidate_2026_08_05]]）:
  「7Bで3点に絞る際に△を除いている。こちらは人気のためだが、**軸を含めた3車が
    飛び抜けているケース**がある。その場合は無理に穴に行かず、**この1点を推奨**
    としても良いのでは。閾値と成り立つか検証してほしい」

## 現行 7B

    軸2車 = ◎◯（wt_overlap_n==2 ∧ order_disagree）
    相手  = 残り5車から △ を除外し pred_prob 上位3車 → 3点=300円

△を切ることが配当を戻す本体（的中中央値 3.4→6.1倍・ガミ率 42.4→10.8%）。
代償として的中率は 47.0→27.3% に落ちる。

## 本案

    集中度 >= 閾値 なら 三連複 {軸1, 軸2, △} の 1点=100円
    未満なら 現行どおり △除外の3点=300円

**1点100円は的中時に1.0倍超で成立する**（3点300円は3.0倍超が必要）ため、
ガミ問題が構造的に消えるのが狙い。ただし ◎◯△ は市場が最も推す組み合わせで
配当は最も沈む帯なので、控除率75%の壁を越えるかは別問題。

## 集中度の候補（すべてオッズ非依存＝発走前に確定）

  A: p3[a1] + p3[a2] + p3[ana]        3車で3枠をどれだけ占めるか
  B: p3[ana] − p3[4位]                △と4番手の差＝3車で切れているか
  C: p3[ana]                          △自体の3着内確率

⚠️ 閾値は本スクリプトで掃引するが、**採否は別窓で一度きり確認すること**。
⚠️ オッズは wt_odds＝最終オッズ（stale）。選出条件は確率のみでオッズ非依存。
DB書き込みなし。予測は exp_axis_rule_decomposition.py と共通のキャッシュを使う。

使い方:
    python scripts/exp_7b_dominant_trio.py [--windows w1,w2,w3,w4]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from scripts.exp_axis_rule_decomposition import (  # noqa: E402
    TRAIN_FROM, WINDOWS, a1_pw, a2_p3_bad, classify_rank, window_preds,
)
from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import rank_7b_select_legs  # noqa: E402

STAKE = 100


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


def collect(df: pd.DataFrame, tf: str, tt: str) -> list[dict]:
    """その窓の 7B レースを集める（本番と同じ軸選定・ランク判定）。"""
    preds = window_preds(df, tf, tt)
    t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
        preds, on=["race_key", "frame_no"], how="inner")
    out = []
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
        r = {
            "rk": rk, "top3": top3, "mark": mark,
            "hon": next((f for f, m in mark.items() if m == 1), None),
            "tai": next((f for f, m in mark.items() if m == 2), None),
            "ana": next((f for f, m in mark.items() if m == 3), None),
            "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
            "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
            "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)},
        }
        a1 = a1_pw(r)
        a2 = a2_p3_bad(r, a1)
        if a2 is None or classify_rank(r, a1, a2) != "7B" or r["ana"] is None:
            continue
        if r["ana"] in (a1, a2):
            continue                       # △が軸と重なるケースは対象外
        r["a1"], r["a2"] = a1, a2
        others = sorted(set(r["p3"]) - {a1, a2})
        rest = [f for f in others if f != r["ana"]]
        r["legs3"] = rank_7b_select_legs(others, r["p3"], r["ana"])
        # 集中度
        r["cA"] = r["p3"][a1] + r["p3"][a2] + r["p3"][r["ana"]]
        r["cB"] = r["p3"][r["ana"]] - max(r["p3"][f] for f in rest) if rest else 0.0
        r["cC"] = r["p3"][r["ana"]]
        out.append(r)
    return out


def settle(r: dict, board: dict, legs: list[int], stake_each: int):
    """買い目の投資・払戻を返す（三連複・軸2車+相手）。"""
    legs = [x for x in legs if frozenset({r["a1"], r["a2"], x}) in board]
    if not legs:
        return None
    bet = len(legs) * stake_each
    rest = r["top3"] - {r["a1"], r["a2"]}
    hit = len(r["top3"] & {r["a1"], r["a2"]}) == 2 and len(rest) == 1 \
        and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return bet, ret, hit


def agg(rows):
    n = len(rows)
    if not n:
        return None
    bet = sum(x[0] for x in rows)
    ret = sum(x[1] for x in rows)
    hits = [x for x in rows if x[2]]
    gami = [x for x in hits if x[1] < x[0]]
    pays = [x[1] / x[0] for x in hits]
    return dict(n=n, hit=100 * len(hits) / n, roi=100 * ret / bet if bet else 0,
                gami=100 * len(gami) / len(hits) if hits else 0,
                med=statistics.median(pays) if pays else 0)


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

    per_window = []
    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        print(f"\n######## 窓 {tf}〜{tt} ########", flush=True)
        races = collect(df, tf, tt)
        trio = load_trio(sorted({r["rk"] for r in races}))
        races = [r for r in races if trio.get(r["rk"])]
        print(f"  7B対象（△あり・△が軸と非重複）: {len(races)} レース")
        per_window.append((races, trio))

    # ---- 基準: 現行3点 と 全件1点 ------------------------------------
    print("\n" + "=" * 84)
    print("【1】閾値なしの比較（7B全件・窓平均）")
    print(f"  {'案':<26}{'n':>6}{'的中':>8}{'ROI':>8}{'ガミ率':>8}{'的中中央値':>10}")
    base = {}
    for lbl, mk in (("現行 △除外の3点(300円)", lambda r: (r["legs3"], STAKE)),
                    ("全件 ◎◯△の1点(100円)", lambda r: ([r["ana"]], STAKE))):
        per = []
        for races, trio in per_window:
            rows = [s for r in races
                    if (s := settle(r, trio[r["rk"]], *mk(r))) is not None]
            a = agg(rows)
            if a:
                per.append(a)
        m = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        base[lbl] = per
        print(f"  {lbl:<26}{m['n']:>6.0f}{m['hit']:>7.1f}%{m['roi']:>7.1f}%"
              f"{m['gami']:>7.1f}%{m['med']:>9.2f}倍")

    # ---- 閾値の掃引 ---------------------------------------------------
    for key, name in (("cA", "A: 3車のp3合計"), ("cB", "B: △と4位のギャップ"),
                      ("cC", "C: △のp3")):
        allv = np.array([r[key] for races, _ in per_window for r in races])
        qs = [np.quantile(allv, q) for q in (0.5, 0.6, 0.7, 0.8, 0.9)]
        print(f"\n【2-{key[-1]}】閾値 {name}")
        print(f"  {'閾値(分位)':<16}{'該当n':>7}{'1点の的中':>10}{'1点のROI':>10}"
              f"{'1点ガミ':>8}{'中央値':>8}   {'混合方針ROI':>12}")
        for q, thr in zip((0.5, 0.6, 0.7, 0.8, 0.9), qs):
            per_hi, per_mix = [], []
            for races, trio in per_window:
                hi, mix = [], []
                for r in races:
                    board = trio[r["rk"]]
                    if r[key] >= thr:
                        s = settle(r, board, [r["ana"]], STAKE)
                        if s:
                            hi.append(s)
                            mix.append(s)
                    else:
                        s = settle(r, board, r["legs3"], STAKE)
                        if s:
                            mix.append(s)
                if hi:
                    per_hi.append(agg(hi))
                if mix:
                    per_mix.append(agg(mix))
            if not per_hi:
                continue
            mh = {k: float(np.mean([p[k] for p in per_hi])) for k in per_hi[0]}
            mm = float(np.mean([p["roi"] for p in per_mix])) if per_mix else 0
            print(f"  上位{100*(1-q):>3.0f}% (>={thr:.3f}){mh['n']:>7.0f}{mh['hit']:>9.1f}%"
                  f"{mh['roi']:>9.1f}%{mh['gami']:>7.1f}%{mh['med']:>7.2f}倍"
                  f"{mm:>12.1f}%")

    # ---- オッズ下限の掃引 ---------------------------------------------
    # ⚠️ ここで使うのは wt_odds＝**最終オッズ**。本番のガミ判定は発走15分前の
    #    実測オッズで行うため、そのままの数字では再現しない（楽観側に出る）。
    #    傾向の確認用と割り切ること。
    print("\n【4】1点(◎◯△)にオッズ下限をかける")
    print("     損益分岐 = 1/的中率。オッズで絞ると母集団が変わり的中率も動くので実測する。")
    print(f"  {'下限':<10}{'n':>7}{'的中':>8}{'ROI':>8}{'平均倍率':>9}{'損益分岐倍率':>12}")
    for flo in (0, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        per = []
        for races, trio in per_window:
            rows, mults = [], []
            for r in races:
                board = trio[r["rk"]]
                key = frozenset({r["a1"], r["a2"], r["ana"]})
                if key not in board or board[key] < flo:
                    continue
                s2 = settle(r, board, [r["ana"]], STAKE)
                if s2:
                    rows.append(s2)
                    mults.append(board[key])
            a = agg(rows)
            if a:
                a["mult"] = float(np.mean(mults))
                per.append(a)
        if not per:
            continue
        m = {k: float(np.mean([x[k] for x in per])) for k in per[0]}
        be = 100 / m["hit"] if m["hit"] else 0
        print(f"  {('なし' if not flo else f'>={flo}倍'):<10}{m['n']:>7.0f}{m['hit']:>7.1f}%"
              f"{m['roi']:>7.1f}%{m['mult']:>8.2f}倍{be:>10.2f}倍")

    print("\n【5】3点(△除外)にオッズ下限をかける")
    print("     3点300円は的中目が3.0倍超で損益分岐。最低オッズ / 合成オッズの両方で見る。")
    for mode in ("min", "syn"):
        nm = "最低オッズ" if mode == "min" else "合成オッズ 1/Σ(1/o)"
        print(f"\n  ── {nm}")
        print(f"  {'下限':<10}{'n':>7}{'的中':>8}{'ROI':>8}{'ガミ率':>8}{'中央値':>8}")
        for flo in (0, 2, 3, 4, 5, 6, 7, 8, 10):
            per = []
            for races, trio in per_window:
                rows = []
                for r in races:
                    board = trio[r["rk"]]
                    legs = [x for x in r["legs3"]
                            if frozenset({r["a1"], r["a2"], x}) in board]
                    if not legs:
                        continue
                    od = [board[frozenset({r["a1"], r["a2"], x})] for x in legs]
                    v = min(od) if mode == "min" else 1.0 / sum(1.0 / o for o in od)
                    if v < flo:
                        continue
                    s2 = settle(r, board, legs, STAKE)
                    if s2:
                        rows.append(s2)
                a = agg(rows)
                if a:
                    per.append(a)
            if not per:
                continue
            m = {k: float(np.mean([x[k] for x in per])) for k in per[0]}
            print(f"  {('なし' if not flo else f'>={flo}倍'):<10}{m['n']:>7.0f}{m['hit']:>7.1f}%"
                  f"{m['roi']:>7.1f}%{m['gami']:>7.1f}%{m['med']:>7.2f}倍")

    print("\n【3】窓別 ROI（符号反転の確認）")
    for lbl, per in base.items():
        print(f"  {lbl:<26}" + "  ".join(f"{p['roi']:6.1f}%" for p in per))


if __name__ == "__main__":
    main()
