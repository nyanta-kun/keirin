"""1着率・3着内率・大敗率の3ヘッドを横並びにして軸を選ぶ設計の検証（2026-08-04）。

ユーザー提案:
  「1着率、3着内率、大敗率を全レース同様に算出の上、横並びに見て
    軸1（勝ちそうで大きく負けなそう）を中心に軸2の検討をする」

現行の軸選定（rank_7s_select_axis）は
    軸2車 = pw上位3 ∩ p3上位3 の重なりから p3 上位2
で、**大敗率という第3の軸を持たない**。本スクリプトは
    軸1 = z(pw) − w1·z(pbad) の最上位   … 勝ちそうで大きく負けなそう
    軸2 = z(p3) − w2·z(pbad) の最上位   … 3着内は堅いが大敗しにくい（軸1を除く）
として w1/w2 を掃引し、現行と比較する。

**軸1と軸2で重みを変えられる**のがこの設計の要点。実測（exp_recent_miss_breakdown.py）で
軸1は外しても4着（惜敗）が62%なのに対し軸2は6着以下の大敗が67%と外し方の質が違うため、
軸2により強い大敗ペナルティ（w2 > w1）をかけるのが理にかなう。

前提と期待値（過大評価しないための注記）:
  3ヘッドは同じ特徴・同じデータから出るため強く相関する。実測で pbad と −p3 の相関は
  0.88、独立情報は約12%。「横並びに見る」といっても使える自由度はその12%の部分。
  ハード除外版（exp_bad_gate_axis_legs.py）ではROIが両窓で改善する一方、的中率は
  窓間で符号が反転した。連続的な合成にすることで的中率側にも効くかを見る
  （隊列位置で「後段補正はゼロ / ベース特徴なら+0.003」だった前例と同じ構図）。

評価は**レース内順位品質を主指標**とする（軸1/軸2の3着内率・両方3着内率・軸1の1着率・
軸2の大敗率）。AUCは採否基準にしない（2026-08-04 に3度、AUC改善と的中率悪化の
乖離を観測したため）。

⚠️ オッズは wt_odds＝最終オッズ（stale）。選出条件は確率のみでオッズ非依存。
DB書き込みなし。

使い方:
    python scripts/exp_three_head_axis.py [--windows w1,w2,w3,w4]
"""
from __future__ import annotations

import argparse
import re
import statistics
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
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    rank_7s_field_entropy, rank_7s_select_axis, rank_7s_wt_overlap_n,
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
N_ENTRIES = 7          # main() で --n-entries から設定（run_window が参照する）


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
                if v <= 0 or v >= 9999:
                    continue
                parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                if len(parts) == 3:
                    out[rk][parts] = v
    return out


def _z(d: dict[int, float]) -> dict[int, float]:
    """レース内 z 化（全車同値なら 0）。"""
    v = np.array(list(d.values()), dtype=float)
    m, s = v.mean(), v.std()
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: (x - m) / s for k, x in d.items()}


def select_axis(r: dict, w1: float | None, w2: float | None) -> tuple[int, int] | None:
    """w1/w2 が None なら現行ロジック（win∩top3 の重なり）。"""
    if w1 is None or w2 is None:
        sel = rank_7s_select_axis(r["pw"], r["p3"])
        return (sel[0], sel[1]) if sel else None
    zw, zp, zb = _z(r["pw"]), _z(r["p3"]), _z(r["bad"])
    s1 = {f: zw[f] - w1 * zb[f] for f in zw}
    a1 = max(s1, key=lambda f: s1[f])
    s2 = {f: zp[f] - w2 * zb[f] for f in zp if f != a1}
    if not s2:
        return None
    a2 = max(s2, key=lambda f: s2[f])
    return a1, a2


def evaluate(races: list[dict], trio: dict, w1: float | None, w2: float | None,
             gate: bool = True) -> dict:
    n = hit = bet = ret = 0
    a1_in = a2_in = both = a1_win = a2_bad = 0
    pays: list[float] = []
    for r in races:
        sel = select_axis(r, w1, w2)
        if not sel:
            continue
        a1, a2 = sel
        if gate:
            ov = rank_7s_wt_overlap_n(
                a1, a2,
                next((f for f, m in r["mark"].items() if m == 1), None),
                next((f for f, m in r["mark"].items() if m == 2), None))
            if ov not in (0, 1):
                continue
            asum = r["p3"][a1] + r["p3"][a2]
            ent = rank_7s_field_entropy(r["p3"])
            if (asum > RANK_7S_AXIS_SUM_MAX) + (ent > RANK_7S_ENTROPY_MAX) > 1:
                continue
        board = trio.get(r["rk"], {})
        legs = [x for x in r["p3"] if x not in (a1, a2)
                and frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        n += 1
        a1_in += a1 in r["top3"]
        a2_in += a2 in r["top3"]
        a1_win += r["fo"].get(a1) == 1
        a2_bad += r["fo"].get(a2, 0) >= 6
        if {a1, a2} <= r["top3"]:
            both += 1
        stake = len(legs) * STAKE
        bet += stake
        rest = r["top3"] - {a1, a2}
        if len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and rest.pop() in legs:
            hit += 1
            got = round(board[frozenset(r["top3"])] * 100) // 10 * 10
            ret += got
            pays.append(got)
    f = lambda x: 100 * x / n if n else 0.0          # noqa: E731
    return {"n": n, "a1": f(a1_in), "a2": f(a2_in), "both": f(both),
            "a1win": f(a1_win), "a2bad": f(a2_bad), "hit": f(hit),
            "roi": 100 * ret / bet if bet else 0.0,
            "med": statistics.median(pays) / 100 if pays else 0.0}


HDR = (f"{'案':28} {'n':>5} {'軸1':>6} {'軸1勝':>7} {'軸2':>6} {'軸2大敗':>8} "
       f"{'両方':>6} {'的中':>6} {'ROI':>7}")


def row(lbl: str, s: dict) -> str:
    return (f"{lbl:28} {s['n']:5d} {s['a1']:5.1f}% {s['a1win']:6.1f}% "
            f"{s['a2']:5.1f}% {s['a2bad']:7.1f}% {s['both']:5.1f}% "
            f"{s['hit']:5.1f}% {s['roi']:6.1f}%")


def run_window(df: pd.DataFrame, tf: str, tt: str, acc: dict) -> None:
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
    test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
    print(f"\n######## 窓 test={tf}〜{tt}  train {len(train):,} / test {len(test):,} ########",
          flush=True)
    p3 = fit_predict(train, test, TARGET_COL_WT)
    pw = fit_predict(train, test, "win_flag")
    pbad = fit_predict(train, test, "bad6")

    t = test.copy()
    t["pp3"], t["ppw"], t["pbad"] = p3, pw, pbad
    races = []
    for rk, g in t.groupby("race_key"):
        if len(g) != N_ENTRIES:
            continue
        fo = {int(r.frame_no): (int(r.finish_order)
                                if r.finish_order is not None and r.finish_order == r.finish_order
                                else 0)
              for r in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue
        races.append({
            "rk": rk, "fo": fo, "top3": top3,
            "p3": {int(r.frame_no): float(r.pp3) for r in g.itertuples(index=False)},
            "pw": {int(r.frame_no): float(r.ppw) for r in g.itertuples(index=False)},
            "bad": {int(r.frame_no): float(r.pbad) for r in g.itertuples(index=False)},
            "mark": {int(r.frame_no): r.prediction_mark for r in g.itertuples(index=False)},
        })
    trio = load_trio(sorted({r["rk"] for r in races}))
    print(f"  評価対象 {len(races)} レース")

    # ヘッド間の相関（自由度がどれだけあるか）
    print(f"  相関: pw–p3 {np.corrcoef(pw, p3)[0,1]:+.4f}  "
          f"p3–pbad {np.corrcoef(p3, pbad)[0,1]:+.4f}  "
          f"pw–pbad {np.corrcoef(pw, pbad)[0,1]:+.4f}")

    print(HDR)
    base = evaluate(races, trio, None, None)
    print(row("現行（win∩top3 重なり）", base))
    acc.setdefault("現行", []).append(base)

    for w1, w2 in ((0.0, 0.0), (0.0, 0.3), (0.0, 0.6),
                   (0.3, 0.3), (0.3, 0.6), (0.6, 0.6), (0.6, 1.0)):
        s = evaluate(races, trio, w1, w2)
        lbl = f"z合成 w1={w1} w2={w2}"
        print(row(lbl, s))
        acc.setdefault(lbl, []).append(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    ap.add_argument("--n-entries", type=int, default=7,
                    help="対象車数（7 または 9）。9車は7車の知見が移る保証が無いため"
                         "別途測るのに使う（2026-08-04 追加）")
    ap.add_argument("--bad-rank", type=int, default=6,
                    help="大敗ヘッドの閾値。既定6=「6着以下」。7車では下位2/7=27.8%%だが"
                         "9車では下位4/9=43.3%% と意味が変わるため、9車では 7 "
                         "（下位3/9=33%%）も試すこと")
    args = ap.parse_args()
    global N_ENTRIES
    N_ENTRIES = args.n_entries
    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= args.bad_rank) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == args.n_entries].copy()
    print(f"{args.n_entries}車立てに限定: {len(df):,}行 "
          f"（大敗ヘッド = {args.bad_rank}着以下・実勢 {100*df['bad6'].mean():.1f}%）")

    acc: dict[str, list] = {}
    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt, acc)

    print("\n" + "=" * 100)
    print("【全窓平均】（窓ごとの単純平均・現行との差）")
    print(HDR)
    b = {k: float(np.mean([s[k] for s in acc["現行"]]))
         for k in ("n", "a1", "a1win", "a2", "a2bad", "both", "hit", "roi")}
    print(row("現行（win∩top3 重なり）", {**b, "n": int(b["n"])}))
    for lbl, lst in acc.items():
        if lbl == "現行":
            continue
        m = {k: float(np.mean([s[k] for s in lst]))
             for k in ("n", "a1", "a1win", "a2", "a2bad", "both", "hit", "roi")}
        print(row(lbl, {**m, "n": int(m["n"])}))
        print(f"{'  └ 差':28} {'':5} {m['a1']-b['a1']:+5.1f}pt {m['a1win']-b['a1win']:+6.1f}pt "
              f"{m['a2']-b['a2']:+5.1f}pt {m['a2bad']-b['a2bad']:+7.1f}pt "
              f"{m['both']-b['both']:+5.1f}pt {m['hit']-b['hit']:+5.1f}pt "
              f"{m['roi']-b['roi']:+6.1f}pt")


if __name__ == "__main__":
    main()
