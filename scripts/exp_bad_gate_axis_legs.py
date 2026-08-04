"""大敗確率を「軸選定」と「相手の絞り込み」に組み込む検証（2026-08-04）。

ユーザー指摘（前版 exp_bad_finish_head.py の設計を是正）:
  「大敗率で軸2に対してのみ見ているが、適正であれば全車に対し有効な条件では。
    大敗率が一定以上に高い場合、軸1・軸2とも選択させず**別の選手を軸にする**必要が
    ある。また相手総流しも軸より閾値を緩くしてよいので、**除外する形で買い目を絞り**
    ROIを確保できないか。**レース自体の除外ではない方向**で検討して」

前版は「大敗しそうなレースを丸ごと見送る」方向だったが、それでは件数が減るだけで
既に枯渇している母集団をさらに削る。本版は件数を減らさずに
  ① 軸選定の母集団から大敗確率の高い車を外す（軸が別の選手に入れ替わる）
  ② 相手（総流し）から大敗確率の高い車を外す（点数が減る＝投資が減る）
を検証する。①はレース数を変えない。②は買い目構成だけを変える。

前提（exp_bad_finish_head.py 窓1）:
  大敗ヘッドは「3着内確率の裏返し」を AUC +0.0126 / PR-AUC +0.0246 上回る。
  相関0.88＝約12%は独立した情報。「4着で惜敗する車」と「7着で沈む車」を
  3着内モデルは区別する動機を持たないため。

⚠️ オッズは wt_odds＝最終オッズ（stale）。ゲートは全て確率のみで構成し
   オッズを条件に使わないため選択バイアスは入らない。DB書き込みなし。

使い方:
    python scripts/exp_bad_gate_axis_legs.py [--windows w1,w2]
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
    # 窓間で的中率の符号が反転したため、ノイズか実体かを切り分ける目的で
    # 過去2窓を追加（2026-08-04・ユーザー指示）。TRAIN_FROM は sb_dyn 充足の
    # 2024-04-01 固定なので、これ以上古い窓は学習期間が痩せて比較にならない。
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
    """テスト期間の三連複オッズを全組み合わせ取得する。"""
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


def build_races(test: pd.DataFrame, p3: np.ndarray, pw: np.ndarray,
                pbad: np.ndarray) -> list[dict]:
    t = test.copy()
    t["pp3"], t["ppw"], t["pbad"] = p3, pw, pbad
    races = []
    for rk, g in t.groupby("race_key"):
        if len(g) != 7:
            continue
        fo = {int(r.frame_no): r.finish_order for r in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if v is not None and v == v and 1 <= int(v) <= 3}
        if len(top3) != 3:
            continue
        races.append({
            "rk": rk,
            "p3": {int(r.frame_no): float(r.pp3) for r in g.itertuples(index=False)},
            "pw": {int(r.frame_no): float(r.ppw) for r in g.itertuples(index=False)},
            "bad": {int(r.frame_no): float(r.pbad) for r in g.itertuples(index=False)},
            "mark": {int(r.frame_no): r.prediction_mark for r in g.itertuples(index=False)},
            "top3": top3,
        })
    return races


def select_axis(r: dict, th_axis: float | None) -> tuple[int, int] | None:
    """大敗確率が th_axis 以上の車を母集団から外してから本番と同じ軸選定を行う。

    全車除外・選定不能になった場合はフィルタ無しへフォールバックする
    （レースを落とさない＝件数を減らさないため）。
    """
    pw, p3 = r["pw"], r["p3"]
    if th_axis is not None:
        ok = {f for f, b in r["bad"].items() if b < th_axis}
        if len(ok) >= 3:
            sel = rank_7s_select_axis({f: pw[f] for f in ok}, {f: p3[f] for f in ok})
            if sel:
                return sel[0], sel[1]
    sel = rank_7s_select_axis(pw, p3)
    return (sel[0], sel[1]) if sel else None


def evaluate(races: list[dict], trio: dict, th_axis: float | None,
             th_leg: float | None, gate: bool = True) -> dict:
    """gate=True なら現行 7S/7A 相当（overlap≤1 かつ axis_sum/entropy ゲート）に絞る。"""
    n = hit = bet = ret = gami = 0
    a1_in = a2_in = both = 0
    pts_sum = 0
    pays: list[float] = []
    for r in races:
        sel = select_axis(r, th_axis)
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
            n_fail = (asum > RANK_7S_AXIS_SUM_MAX) + (ent > RANK_7S_ENTROPY_MAX)
            if n_fail > 1:          # 7S(0個) + 7A(1個) 相当
                continue
        others = [f for f in r["p3"] if f not in (a1, a2)]
        if th_leg is not None:
            kept = [x for x in others if r["bad"][x] < th_leg]
            if kept:
                others = kept
        board = trio.get(r["rk"], {})
        legs = [x for x in others if frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        n += 1
        pts_sum += len(legs)
        a1_in += a1 in r["top3"]
        a2_in += a2 in r["top3"]
        if {a1, a2} <= r["top3"]:
            both += 1
        stake = len(legs) * STAKE
        bet += stake
        rest = r["top3"] - {a1, a2}
        if len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and rest.pop() in legs:
            hit += 1
            od = board[frozenset(r["top3"])]
            got = round(od * 100) // 10 * 10
            ret += got
            pays.append(got)
            if got < stake:
                gami += 1
    return {
        "n": n, "pts": pts_sum / n if n else 0,
        "a1": 100 * a1_in / n if n else 0, "a2": 100 * a2_in / n if n else 0,
        "both": 100 * both / n if n else 0,
        "hit": 100 * hit / n if n else 0,
        "roi": 100 * ret / bet if bet else 0,
        "gami": 100 * gami / hit if hit else 0,
        "med": statistics.median(pays) / 100 if pays else 0,
        "bet": bet,
    }


HDR = (f"{'案':34} {'n':>5} {'点数':>5} {'軸1':>6} {'軸2':>6} {'両方':>6} "
       f"{'的中':>6} {'ROI':>7} {'ガミ':>6} {'中央値':>7}")


def row(lbl: str, s: dict) -> str:
    return (f"{lbl:34} {s['n']:5d} {s['pts']:5.2f} {s['a1']:5.1f}% {s['a2']:5.1f}% "
            f"{s['both']:5.1f}% {s['hit']:5.1f}% {s['roi']:6.1f}% {s['gami']:5.1f}% "
            f"{s['med']:6.1f}倍")


def run_window(df: pd.DataFrame, tf: str, tt: str) -> None:
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
    test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
    print(f"\n######## 窓 test={tf}〜{tt}  train {len(train):,} / test {len(test):,} ########",
          flush=True)
    p3 = fit_predict(train, test, TARGET_COL_WT)
    pw = fit_predict(train, test, "win_flag")
    pbad = fit_predict(train, test, "bad6")
    races = build_races(test, p3, pw, pbad)
    trio = load_trio(sorted({r["rk"] for r in races}))
    print(f"  評価対象 {len(races)} レース（7車立て・3着確定）")

    bad_all = np.concatenate([list(r["bad"].values()) for r in races])
    qs = {q: float(np.quantile(bad_all, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)}
    print("  大敗確率の分位: " + " ".join(f"p{int(q*100)}={v:.3f}" for q, v in qs.items()))

    print("\n【① 軸選定に大敗フィルタ（レース数は減らさない）】")
    print(HDR)
    print(row("現行（フィルタ無し）", evaluate(races, trio, None, None)))
    for q in (0.9, 0.8, 0.7, 0.6):
        th = qs[q]
        print(row(f"軸から bad≥p{int(q*100)}({th:.3f}) を除外",
                  evaluate(races, trio, th, None)))

    print("\n【② 相手から大敗確率の高い車を除外（軸は現行のまま）】")
    print(HDR)
    print(row("現行（総流し）", evaluate(races, trio, None, None)))
    for q in (0.9, 0.8, 0.7, 0.6, 0.5):
        th = qs[q]
        print(row(f"相手から bad≥p{int(q*100)}({th:.3f}) を除外",
                  evaluate(races, trio, None, th)))

    print("\n【③ 組み合わせ（軸は厳しめ・相手は緩め）】")
    print(HDR)
    for qa in (0.8, 0.7):
        for ql in (0.9, 0.8):
            if ql < qa:
                continue
            print(row(f"軸 bad≥p{int(qa*100)} / 相手 bad≥p{int(ql*100)} 除外",
                      evaluate(races, trio, qs[qa], qs[ql])))

    print("\n【④ 参考: ゲート無し（全レース）での相手絞り】")
    print(HDR)
    print(row("ゲート無し・総流し", evaluate(races, trio, None, None, gate=False)))
    for q in (0.8, 0.7, 0.6):
        print(row(f"ゲート無し・相手 bad≥p{int(q*100)} 除外",
                  evaluate(races, trio, None, qs[q], gate=False)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2")
    args = ap.parse_args()
    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    print(f"7車立てに限定: {len(df):,}行")
    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt)


if __name__ == "__main__":
    main()
