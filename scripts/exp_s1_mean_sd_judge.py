"""S1候補のROI判定基準変更: レース単位ROIの平均±2SDで100%を超えるか判定する。

ユーザー指定の基準: 「窓内の平均値から±標準偏差×2の範囲で確認し、100%を超えるかで判定。
実結果としては範囲外も含める」＝ 外れ値（万車券）を除外せず全件を母集団としたまま、
レース単位ROIの分布から mean - 2*SD が 100%（損益分岐）を上回るかを見る、より厳しい基準。

対象: exp_s1_win_axis_trifecta.py で発見した候補（7車・win軸1着固定→top3モデル上位2車
2点流し・目下限なし）。top3_gap閾値 0.08〜0.20 の全域で判定する。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

STAKE = 100
TRAIN_FROM, TRAIN_TO = "2022-12-01", "2025-03-31"
VAL_FROM, VAL_TO = "2025-04-01", "2026-03-31"
TEST_FROM, TEST_TO = "2026-04-01", "2026-07-15"
SEED = 42
N_RIDERS = 7


def train_models():
    print("学習データ読み込み...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=TRAIN_TO))
    df = df[df["finish_order"].notna()]
    X = prepare_X(df)
    win_y = (df["finish_order"] == 1).astype(int)
    top3_y = df["finish_order"].between(1, 3).astype(int)

    def _fit(y):
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(X, y)
        return m

    print("1着モデル学習...", flush=True)
    win_model = _fit(win_y)
    print("3着内モデル学習...", flush=True)
    top3_model = _fit(top3_y)
    return win_model, top3_model


def collect(tf, tt, win_model, top3_model):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_win"] = win_model.predict_proba(X)[:, 1]
    df["p_top3"] = top3_model.predict_proba(X)[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == N_RIDERS]
        fins = {}
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, f, fo in c.execute(q, ch):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(f)))
        tri_bd = defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trifecta' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    tri_bd[rk][parts] = fv
    pm = _load_payouts_wt(rks)

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != N_RIDERS or len(g) != N_RIDERS:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        tri = tri_bd.get(rk)
        if not tri:
            continue
        board = set()
        for k in tri:
            board |= set(k)
        if len(board) != N_RIDERS:
            continue

        g_win = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        axis = int(g_win.iloc[0]["frame_no"])

        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        remainder = g_top3[g_top3["frame_no"] != axis].reset_index(drop=True)
        if len(remainder) < 2:
            continue
        p1 = int(remainder.iloc[0]["frame_no"])
        p2 = int(remainder.iloc[1]["frame_no"])
        top3_gap = float(remainder.iloc[0]["p_top3"] - remainder.iloc[1]["p_top3"])

        order3 = tuple(fno for _, fno in f[:3])
        tri_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        races.append({
            "tri": tri, "board": board, "order3": order3, "tri_pay": tri_pay,
            "axis": axis, "p1": p1, "p2": p2, "top3_gap": top3_gap,
        })
    return races


def race_level_rois(races, th):
    """レース単位ROI配列を返す（目下限なし=2点固定200円stake前提）。"""
    rois = []
    for r in races:
        if r["top3_gap"] < th:
            continue
        a, p1, p2 = r["axis"], r["p1"], r["p2"]
        buy = [(a, p1, p2), (a, p2, p1)]  # leg=0なので常に2点
        bet = len(buy) * STAKE
        pay = r["tri_pay"] * STAKE // 100 if r["order3"] in buy else 0
        rois.append(pay / bet * 100)
    return np.array(rois)


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築（7車）...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model)
    print("テストデータ構築（7車）...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model)
    print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)

    print("\n===== レース単位ROIの mean±2SD 判定（100%=損益分岐） =====", flush=True)
    print(f"{'閾値':>6} | {'期間':>6} {'n':>6} {'mean':>8} {'SD':>10} {'mean-2SD':>10} {'mean+2SD':>10} {'合格':>4}")
    for th100 in range(8, 21):
        th = th100 / 100
        for label, races in (("検証", val), ("テスト", test)):
            rois = race_level_rois(races, th)
            if len(rois) == 0:
                continue
            mean = rois.mean()
            sd = rois.std(ddof=1)
            lo, hi = mean - 2 * sd, mean + 2 * sd
            passed = "OK" if lo > 100 else "NG"
            print(f"{th:6.2f} | {label:>6} {len(rois):6d} {mean:7.1f}% {sd:9.1f}% "
                  f"{lo:9.1f}% {hi:9.1f}% {passed:>4}")


if __name__ == "__main__":
    main()
