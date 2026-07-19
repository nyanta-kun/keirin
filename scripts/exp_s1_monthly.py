"""S1候補（7車・win軸1着固定→top3モデル上位2車2点流し・top3_gap>=0.15）の月次ROI推移。"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

STAKE = 100
TRAIN_FROM, TRAIN_TO = "2022-12-01", "2025-03-31"
FULL_FROM, FULL_TO = "2025-04-01", "2026-07-15"
SEED = 42
N_RIDERS = 7
TOP3_GAP_TH = 0.15


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
        if top3_gap < TOP3_GAP_TH:
            continue

        order3 = tuple(fno for _, fno in f[:3])
        tri_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        buy = [(axis, p1, p2), (axis, p2, p1)]
        bet = len(buy) * STAKE
        hit = order3 in buy
        pay = tri_pay * STAKE // 100 if hit else 0
        race_date = rk.split("_")[0]
        ym = f"{race_date[:4]}-{race_date[4:6]}"
        races.append({"ym": ym, "bet": bet, "hit": int(hit), "pay": pay})
    return races


def main():
    win_model, top3_model = train_models()
    print(f"\n全期間データ構築（7車・{FULL_FROM}〜{FULL_TO}）...", flush=True)
    races = collect(FULL_FROM, FULL_TO, win_model, top3_model)
    print(f"該当レース {len(races)}件（top3_gap>={TOP3_GAP_TH}）", flush=True)

    monthly = defaultdict(lambda: {"n": 0, "hits": 0, "bet": 0, "pay": 0})
    yearly = defaultdict(lambda: {"n": 0, "hits": 0, "bet": 0, "pay": 0})
    for r in races:
        m = monthly[r["ym"]]
        m["n"] += 1
        m["hits"] += r["hit"]
        m["bet"] += r["bet"]
        m["pay"] += r["pay"]
        y = yearly[r["ym"][:4]]
        y["n"] += 1
        y["hits"] += r["hit"]
        y["bet"] += r["bet"]
        y["pay"] += r["pay"]

    print(f"\n===== S1候補 月次ROI（top3_gap>={TOP3_GAP_TH}） =====")
    n_plus = n_minus = 0
    for ym in sorted(monthly.keys()):
        m = monthly[ym]
        roi = m["pay"] / m["bet"] * 100 if m["bet"] else 0
        flag = "+" if roi >= 100 else "-"
        if roi >= 100:
            n_plus += 1
        else:
            n_minus += 1
        print(f"  {ym}: n={m['n']:4d} 的中={m['hits']:3d} ROI={roi:7.1f}% {flag}")
    print(f"  → プラス月 {n_plus}/{n_plus+n_minus}")

    print(f"\n===== S1候補 年次(暦年)ROI（top3_gap>={TOP3_GAP_TH}） =====")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        roi = y["pay"] / y["bet"] * 100 if y["bet"] else 0
        flag = "+" if roi >= 100 else "-"
        print(f"  {yr}: n={y['n']:5d} 的中={y['hits']:4d} ROI={roi:7.1f}% {flag}")


if __name__ == "__main__":
    main()
