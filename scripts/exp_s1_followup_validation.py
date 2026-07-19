"""S1候補（7車・win軸1着固定→top3モデル上位2車2点流し）のフォローアップ検証。

exp_s1_win_axis_trifecta.py で発見した本命候補（top3_gap>=0.15・2点流し・目下限なし・
検証ROI145.8%(n=9949)→テストROI135.3%(n=2851)）について:
  A) 閾値の頑健性: 0.05刻みでなく0.01刻みで近傍を再確認
  B) S2(U)/S3(M)との重複: 同一レースを既に他ランクで拾っていないか
  C) 日次換算・払戻分布: 実務イメージ（1日あたり件数）・巨額配当への依存度チェック
     （上位1/3/5件の払戻を除外してもROIが崩れないか）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
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
        win_probs = g_win["p_win"].tolist()
        win_gap12 = float(win_probs[0] - win_probs[1])

        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        remainder = g_top3[g_top3["frame_no"] != axis].reset_index(drop=True)
        if len(remainder) < 2:
            continue
        p1 = int(remainder.iloc[0]["frame_no"])
        p2 = int(remainder.iloc[1]["frame_no"])
        top3_gap = float(remainder.iloc[0]["p_top3"] - remainder.iloc[1]["p_top3"])

        order3 = tuple(fno for _, fno in f[:3])
        tri_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        race_date = rk.split("_")[0]
        race_date_fmt = f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:8]}"
        races.append({
            "race_key": rk, "race_date": race_date_fmt, "tri": tri, "board": board,
            "order3": order3, "tri_pay": tri_pay,
            "axis": axis, "p1": p1, "p2": p2,
            "win_gap12": win_gap12, "top3_gap": top3_gap,
        })
    return races


def settle_2pt(races, gate_fn, leg=0.0):
    n = hits = bet = pay = 0
    hit_payouts = []
    picked_keys = []
    for r in races:
        if not gate_fn(r):
            continue
        a, p1, p2 = r["axis"], r["p1"], r["p2"]
        buy = []
        for combo in [(a, p1, p2), (a, p2, p1)]:
            ov = r["tri"].get(combo) or 0
            if ov >= leg:
                buy.append(combo)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        picked_keys.append(r["race_key"])
        if r["order3"] in buy:
            hits += 1
            p = r["tri_pay"] * STAKE // 100
            pay += p
            hit_payouts.append(p)
    roi = pay / bet * 100 if bet else 0
    return {"n": n, "hits": hits, "bet": bet, "pay": pay, "roi": roi,
            "hit_payouts": hit_payouts, "picked_keys": picked_keys}


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築（7車）...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model)
    print("テストデータ構築（7車）...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model)
    print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)

    # ---- A) 閾値の頑健性（0.01刻み） ----
    print("\n===== A) top3_gap 閾値の頑健性（0.01刻み） =====", flush=True)
    print(f"{'閾値':>6} | {'検証n':>6} {'検証的中':>8} {'検証ROI':>8} | {'テストn':>7} {'テスト的中':>9} {'テストROI':>9}")
    for th100 in range(8, 21):
        th = th100 / 100
        gf = lambda r, th=th: r["top3_gap"] >= th
        rv = settle_2pt(val, gf)
        rt = settle_2pt(test, gf)
        v_hit = rv["hits"] / rv["n"] * 100 if rv["n"] else 0
        t_hit = rt["hits"] / rt["n"] * 100 if rt["n"] else 0
        print(f"{th:6.2f} | {rv['n']:6d} {v_hit:7.1f}% {rv['roi']:7.1f}% | "
              f"{rt['n']:7d} {t_hit:8.1f}% {rt['roi']:8.1f}%")

    # ---- B) S2/S3との重複確認（テスト期間・採用候補=top3_gap>=0.15） ----
    print("\n===== B) S2(U)/S3(M)との重複確認（テスト期間・top3_gap>=0.15） =====", flush=True)
    best_gate = lambda r: r["top3_gap"] >= 0.15
    rt_best = settle_2pt(test, best_gate)
    picked = set(rt_best["picked_keys"])
    print(f"S1候補レース数: {len(picked)}")
    with get_connection() as c:
        placeholders = ",".join("?" * len(picked)) if picked else "''"
        um_rows = c.execute(
            f"SELECT DISTINCT SUBSTR(race_key, 1, INSTR(race_key,'#')-1) as base_key, rank "
            f"FROM picks_history WHERE rank IN ('7PLUS_U','7PLUS_M') "
            f"AND SUBSTR(race_key, 1, INSTR(race_key,'#')-1) IN ({placeholders})",
            list(picked),
        ).fetchall() if picked else []
    overlap_keys = {row[0] for row in um_rows}
    print(f"うちS2/S3にも該当: {len(overlap_keys)}件 ({len(overlap_keys)/len(picked)*100 if picked else 0:.1f}%)")

    # ---- C) 日次換算・巨額配当依存度チェック ----
    print("\n===== C) 日次換算・払戻分布（テスト期間・top3_gap>=0.15） =====", flush=True)
    n_days = len(set(r["race_date"] for r in test))
    print(f"テスト期間の実施日数: {n_days}日 / 候補レース数: {rt_best['n']} → {rt_best['n']/n_days:.1f}R/日")
    hp = sorted(rt_best["hit_payouts"], reverse=True)
    print(f"的中払戻の分布: 最大={hp[0] if hp else 0}円 中央値={hp[len(hp)//2] if hp else 0}円 "
          f"最小={hp[-1] if hp else 0}円 (n={len(hp)}件)")
    for topn in (1, 3, 5, 10):
        if len(hp) < topn:
            continue
        excl_pay = rt_best["pay"] - sum(hp[:topn])
        excl_roi = excl_pay / rt_best["bet"] * 100 if rt_best["bet"] else 0
        print(f"  上位{topn}件除外後のROI: {excl_roi:.1f}% (除外払戻合計={sum(hp[:topn])}円)")


if __name__ == "__main__":
    main()
