"""S1再検討: 1着モデル軸(1着固定)×3着内モデルで相手2車選出の三連単スイープ（正規プロトコル）。

ユーザー要望: 「1位の指数を固定し、3着モデルから相手1,2車選んでROI100%超える条件検討」

構造:
  軸 = 1着モデル(win model) レース内1位（三連単の1着に固定）
  相手 = 3着内モデル(top3 model)で、軸を除いた残り車の中で上位2頭（p1=上位1頭目, p2=上位2頭目）
  券種: A) 2点流し（軸→p1→p2, 軸→p2→p1）  B) 1点固定（軸→p1→p2・3着内モデル順そのまま）

ゲート探索軸:
  - win_gap12: 軸(win_rank=1)のwin確率 − win_rank=2のwin確率（軸の信頼度）
  - top3_gap: p1のtop3確率 − p2のtop3確率（相手の確信度）
  - agree: 軸(win model 1位) が top3モデル自身の1位とも一致するか
  - 目オッズ下限（三連単配当）
  - 車数（6車 / 7車）

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（条件選択）・
テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証 ROI>=95 ∧ n>=100 で的中率最大（過去のS1/S3スイープと同一基準）。
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


def collect(tf, tt, win_model, top3_model, n_riders):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_win"] = win_model.predict_proba(X)[:, 1]
    df["p_top3"] = top3_model.predict_proba(X)[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == n_riders]
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
        if ne_map.get(rk) != n_riders or len(g) != n_riders:
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
        if len(board) != n_riders:
            continue

        g_win = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        axis = int(g_win.iloc[0]["frame_no"])
        win_probs = g_win["p_win"].tolist()
        win_gap12 = float(win_probs[0] - win_probs[1])

        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        top3_rank1_fno = int(g_top3.iloc[0]["frame_no"])
        agree = (axis == top3_rank1_fno)

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
            "axis": axis, "p1": p1, "p2": p2,
            "win_gap12": win_gap12, "top3_gap": top3_gap, "agree": agree,
        })
    return races


def settle_2pt(races, gate_fn, leg):
    """軸→p1→p2, 軸→p2→p1 の2点流し。"""
    n = hits = bet = pay = 0
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
        if r["order3"] in buy:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def settle_1pt(races, gate_fn, leg):
    """軸→p1→p2 の1点固定（3着内モデル順そのまま）。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gate_fn(r):
            continue
        combo = (r["axis"], r["p1"], r["p2"])
        ov = r["tri"].get(combo) or 0
        if ov < leg:
            continue
        n += 1
        bet += STAKE
        if r["order3"] == combo:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def build_gates():
    gates = [("なし", lambda r: True)]
    for th in (0.05, 0.10, 0.15, 0.20):
        gates.append((f"win_gap12>={th:.2f}", (lambda r, th=th: r["win_gap12"] >= th)))
    for th in (0.05, 0.10, 0.15):
        gates.append((f"top3_gap>={th:.2f}", (lambda r, th=th: r["top3_gap"] >= th)))
    gates.append(("agree(win軸=top3モデル1位)", lambda r: r["agree"]))
    gates.append(("mismatch(win軸≠top3モデル1位)", lambda r: not r["agree"]))
    for wth in (0.10, 0.15):
        for tth in (0.05, 0.10):
            gates.append((f"win_gap12>={wth:.2f}∧top3_gap>={tth:.2f}",
                          (lambda r, wth=wth, tth=tth: r["win_gap12"] >= wth and r["top3_gap"] >= tth)))
    return gates


LEGS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def run(n_riders, val, test, bet_label, settle_fn):
    print(f"\n{'='*70}\n{n_riders}車 / {bet_label}\n{'='*70}", flush=True)
    gates = build_gates()
    results = []
    for name, gf in gates:
        for leg in LEGS:
            n, h, roi = settle_fn(val, gf, leg)
            results.append((name, gf, leg, n, h, roi))
    survivors = [x for x in results if x[3] >= 100 and x[5] >= 95]
    if not survivors:
        best = sorted(results, key=lambda x: -x[5])[:5]
        print("  検証ROI>=95%∧n>=100 の条件なし。参考(検証ROI上位5):")
        for name, gf, leg, n, h, roi in best:
            print(f"    {name} 目>={leg:.0f}: n={n} 的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%")
        return
    survivors.sort(key=lambda x: -(x[4] / x[3]))
    print(f"  検証ROI>=95%∧n>=100: {len(survivors)}件生存。的中率上位5件をテスト評価:")
    for name, gf, leg, n, h, roi in survivors[:5]:
        tn, th, troi = settle_fn(test, gf, leg)
        print(f"    {name} 目>={leg:.0f}: 検証(的中{h/n*100:.1f}%/ROI{roi:.1f}%/n={n}) "
              f"→ テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")


def main():
    win_model, top3_model = train_models()
    for n_riders in (6, 7):
        print(f"\n\n########## データ構築: {n_riders}車 ##########", flush=True)
        val = collect(VAL_FROM, VAL_TO, win_model, top3_model, n_riders)
        test = collect(TEST_FROM, TEST_TO, win_model, top3_model, n_riders)
        print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)
        run(n_riders, val, test, "A) 2点流し(軸→p1→p2, 軸→p2→p1)", settle_2pt)
        run(n_riders, val, test, "B) 1点固定(軸→p1→p2)", settle_1pt)


if __name__ == "__main__":
    main()
