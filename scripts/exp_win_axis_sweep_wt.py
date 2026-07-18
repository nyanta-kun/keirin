"""Phase B: 1着モデル軸×複合ゲートの買い目スイープ（正規プロトコル）。

基礎分析（exp_win_model_base_wt.py）の発見:
  - 不一致レースで軸を「1着モデル1位」に差し替えると1着率+7pt（27.0%→34.0%）
    → 三連単（1着固定）に向く
  - 3着内モデル1位の「1着モデル内相対順位」が悪化するほど段階的に成績が落ちる
    → 消しゲート/信頼度ゲートとして使える

券種2種を検証:
  A) 三連単 軸(1着モデル1位)→相方(同L逃)→残り流し（現行S3の三連複を三連単に変更・
     軸を差し替え）
  B) 三連単 軸(3着内モデル1位=現行S3軸)固定・ただし1着モデル内順位ゲートを追加

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（条件選定）・
テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証ROI>=95 ∧ n>=100 で的中率最大。
"""
import math
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
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT

STAKE = 100
TRAIN_FROM, TRAIN_TO = "2022-12-01", "2025-03-31"
VAL_FROM, VAL_TO = "2025-04-01", "2026-03-31"
TEST_FROM, TEST_TO = "2026-04-01", "2026-07-15"
SEED = 42


def _entropy(probs):
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum(max(p / total, 1e-9) * math.log(max(p / total, 1e-9)) for p in probs)


def train_models():
    """1着モデル・3着内モデルを学習〜2025-03-31で学習し、両モデルを返す。"""
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
        rks = [rk for rk in rks_all if ne_map.get(rk) == 7]
        marks, fins = {}, {}
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, f, pmv, fo in c.execute(q, ch):
                marks.setdefault(rk, {})[int(f)] = pmv
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(f)))
        trio_bd, tri_bd = defaultdict(dict), defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(ch)))
            for rk, bt, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio" and len(parts) == 3 and fv < 9000:
                    trio_bd[rk][frozenset(parts)] = fv
                elif bt == "trifecta" and len(parts) == 3:
                    tri_bd[rk][tuple(parts)] = fv
    pm = _load_payouts_wt(rks)

    def _iv(v):
        return None if v is None or pd.isna(v) else int(v)

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk, {})
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        g_win = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        rows_top3 = list(g_top3.itertuples(index=False))
        rows_win = list(g_win.itertuples(index=False))

        wt_marks = [fno for fno, v in marks.get(rk, {}).items() if v == 1]
        wt_top = min(wt_marks) if wt_marks else None

        top3_m1 = int(rows_top3[0].frame_no)
        win_m1 = int(rows_win[0].frame_no)
        probs_top3 = [float(x) for x in g_top3["p_top3"].tolist()]
        gap12_top3 = probs_top3[0] - probs_top3[1]

        # top3_m1 の win モデル内順位（1-indexed）
        win_rank_map = {int(r.frame_no): i + 1 for i, r in enumerate(rows_win)}
        top3_m1_win_rank = win_rank_map.get(top3_m1)

        def _mate_same_line_nige(rows, axis_fno):
            ax_row = next((r for r in rows if int(r.frame_no) == axis_fno), None)
            if ax_row is None:
                return None
            lg = _iv(getattr(ax_row, "line_group", None))
            if lg is None:
                return None
            lp = _iv(getattr(ax_row, "line_pos", None))
            want = 1 if lp == 2 else 2
            cands = []
            for r in rows:
                fno = int(r.frame_no)
                st = r.style if isinstance(getattr(r, "style", None), str) else ""
                if fno == axis_fno or _iv(getattr(r, "line_group", None)) != lg or st != "逃":
                    continue
                cands.append((fno, _iv(getattr(r, "line_pos", None))))
            if not cands:
                return None
            cands.sort()
            return next((f for f, lp2 in cands if lp2 == want), cands[0][0])

        mate_top3axis = _mate_same_line_nige(rows_top3, top3_m1)
        mate_winaxis = _mate_same_line_nige(rows_top3, win_m1)

        races.append({
            "trio": trio, "tri": tri_bd.get(rk, {}), "board": board,
            "top3": frozenset(fno for _, fno in f[:3]),
            "order3": tuple(fno for _, fno in f[:3]),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "tri_pay": pm.get(rk, {}).get(("trifecta", tuple(fno for _, fno in f[:3])), 0),
            "gap12_top3": gap12_top3,
            "top3_m1": top3_m1, "win_m1": win_m1,
            "wt_top": wt_top,
            "mismatch": wt_top is not None and wt_top != top3_m1,
            "top3_m1_win_rank": top3_m1_win_rank,
            "mate_top3axis": mate_top3axis, "mate_winaxis": mate_winaxis,
        })
    return races


GATES = [
    ("なし", lambda r: True),
    ("不一致", lambda r: r["mismatch"]),
    ("不一致∧gap12>=0.10", lambda r: r["mismatch"] and r["gap12_top3"] >= 0.10),
    ("不一致∧win_rank<=2", lambda r: r["mismatch"] and r["top3_m1_win_rank"] is not None
                                     and r["top3_m1_win_rank"] <= 2),
    ("不一致∧win_rank>=3", lambda r: r["mismatch"] and r["top3_m1_win_rank"] is not None
                                     and r["top3_m1_win_rank"] >= 3),
    ("gap12>=0.10", lambda r: r["gap12_top3"] >= 0.10),
]

AXES = [("top3軸(現行)", lambda r: r["top3_m1"], lambda r: r["mate_top3axis"]),
        ("win軸(新)", lambda r: r["win_m1"], lambda r: r["mate_winaxis"])]


def settle_trifecta(races, gf, ax_fn, mate_fn, leg):
    """三連単 軸(1着固定)→相方→残り の (n-2)通り流し。目オッズ>=leg のみ。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax_fn(r)
        b = mate_fn(r)
        if a is None or b is None or a == b:
            continue
        buy = []
        for t in r["board"] - {a, b}:
            combo = (a, b, t)
            ov = r["tri"].get(combo) or 0
            if ov >= leg:
                buy.append(combo)
            combo2 = (a, t, b)
            ov2 = r["tri"].get(combo2) or 0
            if ov2 >= leg:
                buy.append(combo2)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order3"] in buy:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def settle_trio(races, gf, ax_fn, mate_fn, leg):
    """三連複（比較用）。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax_fn(r)
        b = mate_fn(r)
        if a is None or b is None or a == b:
            continue
        buy = [frozenset({a, b, t}) for t in r["board"] - {a, b}
               if (r["trio"].get(frozenset({a, b, t})) or 0) >= leg]
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["top3"] in buy:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def run_family(label, val, test, cells, settle_fn):
    print(f"\n===== {label} =====", flush=True)
    results = []
    for name, gf, af, mf, leg in cells:
        n, h, roi = settle_fn(val, gf, af, mf, leg)
        if n >= 100:
            results.append((h / n * 100, roi, name, (gf, af, mf, leg), n, h))
    frontier = sorted([x for x in results if x[1] >= 95], reverse=True)
    print("  検証 ROI>=95% の的中率フロンティア:")
    for hit, roi, name, _, n, h in frontier[:8]:
        print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
    if not frontier:
        best = sorted(results, key=lambda x: -x[1])[:5]
        print("  （ROI>=95%なし・参考: 検証ROI上位）")
        for hit, roi, name, _, n, h in best:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
        return
    hit, roi, name, params, n, h = frontier[0]
    tn, th, troi = settle_fn(test, *params)
    print(f"  【選択（的中率最大 s.t. 検証ROI>=95）】{name}（検証 的中{hit:.1f}%・ROI{roi:.1f}%・n={n}）")
    print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for hit2, roi2, name2, params2, n2, _ in frontier[1:3]:
        tn2, th2, troi2 = settle_fn(test, *params2)
        print(f"  [参考] {name2}: 検証{hit2:.1f}%/{roi2:.1f}% → テスト n={tn2} "
              f"的中={th2/tn2*100 if tn2 else 0:.1f}% ROI={troi2:.1f}%")


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model)
    print("テストデータ構築...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model)
    print(f"7車 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    legs = (0.0, 10.0, 15.0, 20.0, 30.0)

    cells_a = [(f"三単[{gl}]×{al}×目>={leg:.0f}", gf, af, mf, leg)
               for gl, gf in GATES for al, af, mf in AXES for leg in legs]
    run_family("A) 三連単 軸×相方(同L逃)×目下限", val, test, cells_a, settle_trifecta)

    cells_b = [(f"三複[{gl}]×{al}×目>={leg:.0f}", gf, af, mf, leg)
               for gl, gf in GATES for al, af, mf in AXES for leg in legs]
    run_family("B) 三連複（比較用・現行S3相当）", val, test, cells_b, settle_trio)


if __name__ == "__main__":
    main()
