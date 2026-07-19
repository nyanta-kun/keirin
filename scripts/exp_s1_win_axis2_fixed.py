"""S1再検討②: 1着固定(win軸)+2軸目固定(ライン or 競走得点)×相手流し 三連単（正規プロトコル）。

ユーザー要望: 1着固定・2軸目も1車に固定（相手2車を流すのはROI確保が困難だったため）。
2軸目の選出方法を2種類試す:
  L) ライン: 軸1と同ライン「逃」の相方（既存S2/S3と同じ _mate_same_line_nige 方式）
  R) 競走得点: 軸1を除いた残り車の中で race_point 最大の1車

3パターン（軸2の位置 × 相手流し）:
  P1) 軸1→軸2(2着固定)→相手流し（3着を残り全車で流す）
  P2) 軸1→相手流し(2着)→軸2(3着固定)
  P3) P1∧P2の総流し（軸2が2着でも3着でも的中扱い・点数は2倍）

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（条件選択）・
テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証 ROI>=95 ∧ n>=100 で的中率最大。
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


def train_win_model():
    print("学習データ読み込み...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=TRAIN_TO))
    df = df[df["finish_order"].notna()]
    X = prepare_X(df)
    win_y = (df["finish_order"] == 1).astype(int)
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)
    print("1着モデル学習...", flush=True)
    m.fit(X, win_y)
    return m


def _iv(v):
    return None if v is None or pd.isna(v) else int(v)


def _mate_same_line_nige(rows, axis_fno):
    """既存 strategy_wt.py と同じロジック: 同ライン「逃」の相方（line_pos相補優先）。"""
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


def collect(tf, tt, win_model, n_riders):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_win"] = win_model.predict_proba(X)[:, 1]
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
        axis1 = int(g_win.iloc[0]["frame_no"])
        win_probs = g_win["p_win"].tolist()
        win_gap12 = float(win_probs[0] - win_probs[1])
        rows = list(g_win.itertuples(index=False))

        axis2_line = _mate_same_line_nige(rows, axis1)

        remainder = g_win[g_win["frame_no"] != axis1]
        axis2_rp = None
        if len(remainder) > 0:
            axis2_rp = int(remainder.loc[remainder["race_point"].idxmax(), "frame_no"])

        order3 = tuple(fno for _, fno in f[:3])
        tri_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        races.append({
            "tri": tri, "board": board, "order3": order3, "tri_pay": tri_pay,
            "axis1": axis1, "axis2_line": axis2_line, "axis2_rp": axis2_rp,
            "win_gap12": win_gap12,
        })
    return races


def settle(races, axis2_key, pattern, gate_fn, leg):
    """pattern: 'p1'(2着固定) / 'p2'(3着固定) / 'p3'(両方総流し)"""
    n = hits = bet = pay = 0
    for r in races:
        if not gate_fn(r):
            continue
        a1 = r["axis1"]
        a2 = r[axis2_key]
        if a2 is None or a2 == a1:
            continue
        others = r["board"] - {a1, a2}
        buy = []
        if pattern in ("p1", "p3"):
            for x in others:
                combo = (a1, a2, x)
                if (r["tri"].get(combo) or 0) >= leg:
                    buy.append(combo)
        if pattern in ("p2", "p3"):
            for x in others:
                combo = (a1, x, a2)
                if (r["tri"].get(combo) or 0) >= leg:
                    buy.append(combo)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order3"] in buy:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


LEGS = (0.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
GAP12_GATES = [("なし", lambda r: True)] + [
    (f"win_gap12>={th:.2f}", (lambda r, th=th: r["win_gap12"] >= th))
    for th in (0.05, 0.10, 0.15, 0.20)
]


def run(n_riders, val, test, axis2_key, axis2_label, pattern, pattern_label):
    print(f"\n{'='*70}\n{n_riders}車 / 2軸目={axis2_label} / {pattern_label}\n{'='*70}", flush=True)
    results = []
    for gname, gf in GAP12_GATES:
        for leg in LEGS:
            n, h, roi = settle(val, axis2_key, pattern, gf, leg)
            results.append((gname, gf, leg, n, h, roi))
    survivors = [x for x in results if x[3] >= 100 and x[5] >= 95]
    if not survivors:
        best = sorted(results, key=lambda x: -x[5])[:5]
        print("  検証ROI>=95%∧n>=100 の条件なし。参考(検証ROI上位5):")
        for gname, gf, leg, n, h, roi in best:
            print(f"    {gname} 目>={leg:.0f}: n={n} 的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%")
        return
    survivors.sort(key=lambda x: -(x[4] / x[3]))
    print(f"  検証ROI>=95%∧n>=100: {len(survivors)}件生存。的中率上位5件をテスト評価:")
    for gname, gf, leg, n, h, roi in survivors[:5]:
        tn, th, troi = settle(test, axis2_key, pattern, gf, leg)
        print(f"    {gname} 目>={leg:.0f}: 検証(的中{h/n*100:.1f}%/ROI{roi:.1f}%/n={n}) "
              f"→ テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")


def main():
    win_model = train_win_model()
    AXIS2 = [("axis2_line", "ライン(同L逃相方)"), ("axis2_rp", "競走得点最大")]
    PATTERNS = [("p1", "P1:軸2=2着固定"), ("p2", "P2:軸2=3着固定"), ("p3", "P3:両方総流し")]
    for n_riders in (6, 7):
        print(f"\n\n########## データ構築: {n_riders}車 ##########", flush=True)
        val = collect(VAL_FROM, VAL_TO, win_model, n_riders)
        test = collect(TEST_FROM, TEST_TO, win_model, n_riders)
        print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)
        for axis2_key, axis2_label in AXIS2:
            for pattern, pattern_label in PATTERNS:
                run(n_riders, val, test, axis2_key, axis2_label, pattern, pattern_label)


if __name__ == "__main__":
    main()
