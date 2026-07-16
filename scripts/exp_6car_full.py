"""S1再検討: 6車の三連複・三連単 全域スイープ。

学習: 〜2025-03-31（lgbm_wt_val25）／検証: 2025-04-01〜2026-03-31（選定はここだけ）
テスト: 2026-04-01〜2026-07-15（最終選定のみ1回）

レバー: 買い目形状（モデル順位サブセット）× 目オッズ帯 × レースゲート
低配当対策 = 目オッズ帯フィルタ（安い目を買わない）を全形状に適用
"""
import math
import re
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

STAKE = 100
MODEL = "lgbm_wt_val25"
VAL = ("2025-04-01", "2026-03-31")
TEST = ("2026-04-01", "2026-07-15")


def _entropy(probs):
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum(max(p / total, 1e-9) * math.log(max(p / total, 1e-9)) for p in probs)


def collect6(tf, tt, model):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rt_map = dict(c.execute(
            "SELECT race_key, race_type FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == 6]
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
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)" % ",".join("?" * len(ch)))
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

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 6 or len(g) != 6:
            continue
        trio = trio_bd.get(rk, {})
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 6:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        m = g["frame_no"].astype(int).tolist()  # モデル順位順
        order3 = tuple(fno for _, fno in f[:3])
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        races.append({
            "m": m, "order3": order3, "top3": frozenset(order3),
            "trio": trio, "tri": tri_bd.get(rk, {}),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(order3)), 0),
            "tri_pay": pm.get(rk, {}).get(("trifecta", order3), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "mto": min(trio.values()),
            "wt_match": wt_top is not None and wt_top == m[0],
            "girls": "ガールズ" in str(rt_map.get(rk) or ""),
        })
    return races


# ── 買い目形状（kind, combo集合を返す） ────────────────────────────────────────
def S_tri_all(m):        return "tri", set(permutations(m, 3))
def S_tri_m1_1st(m):     return "tri", {(m[0], b, c) for b in m[1:] for c in m[1:] if b != c}
def S_tri_m1m2_12(m):    return "tri", {(m[0], m[1], c) for c in m[2:]}
def S_tri_m1_2nd(m):     return "tri", {(b, m[0], c) for b in m[1:] for c in m[1:] if b != c}
def S_tri_m12_12_all(m): return "tri", {(a, b, c) for a in m[:2] for b in m[:2] if a != b for c in m[2:]}
def S_trio_all(m):       return "trio", {frozenset(x) for x in combinations(m, 3)}
def S_trio_m1(m):        return "trio", {frozenset({m[0], *x}) for x in combinations(m[1:], 2)}
def S_trio_m1m2(m):      return "trio", {frozenset({m[0], m[1], c}) for c in m[2:]}
def S_trio_box1234(m):   return "trio", {frozenset(x) for x in combinations(m[:4], 3)}

SHAPES = [
    ("三単全120", S_tri_all), ("三単m1_1着F", S_tri_m1_1st),
    ("三単m1m2_12F", S_tri_m1m2_12), ("三単m1_2着F", S_tri_m1_2nd),
    ("三単m1m2裏表→他", S_tri_m12_12_all),
    ("三複全20", S_trio_all), ("三複m1含む", S_trio_m1),
    ("三複m1m2軸", S_trio_m1m2), ("三複box1-4", S_trio_box1234),
]
BANDS_TRI = [(0, 1e9), (10, 1e9), (10, 50), (20, 100), (30, 300), (50, 1e9)]
BANDS_TRIO = [(0, 1e9), (5, 1e9), (7, 1e9), (10, 1e9), (5, 20), (10, 50)]
GATES = [
    ("なし", lambda r: True),
    ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
    ("gap12<0.06", lambda r: r["gap12"] < 0.06),
    ("mto>=3", lambda r: r["mto"] >= 3.0),
    ("mto>=5", lambda r: r["mto"] >= 5.0),
    ("ent>=1.6", lambda r: r["ent"] >= 1.6),
    ("WT一致", lambda r: r["wt_match"]),
    ("WT不一致", lambda r: not r["wt_match"]),
    ("ガールズ", lambda r: r["girls"]),
    ("非ガールズ", lambda r: not r["girls"]),
]


def settle(races, gf, shape_fn, lo, hi):
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        kind, combos = shape_fn(r["m"])
        book = r["tri"] if kind == "tri" else r["trio"]
        buy = [c for c in combos if lo <= (book.get(c) or 0) < hi]
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        key = r["order3"] if kind == "tri" else r["top3"]
        if key in buy:
            hits += 1
            pay += (r["tri_pay"] if kind == "tri" else r["trio_pay"]) * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def main():
    model = load_model(MODEL)
    val = collect6(*VAL, model)
    test = collect6(*TEST, model)
    n_girls = sum(1 for r in val if r["girls"])
    print(f"検証 6車 {len(val)}R（ガールズ {n_girls}R）/ テスト {len(test)}R")

    results = []
    for gl, gf in GATES:
        for sl, sf in SHAPES:
            kind = sf([1, 2, 3, 4, 5, 6])[0]
            bands = BANDS_TRI if kind == "tri" else BANDS_TRIO
            for lo, hi in bands:
                n, h, roi = settle(val, gf, sf, lo, hi)
                if n >= 150 and 3 <= h / n * 100 <= 35:
                    results.append((roi, gl, sl, lo, hi, n, h))
    results.sort(reverse=True)
    print("\n== 検証上位20（n>=150・的中3-35%） ==")
    for roi, gl, sl, lo, hi, n, h in results[:20]:
        hi_s = "inf" if hi > 1e8 else f"{hi:.0f}"
        print(f"  [{gl}] {sl} 帯[{lo:.0f},{hi_s}): n={n} 的中={h/n*100:.1f}% ROI={roi:.1f}%")

    if not results or results[0][0] < 100:
        print("\n検証でROI100%超なし → 6車に採用条件なし")
        if results:
            roi, gl, sl, lo, hi, n, h = results[0]
            tn, th, troi = settle(test, dict(GATES)[gl], dict(SHAPES)[sl], lo, hi)
            print(f"[参考] 検証ベスト [{gl}]{sl}帯[{lo:.0f},{hi if hi<1e8 else 'inf'}): "
                  f"検証{roi:.1f}% → テスト n={tn} ROI={troi:.1f}%")
        return
    roi, gl, sl, lo, hi, n, h = results[0]
    tn, th, troi = settle(test, dict(GATES)[gl], dict(SHAPES)[sl], lo, hi)
    hi_s = "inf" if hi > 1e8 else f"{hi:.0f}"
    print(f"\n【選択】[{gl}] {sl} 帯[{lo:.0f},{hi_s})（検証 {roi:.1f}%・n={n}・的中{h/n*100:.1f}%）")
    print(f"【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for roi2, gl2, sl2, lo2, hi2, n2, _ in results[1:4]:
        tn2, _, troi2 = settle(test, dict(GATES)[gl2], dict(SHAPES)[sl2], lo2, hi2)
        hi2_s = "inf" if hi2 > 1e8 else f"{hi2:.0f}"
        print(f"[参考] [{gl2}]{sl2}帯[{lo2:.0f},{hi2_s}): 検証{roi2:.1f}% → テスト n={tn2} ROI={troi2:.1f}%")


if __name__ == "__main__":
    main()
