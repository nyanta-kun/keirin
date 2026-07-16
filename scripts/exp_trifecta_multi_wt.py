"""三連単 1車軸マルチ×モデル上位相手（外部予想家スタイル）の検証。

構造: 軸1車を1着/2着/3着の全てに置き、残り2枠を相手3車で埋める18点
（=的中条件は三連複1車軸×相手3車の3点と同一・払戻のみ三連単）。
過去の買い目構造336セルスイープ（exp_bet_structures_sweep_wt）は
1着固定/穴頭固定/2車軸系のみで、マルチ（全着順許容）は未検証だった。

比較: 同一選定の三連複3点。軸=モデル1位/WT◎、相手=モデル2-4位。
正規プロトコル: 学習〜2025-03-31（lgbm_wt_val25）／検証 2025-04-01〜2026-03-31（選定）
／テスト 2026-04-01〜07-15（検証ROI>=95のセルのみ1回評価）。
"""
import math
import re
import sys
from collections import defaultdict
from itertools import permutations
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


def collect(tf, tt, model):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
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
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        frames = g["frame_no"].astype(int).tolist()
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        races.append({
            "trio": trio, "tri": tri_bd.get(rk, {}),
            "top3": frozenset(fno for _, fno in f[:3]),
            "order3": tuple(fno for _, fno in f[:3]),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "tri_pay": pm.get(rk, {}).get(("trifecta", tuple(fno for _, fno in f[:3])), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "mr": frames,  # モデル順位順 frame_no
            "wt_top": wt_top,
            "mto": min(trio.values()) if trio else None,
        })
    return races


GATES = [
    ("なし", lambda r: True),
    ("一致", lambda r: r["wt_top"] is not None and r["wt_top"] == r["mr"][0]),
    ("不一致", lambda r: r["wt_top"] is not None and r["wt_top"] != r["mr"][0]),
    ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
    ("gap12>=0.15", lambda r: r["gap12"] >= 0.15),
    ("ent>=1.84", lambda r: r["ent"] >= 1.84),
    ("ent<1.78", lambda r: r["ent"] < 1.78),
    ("mto>=4.3", lambda r: r["mto"] is not None and r["mto"] >= 4.3),
]

AXES = [("モデル1位", lambda r: r["mr"][0]), ("WT◎", lambda r: r["wt_top"])]

# 目オッズ帯（マルチ18点の各目に適用。Noneオッズは購入対象のまま=保守側）
BANDS = [("全目", 0.0, 1e9), ("目>=10", 10.0, 1e9), ("帯[10,100)", 10.0, 100.0),
         ("帯[20,200)", 20.0, 200.0)]


def settle_multi(races, gf, ax_fn, lo, hi):
    """三連単 軸マルチ×相手3車（モデル上位・軸除く）18点。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax_fn(r)
        if a is None:
            continue
        partners = [f for f in r["mr"][:4] if f != a][:3]
        if len(partners) < 3:
            continue
        buy = set()
        for x, y in permutations(partners, 2):
            for combo in ((a, x, y), (x, a, y), (x, y, a)):
                ov = r["tri"].get(combo)
                if ov is None or lo <= ov < hi:
                    buy.add(combo)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order3"] in buy:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def settle_trio_eq(races, gf, ax_fn, lo, hi):
    """同一選定の三連複3点（軸+相手2車の組合せ・帯は trio オッズに適用）。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax_fn(r)
        if a is None:
            continue
        partners = [f for f in r["mr"][:4] if f != a][:3]
        if len(partners) < 3:
            continue
        buy = []
        for i in range(3):
            for j in range(i + 1, 3):
                key = frozenset({a, partners[i], partners[j]})
                ov = r["trio"].get(key)
                if ov is None or lo <= ov < hi:
                    buy.append(key)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["top3"] in buy:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def main():
    model = load_model(MODEL)
    val = collect(*VAL, model)
    test = collect(*TEST, model)
    print(f"7車 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    cells = []
    for label, fn in (("マルチ18点", settle_multi), ("三複3点", settle_trio_eq)):
        for gl, gf in GATES:
            for al, af in AXES:
                for bl, lo, hi in BANDS:
                    n, h, roi = fn(val, gf, af, lo, hi)
                    if n >= 300:
                        cells.append((roi, h / n * 100, n,
                                      f"{label}[{gl}]軸={al}×{bl}", fn, (gf, af, lo, hi)))

    cells.sort(reverse=True)
    print("\n===== 検証ROI上位20（n>=300） =====")
    for roi, hit, n, name, _, _ in cells[:20]:
        print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")

    passed = [c for c in cells if c[0] >= 95]
    if not passed:
        print("\n検証ROI>=95のセルなし → テスト評価なし（不採用）")
        return
    print("\n===== 検証ROI>=95 → テスト1回評価 =====")
    for roi, hit, n, name, fn, params in passed[:5]:
        tn, th, troi = fn(test, *params)
        print(f"    {name}: 検証{hit:.1f}%/{roi:.1f}%(n={n}) → "
              f"テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")


if __name__ == "__main__":
    main()
