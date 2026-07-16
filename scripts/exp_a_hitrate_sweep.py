"""A域（7車・WT◎=システム◎一致）の的中率優先スイープ。

目的: 「1日約10Rの推奨量を確保しつつ的中率を最大化（ROIは90%以上許容）」という
A ランク本来の目的での推奨条件を探す。

正規プロトコル: 学習〜2025-03-31（lgbm_wt_val25）／検証 2025-04-01〜2026-03-31（選定）
／テスト 2026-04-01〜07-15（選定セルのみ1回評価）。
選定基準: 検証 ROI>=90 ∧ n>=閾値（2500≒8R/日・1000≒3R/日の2段）で【的中率最大】。
"""
import math
import re
import sys
from collections import defaultdict
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
        trio_bd, ex_bd = defaultdict(dict), defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','exacta') AND race_key IN (%s)"
                 % ",".join("?" * len(ch)))
            for rk, bt, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or not (0 < fv < 9000):
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→>]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio" and len(parts) == 3:
                    trio_bd[rk][frozenset(parts)] = fv
                elif bt == "exacta" and len(parts) == 2:
                    ex_bd[rk][tuple(parts)] = fv
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
        if len(f) < 2:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        rows_g = list(g.itertuples(index=False))
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        m1 = int(rows_g[0].frame_no)
        if wt_top is None or m1 != wt_top:
            continue  # A域 = 一致レースのみ
        model_rank = {int(r.frame_no): i + 1 for i, r in enumerate(rows_g)}
        # 別ライン先頭のうち競走得点最上位（現行Aの軸）
        lg1 = _iv(getattr(rows_g[0], "line_group", None))
        rivals = []
        for row in rows_g[1:]:
            if _iv(getattr(row, "line_pos", None)) != 1:
                continue
            lg = _iv(getattr(row, "line_group", None))
            if lg1 is not None and lg == lg1:
                continue
            rp = getattr(row, "race_point", None)
            rivals.append((float(rp) if rp is not None and rp == rp else -1.0,
                           int(row.frame_no)))
        axis_rp = max(rivals)[1] if rivals else None
        # 別ライン先頭のうちモデル順位最上位（軸バリアント）
        axis_mdl = None
        if rivals:
            axis_mdl = min((model_rank[f2], f2) for _, f2 in rivals)[1]
        races.append({
            "board": board, "ex": ex_bd.get(rk, {}),
            "order2": tuple(fno for _, fno in f[:2]),
            "ex_pay": pm.get(rk, {}).get(("exacta", tuple(fno for _, fno in f[:2])), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "m1": m1, "axis_rp": axis_rp, "axis_mdl": axis_mdl,
            "mto": min(trio.values()) if trio else None,
        })
    return races


AXES = [("別L先頭・得点最上位", lambda r: r["axis_rp"]),
        ("別L先頭・モデル最上位", lambda r: r["axis_mdl"]),
        ("システム◎(=WT◎)", lambda r: r["m1"])]

GATES = [
    ("なし", lambda r: True),
    ("ent>=1.84", lambda r: r["ent"] >= 1.84),
    ("ent>=1.88", lambda r: r["ent"] >= 1.88),
    ("ent<1.84", lambda r: r["ent"] < 1.84),
    ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
    ("gap12<0.05", lambda r: r["gap12"] < 0.05),
    ("mto>=4.3", lambda r: r["mto"] is not None and r["mto"] >= 4.3),
]

BANDS = [(2, 20), (3, 20), (3, 30), (5, 30), (5, 50), (3, 50), (2, 10)]


def settle(races, gf, axis_fn, lo, hi):
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = axis_fn(r)
        if a is None:
            continue
        buy = []
        for x in sorted(r["board"] - {a}):
            ov = r["ex"].get((a, x))
            if ov is not None and lo <= ov < hi:
                buy.append(x)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order2"][0] == a and r["order2"][1] in buy:
            hits += 1
            pay += r["ex_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def main():
    model = load_model(MODEL)
    val = collect(*VAL, model)
    test = collect(*TEST, model)
    print(f"A域（7車一致） 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    cells = []
    for gl, gf in GATES:
        for al, af in AXES:
            for lo, hi in BANDS:
                n, h, roi = settle(val, gf, af, lo, hi)
                if n >= 300:
                    cells.append((h / n * 100, roi, n, f"[{gl}]×{al}×帯[{lo},{hi})",
                                  (gf, af, lo, hi)))

    for n_min, label in ((2500, "n>=2500（≒8R/日）"), (1000, "n>=1000（≒3R/日）")):
        frontier = sorted([c for c in cells if c[1] >= 90 and c[2] >= n_min], reverse=True)
        print(f"\n===== 検証 ROI>=90 ∧ {label} の的中率フロンティア =====")
        if not frontier:
            best = sorted([c for c in cells if c[2] >= n_min], key=lambda x: -x[1])[:5]
            print("  （該当なし・参考: ROI上位）")
            for hit, roi, n, name, _ in best:
                print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
            continue
        for hit, roi, n, name, _ in frontier[:8]:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
        hit, roi, n, name, params = frontier[0]
        gf, af, lo, hi = params
        tn, th, troi = settle(test, gf, af, lo, hi)
        print(f"  【選択】{name}（検証 的中{hit:.1f}%・ROI{roi:.1f}%・n={n}）")
        print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
        for hit2, roi2, n2, name2, params2 in frontier[1:3]:
            gf2, af2, lo2, hi2 = params2
            tn2, th2, troi2 = settle(test, gf2, af2, lo2, hi2)
            print(f"  [参考] {name2}: 検証{hit2:.1f}%/{roi2:.1f}% → テスト n={tn2} "
                  f"的中={th2/tn2*100 if tn2 else 0:.1f}% ROI={troi2:.1f}%")


if __name__ == "__main__":
    main()
