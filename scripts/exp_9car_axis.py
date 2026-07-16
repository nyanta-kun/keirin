"""9車立て: 2車軸（三連複流し・7点母体）戦略の検証。

学習: 〜2025-03-31（lgbm_wt_val25）／検証: 2025-04-01〜2026-03-31（選定はここだけ）
テスト: 2026-04-01〜2026-07-15（最終選定のみ1回）

①軸候補×レース条件の3着内率 → ②2軸目×目オッズ下限 → 選定→テスト。
ゲート閾値は検証期間の9車分布の分位点から取る（選定の一部＝検証内で完結）。
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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


def collect9(tf, tt, model):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        gr_map = dict(c.execute(
            "SELECT race_key, grade FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == 9]
        marks, fins = {}, {}
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, f, pmv, fo in c.execute(q, ch):
                marks.setdefault(rk, {})[int(f)] = pmv
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(f)))
        trio_bd = defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or not (0 < fv < 90000):
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio_bd[rk][parts] = fv
    pm = _load_payouts_wt(rks)

    def _iv(v):
        return None if v is None or pd.isna(v) else int(v)

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 9 or len(g) != 9:
            continue
        trio = trio_bd.get(rk, {})
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 9:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        rows_g = list(g.itertuples(index=False))
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        top3 = frozenset(fno for _, fno in f[:3])
        grade = str(gr_map.get(rk) or "")
        races.append({
            "trio": trio, "board": board,
            "ent": _entropy(probs), "mto": min(trio.values()),
            "gap12": probs[0] - probs[1],
            "mkt_rank": {fo_: i + 1 for i, fo_ in
                         enumerate(sorted(board, key=lambda x: (-q[x], x)))},
            "rows": rows_g, "top3": top3,
            "trio_pay": pm.get(rk, {}).get(("trio", top3), 0),
            "wt_top": wt_top, "m1": int(rows_g[0].frame_no),
            "grade": grade, "is_g": grade.startswith("G") or grade in ("GP",),
            "iv": _iv,
        })
    return races


def axis_m1(r):
    return r["m1"]

def axis_rival(r):
    iv = r["iv"]
    lg1 = iv(getattr(r["rows"][0], "line_group", None))
    rivals = []
    for row in r["rows"][1:]:
        if iv(getattr(row, "line_pos", None)) != 1:
            continue
        lg = iv(getattr(row, "line_group", None))
        if lg1 is not None and lg == lg1:
            continue
        rp = getattr(row, "race_point", None)
        rivals.append((float(rp) if rp is not None and rp == rp else -1.0,
                       int(row.frame_no)))
    return max(rivals)[1] if rivals else None

def axis_ana(r):
    """穴軸: 市場4-7位∧モデル3位内∧(単騎 or ライン先頭/番手)。"""
    iv = r["iv"]
    for row in r["rows"][:3]:
        f = int(row.frame_no)
        ls = iv(getattr(row, "line_size", None))
        lp = iv(getattr(row, "line_pos", None))
        if not (ls == 1 or lp in (1, 2)):
            continue
        if 4 <= r["mkt_rank"].get(f, 10) <= 7:
            return f
    return None


AXES = [("モデル1位", axis_m1), ("別L先頭", axis_rival), ("穴軸", axis_ana)]


def mate_model2(r, axis):
    for row in r["rows"]:
        f = int(row.frame_no)
        if f != axis:
            return f
    return None

def mate_same_line(r, axis):
    iv = r["iv"]
    ax = next((row for row in r["rows"] if int(row.frame_no) == axis), None)
    if ax is None:
        return None
    lg = iv(getattr(ax, "line_group", None))
    lp = iv(getattr(ax, "line_pos", None))
    if lg is None:
        return None
    want = 1 if lp == 2 else 2
    cands = [(int(row.frame_no), iv(getattr(row, "line_pos", None)))
             for row in r["rows"]
             if int(row.frame_no) != axis and iv(getattr(row, "line_group", None)) == lg]
    if not cands:
        return None
    cands.sort()
    return next((f for f, lp2 in cands if lp2 == want), cands[0][0])

def mate_same_line_nige(r, axis):
    iv = r["iv"]
    ax = next((row for row in r["rows"] if int(row.frame_no) == axis), None)
    if ax is None:
        return None
    lg = iv(getattr(ax, "line_group", None))
    if lg is None:
        return None
    lp = iv(getattr(ax, "line_pos", None))
    want = 1 if lp == 2 else 2
    cands = [(int(row.frame_no), iv(getattr(row, "line_pos", None)))
             for row in r["rows"]
             if int(row.frame_no) != axis
             and iv(getattr(row, "line_group", None)) == lg
             and (row.style if isinstance(getattr(row, "style", None), str) else "") == "逃"]
    if not cands:
        return None
    cands.sort()
    return next((f for f, lp2 in cands if lp2 == want), cands[0][0])

def mate_mkt2(r, axis):
    for f in sorted(r["board"], key=lambda x: r["mkt_rank"].get(x, 99)):
        if f != axis:
            return f
    return None

def mate_rival(r, axis):
    rv = axis_rival(r)
    return rv if rv is not None and rv != axis else None


MATES = [("モデル2位", mate_model2), ("同L相方", mate_same_line),
         ("同L逃相方", mate_same_line_nige), ("市場2位", mate_mkt2),
         ("別L先頭", mate_rival)]


def settle(races, gf, axis_fn, mate_fn, leg_th):
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = axis_fn(r)
        if a is None:
            continue
        b = mate_fn(r, a)
        if b is None or b == a:
            continue
        buy = [frozenset({a, b, t}) for t in r["board"] - {a, b}
               if (r["trio"].get(frozenset({a, b, t})) or 0) >= leg_th]
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
    val = collect9(*VAL, model)
    test = collect9(*TEST, model)
    ents = np.array([r["ent"] for r in val])
    mtos = np.array([r["mto"] for r in val])
    gaps = np.array([r["gap12"] for r in val])
    e_q1, e_q3 = np.quantile(ents, [0.25, 0.75])
    m_q3 = np.quantile(mtos, 0.75)
    g_q3 = np.quantile(gaps, 0.75)
    n_g = sum(1 for r in val if r["is_g"])
    print(f"検証 9車 {len(val)}R（G級 {n_g}R）/ テスト {len(test)}R")
    print(f"分位点: ent Q1={e_q1:.3f} Q3={e_q3:.3f} / mto Q3={m_q3:.1f} / gap12 Q3={g_q3:.3f}")

    GATES = [
        ("なし", lambda r: True),
        (f"ent>=Q3({e_q3:.2f})", lambda r: r["ent"] >= e_q3),
        (f"ent<Q1({e_q1:.2f})", lambda r: r["ent"] < e_q1),
        (f"mto>=Q3({m_q3:.0f})", lambda r: r["mto"] >= m_q3),
        (f"gap12>=Q3({g_q3:.3f})", lambda r: r["gap12"] >= g_q3),
        ("WT一致", lambda r: r["wt_top"] is not None and r["m1"] == r["wt_top"]),
        ("WT不一致", lambda r: r["wt_top"] is not None and r["m1"] != r["wt_top"]),
        ("G級", lambda r: r["is_g"]),
        ("F級", lambda r: not r["is_g"]),
        (f"gap12>=Q3∧WT一致", lambda r: r["gap12"] >= g_q3
         and r["wt_top"] is not None and r["m1"] == r["wt_top"]),
    ]

    print("\n== ①軸の3着内率×レース条件（検証1年） ==")
    step1 = {}
    for al, af in AXES:
        rows = []
        for gl, gf in GATES:
            sel = [(r, af(r)) for r in val if gf(r)]
            sel = [(r, a) for r, a in sel if a is not None]
            if len(sel) < 80:
                continue
            t3 = sum(1 for r, a in sel if a in r["top3"]) / len(sel)
            rows.append((t3, gl, gf))
            print(f"  [{al}] {gl:<22} n={len(sel):5d} 軸3着内={t3*100:5.1f}%")
        rows.sort(reverse=True)
        step1[al] = rows

    print("\n== ②2軸目×目オッズ下限（軸別・上位ゲート4つ） ==")
    results = []
    for al, af in AXES:
        for t3, gl, gf in step1.get(al, [])[:4]:
            for ml, mf in MATES:
                for leg in (10.0, 20.0, 30.0, 50.0):
                    n, h, roi = settle(val, gf, af, mf, leg)
                    if n >= 100 and 3 <= h / n * 100 <= 30:
                        results.append((roi, al, gl, ml, leg, n, h, gf, af, mf))
    results.sort(key=lambda x: -x[0])
    for roi, al, gl, ml, leg, n, h, *_ in results[:15]:
        print(f"  {al}×[{gl}]×{ml}×目>={leg:.0f}: n={n} 的中={h/n*100:.1f}% ROI={roi:.1f}%")

    if not results or results[0][0] < 100:
        print("\n検証でROI100%超なし → 9車に採用条件なし")
        return
    roi, al, gl, ml, leg, n, h, gf, af, mf = results[0]
    tn, th, troi = settle(test, gf, af, mf, leg)
    print(f"\n【選択】{al}×[{gl}]×{ml}×目>={leg:.0f}（検証 {roi:.1f}%・n={n}・的中{h/n*100:.1f}%）")
    print(f"【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for roi2, al2, gl2, ml2, leg2, n2, _, gf2, af2, mf2 in results[1:4]:
        tn2, _, troi2 = settle(test, gf2, af2, mf2, leg2)
        print(f"[参考] {al2}×[{gl2}]×{ml2}×目>={leg2:.0f}: 検証{roi2:.1f}% → テスト n={tn2} ROI={troi2:.1f}%")


if __name__ == "__main__":
    main()
