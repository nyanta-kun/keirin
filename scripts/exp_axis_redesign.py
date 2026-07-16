"""S1/S3/A 再設計: ①軸の3着内レース条件選定 → ②2軸目選定 → 三連複2車軸流し。

学習: 〜2025-03-31（lgbm_wt_val25）
検証: 2025-04-01〜2026-03-31（1年・全ての選定はここだけ）
テスト: 2026-04-01〜2026-07-15（各ランク最終選定のみ1回評価）

ドメイン（非重複維持）:
  S1: 6車全レース・軸=モデル1位
  S3: 7車 ∧ WT◎≠モデル1位（不一致）・軸=システム◎（モデル1位）
  A : 7車 ∧ WT◎=モデル1位（一致）・軸=別ライン先頭・競走得点最上位
（S2=波乱×穴×同L逃は現行のまま採用済み・本検討の対象外）
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


def collect(tf, tt, model, ne_want):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
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
                if fv is None or not (0 < fv < 9000):
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
        ne = ne_map.get(rk)
        if ne not in ne_want or len(g) != ne:
            continue
        trio = trio_bd.get(rk, {})
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != ne:
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
        mkt_sorted = sorted(board, key=lambda x: (-q[x], x))
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        top3 = frozenset(fno for _, fno in f[:3])
        races.append({
            "rk": rk, "ne": ne, "trio": trio, "board": board,
            "ent": _entropy(probs), "mto": min(trio.values()),
            "gap12": probs[0] - probs[1],
            "mkt_rank": {fo_: i + 1 for i, fo_ in enumerate(mkt_sorted)},
            "mkt2": next((x for x in mkt_sorted if x != int(rows_g[0].frame_no)), None),
            "rows": rows_g, "top3": top3,
            "trio_pay": pm.get(rk, {}).get(("trio", top3), 0),
            "wt_top": wt_top, "m1": int(rows_g[0].frame_no),
            "iv": _iv,
        })
    return races


# ── 軸定義 ──────────────────────────────────────────────────────────────────
def axis_m1(r):
    return r["m1"]

def axis_rival(r):
    """別ライン先頭・競走得点最上位（Aの軸）。"""
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


# ── 2軸目定義（axis に対して） ────────────────────────────────────────────────
def mate_model2(r, axis):
    for row in r["rows"]:
        f = int(row.frame_no)
        if f != axis:
            return f
    return None

def mate_same_line(r, axis):
    """同ライン相方（先頭⇔番手・脚質不問）。"""
    iv = r["iv"]
    ax_row = next((row for row in r["rows"] if int(row.frame_no) == axis), None)
    if ax_row is None:
        return None
    lg = iv(getattr(ax_row, "line_group", None))
    lp = iv(getattr(ax_row, "line_pos", None))
    if lg is None:
        return None
    want = 1 if lp == 2 else 2
    cands = []
    for row in r["rows"]:
        f = int(row.frame_no)
        if f == axis or iv(getattr(row, "line_group", None)) != lg:
            continue
        cands.append((f, iv(getattr(row, "line_pos", None))))
    if not cands:
        return None
    cands.sort()
    return next((f for f, lp2 in cands if lp2 == want), cands[0][0])

def mate_same_line_nige(r, axis):
    """同ライン「逃」相方（S2/S3現行方式）。"""
    iv = r["iv"]
    ax_row = next((row for row in r["rows"] if int(row.frame_no) == axis), None)
    if ax_row is None:
        return None
    lg = iv(getattr(ax_row, "line_group", None))
    if lg is None:
        return None
    lp = iv(getattr(ax_row, "line_pos", None))
    want = 1 if lp == 2 else 2
    cands = []
    for row in r["rows"]:
        f = int(row.frame_no)
        st = row.style if isinstance(getattr(row, "style", None), str) else ""
        if f == axis or iv(getattr(row, "line_group", None)) != lg or st != "逃":
            continue
        cands.append((f, iv(getattr(row, "line_pos", None))))
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

# ── レースゲート ──────────────────────────────────────────────────────────────
GATES_7 = [
    ("なし",              lambda r: True),
    ("ent<1.7",           lambda r: r["ent"] < 1.7),
    ("ent<1.84",          lambda r: r["ent"] < 1.84),
    ("ent>=1.84",         lambda r: r["ent"] >= 1.84),
    ("ent>=1.84∧mto>=4.3", lambda r: r["ent"] >= 1.84 and r["mto"] >= 4.3),
    ("ent>=1.84∧mto>=5",  lambda r: r["ent"] >= 1.84 and r["mto"] >= 5.0),
    ("mto>=5",            lambda r: r["mto"] >= 5.0),
    ("mto<3",             lambda r: r["mto"] < 3.0),
    ("gap12>=0.10",       lambda r: r["gap12"] >= 0.10),
    ("gap12<0.05",        lambda r: r["gap12"] < 0.05),
]
GATES_6 = [
    ("なし",         lambda r: True),
    ("gap12>=0.06",  lambda r: r["gap12"] >= 0.06),
    ("gap12>=0.10",  lambda r: r["gap12"] >= 0.10),
    ("mto>=3",       lambda r: r["mto"] >= 3.0),
    ("mto>=5",       lambda r: r["mto"] >= 5.0),
    ("gap12>=0.10∧mto>=3", lambda r: r["gap12"] >= 0.10 and r["mto"] >= 3.0),
    ("ent>=1.6",     lambda r: r["ent"] >= 1.6),
]


def step1(label, races, axis_fn, gates):
    print(f"\n== {label}: ①軸の3着内率×レース条件（検証1年） ==")
    out = []
    for gl, gf in gates:
        sel = [r for r in races if gf(r)]
        ax = [(r, axis_fn(r)) for r in sel]
        ax = [(r, a) for r, a in ax if a is not None]
        if len(ax) < 100:
            continue
        t3 = sum(1 for r, a in ax if a in r["top3"]) / len(ax)
        print(f"  {gl:<22} n={len(ax):5d}  軸3着内率={t3*100:5.1f}%")
        out.append((t3, gl, gf))
    return out


def settle_pair(races, gf, axis_fn, mate_fn, leg_th):
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
        combos = [frozenset({a, b, t}) for t in r["board"] - {a, b}
                  if (r["trio"].get(frozenset({a, b, t})) or 0) >= leg_th]
        if not combos:
            continue
        n += 1
        bet += len(combos) * STAKE
        if r["top3"] in combos:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def step2(label, val, test, axis_fn, gate_list, legs):
    print(f"\n== {label}: ②2軸目×目オッズ下限（検証1年・軸3着内率上位ゲート） ==")
    results = []
    for t3, gl, gf in gate_list:
        for ml, mf in MATES:
            for leg in legs:
                n, h, roi = settle_pair(val, gf, axis_fn, mf, leg)
                if n >= 100 and 5 <= h / n * 100 <= 30:
                    results.append((roi, gl, ml, leg, n, h))
    results.sort(reverse=True)
    for roi, gl, ml, leg, n, h in results[:8]:
        print(f"  [{gl}] × {ml} × 目>={leg:.0f}: n={n} 的中={h/n*100:.1f}% ROI={roi:.1f}%")
    if not results:
        print("  基準を満たす組合せなし")
        return
    roi, gl, ml, leg, n, h = results[0]
    gf = dict((g, f) for _, g, f in gate_list)[gl]
    mf = dict(MATES)[ml]
    tn, th, troi = settle_pair(test, gf, axis_fn, mf, leg)
    print(f"  【選択】[{gl}] × {ml} × 目>={leg:.0f}（検証 {roi:.1f}%・n={n}）")
    print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for roi2, gl2, ml2, leg2, n2, _ in results[1:3]:
        gf2 = dict((g, f) for _, g, f in gate_list)[gl2]
        mf2 = dict(MATES)[ml2]
        tn2, _, troi2 = settle_pair(test, gf2, axis_fn, mf2, leg2)
        print(f"  [参考] [{gl2}]×{ml2}×目>={leg2:.0f}: 検証{roi2:.1f}% → テスト n={tn2} ROI={troi2:.1f}%")


def main():
    model = load_model(MODEL)
    val7 = collect(*VAL, model, {7})
    test7 = collect(*TEST, model, {7})
    val6 = collect(*VAL, model, {6})
    test6 = collect(*TEST, model, {6})
    print(f"検証: 7車 {len(val7)}R / 6車 {len(val6)}R  テスト: 7車 {len(test7)}R / 6車 {len(val6) and len(test6)}R")

    # S1: 6車・軸=モデル1位
    g1 = step1("S1(6車・軸=モデル1位)", val6, axis_m1, GATES_6)
    g1.sort(reverse=True)
    step2("S1(6車)", val6, test6, axis_m1, g1[:4], legs=[0.0, 5.0, 7.0, 10.0])

    # S3: 7車不一致・軸=システム◎
    val_mm = [r for r in val7 if r["wt_top"] is not None and r["m1"] != r["wt_top"]]
    test_mm = [r for r in test7 if r["wt_top"] is not None and r["m1"] != r["wt_top"]]
    g3 = step1(f"S3(7車不一致 {len(val_mm)}R・軸=システム◎)", val_mm, axis_m1, GATES_7)
    g3.sort(reverse=True)
    step2("S3(不一致)", val_mm, test_mm, axis_m1, g3[:4], legs=[10.0, 15.0, 20.0])

    # A: 7車一致・軸=別L先頭得点最上位
    val_ma = [r for r in val7 if r["wt_top"] is not None and r["m1"] == r["wt_top"]]
    test_ma = [r for r in test7 if r["wt_top"] is not None and r["m1"] == r["wt_top"]]
    ga = step1(f"A(7車一致 {len(val_ma)}R・軸=別L先頭)", val_ma, axis_rival, GATES_7)
    ga.sort(reverse=True)
    step2("A(一致)", val_ma, test_ma, axis_rival, ga[:4], legs=[10.0, 15.0, 20.0])


if __name__ == "__main__":
    main()
