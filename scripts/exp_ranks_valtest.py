"""S2/S3/A 条件選択のやり直し（正規プロトコル・7車）。

学習: 〜2025-12-31（lgbm_wt_val26・落車込み）
検証: 2026-01-01〜2026-03-31 — 閾値グリッドの選択はここだけ
テスト: 2026-04-01〜2026-07-15 — 選択条件のみ最終確認

S2: ent_th × mto_th × leg_th（穴×同L逃相方・三連複流し）
S3: 同グリッド（◎不一致×システム◎軸・S2重複排除なしの純定義で選択）
A : ent_th × 二連単帯 [lo,hi)（◎一致×別L先頭軸）
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


def collect7(tf, tt, model):
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
        trio_bd, exa_bd = defaultdict(dict), defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','exacta') AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, bt, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or not (0 < fv < 9000):
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio" and len(parts) == 3:
                    trio_bd[rk][frozenset(parts)] = fv
                elif bt == "exacta" and len(parts) == 2:
                    exa_bd[rk][tuple(parts)] = fv
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
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        rows_g = list(g.itertuples(index=False))
        top3 = frozenset(fno for _, fno in f[:3])
        order2 = (f[0][1], f[1][1])
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        mkt = {fo_: i + 1 for i, fo_ in enumerate(sorted(board, key=lambda x: (-q[x], x)))}
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        races.append({
            "rk": rk, "trio": trio, "exa": exa_bd.get(rk, {}), "board": board,
            "ent": _entropy(probs), "mto": min(trio.values()),
            "mkt": mkt, "rows": rows_g, "top3": top3, "order2": order2,
            "trio_pay": pm.get(rk, {}).get(("trio", top3), 0),
            "exa_pay": pm.get(rk, {}).get(("exacta", order2), 0),
            "wt_top": wt_top, "m1": int(rows_g[0].frame_no),
            "iv": _iv,
        })
    return races


def _u_pair(r, iv):
    """S2の穴×同L逃相方ペア（モデル順位最小→車番最小）。"""
    eligible = []
    for ri, row in enumerate(r["rows"][:3], start=1):
        lg = iv(getattr(row, "line_group", None))
        ls = iv(getattr(row, "line_size", None))
        lp = iv(getattr(row, "line_pos", None))
        if not (ls == 1 or lp in (1, 2)) or lg is None:
            continue
        dark = int(row.frame_no)
        if not (4 <= r["mkt"].get(dark, 8) <= 7):
            continue
        for m in r["rows"]:
            mf = int(m.frame_no)
            mlg = iv(getattr(m, "line_group", None))
            mst = m.style if isinstance(getattr(m, "style", None), str) else ""
            if mf == dark or mlg is None or mlg != lg or mst != "逃":
                continue
            eligible.append((ri, dark, mf))
    if not eligible:
        return None
    eligible.sort()
    return eligible[0][1], eligible[0][2]


def _m_pair(r, iv):
    """S3のシステム◎×同L逃相方ペア。"""
    if r["wt_top"] is None or r["m1"] == r["wt_top"]:
        return None
    r1 = r["rows"][0]
    lg1 = iv(getattr(r1, "line_group", None))
    if lg1 is None:
        return None
    lp1 = iv(getattr(r1, "line_pos", None))
    want = 1 if lp1 == 2 else 2
    mates = []
    for row in r["rows"]:
        fno = int(row.frame_no)
        lg = iv(getattr(row, "line_group", None))
        st = row.style if isinstance(getattr(row, "style", None), str) else ""
        if fno == r["m1"] or lg is None or lg != lg1 or st != "逃":
            continue
        mates.append((fno, iv(getattr(row, "line_pos", None))))
    if not mates:
        return None
    mates.sort()
    mate = next((x for x, lp in mates if lp == want), mates[0][0])
    return r["m1"], mate


def eval_pair_trio(races, pair_fn, ent_th, mto_th, leg_th):
    n = hits = bet = pay = 0
    for r in races:
        if r["ent"] < ent_th or r["mto"] < mto_th:
            continue
        pr = pair_fn(r, r["iv"])
        if pr is None:
            continue
        a, b = pr
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


def eval_a(races, ent_th, lo, hi):
    n = hits = bet = pay = 0
    for r in races:
        if r["ent"] < ent_th:
            continue
        if r["wt_top"] is None or r["m1"] != r["wt_top"]:
            continue
        iv = r["iv"]
        r1 = r["rows"][0]
        lg1 = iv(getattr(r1, "line_group", None))
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
        if not rivals:
            continue
        _, axis = max(rivals)
        partners = [x for x in sorted(r["board"] - {axis})
                    if r["exa"].get((axis, x)) is not None
                    and lo <= r["exa"][(axis, x)] < hi]
        if not partners:
            continue
        n += 1
        bet += len(partners) * STAKE
        if r["order2"][0] == axis and r["order2"][1] in partners:
            hits += 1
            pay += r["exa_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def sweep(label, val_races, test_races, grid, ev):
    print(f"\n===== {label} =====")
    results = []
    for params in grid:
        n, h, roi = ev(val_races, *params)
        if n >= 25:
            results.append((roi, params, n, h))
            print(f"  {params}: 検証 n={n:4d} 的中={h/n*100:5.1f}% ROI={roi:6.1f}%")
    if not results:
        print("  検証で母数を満たす条件なし")
        return
    results.sort(reverse=True)
    roi, params, n, h = results[0]
    tn, th, troi = ev(test_races, *params)
    print(f"  【選択】{params}（検証ROI {roi:.1f}%・n={n}）")
    print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for roi2, params2, n2, _ in results[1:3]:
        tn2, _, troi2 = ev(test_races, *params2)
        print(f"  [参考] 検証次点 {params2}: 検証{roi2:.1f}% → テスト n={tn2} ROI={troi2:.1f}%")


def main():
    model = load_model(MODEL)
    val = collect7(*VAL, model)
    test = collect7(*TEST, model)
    print(f"検証 7車 {len(val)}R / テスト 7車 {len(test)}R")

    grid_um = [(e, m, l) for e in (1.75, 1.80, 1.84, 1.88)
               for m in (3.5, 4.3, 5.0) for l in (10.0, 15.0, 20.0)]
    sweep("S2（波乱ライン連れ込み・三連複）", val, test, grid_um,
          lambda rs, e, m, l: eval_pair_trio(rs, _u_pair, e, m, l))
    sweep("S3（◎不一致×システム◎・三連複）", val, test, grid_um,
          lambda rs, e, m, l: eval_pair_trio(rs, _m_pair, e, m, l))
    grid_a = [(e, lo, hi) for e in (1.75, 1.80, 1.84, 1.88)
              for lo, hi in ((3, 30), (5, 50), (10, 50), (5, 99999))]
    sweep("A（◎一致×別L先頭軸・二連単）", val, test, grid_a, eval_a)


if __name__ == "__main__":
    main()
