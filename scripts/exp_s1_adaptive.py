"""新S1候補: 適応型2車軸トリオ / m1 1着固定三連単（7車）。

学習: 〜2025-03-31（lgbm_wt_val25）／検証: 2025-04-01〜2026-03-31（選定）
テスト: 2026-04-01〜2026-07-15（各ファミリー選定1条件のみ）
選定基準: 検証ROI>=95% ∧ n>=300 の中で【的中率最大】（ユーザー基準: ROI90%許容・的中率重視）
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
ENT_LOW = 1.78  # 7車 ent 下位1/4 近似（検証分布Q1）


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
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        mkt = {fo_: i + 1 for i, fo_ in enumerate(sorted(board, key=lambda x: (-q[x], x)))}
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        m1, m2 = int(rows_g[0].frame_no), int(rows_g[1].frame_no)
        lg = {int(r.frame_no): _iv(getattr(r, "line_group", None)) for r in rows_g}
        lp = {int(r.frame_no): _iv(getattr(r, "line_pos", None)) for r in rows_g}
        # m1 の同ライン相方（先頭⇔番手・車番昇順フォールバック）
        mate = None
        if lg[m1] is not None:
            want = 1 if lp[m1] == 2 else 2
            cands = sorted((fno for fno in board
                            if fno != m1 and lg.get(fno) == lg[m1]))
            mate = next((f2 for f2 in cands if lp.get(f2) == want),
                        cands[0] if cands else None)
        races.append({
            "board": board, "trio": trio, "tri": tri_bd.get(rk, {}),
            "top3": frozenset(fno for _, fno in f[:3]),
            "order3": tuple(fno for _, fno in f[:3]),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "tri_pay": pm.get(rk, {}).get(("trifecta", tuple(fno for _, fno in f[:3])), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "m1": m1, "m2": m2, "m3": int(rows_g[2].frame_no),
            "m2_mkt": mkt.get(m2, 99),
            "same12": lg[m1] is not None and lg[m1] == lg.get(m2),
            "mate": mate,
            "wt_match": wt_top is not None and wt_top == m1,
            "mismatch": wt_top is not None and wt_top != m1,
        })
    return races


# ── 相方ルール ──
def R_m2(r):
    return r["m2"]

def R_mate(r):
    return r["mate"]

def R_adapt_mkt(r):
    """同ライン→m2 / 別ライン∧m2市場4位以下→同L相方 / それ以外→m2"""
    if r["same12"]:
        return r["m2"]
    if r["m2_mkt"] >= 4 and r["mate"] is not None:
        return r["mate"]
    return r["m2"]

def R_adapt_line(r):
    """同ライン→m2 / 別ライン→同L相方（無ければm2）"""
    if r["same12"]:
        return r["m2"]
    return r["mate"] if r["mate"] is not None else r["m2"]

RULES = [("常にm2", R_m2), ("常に同L相方", R_mate),
         ("適応(市場乖離)", R_adapt_mkt), ("適応(ライン)", R_adapt_line)]

# S3新定義（不一致∧gap12>=0.10）との重複を除外するゲートを含む
GATES = [
    ("なし", lambda r: True),
    ("非S3域", lambda r: not (r["mismatch"] and r["gap12"] >= 0.10)),
    ("WT一致", lambda r: r["wt_match"]),
    ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
    ("gap12>=0.15", lambda r: r["gap12"] >= 0.15),
    ("gap12>=0.15∧一致", lambda r: r["gap12"] >= 0.15 and r["wt_match"]),
    ("gap12>=0.15∧ent低", lambda r: r["gap12"] >= 0.15 and r["ent"] < ENT_LOW),
]


def settle_trio(races, gf, rule, leg):
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        b = rule(r)
        if b is None or b == r["m1"]:
            continue
        buy = [frozenset({r["m1"], b, t}) for t in r["board"] - {r["m1"], b}
               if (r["trio"].get(frozenset({r["m1"], b, t})) or 0) >= leg]
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["top3"] in buy:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def settle_tri(races, gf, second_fn, leg_lo, leg_hi):
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        seconds = second_fn(r)
        seconds = [s for s in seconds if s is not None and s != r["m1"]]
        if not seconds:
            continue
        buy = set()
        for s in set(seconds):
            for t in r["board"] - {r["m1"], s}:
                ov = r["tri"].get((r["m1"], s, t)) or 0
                if leg_lo <= ov < leg_hi:
                    buy.add((r["m1"], s, t))
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order3"] in buy:
            hits += 1
            pay += r["tri_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


SECONDS = [
    ("2着=m2", lambda r: [r["m2"]]),
    ("2着=適応(市場乖離)", lambda r: [R_adapt_mkt(r)]),
    ("2着={m2,m3}", lambda r: [r["m2"], r["m3"]]),
    ("2着={m2,同L相方}", lambda r: [r["m2"], r["mate"]]),
]


def run_family(label, val, test, cells, settle_fn):
    print(f"\n===== {label} =====")
    results = []
    for params in cells:
        n, h, roi = settle_fn(val, *params[1:])
        if n >= 300:
            results.append((h / n * 100, roi, params[0], params[1:], n, h))
    # 表示: ROI>=95 のフロンティア（的中率順）
    frontier = sorted([x for x in results if x[1] >= 95], reverse=True)
    print("  検証 ROI>=95% の的中率フロンティア:")
    for hit, roi, name, _, n, h in frontier[:10]:
        print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
    if not frontier:
        best = sorted(results, key=lambda x: -x[1])[:5]
        print("  （ROI>=95%なし・参考: 検証ROI上位）")
        for hit, roi, name, _, n, h in best:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
        return
    hit, roi, name, params, n, h = frontier[0]
    tn, th, troi = settle_fn(test, *params)
    print(f"  【選択（的中率最大 s.t. ROI>=95）】{name}（検証 的中{hit:.1f}%・ROI{roi:.1f}%・n={n}）")
    print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    for hit2, roi2, name2, params2, n2, _ in frontier[1:3]:
        tn2, th2, troi2 = settle_fn(test, *params2)
        print(f"  [参考] {name2}: 検証{hit2:.1f}%/{roi2:.1f}% → テスト n={tn2} "
              f"的中={th2/tn2*100 if tn2 else 0:.1f}% ROI={troi2:.1f}%")


def main():
    model = load_model(MODEL)
    val = collect(*VAL, model)
    test = collect(*TEST, model)
    print(f"検証 7車 {len(val)}R / テスト {len(test)}R")

    cells1 = [(f"[{gl}]×{rl}×目>={leg:.0f}", gf, rf, leg)
              for gl, gf in GATES for rl, rf in RULES
              for leg in (0.0, 5.0, 7.0, 10.0, 15.0)]
    run_family("案1: 適応型2車軸トリオ", val, test, cells1, settle_trio)

    cells2 = [(f"[{gl}]×{sl}×帯[{lo:.0f},{'inf' if hi>1e8 else int(hi)})", gf, sf, lo, hi)
              for gl, gf in GATES for sl, sf in SECONDS
              for lo, hi in ((0, 1e9), (10, 1e9), (15, 1e9), (10, 100), (20, 200))]
    run_family("案2: m1 1着固定三連単", val, test, cells2, settle_tri)


if __name__ == "__main__":
    main()
