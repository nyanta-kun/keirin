"""A域（7車・WT◎=システム◎一致）の2軸目（相手）構造選定スイープ。

exp_a_hitrate_sweep（帯全流し）の続き。「2軸目条件の見直しで 5R/日×ROI100% が
確保できるか」を検証する。券種は 二連単 軸→構造選定相手（1点/2点）と
三連複 2車軸流し（相方構造選定×目下限）。

正規プロトコル: 学習〜2025-03-31（lgbm_wt_val25）／検証 2025-04-01〜2026-03-31（選定）
／テスト 2026-04-01〜07-15（選定セルのみ1回評価）。
選定: 検証 ROI>=100 ∧ n>=1825（5R/日）で的中率最大。なければ ROI>=90、それもなければ参考表示。
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
N_MIN = 1825  # 5R/日 × 365日


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
        if len(f) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        rows_g = list(g.itertuples(index=False))
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        m1 = int(rows_g[0].frame_no)
        if wt_top is None or m1 != wt_top:
            continue  # A域 = 一致レースのみ
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        mkt = {fo_: i + 1 for i, fo_ in enumerate(sorted(board, key=lambda x: (-q[x], x)))}
        rows = [(int(r.frame_no), _iv(getattr(r, "line_group", None)),
                 _iv(getattr(r, "line_pos", None)),
                 r.style if isinstance(getattr(r, "style", None), str) else "",
                 float(getattr(r, "race_point", None) or -1.0)
                 if getattr(r, "race_point", None) == getattr(r, "race_point", None) else -1.0)
                for r in rows_g]
        races.append({
            "board": board, "ex": ex_bd.get(rk, {}), "trio": trio,
            "rows": rows, "m1": m1, "mkt": mkt,
            "order2": tuple(fno for _, fno in f[:2]),
            "top3": frozenset(fno for _, fno in f[:3]),
            "ex_pay": pm.get(rk, {}).get(("exacta", tuple(fno for _, fno in f[:2])), 0),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "mto": min(trio.values()) if trio else None,
        })
    return races


# ── 軸 ──
def ax_rival(r):
    """別L先頭・競走得点最上位（旧Aの軸）。"""
    lg1 = next(lg for f, lg, lp, st, rp in r["rows"] if f == r["m1"])
    cands = [(rp, f) for f, lg, lp, st, rp in r["rows"]
             if f != r["m1"] and lp == 1 and (lg1 is None or lg != lg1)]
    return max(cands)[1] if cands else None

def ax_m1(r):
    return r["m1"]

AXES = [("別L先頭得点", ax_rival), ("◎(m1)", ax_m1)]


# ── 2軸目（相手）候補 ──
def _row(r, fno):
    return next((t for t in r["rows"] if t[0] == fno), None)

def mate_m1(r, a):
    return r["m1"] if a != r["m1"] else None

def mate_model2(r, a):
    for f, *_ in r["rows"]:
        if f != a:
            return f
    return None

def mate_same_line(r, a):
    """軸と同ライン相方（先頭⇔番手優先→車番最小・脚質不問）。"""
    ar = _row(r, a)
    if ar is None or ar[1] is None:
        return None
    want = 1 if ar[2] == 2 else 2
    cands = sorted((f, lp) for f, lg, lp, st, rp in r["rows"]
                   if f != a and lg == ar[1])
    if not cands:
        return None
    return next((f for f, lp in cands if lp == want), cands[0][0])

def mate_same_line_nige(r, a):
    """軸と同ライン「逃」相方（S2/S3方式）。"""
    ar = _row(r, a)
    if ar is None or ar[1] is None:
        return None
    want = 1 if ar[2] == 2 else 2
    cands = sorted((f, lp) for f, lg, lp, st, rp in r["rows"]
                   if f != a and lg == ar[1] and st == "逃")
    if not cands:
        return None
    return next((f for f, lp in cands if lp == want), cands[0][0])

def mate_mkt2(r, a):
    for f in sorted(r["board"], key=lambda x: r["mkt"].get(x, 99)):
        if f != a:
            return f
    return None

def mate_dark(r, a):
    """市場4-7位∧モデル3位内（S2の穴条件・軸と別人）。"""
    for i, (f, *_ ) in enumerate(r["rows"][:3]):
        if f != a and 4 <= r["mkt"].get(f, 99) <= 7:
            return f
    return None

def mate_rival2(r, a):
    """別L先頭のうち得点2番目（軸=別L先頭得点1位のとき）。"""
    lg1 = next(lg for f, lg, lp, st, rp in r["rows"] if f == r["m1"])
    cands = sorted(((rp, f) for f, lg, lp, st, rp in r["rows"]
                    if f != r["m1"] and f != a and lp == 1 and (lg1 is None or lg != lg1)),
                   reverse=True)
    return cands[0][1] if cands else None

MATES = [("◎(m1)", mate_m1), ("モデル2位", mate_model2), ("同L相方", mate_same_line),
         ("同L逃相方", mate_same_line_nige), ("市場2位", mate_mkt2),
         ("穴(市場4-7×モデル3位内)", mate_dark), ("別L先頭2番手", mate_rival2)]

GATES = [
    ("なし", lambda r: True),
    ("ent>=1.84", lambda r: r["ent"] >= 1.84),
    ("ent<1.84", lambda r: r["ent"] < 1.84),
    ("gap12<0.05", lambda r: r["gap12"] < 0.05),
    ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
    ("mto>=4.3", lambda r: r["mto"] is not None and r["mto"] >= 4.3),
]


def settle_exacta(races, gf, ax, mates, band):
    """二連単 軸→構造選定相手（mates が返す1〜2車・band でオッズフィルタ）。"""
    lo, hi = band
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax(r)
        if a is None:
            continue
        bs = []
        for mf in mates:
            b = mf(r, a)
            if b is not None and b != a and b not in bs:
                bs.append(b)
        buy = []
        for b in bs:
            ov = r["ex"].get((a, b))
            if ov is not None and lo <= ov < hi:
                buy.append(b)
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["order2"][0] == a and r["order2"][1] in buy:
            hits += 1
            pay += r["ex_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def settle_trio(races, gf, ax, mate_fn, leg):
    """三連複 {軸, 相方} 2車軸流し（目オッズ>=leg のみ）。"""
    n = hits = bet = pay = 0
    for r in races:
        if not gf(r):
            continue
        a = ax(r)
        if a is None:
            continue
        b = mate_fn(r, a)
        if b is None or b == a:
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


def main():
    model = load_model(MODEL)
    val = collect(*VAL, model)
    test = collect(*TEST, model)
    print(f"A域（7車一致） 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    cells = []  # (hit%, roi, n, name, settle_fn, params)
    # 二連単: 相手1点 / 2点（構造選定）× 帯
    BANDS = [(0.0, 1e9), (2.0, 1e9), (3.0, 30.0), (5.0, 50.0)]
    for gl, gf in GATES:
        for al, af in AXES:
            for ml, mf in MATES:
                for band in BANDS:
                    n, h, roi = settle_exacta(val, gf, af, [mf], band)
                    if n >= 300:
                        bl = f"帯[{band[0]:.0f},{'inf' if band[1] > 1e8 else int(band[1])})"
                        cells.append((h / n * 100, roi, n,
                                      f"二単[{gl}]{al}→{ml}×{bl}",
                                      settle_exacta, (gf, af, [mf], band)))
            # 2点（代表ペア組合せ）
            for (m1l, m1f), (m2l, m2f) in (
                (("◎(m1)", mate_m1), ("同L相方", mate_same_line)),
                (("◎(m1)", mate_m1), ("市場2位", mate_mkt2)),
                (("モデル2位", mate_model2), ("同L相方", mate_same_line)),
                (("市場2位", mate_mkt2), ("穴(市場4-7×モデル3位内)", mate_dark)),
            ):
                for band in BANDS:
                    n, h, roi = settle_exacta(val, gf, af, [m1f, m2f], band)
                    if n >= 300:
                        bl = f"帯[{band[0]:.0f},{'inf' if band[1] > 1e8 else int(band[1])})"
                        cells.append((h / n * 100, roi, n,
                                      f"二単[{gl}]{al}→{{{m1l},{m2l}}}×{bl}",
                                      settle_exacta, (gf, af, [m1f, m2f], band)))
            # 三連複 2車軸流し
            for ml, mf in MATES:
                for leg in (0.0, 5.0, 10.0, 15.0):
                    n, h, roi = settle_trio(val, gf, af, mf, leg)
                    if n >= 300:
                        cells.append((h / n * 100, roi, n,
                                      f"三複[{gl}]{al}={ml}×目>={leg:.0f}",
                                      settle_trio, (gf, af, mf, leg)))

    print(f"総セル {len(cells)}（n>=300）", flush=True)
    for roi_min in (100.0, 90.0):
        frontier = sorted([c for c in cells if c[1] >= roi_min and c[2] >= N_MIN],
                          reverse=True)
        print(f"\n===== 検証 ROI>={roi_min:.0f} ∧ n>={N_MIN}（5R/日） 的中率フロンティア =====")
        if not frontier:
            print("  該当なし")
            continue
        for hit, roi, n, name, _, _ in frontier[:8]:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
        hit, roi, n, name, fn, params = frontier[0]
        tn, th, troi = fn(test, *params)
        print(f"  【選択】{name}（検証 的中{hit:.1f}%・ROI{roi:.1f}%・n={n}）")
        print(f"  【テスト】n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
        break
    else:
        # どちらの基準でも該当なし → 参考: n>=N_MIN の ROI 上位
        best = sorted([c for c in cells if c[2] >= N_MIN], key=lambda x: -x[1])[:8]
        print(f"\n（参考: n>={N_MIN} の検証ROI上位）")
        for hit, roi, n, name, _, _ in best:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")
        # 量を無視した場合の上限も表示
        best_any = sorted(cells, key=lambda x: -x[1])[:5]
        print("（参考: 量制約なし n>=300 の検証ROI上位）")
        for hit, roi, n, name, _, _ in best_any:
            print(f"    {name}: n={n} 的中={hit:.1f}% ROI={roi:.1f}%")


if __name__ == "__main__":
    main()
