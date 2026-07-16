"""軸=モデル1位の2つの質問に答える分析（検証1年のみ・テスト不使用）。

Q1: 軸が1着になるレースは事前条件で絞りやすいか（1着率の可動域）
Q2: 軸3着内∧モデル2位が3着圏外、のレースに事前シグナルはあるか
7車・9車の両方で確認。
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
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

MODEL = "lgbm_wt_val25"
VAL = ("2025-04-01", "2026-03-31")


def _entropy(probs):
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum(max(p / total, 1e-9) * math.log(max(p / total, 1e-9)) for p in probs)


def collect(tf, tt, model, ne_want):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) in ne_want]
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
        r0, r1 = g.iloc[0], g.iloc[1]
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        mkt = {fo_: i + 1 for i, fo_ in enumerate(sorted(board, key=lambda x: (-q[x], x)))}
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        m1, m2 = int(r0["frame_no"]), int(r1["frame_no"])
        top3 = frozenset(fno for _, fno in f[:3])
        races.append({
            "ne": ne, "m1": m1, "m2": m2,
            "win1": f[0][1] == m1, "in3_1": m1 in top3, "in3_2": m2 in top3,
            "gap12": probs[0] - probs[1], "gap23": (probs[1] - probs[2]) * 100,
            "ent": _entropy(probs), "mto": min(trio.values()),
            "wt_match": wt_top is not None and wt_top == m1,
            "m1_mkt": mkt.get(m1, 99), "m2_mkt": mkt.get(m2, 99),
            "m1_lp": _iv(r0.get("line_pos")), "m1_ls": _iv(r0.get("line_size")),
            "m2_lp": _iv(r1.get("line_pos")), "m2_ls": _iv(r1.get("line_size")),
            "same12": (_iv(r0.get("line_group")) is not None
                       and _iv(r0.get("line_group")) == _iv(r1.get("line_group"))),
            "m1_style": r0.get("style") if isinstance(r0.get("style"), str) else "?",
        })
    return races


def table(races, label, cond_list, metric):
    print(f"\n-- {label} --")
    for cl, cf in cond_list:
        sel = [r for r in races if cf(r)]
        if len(sel) < 120:
            continue
        v = sum(1 for r in sel if metric(r)) / len(sel)
        print(f"  {cl:<28} n={len(sel):5d}  {v*100:5.1f}%")


def main():
    model = load_model(MODEL)
    for ne in (7, 9):
        races = collect(*VAL, model, {ne})
        print(f"\n===== {ne}車（検証 {len(races)}R） =====")
        base_w = sum(1 for r in races if r["win1"]) / len(races)
        base_3 = sum(1 for r in races if r["in3_1"]) / len(races)
        print(f"ベース: モデル1位 1着率 {base_w*100:.1f}% / 3着内率 {base_3*100:.1f}%")

        conds = [
            ("全体", lambda r: True),
            ("gap12>=0.10", lambda r: r["gap12"] >= 0.10),
            ("gap12>=0.15", lambda r: r["gap12"] >= 0.15),
            ("gap12>=0.20", lambda r: r["gap12"] >= 0.20),
            ("gap12>=0.15∧WT一致", lambda r: r["gap12"] >= 0.15 and r["wt_match"]),
            ("gap12>=0.15∧市場1位", lambda r: r["gap12"] >= 0.15 and r["m1_mkt"] == 1),
            ("gap12>=0.15∧ent低(下1/4)", lambda r: r["gap12"] >= 0.15 and r["ent"] < (1.78 if r["ne"] == 7 else 1.99)),
            ("gap12>=0.15∧mto<3", lambda r: r["gap12"] >= 0.15 and r["mto"] < 3.0),
            ("1位=ライン先頭", lambda r: r["m1_lp"] == 1 and (r["m1_ls"] or 1) > 1),
            ("1位=番手", lambda r: r["m1_lp"] == 2),
            ("1位=単騎", lambda r: (r["m1_ls"] or 0) == 1),
            ("1位=逃", lambda r: r["m1_style"] == "逃"),
            ("gap12>=0.15∧1位=番手", lambda r: r["gap12"] >= 0.15 and r["m1_lp"] == 2),
        ]
        table(races, "Q1: モデル1位の【1着率】", conds, lambda r: r["win1"])

        # Q2: 軸3着内 ∧ 2位圏外
        in3 = [r for r in races if r["in3_1"]]
        base_q2 = sum(1 for r in in3 if not r["in3_2"]) / len(in3)
        print(f"\nQ2ベース: 軸3着内のうちモデル2位が圏外 = {base_q2*100:.1f}% "
              f"(全レース比 {base_q2*base_3*100:.1f}%)")
        conds2 = [
            ("全体（軸3着内のみ）", lambda r: True),
            ("gap23<=0.5pt（2-3位僅差）", lambda r: r["gap23"] <= 0.5),
            ("gap23>=3pt（2位優位）", lambda r: r["gap23"] >= 3.0),
            ("2位の市場評価が低い(mkt>=4)", lambda r: r["m2_mkt"] >= 4),
            ("2位の市場評価が高い(mkt<=2)", lambda r: r["m2_mkt"] <= 2),
            ("1-2位同ライン", lambda r: r["same12"]),
            ("1-2位別ライン", lambda r: not r["same12"]),
            ("2位=単騎", lambda r: (r["m2_ls"] or 0) == 1),
            ("2位=ライン先頭", lambda r: r["m2_lp"] == 1 and (r["m2_ls"] or 1) > 1),
            ("2位=番手", lambda r: r["m2_lp"] == 2),
            ("2位=3番手+", lambda r: (r["m2_lp"] or 0) >= 3),
            ("ent高(上1/4)", lambda r: r["ent"] >= (1.86 if r["ne"] == 7 else 2.06)),
            ("2位mkt>=4∧別ライン", lambda r: r["m2_mkt"] >= 4 and not r["same12"]),
        ]
        table(in3, "Q2: 軸3着内レースでの【モデル2位圏外率】", conds2,
              lambda r: not r["in3_2"])


if __name__ == "__main__":
    main()
