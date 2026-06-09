"""7車以上: 軸信頼度による選別（誤差分解の続き）

分解で7+のミス主因は「r2(pred2位)落ち26.8%」と判明。広げてもROIは控除率に収束。
残る一手＝「広げる」でなく「軸が信頼できるレースを選別」する。
ただし軸信頼=堅い=低オッズの懸念があり、的中↑とオッズ↓のどちらが勝つかは実測で判定。
選別軸: gap23(pred2-3位差・大=r2がr3より明確に上=r2安定) / r2_prob絶対値 / 上位2信頼の複合。
買い目: 2軸流しr3-6(4点・相手抜けも安く拾う) と 1軸流しr1+top5(6点)。
pooled lgbm_wt・7+・最終オッズ上限値・train→test。★再現=TR/TE>100%&TE≥30R。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
import statistics
from collections import defaultdict
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection
from roi_robustness_wt import roi_summary

model = load_model("lgbm_wt")


def load_trio_board(race_keys):
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None:
                    board[rk][frozenset(int(x) for x in comb.split("="))] = od
    return board


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz >= 7].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks = df["race_key"].unique().tolist()
    board = load_trio_board(rks)
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 7:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        rows.append({"n": n, "fr": fr, "p": p,
                     "gap12": p[0] - p[1], "gap23": p[1] - p[2], "r2p": p[1],
                     "win": frozenset(int(x) for x in fin["frame_no"]),
                     "board": board.get(rk, {})})
    return rows


B26 = lambda fr: [frozenset((fr[0], fr[1], x)) for x in fr[2:6]]              # 2軸流しr3-6 4点
B1t5 = lambda fr: [frozenset((fr[0], a, b)) for a, b in itertools.combinations(fr[1:5], 2)]  # 1軸r1+top5 6点
BUYS = {"2軸r3-6(4点)": B26, "1軸r1+top5(6点)": B1t5}


def roi_for(rows, cond, buy):
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        combos = list({frozenset(c) for c in buy(r["fr"]) if len(set(c)) == 3})
        if not combos:
            continue
        bets.append(len(combos) * 100)
        o = r["board"].get(r["win"], 0.0)
        pays.append(o * 100 if r["win"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def qtiles(tr, key):
    v = sorted(r[key] for r in tr)
    return v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4]


def report(tr, te):
    g23_q = qtiles(tr, "gap23"); r2_q = qtiles(tr, "r2p"); g12_q = qtiles(tr, "gap12")
    CONDS = [
        ("ALL", lambda r: True),
        (f"gap23>=p75({g23_q[2]:.3f})軸2明確", lambda r, v=g23_q[2]: r["gap23"] >= v),
        (f"gap23>=中央({g23_q[1]:.3f})", lambda r, v=g23_q[1]: r["gap23"] >= v),
        (f"r2_prob>=p75({r2_q[2]:.3f})", lambda r, v=r2_q[2]: r["r2p"] >= v),
        (f"r2_prob>=中央({r2_q[1]:.3f})", lambda r, v=r2_q[1]: r["r2p"] >= v),
        (f"gap12>=p75({g12_q[2]:.3f})本命強", lambda r, v=g12_q[2]: r["gap12"] >= v),
        ("複合: gap23>=中央 & r2p>=中央", lambda r, a=g23_q[1], b=r2_q[1]: r["gap23"] >= a and r["r2p"] >= b),
        ("複合: gap23>=p75 & gap12>=中央", lambda r, a=g23_q[2], b=g12_q[1]: r["gap23"] >= a and r["gap12"] >= b),
    ]
    print(f"\n{'='*108}\n  7+ 軸信頼度選別 × 買い目（的中↑とオッズ↓の綱引きを実測）  TR {len(tr)}R / TE {len(te)}R（上限値）\n{'='*108}")
    for bn, buy in BUYS.items():
        print(f"\n  ■ 買い目: {bn}")
        print(f"    {'選別':<34}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}{'再現':>6}")
        for cn, cond in CONDS:
            s1, n1 = roi_for(tr, cond, buy); s2, n2 = roi_for(te, cond, buy)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"    {cn:<34}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                  f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ 軸信頼選別で的中は上がるはず。だが堅い=低オッズなら相殺。★再現が出れば7+に活路。")


if __name__ == "__main__":
    report(collect("2023-07-01", "2026-02-28"), collect("2026-03-01", "2026-06-08"))
