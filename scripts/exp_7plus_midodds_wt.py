"""7車以上限定: オッズ帯 × レース選別(top3_sum/gap12) のクロス検証

≤6車は運用モデル不変。本検証は7+セグメント限定。
中間オッズ集中(≤6で★再現 20-80倍)は7+で唯一未適用のレバー。レース選別と掛けて
再現黒字セル(TR/TE>100%・TE十分R)が出るか。モデルは7+でAUC最良の pooled lgbm_wt。
買い目=2軸流し全点(wide)を基準にオッズ帯で絞る。払戻/オッズ=wt_odds最終=上限値。
train(2023-07〜2026-02)→test(2026-03〜)。
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
    df = df[df["race_key"].isin(sz[sz >= 7].index)].copy()      # 7車以上
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
        bd = board.get(rk, {})
        if not bd:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        win = frozenset(int(x) for x in fin["frame_no"])
        win_odds = bd.get(win, 0.0)
        wide = [frozenset((fr[0], fr[1], x)) for x in fr[2:]]   # 2軸流し全点(n-2)
        rows.append({"gap12": p[0] - p[1], "top3_sum": p[0] + p[1] + p[2],
                     "win": win, "win_odds": win_odds, "board": bd, "wide": wide})
    return rows


def band_roi(rows, cond, lo, hi):
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        bd = r["board"]
        sub = [c for c in r["wide"] if (o := bd.get(c)) is not None and lo <= o < hi]
        if not sub:
            continue
        bets.append(len(sub) * 100)
        pays.append(r["win_odds"] * 100 if r["win"] in sub else 0.0)
    return roi_summary(pays, bets), len(pays)


def fmt(s, n):
    return f"{n:>5}R {s['roi']:>5.0%} 的中{s['hit_rate']:>4.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]"


def report(tr, te):
    med = statistics.median([r["top3_sum"] for r in tr])
    p25 = sorted(r["top3_sum"] for r in tr)[len(tr) // 4]
    CONDS = [
        ("ALL", lambda r: True),
        ("gap12<0.10", lambda r: r["gap12"] < 0.10),
        ("gap12<0.06", lambda r: r["gap12"] < 0.06),
        (f"top3_sum<中央({med:.2f})", lambda r, m=med: r["top3_sum"] < m),
        (f"top3_sum<=p25({p25:.2f})", lambda r, v=p25: r["top3_sum"] <= v),
    ]
    RANGES = [(0, 1e9), (5, 40), (10, 80), (20, 80), (20, 200), (40, 200), (10, 1e9), (20, 1e9), (40, 1e9)]
    RLAB = {(0, 1e9): "全帯", (10, 1e9): ">=10", (20, 1e9): ">=20", (40, 1e9): ">=40"}
    print(f"\n{'='*104}\n  7+ オッズ帯×レース選別クロス(2軸流し全点・wt最終オッズ上限値)  TR {len(tr)}R / TE {len(te)}R\n{'='*104}")
    for cn, cond in CONDS:
        print(f"\n  ▼ {cn}")
        print(f"    {'帯(倍)':<9}{'TRAIN':<34}{'TEST':<34}{'再現':>5}")
        for lo, hi in RANGES:
            s1, n1 = band_roi(tr, cond, lo, hi)
            s2, n2 = band_roi(te, cond, lo, hi)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            lab = RLAB.get((lo, hi), f"{lo}-{hi}")
            print(f"    {lab:<9}{fmt(s1,n1):<34}{fmt(s2,n2):<34}{flag:>5}")
    print("\n  ※ 7+で中間/中高オッズ集中が選別と掛けて★再現を出すか。出なければ7+撤退は確定（odds帯レバーでも不可）。")


if __name__ == "__main__":
    report(collect("2023-07-01", "2026-02-28"), collect("2026-03-01", "2026-06-08"))
