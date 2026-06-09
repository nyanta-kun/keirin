"""7車以上: 本命バスト狙いの極限追求（favbust の続き）

favbustで最良=r1_prob≤p25選別×1位除外買いでTRAIN82%(壁手前)。最後の一押し:
  ①バスト選別を極限化(r1_prob≤p25/p15/p10/p05)＝より「飛ぶ」レースに集中
  ②1位除外の相手も中間オッズ帯[10,80]/[20,200]に絞る(≤6で効いたレバー)
  ③①②の複合
これでTRAINが100%を越えれば7+に活路。越えなければ本命バストでも控除率の壁。
pooled lgbm_wt・7+・最終オッズ上限値・train→test。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
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
        win = frozenset(int(x) for x in fin["frame_no"])
        rows.append({"fr": fr, "r1p": p[0], "gap12": p[0] - p[1],
                     "win": win, "win_odds": board.get(rk, {}).get(win, 0.0),
                     "board": board.get(rk, {})})
    return rows


def excl_r1_combos(fr, kmax):
    """1位(fr[0])を除外し、r2..r(kmax)のbox。"""
    return list(itertools.combinations(fr[1:kmax], 3))


def roi_for(rows, cond, kmax, oband=None):
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        bd = r["board"]
        combos = []
        for c in excl_r1_combos(r["fr"], kmax):
            fc = frozenset(c)
            if len(fc) != 3:
                continue
            if oband is not None:
                o = bd.get(fc)
                if o is None or not (oband[0] <= o < oband[1]):
                    continue
            combos.append(fc)
        combos = list(set(combos))
        if not combos:
            continue
        bets.append(len(combos) * 100)
        pays.append(r["win_odds"] * 100 if r["win"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def report(tr, te):
    r1p = sorted(r["r1p"] for r in tr)
    P = {f"p{q}": r1p[int(len(r1p) * q / 100)] for q in (25, 15, 10, 5)}
    SEL = [("ALL", lambda r: True)] + [
        (f"r1p<=p{q}({P[f'p{q}']:.3f})", (lambda r, v=P[f'p{q}']: r["r1p"] <= v)) for q in (25, 15, 10, 5)]
    OB = [("帯なし", None), ("[10,80]", (10, 80)), ("[20,200]", (20, 200)), ("[40,1e9]", (40, 1e9))]
    for kmax, pts_label in [(5, "box r2-5(4点)"), (6, "box r2-6(10点)")]:
        print(f"\n{'='*112}\n  本命バスト極限×中間オッズ: 1位除外 {pts_label}  TR {len(tr)}R / TE {len(te)}R（上限値）\n{'='*112}")
        for obn, ob in OB:
            print(f"\n  ▼ オッズ帯 {obn}")
            print(f"    {'バスト選別':<22}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}{'再現':>6}")
            for sn, cond in SEL:
                s1, n1 = roi_for(tr, cond, kmax, ob); s2, n2 = roi_for(te, cond, kmax, ob)
                flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
                print(f"    {sn:<22}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                      f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ バスト選別を極限化+中間オッズ集中でもTRAIN<100%なら、本命バストでも控除率の壁＝7+は構造的に不可。")


if __name__ == "__main__":
    report(collect("2023-07-01", "2026-02-28"), collect("2026-03-01", "2026-06-08"))
