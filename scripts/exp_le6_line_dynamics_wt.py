"""6車以下: ライン力学(本命ライン道連れ崩壊・別ライン台頭)を転用できるか【検証のみ・運用不変】

7+で実証した副産物=「本命(逃)ラインは飛ぶ時に番手も道連れで崩れ、別ライン1-2が台頭」。
収益実体のある≤6車側で活かせるか検証(運用モデル・買い目は変更しない・あくまで検証)。
 Part A 精度: ≤6車で①pred1位バスト率 ②バスト時の別ライン最上位(RLH) vs pred2位 top3率。
 Part B ROI: 現行2軸流し(pred1,pred2+pred3-5流し) vs ライン考慮版(同ライン依存回避)を
   ALL / 開いたレース(top3_sum下位) で比較。再現的に上回るかのみ判定（採用は別途）。
払戻=wt_odds最終(三連複・上限値)。pooled lgbm_wt・≤6車・train→test。
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
    df = df[df["race_key"].isin(sz[sz <= 6].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks = df["race_key"].unique().tolist()
    board = load_trio_board(rks)
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 4:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = [int(x) for x in g["frame_no"]]
        line = {int(r["frame_no"]): r["line_group"] for _, r in g.iterrows()}
        win = frozenset(int(x) for x in fin["frame_no"])
        r1 = fr[0]; r1L = line[r1]
        rlh = next((x for x in fr[1:] if line[x] != r1L), None)       # 別ライン最上位
        # pred2が本命と同ラインか
        pred2 = fr[1]; pred2_same = (line[pred2] == r1L)
        rows.append({"fr": fr, "line": line, "win": win, "n": n,
                     "r1": r1, "rlh": rlh, "pred2": pred2, "pred2_same": pred2_same,
                     "bust": r1 not in win, "gap12": p[0] - p[1],
                     "top3_sum": p[0] + p[1] + p[2], "board": board.get(rk, {})})
    return rows


# ---------- Part A 精度 ----------
def partA(rows):
    N = len(rows)
    bust = [r for r in rows if r["bust"]]
    print(f"\n{'='*80}\n  Part A ≤6車 ライン力学 精度  N={N}R\n{'='*80}")
    print(f"  pred1位バスト率(3着外): {len(bust)/N:.1%}  (7+は20%)")
    print(f"  pred2位が本命と同ライン: {sum(1 for r in rows if r['pred2_same'])/N:.1%}")
    nb = len(bust)
    def t3(key):
        s = [r for r in bust if r[key] is not None]
        return sum(1 for r in s if r[key] in r["win"]) / max(len(s), 1), len(s)
    rlh_t3, mrlh = t3("rlh"); p2_t3, mp2 = t3("pred2")
    print(f"  バスト時 top3率: 別ライン最上位RLH {rlh_t3:.1%}(該当{mrlh}R) / pred2位 {p2_t3:.1%}(該当{mp2}R)")
    # バスト時、本命と同ライン vs 別ライン の top3率
    sl = [(r, x) for r in bust for x in r["fr"][1:]]
    same = [(r, x) for r, x in sl if r["line"][x] == r["line"][r["r1"]]]
    riv = [(r, x) for r, x in sl if r["line"][x] != r["line"][r["r1"]]]
    def rate(lst): return sum(1 for r, x in lst if x in r["win"]) / max(len(lst), 1)
    print(f"  バスト時 非r1選手top3率: 本命と同ライン {rate(same):.1%} / 別ライン {rate(riv):.1%}")
    print("   ※ 7+と同じ『同ライン道連れ・別ライン台頭』が≤6車でも出るか。差が小さければライン転用の余地小。")


# ---------- Part B ROI ----------
def remaining(r, ex):
    return [x for x in r["fr"] if x not in ex]


def build(r, kind):
    r1, pred2, rlh = r["r1"], r["pred2"], r["rlh"]
    if kind == "現行:2軸流しpred1,pred2+pred3-5":
        return [(r1, pred2, x) for x in r["fr"][2:5]]
    if kind == "ライン版:pred1+別ライン頭RLH+流し":
        if rlh is None: return []
        pool = remaining(r, {r1, rlh})[:3]
        return [(r1, rlh, x) for x in pool]
    if kind == "ライン版:pred2同ラインならRLHに差替":
        a2 = rlh if (r["pred2_same"] and rlh is not None) else pred2
        pool = remaining(r, {r1, a2})[:3]
        return [(r1, a2, x) for x in pool]
    return []


KINDS = ["現行:2軸流しpred1,pred2+pred3-5", "ライン版:pred1+別ライン頭RLH+流し",
         "ライン版:pred2同ラインならRLHに差替"]


def roi_for(rows, cond, kind):
    pays, bets = [], []
    for r in rows:
        if not cond(r): continue
        combos = list({frozenset(c) for c in build(r, kind) if len(set(c)) == 3})
        if not combos: continue
        bets.append(len(combos) * 100)
        o = r["board"].get(r["win"], 0.0)
        pays.append(o * 100 if r["win"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def partB(tr, te):
    med = statistics.median([r["top3_sum"] for r in tr])
    p25 = sorted(r["top3_sum"] for r in tr)[len(tr) // 4]
    SEL = [("ALL(≤6車)", lambda r: True),
           (f"開: top3_sum<中央({med:.2f})", lambda r, m=med: r["top3_sum"] < m),
           (f"開: top3_sum<=p25({p25:.2f})Q1相当", lambda r, v=p25: r["top3_sum"] <= v)]
    for sn, cond in SEL:
        print(f"\n{'='*112}\n  Part B ≤6車 ROI [{sn}] 現行 vs ライン版（最終オッズ上限値）\n{'='*112}")
        print(f"    {'買い目':<36}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}")
        for kind in KINDS:
            s1, n1 = roi_for(tr, cond, kind); s2, n2 = roi_for(te, cond, kind)
            print(f"    {kind:<36}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                  f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]")
    print("\n  ※ ライン版が現行をTRAIN・TESTとも一貫して上回れば≤6車改善の候補。下回れば現行維持が正。")


if __name__ == "__main__":
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    partA(tr)
    partB(tr, te)
