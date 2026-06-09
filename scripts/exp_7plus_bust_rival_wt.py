"""7車以上: 本命バスト×別ライン軸 のROI再検証（追検証8の90%天井を別ライン構造で越えるか）

追検証9の精度発見: バスト時の軸は「別ライン最上位(RLH)」が最良(top3率76.1%>pred2位69.6%)。
本検証(三連複・1位完全除外):
 Part1 精度: 第2軸は何が最良か（RLHと同ライン番手 / 別の第3ライン頭 / pred順）。
   1人精度(top3率)＋2軸ペア精度(両方top3率＝2軸流しの天井)。
 Part2 ROI: 1軸流し(RLH) vs 2軸流し(RLH+第2軸)。第2軸が同ラインか別ラインかも比較。
   選別 ALL / r1_prob≤p25(飛びやすい)。払戻=wt_odds最終(上限値)。
リーダー定義(pred降順): r1=本命 / RLH=別ライン最上位 / rlh_bante=RLHと同ライン最上位(勝ちライン番手)
 / rival2=r1とRLH以外のライン頭 / pred_next=非r1のpred次位(RLH以外)。
pooled lgbm_wt・7+・train(2023-07〜2026-02)→test(2026-03〜)。
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


def leaders(fr, line):
    """pred降順fr・line(frame->group)から本命/別ライン頭/勝ちライン番手/第3ライン頭/pred次位。"""
    r1 = fr[0]; r1L = line[r1]
    rlh = next((x for x in fr[1:] if line[x] != r1L), None)
    rlhL = line[rlh] if rlh is not None else None
    rlh_bante = next((x for x in fr[1:] if x != rlh and line[x] == rlhL), None) if rlh else None
    rival2 = next((x for x in fr[1:] if line[x] not in (r1L, rlhL)), None)
    pred_next = next((x for x in fr[1:] if x != rlh), None)
    return r1, rlh, rlh_bante, rival2, pred_next


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
        fr = [int(x) for x in g["frame_no"]]
        line = {int(r["frame_no"]): r["line_group"] for _, r in g.iterrows()}
        win = frozenset(int(x) for x in fin["frame_no"])
        r1, rlh, rlh_bante, rival2, pred_next = leaders(fr, line)
        bust = r1 not in win
        rows.append({
            "fr": fr, "line": line, "win": win, "bust": bust, "r1p": p[0],
            "r1": r1, "rlh": rlh, "rlh_bante": rlh_bante, "rival2": rival2,
            "pred_next": pred_next, "pred2": fr[1],
            "board": board.get(rk, {}),
        })
    return rows


# ---------- Part1 精度（バスト時） ----------
def part1(rows):
    bust = [r for r in rows if r["bust"]]
    N = len(bust)
    print(f"\n{'='*84}\n  Part1 バスト時の軸精度  TRAIN バスト{N}R\n{'='*84}")
    def t3(key):
        s = [r for r in bust if r[key] is not None]
        return sum(1 for r in s if r[key] in r["win"]) / max(len(s), 1), len(s)
    print("  ▼ 第1軸/単独 top3率:")
    for key, lab in [("rlh", "別ライン最上位RLH"), ("pred2", "pred2位"), ("pred_next", "pred次位(非r1)"),
                     ("rlh_bante", "勝ちライン番手(RLH同ライン)"), ("rival2", "第3ライン頭")]:
        r, m = t3(key)
        print(f"    {lab:<28}{r:>7.1%}  (該当{m}R)")
    print("  ▼ 2軸ペア『両方top3』率（=2軸流しの的中天井）:")
    def pair(a, b):
        s = [r for r in bust if r[a] is not None and r[b] is not None and r[a] != r[b]]
        return sum(1 for r in s if r[a] in r["win"] and r[b] in r["win"]) / max(len(s), 1), len(s)
    for a, b, lab in [("rlh", "rlh_bante", "RLH + 勝ちライン番手(同ライン)"),
                      ("rlh", "rival2", "RLH + 第3ライン頭(別ライン)"),
                      ("rlh", "pred_next", "RLH + pred次位"),
                      ("rlh", "pred2", "RLH + pred2位")]:
        r, m = pair(a, b)
        print(f"    {lab:<32}{r:>7.1%}  (該当{m}R)")
    print("   ※ 第2軸は同ライン番手か別ライン頭か—『両方top3』率が高い方が2軸流しに適する。")


# ---------- Part2 ROI ----------
def remaining(r, exclude):
    return [x for x in r["fr"] if x not in exclude]


def build(r, kind):
    r1 = r["r1"]; rlh = r["rlh"]
    ex = {r1}
    if kind == "1軸RLH流しtop4(C4,2=6点)":
        if rlh is None: return []
        pool = remaining(r, {r1, rlh})[:4]
        return [(rlh, a, b) for a, b in itertools.combinations(pool, 2)]
    if kind == "1軸RLH流しtop3(C3,2=3点)":
        if rlh is None: return []
        pool = remaining(r, {r1, rlh})[:3]
        return [(rlh, a, b) for a, b in itertools.combinations(pool, 2)]
    if kind == "2軸RLH+番手(同)流し3点":
        a2 = r["rlh_bante"]
        if rlh is None or a2 is None: return []
        pool = remaining(r, {r1, rlh, a2})[:3]
        return [(rlh, a2, x) for x in pool]
    if kind == "2軸RLH+第3ライン頭(別)流し3点":
        a2 = r["rival2"]
        if rlh is None or a2 is None: return []
        pool = remaining(r, {r1, rlh, a2})[:3]
        return [(rlh, a2, x) for x in pool]
    if kind == "2軸RLH+pred次位 流し3点":
        a2 = r["pred_next"]
        if rlh is None or a2 is None: return []
        pool = remaining(r, {r1, rlh, a2})[:3]
        return [(rlh, a2, x) for x in pool]
    if kind == "baseline:box r2-5(1位除外4点)":
        pool = remaining(r, {r1})[:4]
        return list(itertools.combinations(pool, 3))
    return []


KINDS = ["baseline:box r2-5(1位除外4点)", "1軸RLH流しtop3(C3,2=3点)", "1軸RLH流しtop4(C4,2=6点)",
         "2軸RLH+番手(同)流し3点", "2軸RLH+第3ライン頭(別)流し3点", "2軸RLH+pred次位 流し3点"]


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


def part2(tr, te):
    r1p = sorted(r["r1p"] for r in tr); q25 = r1p[len(r1p) // 4]
    SEL = [("ALL(7+全レース)", lambda r: True),
           (f"r1_prob<=p25({q25:.3f})飛びやすい", lambda r, v=q25: r["r1p"] <= v)]
    for sn, cond in SEL:
        print(f"\n{'='*112}\n  Part2 ROI [{sn}] 三連複・1位完全除外  TR / TE（最終オッズ上限値）\n{'='*112}")
        print(f"    {'買い目':<32}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}{'再現':>6}")
        for kind in KINDS:
            s1, n1 = roi_for(tr, cond, kind); s2, n2 = roi_for(te, cond, kind)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"    {kind:<32}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                  f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ 別ライン軸でTRAINが追検証8の90%天井を越え100%に届くか。届かなければバスト+ライン構造でも控除率の壁。")


if __name__ == "__main__":
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    part1(tr)
    part2(tr, te)
