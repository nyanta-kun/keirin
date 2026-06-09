"""7車以上: 本命バスト×別ライン構造の『三連単』検証（1位を2-3着に許容）

三連複(1位完全除外)はTRAIN天井91%。三連単で勝ちライン(RLH 1着・番手 2着)の順序を突き、
点数を絞れば高配当で100%を越えるか。1位は飛ぶ前提だが完全に消えるとは限らない→2-3着に許容。
リーダー(pred降順): r1=本命 / RLH=別ライン最上位 / bante=RLH同ライン最上位(勝ちライン番手)
 / next=非r1のpred次位(RLH以外)。
払戻=wt_odds最終(三連単・上限値)。pooled lgbm_wt・7+・train→test。
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
import re

model = load_model("lgbm_wt")


def load_tri_winorder(want):
    """want={rk:order_tuple} -> {rk: 払戻(円)}（実着順の三連単オッズのみ・省メモリ）。"""
    out = {}
    keys = list(want.keys())
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trifecta' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is None or rk in out:
                    continue
                parts = [p for p in re.split(r"[-=→]", str(comb)) if p != ""]
                try:
                    t = tuple(int(p) for p in parts)
                except ValueError:
                    continue
                if t == want[rk]:
                    out[rk] = int(round(od * 100))
    return out


def leaders(fr, line):
    r1 = fr[0]; r1L = line[r1]
    rlh = next((x for x in fr[1:] if line[x] != r1L), None)
    rlhL = line[rlh] if rlh is not None else None
    bante = next((x for x in fr[1:] if x != rlh and line[x] == rlhL), None) if rlh else None
    nxt = next((x for x in fr[1:] if x != rlh), None)
    nxt2 = next((x for x in fr[1:] if x not in (rlh, nxt)), None)
    return r1, rlh, bante, nxt, nxt2


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz >= 7].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rows = []
    want = {}
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 7:
            continue
        fin = g[g["finish_order"].between(1, 3)].sort_values("finish_order")
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = [int(x) for x in g["frame_no"]]
        line = {int(r["frame_no"]): r["line_group"] for _, r in g.iterrows()}
        order = tuple(int(x) for x in fin["frame_no"])
        r1, rlh, bante, nxt, nxt2 = leaders(fr, line)
        want[rk] = order
        rows.append({"rk": rk, "order": order, "r1p": p[0],
                     "r1": r1, "rlh": rlh, "bante": bante, "nxt": nxt, "nxt2": nxt2})
    tri = load_tri_winorder(want)
    for r in rows:
        r["pay"] = tri.get(r["rk"], 0)
    return rows


def build(r, kind):
    r1, rlh, bante, nxt, nxt2 = r["r1"], r["rlh"], r["bante"], r["nxt"], r["nxt2"]
    if rlh is None:
        return []
    def perm(heads, seconds, thirds):
        cs = [(a, b, c) for a in heads for b in seconds for c in thirds if len({a, b, c}) == 3]
        return list(dict.fromkeys(cs))
    if kind == "RLH1-番手2-3着{r1,next}(完全除外せず)":
        if bante is None: return []
        return perm([rlh], [bante], [x for x in [r1, nxt, nxt2] if x])
    if kind == "勝ちライン1-2両順×3着{r1,next}":
        if bante is None: return []
        return perm([rlh, bante], [rlh, bante], [x for x in [r1, nxt] if x])
    if kind == "RLH1固定-2,3着∈{番手,r1,next}box":
        pool = [x for x in [bante, r1, nxt] if x]
        return perm([rlh], pool, pool)
    if kind == "RLH1-番手2-rest3(1位完全除外)":
        if bante is None: return []
        return perm([rlh], [bante], [x for x in [nxt, nxt2] if x])
    if kind == "本命2-3着固定・RLH頭(F2型):{RLH}1-{r1,bante}23":
        pool = [x for x in [r1, bante, nxt] if x]
        return perm([rlh], pool, pool)
    return []


KINDS = ["RLH1-番手2-3着{r1,next}(完全除外せず)", "勝ちライン1-2両順×3着{r1,next}",
         "RLH1固定-2,3着∈{番手,r1,next}box", "RLH1-番手2-rest3(1位完全除外)"]


def roi_for(rows, cond, kind):
    pays, bets = [], []
    for r in rows:
        if not cond(r): continue
        combos = list(dict.fromkeys(tuple(c) for c in build(r, kind) if len(set(c)) == 3))
        if not combos: continue
        bets.append(len(combos) * 100)
        pays.append(r["pay"] if r["order"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def report(tr, te):
    r1p = sorted(r["r1p"] for r in tr); q25 = r1p[len(r1p) // 4]
    SEL = [("ALL(7+)", lambda r: True),
           (f"r1_prob<=p25({q25:.3f})飛びやすい", lambda r, v=q25: r["r1p"] <= v)]
    for sn, cond in SEL:
        print(f"\n{'='*114}\n  7+ 三連単 本命バスト×別ライン [{sn}]  TR / TE（最終オッズ上限値）\n{'='*114}")
        print(f"    {'買い目(三連単)':<40}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}{'再現':>6}")
        for kind in KINDS:
            s1, n1 = roi_for(tr, cond, kind); s2, n2 = roi_for(te, cond, kind)
            ppr = sum(len(list(dict.fromkeys(tuple(c) for c in build(r, kind) if len(set(c)) == 3)))
                      for r in tr if cond(r)) / max(n1, 1)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"    {kind:<40}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                  f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ 三連単で勝ちライン1-2順序を突いてもTRAIN<100%なら、三連単でもバスト+ライン構造は控除率の壁。")


if __name__ == "__main__":
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    report(tr, te)
