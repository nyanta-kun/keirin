"""7車以上: 「1位を1着で買わない」ライン構造ベースの波乱戦略

指数1位=市場織込済で高配当取りづらい。ユーザー案:
  T_noFav1 : 三連単 1着=別ライン頭/番手, 1位は2-3着固定（1位は1着で買わない）
  trio_bante : 三連複 軸=1位ライン番手 流し{1位,別頭r1,r2}
  trio_rhead : 三連複 軸=別ライン1番手(r1) 流し{1位,番手,r2}
  vbox4 : 三連複 box{1位,番手,r1,r2}（価値選手4頭BOX=4点）
比較: std3(三連複 軸1位-指数2位 流し)。
リーダー定義: fav=指数1位 / bante=1位ライン内の指数最上位(1位以外) / r1,r2=1位以外の各ライン頭(指数最上位)を指数順。
波乱選別(open)で7+を train→test。最終オッズ上限値。
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _load_payouts_wt
from roi_robustness_wt import roi_summary

model = load_model("lgbm_wt")


def riders(g):
    """指数降順のgから fav/bante/rival_leaders を返す。"""
    fr = g["frame_no"].astype(int).tolist()
    line = dict(zip(g["frame_no"].astype(int), g["line_group"]))
    fav = fr[0]; fav_line = line[fav]
    bante = next((x for x in fr if line[x] == fav_line and x != fav), None)  # 1位ライン番手
    seen = set(); rlead = []
    for x in fr:                          # 指数順に各「1位以外ライン」の頭
        L = line[x]
        if L == fav_line or L in seen:
            continue
        seen.add(L); rlead.append(x)
    return fav, bante, rlead


def collect(f, t, max_riders_lo=7, hi=99):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[(sz >= max_riders_lo) & (sz <= hi)].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False); n = len(g)
        if n < max_riders_lo: continue
        p = g["pred_prob"].tolist()
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3: continue
        fr = g["frame_no"].astype(int).tolist()
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order); po = pm.get(rk, {})
        fav, bante, rl = riders(g)
        r1 = rl[0] if rl else None; r2 = rl[1] if len(rl) > 1 else None
        def trio(c): return po.get(("trio", frozenset(c)), 0)
        def tri(o): return po.get(("trifecta", o), 0)
        def mk_trio(combos):
            combos = [frozenset(c) for c in combos if len(set(c)) == 3 and all(x is not None for x in c)]
            combos = list({frozenset(c) for c in combos})
            h = top3 in combos; return (h, trio(top3) if h else 0, len(combos)*100)
        def mk_tri(combos):
            combos = [tuple(c) for c in combos if len(set(c)) == 3 and all(x is not None for x in c)]
            combos = list(dict.fromkeys(combos)); h = order in combos
            return (h, tri(order) if h else 0, len(combos)*100)

        # std3
        std3 = mk_trio([(fav, fr[1], x) for x in fr[2:5]])
        # T_noFav1: 1着∈{r1,r2,bante}, 1位は2-3着に固定, 残り∈{fav,bante,r1,r2}
        heads = [h for h in [r1, r2, bante] if h is not None]
        pool = [x for x in [fav, bante, r1, r2] if x is not None]
        t_combos = []
        for h in heads:
            for a, b in itertools.permutations([x for x in pool if x != h], 2):
                if fav in (a, b):           # 1位を含む(=1位は2or3着) / 1着には入れない
                    t_combos.append((h, a, b))
        tnf = mk_tri(t_combos)
        # trio_bante: 軸=番手 流し{fav,r1,r2}
        tb = mk_trio([(bante, fav, r1), (bante, fav, r2), (bante, r1, r2)]) if bante else (False, 0, 0)
        # trio_rhead: 軸=r1 流し{fav,bante,r2}
        trh = mk_trio([(r1, fav, bante), (r1, fav, r2), (r1, bante, r2)]) if r1 else (False, 0, 0)
        # vbox4: {fav,bante,r1,r2} box
        v4 = [x for x in [fav, bante, r1, r2] if x is not None]
        vbox = mk_trio(list(itertools.combinations(v4, 3))) if len(v4) >= 3 else (False, 0, 0)
        rows.append({"gap12": p[0]-p[1], "top3_sum": p[0]+p[1]+p[2],
                     "std3": std3, "tnf": tnf, "tb": tb, "trh": trh, "vbox": vbox})
    return rows


def agg(rows, cond, k):
    s = [r for r in rows if cond(r) and r[k][2] > 0]
    return roi_summary([r[k][1] for r in s], [r[k][2] for r in s]), len(s)

def report(tr, te, title):
    print(f"\n{'='*92}\n  {title}  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）\n{'='*92}")
    import statistics
    med = statistics.median([r["top3_sum"] for r in tr])
    CONDS = {"ALL": lambda r: True, "gap12<0.10": lambda r: r["gap12"] < 0.10,
             "top3_sum<中央": lambda r, m=med: r["top3_sum"] < m}
    LAB = {"std3": "std3(基準)", "tnf": "1位2-3着(三単)", "tb": "番手軸(三複)",
           "trh": "別ライン頭軸(三複)", "vbox": "価値4頭box(三複)"}
    for cn, cond in CONDS.items():
        print(f"\n  ▼ {cn}")
        for k in ["std3", "tnf", "tb", "trh", "vbox"]:
            s1, n1 = agg(tr, cond, k); s2, n2 = agg(te, cond, k)
            ppr = sum(r[k][2] for r in te if cond(r) and r[k][2] > 0)/max(n2, 1)/100
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"    {LAB[k]:<16}{ppr:>3.0f}点 TR {n1:>5}R {s1['roi']:>5.0%} | TE {n2:>5}R {s2['roi']:>5.0%} "
                  f"[{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}] {flag}")

# 7+
report(collect("2023-07-01", "2026-02-28"), collect("2026-03-01", "2026-06-08"), "7車以上 ライン軸波乱戦略")
