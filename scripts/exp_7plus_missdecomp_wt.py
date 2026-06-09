"""7車以上: 的中率ミス要因の分解と買い目最適化（error decomposition）

現行2軸流し三連複(軸=pred上位2・相手=pred3-5位の3点)が外す理由を分解:
  ① 軸落ち : 実着top3に軸(r1/r2)が不在（r1落ち/r2落ち/両落ち）
  ② 着順違い: 三連複では無関係（三連単に行く場合のみ。参考表示）
  ③ 相手抜け: 軸2人は来たが3人目が流し範囲(r3-5)の外
→ ③で「3人目がpred何位に落ちたか」分布を見て、最小点数増で的中率を最大化する買い目を特定。
さらに候補買い目の的中率×ROI(wt最終オッズ上限値)を併記し、的中率向上が+EVに繋がるか検証。
pooled lgbm_wt・7+・finish_order≥1・train(2023-07〜2026-02)→test(2026-03〜)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
from collections import defaultdict, Counter
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
        fr = g["frame_no"].astype(int).tolist()        # pred降順の車番(rank0..)
        rankof = {f: i for i, f in enumerate(fr)}
        Wframes = [int(x) for x in fin["frame_no"]]
        ranks_W = sorted(rankof[f] for f in Wframes)   # 実着top3のpred順位(0始まり)
        rows.append({"n": n, "fr": fr, "ranks_W": ranks_W,
                     "win": frozenset(Wframes), "board": board.get(rk, {})})
    return rows


def decompose(rows, label):
    """現行2軸流し(軸r1,r2 / 相手r3-5)の的中・ミス要因を分類。"""
    N = len(rows)
    hit = 0
    a_none = a_r1out = a_r2out = 0      # 軸落ち
    partner_miss = 0                   # 相手抜け
    partner_rank_dist = Counter()      # 相手抜け時の3人目pred順位
    third_rank_when_axes_in = Counter()
    for r in rows:
        rw = r["ranks_W"]
        s = set(rw)
        a1 = 0 in s; a2 = 1 in s
        if a1 and a2:
            third = [x for x in rw if x not in (0, 1)][0]
            third_rank_when_axes_in[third] += 1
            if third in (2, 3, 4):
                hit += 1
            else:
                partner_miss += 1
                partner_rank_dist[third] += 1
        else:
            if not a1 and not a2:
                a_none += 1
            elif a1 and not a2:
                a_r2out += 1
            else:
                a_r1out += 1
    print(f"\n{'='*92}\n  {label}: 2軸流し三連複(3点) ミス要因分解  N={N}R\n{'='*92}")
    print(f"  的中(軸2人+相手r3-5)        {hit:>6} ({hit/N:>5.1%})")
    print(f"  ③相手抜け(軸2人来たが3人目が範囲外){partner_miss:>6} ({partner_miss/N:>5.1%})")
    axes_in = hit + partner_miss
    print(f"   └ 軸2人とも来た計          {axes_in:>6} ({axes_in/N:>5.1%})  ＝2軸流しの的中天井")
    print(f"  ①軸落ち計                 {a_none+a_r1out+a_r2out:>6} ({(a_none+a_r1out+a_r2out)/N:>5.1%})")
    print(f"   ├ r1(pred1位)落ち          {a_r1out:>6} ({a_r1out/N:>5.1%})")
    print(f"   ├ r2(pred2位)落ち          {a_r2out:>6} ({a_r2out/N:>5.1%})")
    print(f"   └ 両軸落ち                {a_none:>6} ({a_none/N:>5.1%})")
    print(f"\n  ③相手抜け時の『3人目』pred順位分布（rank0始まり・5位以降が抜け）:")
    for k in sorted(partner_rank_dist):
        c = partner_rank_dist[k]
        print(f"     rank{k}(={k+1}位): {c:>5} ({c/max(partner_miss,1):>5.1%})  ※相手をrank{k}まで広げれば追加で{c/N:>4.1%}的中")
    # 累積: 相手をtopK位まで広げた時の的中率(軸2人来た前提)
    print(f"\n  ▼ 相手の流し範囲を広げた時の的中率(軸2人来た{axes_in/N:.1%}が上限)・点数(7車基準)")
    for kmax in [4, 5, 6, 7, 8]:   # rank index上限(相手を rank2..kmax)
        cum = sum(third_rank_when_axes_in[x] for x in range(2, kmax + 1))
        pts = kmax - 1             # 相手 = rank2..kmax = (kmax-1)点
        print(f"     相手rank2-{kmax}({pts}点): 的中{cum/N:>5.1%}  (軸2人来た中の{cum/max(axes_in,1):>5.1%}捕捉)")


def coverage(rows, label):
    """実着top3がpred top-k に収まる確率(=各種box/流しの的中天井)。"""
    N = len(rows)
    print(f"\n{'='*92}\n  {label}: 実着top3がpred top-k に収まる率（買い目天井）  N={N}R\n{'='*92}")
    print(f"  {'top-k':<8}{'W⊆topk率':<12}{'box点数C(k,3)':<14}{'1軸流し(r1固定)点数':<18}")
    for k in range(3, 10):
        cov = sum(1 for r in rows if max(r["ranks_W"]) <= k - 1 and r["n"] >= k) / N
        box = k * (k - 1) * (k - 2) // 6
        ax1 = (k - 1) * (k - 2) // 2   # r1固定+残りtop(k)から2 = C(k-1,2)
        print(f"  top{k:<5}{cov:>9.1%}   {box:>10}   {ax1:>14}")
    # 1着・2着・3着 各々のpred順位中央
    import statistics
    r1pos = [r["ranks_W"][0] for r in rows]
    print(f"\n  実着top3のpred順位(最良/中/最悪)中央値: "
          f"{statistics.median(r['ranks_W'][0] for r in rows):.0f}/"
          f"{statistics.median(r['ranks_W'][1] for r in rows):.0f}/"
          f"{statistics.median(r['ranks_W'][2] for r in rows):.0f} (0始まり)")


def buy_roi(rows, builder, label_set):
    """builder(fr,n)->list[frozenset] の買い目で的中率×ROI。"""
    res = {}
    for name, fn in builder.items():
        pays, bets = [], []
        for r in rows:
            combos = fn(r["fr"], r["n"])
            combos = list({frozenset(c) for c in combos if len(set(c)) == 3})
            if not combos:
                continue
            bets.append(len(combos) * 100)
            o = r["board"].get(r["win"], 0.0)
            pays.append(o * 100 if r["win"] in combos else 0.0)
        res[name] = (roi_summary(pays, bets), len(pays),
                     sum(bets) / max(len(bets), 1) / 100)
    return res


BUILDERS = {
    "2軸流しr3-5(現行3点)": lambda fr, n: [(fr[0], fr[1], x) for x in fr[2:5]],
    "2軸流しr3-6(4点)": lambda fr, n: [(fr[0], fr[1], x) for x in fr[2:6]],
    "2軸流しr3-7(5点)": lambda fr, n: [(fr[0], fr[1], x) for x in fr[2:7]],
    "1軸流しr1+top5(C(4,2)6点)": lambda fr, n: [(fr[0], a, b) for a, b in itertools.combinations(fr[1:5], 2)],
    "1軸流しr1+top6(C(5,2)10点)": lambda fr, n: [(fr[0], a, b) for a, b in itertools.combinations(fr[1:6], 2)],
    "box top4(4点)": lambda fr, n: list(itertools.combinations(fr[:4], 3)),
    "box top5(10点)": lambda fr, n: list(itertools.combinations(fr[:5], 3)),
}


def report_buys(tr, te):
    rtr = buy_roi(tr, BUILDERS, None)
    rte = buy_roi(te, BUILDERS, None)
    print(f"\n{'='*100}\n  7+ 候補買い目: 的中率×ROI(wt最終オッズ上限値)  TR {len(tr)}R / TE {len(te)}R\n{'='*100}")
    print(f"  {'買い目':<26}{'点':>4}  {'TR的中':>6}{'TR_ROI':>8}   {'TE的中':>6}{'TE_ROI':>8}{'TE_CI':>16}{'再現':>6}")
    for name in BUILDERS:
        s1, n1, pp = rtr[name]; s2, n2, _ = rte[name]
        flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ""
        print(f"  {name:<26}{pp:>4.0f}  {s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}   "
              f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ 的中率を上げても点数増でROIは控除率に収束。的中↑が+EVに繋がる(★再現)買い目があるかが焦点。")


if __name__ == "__main__":
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    decompose(tr, "TRAIN 7+")
    coverage(tr, "TRAIN 7+")
    report_buys(tr, te)
