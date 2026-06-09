"""7車以上: 「本命(pred1位)バスト」を狙い1位を外して残りで組む（軸落ち46%の高配当側）

ユーザー仮説: 軸落ちの中でも『1位飛び』は人気が飛ぶため高配当。1位を買い目から外し
相手も残り(r2以降)から選定する。
構造: pred1位は7+で~80%が3着内→1位除外買いはその80%が自動的に外れ(コスト発生)。
勝負は ①1位が飛ぶ20%を市場超で予測できるか ②飛んだ時に残り2人を当てられるか。
選別: gap12低/1位prob低/1位が逃(style_enc==0)/n_senko多。買い目は全て1位除外。
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
    has_style = "style_enc" in df.columns
    has_ns = "n_senko" in df.columns
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
        r1 = fr[0]
        r1_style = int(g.iloc[0]["style_enc"]) if has_style else -1
        n_senko = int(g.iloc[0]["n_senko"]) if has_ns else -1
        rows.append({
            "n": n, "fr": fr, "p": p, "gap12": p[0] - p[1], "r1p": p[0],
            "r1_style": r1_style, "n_senko": n_senko,
            "r1_in": r1 in win, "win": win, "win_odds": board.get(rk, {}).get(win, 0.0),
            "board": board.get(rk, {}),
        })
    return rows


# 1位(fr[0])を除外した買い目。frの[1:]=r2以降で組む
BUYS = {
    "box r2-4(1点)": lambda fr: list(itertools.combinations(fr[1:4], 3)),
    "box r2-5(4点)": lambda fr: list(itertools.combinations(fr[1:5], 3)),
    "box r2-6(10点)": lambda fr: list(itertools.combinations(fr[1:6], 3)),
    "r2軸流しr3-6(C(4,2)6点)": lambda fr: [(fr[1], a, b) for a, b in itertools.combinations(fr[2:6], 2)],
    "2軸r2r3流しr4-6(3点)": lambda fr: [(fr[1], fr[2], x) for x in fr[3:6]],
}


def roi_for(rows, cond, buy):
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        combos = list({frozenset(c) for c in buy(r["fr"]) if len(set(c)) == 3})
        if not combos:
            continue
        bets.append(len(combos) * 100)
        pays.append(r["win_odds"] * 100 if r["win"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def bust_rate(rows, cond):
    s = [r for r in rows if cond(r)]
    if not s:
        return 0.0, 0
    return sum(1 for r in s if not r["r1_in"]) / len(s), len(s)


def report(tr, te):
    g12 = sorted(r["gap12"] for r in tr); q25g = g12[len(g12) // 4]
    r1p = sorted(r["r1p"] for r in tr); q25p = r1p[len(r1p) // 4]
    CONDS = [
        ("ALL", lambda r: True),
        (f"gap12<=p25({q25g:.3f})1位非優勢", lambda r, v=q25g: r["gap12"] <= v),
        (f"r1_prob<=p25({q25p:.3f})", lambda r, v=q25p: r["r1p"] <= v),
        ("1位が逃(style=0)", lambda r: r["r1_style"] == 0),
        ("n_senko>=3(撹乱)", lambda r: r["n_senko"] >= 3),
        ("1位逃 & gap12<=p25", lambda r, v=q25g: r["r1_style"] == 0 and r["gap12"] <= v),
        ("n_senko>=3 & gap12<=p25", lambda r, v=q25g: r["n_senko"] >= 3 and r["gap12"] <= v),
    ]
    # ① 1位バスト率（選別で20%超に上げられるか）
    print(f"\n{'='*96}\n  ① pred1位バスト率(3着外)を選別で上げられるか  TRAIN {len(tr)}R / TEST {len(te)}R\n{'='*96}")
    print(f"  {'選別':<30}{'TR_bust率':>10}{'TR_R':>8}{'TE_bust率':>10}{'TE_R':>8}")
    for cn, cond in CONDS:
        b1, n1 = bust_rate(tr, cond); b2, n2 = bust_rate(te, cond)
        print(f"  {cn:<30}{b1:>9.1%}{n1:>8}{b2:>9.1%}{n2:>8}")
    print("  ※ 全体バスト~20%。選別で大きく上がらなければ『飛び』は予測不能＝1位除外買いの母数が薄い。")

    # ② 1位除外買いの的中率×ROI
    for bn, buy in BUYS.items():
        print(f"\n{'='*108}\n  ② 1位除外買い: {bn}  TR {len(tr)}R / TE {len(te)}R（最終オッズ上限値）\n{'='*108}")
        print(f"    {'選別':<30}{'TR的中':>6}{'TR_ROI':>8}{'TR_R':>7}   {'TE的中':>6}{'TE_ROI':>8}{'TE_R':>7}{'TE_CI':>16}{'再現':>6}")
        for cn, cond in CONDS:
            s1, n1 = roi_for(tr, cond, buy); s2, n2 = roi_for(te, cond, buy)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"    {cn:<30}{s1['hit_rate']:>6.1%}{s1['roi']:>8.0%}{n1:>7}   "
                  f"{s2['hit_rate']:>6.1%}{s2['roi']:>8.0%}{n2:>7} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{flag:>6}")
    print("\n  ※ 1位除外は的中率が低い(80%は1位が来て自動外れ)が高配当。的中×配当が控除率を超え★再現が出るかが焦点。")


if __name__ == "__main__":
    report(collect("2023-07-01", "2026-02-28"), collect("2026-03-01", "2026-06-08"))
