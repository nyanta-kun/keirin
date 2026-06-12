"""≤6車: 1位指数(pred1)の着順別レース選定 × 三連単推奨の検証

ユーザーの問い:
  「pred1(指数1位)が ①2-3着 / ②4着以下(バスト) になるレースを事前選定でき、
    そのレースで三連単の推奨(=黒字)が取れるか」を、7+の本命バスト検証を踏まえて ≤6車 で検証。

7+ の確定事実(`docs/analysis/05`): pred1 は7+で~80%top3、バストは r1_prob で部分予測可(AUC0.71)、
  バスト時の最良軸は別ライン最上位RLH(76.1%)・第2軸は勝ちライン番手。だが市場が同構造を織込み
  三連複91%/三連単88%天井(=フィールドサイズ=市場効率の壁)。**≤6車は非効率ポケット有り**＝
  同じ手法が壁を越えるかが本検証の核心。

3部:
  Part1 着順分類の基準率: pred1 の着順 {1着/2着/3着/4着以下} を ≤6車 で測る(7+比較)。
  Part2 選定可能性: 事前情報(r1_prob/gap/ratio/top3_sum/n_senko/n_lines)で「pred1≠1着」「バスト」を
        当てられるか(単一特徴AUC + 層別実現率)。
  Part3 三連単推奨: pred1着順を踏まえた三連単構築が現行(trio 2軸流し / SS三連単 pred1頭)を
        ★再現(TRAIN&TEST>100%)で上回るか。選定条件別に ROI/CI を比較。

規律: model=lgbm_wt_eval(holdout・test>=2026-03 はOOS)。★再現=TRAIN(大標本)&TEST 両方>100%・
  bootstrap CI(roi_robustness_wt)。払戻=wt_odds最終=上限値(実運用は下振れ)。
  train 2023-07〜2026-02 / test 2026-03〜。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import numpy as np
import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _filter_by_n_riders, _assign_tier
from roi_robustness_wt import roi_summary

MODEL = "lgbm_wt_eval"


def manual_auc(y, s):
    pairs = sorted(zip(s, y)); n = len(pairs)
    npos = sum(y); nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (sum_pos - npos * (npos + 1) / 2) / (npos * nneg)


def load_win_payouts(want_tri, want_trio):
    """実着順の三連単払戻 と 実top3の三連複払戻 を一括取得（勝ち組のみ・省メモリ）。"""
    tri, trio = {}, {}
    keys = list(set(want_tri) | set(want_trio))
    from src.database import get_connection
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            ph = ",".join("?" * len(chunk))
            q = (f"SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 f"WHERE bet_type IN ('trifecta','trio') AND race_key IN ({ph})")
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None:
                    continue
                parts = [p for p in re.split(r"[-=→]", str(comb)) if p != ""]
                try:
                    nums = [int(p) for p in parts]
                except ValueError:
                    continue
                if bt == "trifecta" and rk in want_tri and tuple(nums) == want_tri[rk] and rk not in tri:
                    tri[rk] = int(round(od * 100))
                elif bt == "trio" and rk in want_trio and frozenset(nums) == want_trio[rk] and rk not in trio:
                    trio[rk] = int(round(od * 100))
    return tri, trio


def leaders(fr, line):
    """pred降順 fr とライン辞書から軸候補を返す。"""
    r1 = fr[0]; r1L = line[r1]
    r2 = fr[1] if len(fr) > 1 else None
    rlh = next((x for x in fr[1:] if line[x] != r1L), None)         # 別ライン最上位
    rlhL = line[rlh] if rlh is not None else None
    bante = next((x for x in fr[1:] if x != rlh and line[x] == rlhL), None) if rlh else None
    nxt = next((x for x in fr[1:] if x != rlh), None)
    nxt2 = next((x for x in fr[1:] if x not in (rlh, nxt)), None)
    return {"r1": r1, "r2": r2, "rlh": rlh, "bante": bante, "nxt": nxt, "nxt2": nxt2,
            "p3": fr[2] if len(fr) > 2 else None, "p4": fr[3] if len(fr) > 3 else None,
            "p5": fr[4] if len(fr) > 4 else None}


def collect(f, t):
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, 6)
    rows, want_tri, want_trio = [], {}, {}
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 4:
            continue
        fin = g[g["finish_order"].between(1, 3)].sort_values("finish_order")
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = [int(x) for x in g["frame_no"]]
        line = {int(r["frame_no"]): r["line_group"] for _, r in g.iterrows()}
        order = tuple(int(x) for x in fin["frame_no"])    # 実1-2-3着
        top3 = frozenset(order)
        ld = leaders(fr, line)
        # pred1 の実着順（1,2,3 / 4=4着以下バスト）
        r1_fin = g[g["frame_no"] == ld["r1"]]["finish_order"].iloc[0]
        r1_pos = int(r1_fin) if int(r1_fin) in (1, 2, 3) else 4
        r1_row = g.iloc[0]
        want_tri[rk] = order
        want_trio[rk] = top3
        rows.append({
            "rk": rk, "n": n, "order": order, "top3": top3, "ld": ld,
            "r1_pos": r1_pos, "bust": int(r1_pos == 4), "not1st": int(r1_pos != 1),
            "r1_prob": p[0], "gap12": p[0] - p[1], "gap13": p[0] - (p[2] if n > 2 else 0),
            "ratio": p[0] / (3.0 / n), "top3_sum": p[0] + p[1] + (p[2] if n > 2 else 0),
            "n_senko": int(r1_row["n_senko"]), "n_lines": g["line_group"].nunique(),
            "tier": _assign_tier(p[0] - p[1], p[0] / (3.0 / n)),
        })
    tri, trio = load_win_payouts(want_tri, want_trio)
    for r in rows:
        r["tri_pay"] = tri.get(r["rk"], 0)
        r["trio_pay"] = trio.get(r["rk"], 0)
    return rows


# ---------- Part 1: 基準率 ----------
def part1(tr, te):
    print(f"\n{'='*90}\n  Part1. pred1(指数1位)の着順分類 基準率  ≤6車  TR {len(tr)}R / TE {len(te)}R\n{'='*90}")
    for nm, rows in [("TRAIN", tr), ("TEST", te)]:
        n = len(rows)
        c = {1: 0, 2: 0, 3: 0, 4: 0}
        for r in rows:
            c[r["r1_pos"]] += 1
        print(f"  {nm}: pred1着順  1着 {c[1]/n:.1%} / 2着 {c[2]/n:.1%} / 3着 {c[3]/n:.1%} / "
              f"4着以下(バスト) {c[4]/n:.1%}   [top3={1-c[4]/n:.1%}]")
    print("  ※ 7+参考: pred1 ~80%top3(飛び~20%)。≤6車は車数少で top3率↑・1着率↑が期待値。")


# ---------- Part 2: 選定可能性 ----------
def part2(tr, te):
    print(f"\n{'='*90}\n  Part2. 選定可能性: 事前情報で『pred1≠1着』『バスト』を当てられるか\n{'='*90}")
    dtr, dte = pd.DataFrame(tr), pd.DataFrame(te)
    print(f"  基準率(TEST): pred1≠1着 {dte['not1st'].mean():.1%} / バスト(4着以下) {dte['bust'].mean():.1%}")
    print(f"\n  単一特徴の判別AUC(TEST・符号はイベントと正相関に調整):")
    print(f"    {'特徴':<12}{'pred1≠1着AUC':>14}{'バストAUC':>12}")
    for col, sign in [("r1_prob", -1), ("gap12", -1), ("gap13", -1), ("ratio", -1),
                      ("top3_sum", -1), ("n_senko", +1), ("n_lines", +1)]:
        a1 = manual_auc(dte["not1st"].tolist(), (sign * dte[col]).tolist())
        a2 = manual_auc(dte["bust"].tolist(), (sign * dte[col]).tolist())
        print(f"    {col:<12}{a1:>13.4f}{a2:>12.4f}")
    # r1_prob 四分位ごとの実現率(TEST)
    dte = dte.copy()
    dte["q"] = pd.qcut(dte["r1_prob"], 4, labels=["Q1_低", "Q2", "Q3", "Q4_高"])
    print(f"\n  r1_prob四分位ごとの実現率(TEST):")
    print(f"    {'帯':<8}{'R':>6}{'pred1=1着':>10}{'pred1 top3':>12}{'バスト':>9}{'中央三連単配当':>14}")
    for q in ["Q1_低", "Q2", "Q3", "Q4_高"]:
        s = dte[dte["q"] == q]
        p1 = (s["r1_pos"] == 1).mean(); pt3 = (s["r1_pos"] <= 3).mean(); pb = s["bust"].mean()
        med = s[s["tri_pay"] > 0]["tri_pay"].median() if (s["tri_pay"] > 0).any() else 0
        print(f"    {q:<8}{len(s):>6}{p1:>9.1%}{pt3:>11.1%}{pb:>8.1%}{med:>12,.0f}円")
    print("   ※ AUC>0.6 & 単調なら選定可能。Q1_低(指数1位が弱い)で バスト率↑・1着率↓ なら『pred1≠1着』選定が立つ。")


# ---------- Part 3: 三連単推奨 ----------
def _dedup(combos):
    return list(dict.fromkeys(tuple(c) for c in combos if len(set(c)) == 3))


def build_tri(r, kind):
    """三連単の買い目リスト(順序tuple)。"""
    ld = r["ld"]
    r1, r2, rlh, bante = ld["r1"], ld["r2"], ld["rlh"], ld["bante"]
    p3, p4, p5, nxt, nxt2 = ld["p3"], ld["p4"], ld["p5"], ld["nxt"], ld["nxt2"]
    def perm(heads, seconds, thirds):
        return _dedup((a, b, c) for a in heads if a for b in seconds if b for c in thirds if c)
    th = [x for x in [p3, p4, p5] if x]
    if kind == "SS現行: pred1→pred2→thirds":
        return perm([r1], [r2], th)
    if kind == "pred1,pred2 1-2着BOX×thirds":
        return perm([r1, r2], [r1, r2], th)
    if kind == "pred1を2-3着許容(head∈{pred2,RLH})":
        # 別ライン/2位を頭に、pred1 は2-3着に置く（pred1が勝たない前提）
        return perm([x for x in [r2, rlh] if x], [r1, r2, rlh], [r1, r2, rlh, p3])
    if kind == "バストRLH軸(pred1除外)RLH→bante→{nxt,nxt2}":
        if not (rlh and bante):
            return []
        return perm([rlh], [bante], [x for x in [nxt, nxt2] if x])
    if kind == "頭box{pred1,pred2,RLH}→相手{thirds}":
        return perm([x for x in [r1, r2, rlh] if x], [x for x in [r1, r2, rlh] if x],
                    [x for x in [r1, r2, rlh, p3] if x])
    return []


def roi_tri(rows, cond, kind):
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        combos = build_tri(r, kind)
        if not combos:
            continue
        bets.append(len(combos) * 100)
        pays.append(r["tri_pay"] if r["order"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def roi_trio_baseline(rows, cond):
    """現行 trio 2軸流し {pred1,pred2,x} x∈thirds（3点）。"""
    pays, bets = [], []
    for r in rows:
        if not cond(r):
            continue
        ld = r["ld"]; r1, r2 = ld["r1"], ld["r2"]
        th = [x for x in [ld["p3"], ld["p4"], ld["p5"]] if x]
        combos = [frozenset((r1, r2, x)) for x in th]
        combos = [c for c in combos if len(c) == 3]
        if not combos:
            continue
        bets.append(len(combos) * 100)
        pays.append(r["trio_pay"] if r["top3"] in combos else 0.0)
    return roi_summary(pays, bets), len(pays)


def part3(tr, te):
    r1ps = sorted(r["r1_prob"] for r in tr)
    q25 = r1ps[len(r1ps) // 4]; q50 = r1ps[len(r1ps) // 2]
    SEL = [
        ("ALL(≤6車)", lambda r: True),
        (f"r1_prob<=p25({q25:.3f})pred1弱", lambda r, v=q25: r["r1_prob"] <= v),
        (f"r1_prob<=p50({q50:.3f})", lambda r, v=q50: r["r1_prob"] <= v),
        ("S/A層のみ", lambda r: r["tier"] in ("S", "A")),
        ("SS層のみ", lambda r: r["tier"] == "SS"),
    ]
    TRI_KINDS = ["SS現行: pred1→pred2→thirds", "pred1,pred2 1-2着BOX×thirds",
                 "pred1を2-3着許容(head∈{pred2,RLH})", "バストRLH軸(pred1除外)RLH→bante→{nxt,nxt2}",
                 "頭box{pred1,pred2,RLH}→相手{thirds}"]
    print(f"\n{'='*118}\n  Part3. 三連単推奨 ROI（選定条件×構築・TRAIN→TEST・最終オッズ上限値）\n{'='*118}")
    for sn, cond in SEL:
        print(f"\n  ── 選定: [{sn}] ──")
        # 現行 trio 2軸流し baseline（参照）
        b1, bn1 = roi_trio_baseline(tr, cond); b2, bn2 = roi_trio_baseline(te, cond)
        print(f"    {'[基準]trio現行2軸流し':<42}TR {b1['roi']:>5.0%}({bn1})  "
              f"TE {b2['roi']:>5.0%}({bn2}) 的中{b2['hit_rate']:>4.0%} [{b2['ci_lo']:>4.0%},{b2['ci_hi']:>5.0%}] 最大除{b2['roi_ex_max']:>4.0%}")
        print(f"    {'三連単 構築':<42}{'TR_ROI(R)':>13}  {'TE_ROI(R)':>13}{'TE的中':>7}{'TE_CI':>15}{'最大除':>7}{'点':>4}{'再現':>6}")
        for kind in TRI_KINDS:
            s1, n1 = roi_tri(tr, cond, kind); s2, n2 = roi_tri(te, cond, kind)
            if n1 == 0 and n2 == 0:
                continue
            ppr = np.mean([len(build_tri(r, kind)) for r in tr if cond(r) and build_tri(r, kind)]) if n1 else 0
            # ★頑健再現: TR&TE>100% かつ TE最大払戻除去後も>100%（単発高配当依存でない）
            robust = (s1["roi"] > 1 and s2["roi"] > 1 and n2 >= 30 and s2["roi_ex_max"] > 1)
            flag = "★頑健" if robust else ("★単発" if (s1["roi"] > 1 and s2["roi"] > 1 and n2 >= 30) else ("小標本" if n2 < 30 else ""))
            print(f"    {kind:<42}{s1['roi']:>6.0%}({n1:>4}) {s2['roi']:>6.0%}({n2:>4})"
                  f"{s2['hit_rate']:>7.0%} [{s2['ci_lo']:>4.0%},{s2['ci_hi']:>5.0%}]{s2['roi_ex_max']:>7.0%}{ppr:>4.0f}{flag:>6}")
    print(f"\n{'='*118}")
    print("  判定: ★再現(TR&TE>100%)が trio基準を上回れば三連単推奨の余地。")
    print("   pred1弱(r1_prob低)選定×pred1除外/2-3着許容 で壁(7+の88%天井)を ≤6車 が越えるかが核心。")


if __name__ == "__main__":
    print("collecting TRAIN (2023-07〜2026-02)...", flush=True)
    tr = collect("2023-07-01", "2026-02-28")
    print(f"  TRAIN {len(tr)} races", flush=True)
    print("collecting TEST (2026-03〜)...", flush=True)
    te = collect("2026-03-01", "2026-06-08")
    print(f"  TEST {len(te)} races", flush=True)
    part1(tr, te)
    part2(tr, te)
    part3(tr, te)
