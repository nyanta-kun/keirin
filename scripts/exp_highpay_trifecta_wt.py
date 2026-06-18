"""車数制限なし: 高配当レースの事前検知 × 三連単~10点構造のROI検証。

ユーザーの問い(2026-06-11): 高配当の発生するレースを検知し、三連単の買い目(~10点)で
ROIを確保できるか。車数の制限なしで検討。

既存資産との関係（事前登録の根拠）:
  - 7+は19軸+EV原理+独立再現でクローズ(docs/analysis/05)。三連単天井88%。
    → 7+側は唯一の未検証新レバー fav_mismatch(2026-06-10発見・≤6車のみ検証済) に絞る。
      既存軸(top3_sum等)の再試行はしない（結果は対照として併記のみ）。
  - ≤6車の三連単は pred1,pred2 1-2着BOX→thirds(6点) が唯一の★頑健(docs/analysis/10)。
    10点への拡張形(ヒモ拡張/p3の2着替わりカバー)が本実験の新規部分。

事前登録（構造4種のみ・後出し追加はしない）:
  S0: pred1→pred2→thirds3            3点（現行SS基準）
  S1: BOX(p1,p2)→thirds3             6点（現行opt-in・★頑健既知）
  S2: BOX(p1,p2)→thirds全(max5)      ≤6車8点/7+10点（ヒモを広げ高配当3着を拾う）
  S3: 3頭BOX{p1,p2,p3}＋BOX(p1,p2)→{r4,r5}  10点（p3の2着替わりをカバー）
選別6種: ALL / RANK(現行SS/S/A) / UPSET(top3_sum≤TRAIN p25・車数層別カット) /
        MM(fav_mismatch) / RANK∩MM / RANK∩UPSET。車数 ≤6 / 7+ で層別。
判定: ★頑健 = TRAIN&TEST>100% かつ TEST最大払戻除去後>100%。

model=lgbm_wt_eval(OOS) / TRAIN 2023-07〜2026-02 / TEST 2026-03〜06-08。
払戻=最終オッズ=上限値（朝ドリフトの留保は他実験と同じ・三連単高配当帯は影響最大）。
"""
import itertools
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt, _assign_tier
from src.database import get_connection
from roi_robustness_wt import roi_summary

HIGH_PAY = 10000  # 万車券


def _market_favs(race_keys):
    """trio盤面から市場P(top3)を逆算し {rk: (市場本命frame, gap12_mkt)} を返す（batch）。"""
    acc: dict[str, dict[int, float]] = {}
    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ({ph})", chunk).fetchall()
            for rk, combo, ov in rows:
                if ov is None or ov <= 0 or ov >= 9000:
                    continue
                try:
                    fr = [int(x) for x in re.split(r"[-=]", str(combo))]
                except ValueError:
                    continue
                if len(fr) != 3:
                    continue
                w = 1.0 / ov
                q = acc.setdefault(rk, {})
                for f in fr:
                    q[f] = q.get(f, 0.0) + w
    out = {}
    for rk, q in acc.items():
        if len(q) < 4:
            continue
        s = sorted(q.items(), key=lambda x: -x[1])
        tot = sum(q.values())
        out[rk] = (s[0][0], 3.0 * (s[0][1] - s[1][1]) / tot)
    return out


def _score(order, rp, combos):
    """combos=[(a,b,c),...] を採点。pay, bet を返す。"""
    bet = len(combos) * 100
    pay = 0
    for c in combos:
        if order == c:
            pay = rp.get(("trifecta", c), 0)
            break
    return pay, bet


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)   # 車数フィルタなし
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    mf = _market_favs(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 5:
            continue  # S3(10点)が組めない4車以下は除外（点数前提を揃える）
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        gap12 = p[0] - p[1]
        tier = _assign_tier(gap12, p[0] / (3.0 / n))
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        rp = pm.get(rk, {})
        pay_win = rp.get(("trifecta", order), 0)   # 勝ち目の実払戻（高配当判定用）
        mkt = mf.get(rk)
        p1, p2, p3 = fr[0], fr[1], fr[2]
        thirds3 = fr[2:5]
        thirds_full = fr[2:7]                       # max5 → ≤6車8点/7+10点
        s0 = _score(order, rp, [(p1, p2, x) for x in thirds3])
        s1 = _score(order, rp, [(a, b, x) for a, b in ((p1, p2), (p2, p1)) for x in thirds3])
        s2 = _score(order, rp, [(a, b, x) for a, b in ((p1, p2), (p2, p1)) for x in thirds_full])
        c3 = [(a, b, c) for a, b, c in itertools.permutations((p1, p2, p3))]
        c3 += [(a, b, x) for a, b in ((p1, p2), (p2, p1)) for x in fr[3:5]]
        s3 = _score(order, rp, c3)
        rows.append({
            "n": n, "tier": tier, "gap12": gap12,
            "r1_prob": p[0], "top3_sum": p[0] + p[1] + p[2],
            "mm": (mkt is not None and mkt[0] != p1),
            "has_mkt": mkt is not None,
            "d_gap": (gap12 - mkt[1]) if mkt else None,
            "pay_win": pay_win, "high": pay_win >= HIGH_PAY,
            "S0": s0, "S1": s1, "S2": s2, "S3": s3,
        })
    return rows


def _auc(y, x):
    """rank-based AUC（タイは平均順位・scipy不要）。"""
    y = np.asarray(y, dtype=bool); x = np.asarray(x, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # タイの平均順位化
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def line(lab, s, n, star=""):
    return (f"  {lab:<30}{n:>6}{s['hit_rate']:>8.1%}{s['roi']:>8.0%}"
            f" [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>9.0%}{star}")


def main():
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    print(f"\n車数制限なし 高配当検知×三連単~10点  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値・5車以上）")

    # ---- Part1: 高配当(万車券)の事前検知 ----
    print(f"\n【Part1】三連単万車券(≥{HIGH_PAY:,}円)の事前検知（TRAIN）")
    for szlab, cond in [("≤6車", lambda r: r["n"] <= 6), ("7+", lambda r: r["n"] >= 7)]:
        sub = [r for r in tr if cond(r) and r["pay_win"] > 0]
        if not sub:
            continue
        base = np.mean([r["high"] for r in sub])
        med = np.median([r["pay_win"] for r in sub])
        print(f"  ── {szlab}: n={len(sub)} 万車券率={base:.1%} 中央配当={med:,.0f}円 ──")
        y = [r["high"] for r in sub]
        for sig, key, neg in [("-top3_sum", "top3_sum", True), ("-r1_prob", "r1_prob", True),
                              ("-gap12", "gap12", True)]:
            x = [-r[key] if neg else r[key] for r in sub]
            print(f"    AUC({sig:<10}) = {_auc(y, x):.3f}")
        dsub = [r for r in sub if r["d_gap"] is not None]
        if dsub:
            print(f"    AUC(-d_gap    ) = {_auc([r['high'] for r in dsub], [-r['d_gap'] for r in dsub]):.3f}")
        mmsub = [r for r in sub if r["has_mkt"]]
        hm = np.mean([r["high"] for r in mmsub if r["mm"]]) if any(r["mm"] for r in mmsub) else 0
        hs = np.mean([r["high"] for r in mmsub if not r["mm"]])
        print(f"    fav_mismatch: 万車券率 不一致{hm:.1%} vs 一致{hs:.1%}"
              f"（不一致率 {np.mean([r['mm'] for r in mmsub]):.1%}）")
        # top3_sum 四分位 → 万車券率/中央配当
        qs = np.quantile([r["top3_sum"] for r in sub], [0.25, 0.5, 0.75])
        for qlab, lo, hi in [("Q1_低(波乱)", -9, qs[0]), ("Q2", qs[0], qs[1]),
                             ("Q3", qs[1], qs[2]), ("Q4_高(堅)", qs[2], 9)]:
            ss = [r for r in sub if lo <= r["top3_sum"] < hi]
            print(f"    top3_sum {qlab:<10} 万車券率{np.mean([r['high'] for r in ss]):>6.1%}"
                  f" 中央配当{np.median([r['pay_win'] for r in ss]):>9,.0f}円")

    # ---- Part2: 選別×構造 ROI ----
    t3cut = {}
    for szlab, cond in [("≤6車", lambda r: r["n"] <= 6), ("7+", lambda r: r["n"] >= 7)]:
        t3cut[szlab] = np.quantile([r["top3_sum"] for r in tr if cond(r)], 0.25)
    sels = [
        ("ALL", lambda r, c: True),
        ("RANK(SS/S/A)", lambda r, c: r["tier"] is not None),
        ("UPSET(t3s≤p25)", lambda r, c: r["top3_sum"] <= c),
        ("MM(fav不一致)", lambda r, c: r["mm"]),
        ("RANK∩MM", lambda r, c: r["tier"] is not None and r["mm"]),
        ("RANK∩UPSET", lambda r, c: r["tier"] is not None and r["top3_sum"] <= c),
    ]
    print(f"\n【Part2】選別×構造 三連単ROI（★=TRAIN&TE>100%かつTE最大除>100%）")
    for szlab, cond in [("≤6車", lambda r: r["n"] <= 6), ("7+", lambda r: r["n"] >= 7)]:
        c = t3cut[szlab]
        print(f"\n  ════ {szlab}（top3_sum p25カット={c:.3f}）════")
        for slab, scond in sels:
            print(f"  ── {slab} ──")
            print(f"  {'構造':<30}{'R':>6}{'的中率':>8}{'ROI':>8}{'95%CI':>13}{'最大除':>8}")
            for st in ["S0", "S1", "S2", "S3"]:
                res = {}
                for name, rows in [("TR", tr), ("TE", te)]:
                    sub = [r for r in rows if cond(r) and scond(r, c)]
                    res[name] = (roi_summary([r[st][0] for r in sub], [r[st][1] for r in sub]),
                                 len(sub))
                str_tr, n_tr = res["TR"]; str_te, n_te = res["TE"]
                star = " ★" if (str_tr["roi"] > 1 and str_te["roi"] > 1
                                and str_te["roi_ex_max"] > 1) else ""
                npts = {"S0": "3点", "S1": "6点", "S2": "8-10点", "S3": "10点"}[st]
                print(line(f"{st}({npts}) TR", str_tr, n_tr))
                print(line(f"{'':>10} TE", str_te, n_te, star))


if __name__ == "__main__":
    main()
