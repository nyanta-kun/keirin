"""≤6車: 「オッズの抜け具合 vs 指数の抜け具合」のギャップ（不一致）で回収率を取れるか。

ユーザーの問い(2026-06-10): モデルの本命優位度と市場の本命優位度の不一致を
レースレベルの選別シグナルにできないか。
  - per-combo value は Kelly検証(docs/analysis/09)で死亡確認（較正Pは市場価格をなぞる）。
  - 本実験は combo単位でなく「レースの本命がどれだけ抜けているか」のモデルvs市場の
    不一致 = race-level signal。≤6車では未検証の切り口。

市場側の P(top3) は trio 全盤面から逆算:
  q_i = Σ_{trioにiを含む組} 1/odds,  P_mkt(i) = 3 * q_i / Σ_j q_j   （top3確率なので合計=3）
モデル側は pred_prob(=P(top3))。同じ土俵で
  gap12_model = p1 - p2（モデル降順）
  gap12_mkt   = m1 - m2（市場降順）
  D_gap = gap12_model - gap12_mkt   （>0: モデルの方が抜けてると見る / <0: 市場の方が抜けてると見る）
  same_fav = モデル1位とオッズ1位が同一選手か
を計算し、本番戦略(SS/S/A 2軸流し3点・SS=三連単)のROIを D_gap で層別する。

model=lgbm_wt_eval(OOS) / TRAIN 2023-07〜2026-02 / TEST 2026-03〜06-08。
払戻=最終オッズ=上限値。市場オッズも最終値（朝ドリフトの留保は他実験と同じ）。
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from src.database import get_connection
from roi_robustness_wt import roi_summary


def _market_top3_probs(conn, rk):
    """trio盤面から各車の市場P(top3)を逆算。{frame: P_mkt}。盤面不足は None。"""
    rows = conn.execute(
        "SELECT combination, odds_value FROM wt_odds WHERE race_key=? AND bet_type='trio'",
        (rk,)).fetchall()
    q = {}
    n_combo = 0
    for combo, ov in rows:
        if ov is None or ov <= 0 or ov >= 9000:
            continue
        try:
            fr = [int(x) for x in re.split(r"[-=]", str(combo))]
        except ValueError:
            continue
        if len(fr) != 3:
            continue
        n_combo += 1
        w = 1.0 / ov
        for f in fr:
            q[f] = q.get(f, 0.0) + w
    if n_combo < 4 or not q:
        return None
    tot = sum(q.values())
    return {f: 3.0 * v / tot for f, v in q.items()}


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes <= 6].index)]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    with get_connection() as conn:
        for rk, g in df.groupby("race_key"):
            g = g.sort_values("pred_prob", ascending=False)
            n = len(g)
            if n < 3:
                continue
            p = g["pred_prob"].tolist()
            fr = g["frame_no"].astype(int).tolist()
            gap12 = p[0] - p[1]
            ratio = p[0] / (3.0 / n)
            # 本番ランク
            if gap12 < 0.06:
                tier = None
            elif gap12 >= 0.15 and ratio < 1.3:
                tier = "SS"
            elif gap12 >= 0.15 and ratio < 1.6:
                tier = "S"
            elif gap12 >= 0.15:
                tier = None
            else:
                tier = "A"
            # 市場P(top3)
            mkt = _market_top3_probs(conn, rk)
            if mkt is None or len(mkt) < n:
                continue
            mkt_sorted = sorted(mkt.items(), key=lambda x: -x[1])
            gap12_mkt = mkt_sorted[0][1] - mkt_sorted[1][1]
            same_fav = (mkt_sorted[0][0] == fr[0])
            d_gap = gap12 - gap12_mkt
            d_fav = p[0] - mkt.get(fr[0], 0.0)   # モデル本命に対する市場の評価差
            # 結果
            fin = g[g["finish_order"].between(1, 3)]
            if len(fin) < 3:
                continue
            order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
            top3 = frozenset(order)
            rp = pm.get(rk, {})
            # 本番戦略の採点（SS=三連単1→2→thirds / S,A=trio 2軸流し）
            pay, bet = 0, 0
            if tier:
                p1f, p2f = fr[0], fr[1]
                thirds = fr[2:5]
                bet = len(thirds) * 100
                for x in thirds:
                    if tier == "SS":
                        if order == (p1f, p2f, x):
                            pay = rp.get(("trifecta", (p1f, p2f, x)), 0); break
                    else:
                        if frozenset((p1f, p2f, x)) == top3:
                            pay = rp.get(("trio", frozenset((p1f, p2f, x))), 0); break
            rows.append({
                "tier": tier, "gap12": gap12, "gap12_mkt": gap12_mkt,
                "d_gap": d_gap, "d_fav": d_fav, "same_fav": same_fav,
                "top3_sum": p[0] + p[1] + p[2],
                "fav_bust": fr[0] not in top3,
                "pay": pay, "bet": bet,
            })
    return rows


def seg(rows, cond):
    sub = [r for r in rows if r["tier"] and cond(r)]
    return roi_summary([r["pay"] for r in sub], [r["bet"] for r in sub]), len(sub)


def line(lab, s, n, extra=""):
    return (f"  {lab:<26}{n:>5}{s['hit_rate']:>8.1%}{s['roi']:>8.0%}"
            f" [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>9.0%}{extra}")


tr = collect("2023-07-01", "2026-02-28")
te = collect("2026-03-01", "2026-06-08")
print(f"\n≤6車 本命優位度のモデルvs市場ギャップ  TRAIN {len(tr)}R / TEST {len(te)}R（最終オッズ上限値）")

# ---- Part1: D_gap は既知レバーと独立か ----
import numpy as np
d = np.array([r["d_gap"] for r in tr])
t3 = np.array([r["top3_sum"] for r in tr])
g12 = np.array([r["gap12"] for r in tr])
print(f"\n【Part1】D_gap の素性（TRAIN）")
print(f"  corr(D_gap, top3_sum) = {np.corrcoef(d, t3)[0,1]:+.3f}")
print(f"  corr(D_gap, gap12)    = {np.corrcoef(d, g12)[0,1]:+.3f}")
print(f"  corr(gap12, gap12_mkt)= {np.corrcoef(g12, np.array([r['gap12_mkt'] for r in tr]))[0,1]:+.3f}")
same = sum(1 for r in tr if r["same_fav"]) / len(tr)
print(f"  モデル1位=オッズ1位 一致率 = {same:.1%}")
# D_gap別の本命バスト率（市場が抜けてると見る側で本命は飛びにくいか）
cuts = np.quantile(d, [0.25, 0.5, 0.75])
print(f"  D_gap四分位カット = {[round(c,3) for c in cuts]}")
for lab, lo, hi in [("Q1(市場の方が抜け)", -9, cuts[0]), ("Q2", cuts[0], cuts[1]),
                    ("Q3", cuts[1], cuts[2]), ("Q4(モデルの方が抜け)", cuts[2], 9)]:
    sub = [r for r in tr if lo <= r["d_gap"] < hi]
    br = sum(1 for r in sub if r["fav_bust"]) / len(sub) if sub else 0
    print(f"    {lab:<22} 本命バスト率 {br:.1%}")

# ---- Part2: 本番戦略ROIを D_gap 四分位で層別（TRAINカット→TEST適用）----
print(f"\n【Part2】本番戦略(SS/S/A)のROI × D_gap四分位（TRAINカット→TEST・★=単調なら新レバー）")
print(f"  {'帯':<26}{'R':>5}{'的中率':>8}{'ROI':>8}{'95%CI':>13}{'最大除':>8}")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for lab, lo, hi in [("Q1(市場の方が抜け)", -9, cuts[0]), ("Q2", cuts[0], cuts[1]),
                        ("Q3", cuts[1], cuts[2]), ("Q4(モデルの方が抜け)", cuts[2], 9)]:
        s, n = seg(rows, lambda r, a=lo, b=hi: a <= r["d_gap"] < b)
        print(line(lab, s, n))

# ---- Part3: 本命人物の不一致（モデル1位≠オッズ1位）----
print(f"\n【Part3】モデル1位とオッズ1位の一致/不一致 × 本番戦略ROI")
print(f"  {'帯':<26}{'R':>5}{'的中率':>8}{'ROI':>8}{'95%CI':>13}{'最大除':>8}")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for lab, cond in [("一致(same_fav)", lambda r: r["same_fav"]),
                      ("不一致(モデル独自本命)", lambda r: not r["same_fav"])]:
        s, n = seg(rows, cond)
        print(line(lab, s, n))

# ---- Part4: top3_sum(既知レバー)と直交させた付加価値 ----
print(f"\n【Part4】既知レバー(top3_sum Q1_loose相当=下位25%)内での D_gap 追加判別力（TEST）")
t3cut = np.quantile([r["top3_sum"] for r in tr], 0.25)
print(f"  top3_sum<{t3cut:.3f}(loose) 内:")
print(f"  {'帯':<26}{'R':>5}{'的中率':>8}{'ROI':>8}{'95%CI':>13}{'最大除':>8}")
for lab, cond in [("D_gap<中央値", lambda r: r["top3_sum"] < t3cut and r["d_gap"] < cuts[1]),
                  ("D_gap≥中央値", lambda r: r["top3_sum"] < t3cut and r["d_gap"] >= cuts[1])]:
    s, n = seg(te, cond)
    print(line(lab, s, n))
print(f"  top3_sum≥{t3cut:.3f}(non-loose) 内:")
for lab, cond in [("D_gap<中央値", lambda r: r["top3_sum"] >= t3cut and r["d_gap"] < cuts[1]),
                  ("D_gap≥中央値", lambda r: r["top3_sum"] >= t3cut and r["d_gap"] >= cuts[1])]:
    s, n = seg(te, cond)
    print(line(lab, s, n))
print()
