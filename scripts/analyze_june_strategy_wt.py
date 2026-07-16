"""6月実績で各戦略をシミュレーション比較

picks_history の 7PLUS_S/A レースを基準に、
現行(全相手流し)と代替戦略（B10, B12 + R filter）を遡及比較する。

比較軸:
  現行ライブ    : picks_history の実績値（7PLUS_S/A）
  BT_B0        : バックテスト再現（全相手流し・eval model）
  BT_B10       : 上位3相手 & オッズ≥10
  BT_B12_R0    : 3連単 odds上位3 全レース
  BT_B12_R6    : 3連単 odds上位3 × 波乱ゲート(top3_sum≤Q1)
  BT_B12_R9    : 3連単 odds上位3 × gap12≥0.10 & syn_min≥1.2
  BT_B12_R12   : 3連単 odds上位3 × 9車 & gap12≥0.10
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from src.database import get_connection
from roi_robustness_wt import roi_summary

JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-25"
TRAIN_END  = "2026-02-28"
GAMI_MIN   = 5.0


# ──── TRAIN で top3_sum Q1 カットを決める ────

def get_q1_cut():
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date="2023-07-01", max_date=TRAIN_END))
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()
    df = _apply_pred_prob_wt(model, df)
    t3_list = []
    for rk, g in df.groupby("race_key"):
        p = g.sort_values("pred_prob", ascending=False)["pred_prob"].tolist()
        if len(p) >= 3:
            t3_list.append(p[0]+p[1]+p[2])
    return float(np.percentile(t3_list, 25)), float(np.percentile(t3_list, 50))


# ──── 6月レース収集 ────

def collect_june(q1: float, q2: float) -> list[dict]:
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=JUNE_START, max_date=JUNE_END))
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()
    df = _apply_pred_prob_wt(model, df)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        n_entries = len(g)
        if n_entries < 7:
            continue

        dns = frozenset(g[g["finish_order"] == 0]["frame_no"].astype(int))
        g_s   = g.sort_values("pred_prob", ascending=False)
        probs = g_s["pred_prob"].tolist()
        frs   = g_s["frame_no"].astype(int).tolist()

        if len(probs) < 3:
            continue

        gap12    = probs[0] - probs[1]
        top3_sum = probs[0] + probs[1] + probs[2]
        p1, p2   = frs[0], frs[1]

        if p1 in dns or p2 in dns:
            continue

        opponents = [f for f in frs[2:] if f not in dns]
        if not opponents:
            continue

        rp = pm.get(rk, {})

        # trio オッズ (全相手)
        opp_with_odds = []
        for opp in opponents:
            o = rp.get(("trio", frozenset((p1, p2, opp))))
            if o is not None and o > 0:
                opp_with_odds.append((opp, o / 100.0))

        if not opp_with_odds:
            continue

        gami = min(o for _, o in opp_with_odds)
        if gami < GAMI_MIN:
            continue

        n_bets   = len(opp_with_odds)
        syn_min  = gami / n_bets

        # trifecta オッズ (p1→p2→相手)
        trifecta_with_odds = []
        for opp, _ in opp_with_odds:
            o = rp.get(("trifecta", (p1, p2, opp)))
            if o is not None and o > 0:
                trifecta_with_odds.append((opp, o / 100.0))

        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3  = frozenset(fin["frame_no"].astype(int))
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int))

        rows.append({
            "race_key":             rk,
            "race_date":            g["race_date"].iloc[0],
            "n_entries":            n_entries,
            "gap12":                gap12,
            "top3_sum":             top3_sum,
            "syn_min":              syn_min,
            "p1": p1, "p2": p2,
            "opp_with_odds":        opp_with_odds,
            "trifecta_with_odds":   trifecta_with_odds,
            "top3": top3, "order": order,
            # フィルターフラグ（事前計算）
            "f_s":   gap12 >= 0.10,
            "f_r6":  top3_sum <= q1,
            "f_r9":  gap12 >= 0.10 and syn_min >= 1.2,
            "f_r12": n_entries == 9 and gap12 >= 0.10,
        })

    return rows


# ──── 買い目別 (pay, bet) ────

def apply_bet(race: dict, bt: str):
    p1, p2 = race["p1"], race["p2"]
    top3   = race["top3"]
    order  = race["order"]
    owd    = race["opp_with_odds"]
    twd    = race["trifecta_with_odds"]

    if bt == "B0":
        legs = owd
    elif bt == "B10":
        legs = [(f, o) for f, o in owd[:3] if o >= 10]
    elif bt == "B12":
        legs = sorted(twd, key=lambda x: -x[1])[:3]
    else:
        return None

    if not legs:
        return None

    if bt in ("B0", "B10"):
        pay = next((o * 100 for f, o in legs if frozenset((p1, p2, f)) == top3), 0.0)
        return pay, len(legs) * 100
    else:  # B12 三連単
        pay = next((o * 100 for f, o in legs if (p1, p2, f) == order), 0.0)
        return pay, len(legs) * 100


def roi_str(rows, filt_fn, bt):
    sub = [(apply_bet(r, bt)) for r in rows if filt_fn(r)]
    sub = [x for x in sub if x is not None]
    if not sub:
        return f"{'0':>4}R  --"
    pays = [p for p, _ in sub]
    bets = [b for _, b in sub]
    n    = len(sub)
    hits = sum(1 for p in pays if p > 0)
    s    = roi_summary(pays, bets)
    flag = "★" if s["roi"] > 1.0 else " "
    return f"{n:>4}R {s['roi']:>6.0%}{flag} 的{hits/n:>5.1%} 投{sum(bets):>7,}円 払{sum(pays):>7,.0f}円"


# ──── picks_history 実績（参考値）────

def live_actual():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT rank,
                   COUNT(*) as n,
                   SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END) as hits,
                   SUM(bet_amount) as bet,
                   SUM(payout) as pay
            FROM picks_history
            WHERE race_date >= ? AND race_date <= ?
              AND route='wt' AND miwokuri=0
              AND rank IN ('7PLUS_S','7PLUS_A','7PLUS_SS')
            GROUP BY rank ORDER BY rank
        """, (JUNE_START, JUNE_END)).fetchall()
    return [dict(r) for r in rows]


def main():
    print("TRAIN の top3_sum Q1/Q2 カットを計算中...", flush=True)
    q1, q2 = get_q1_cut()
    print(f"  Q1={q1:.3f}  Q2={q2:.3f}", flush=True)

    print("6月データ収集中...", flush=True)
    june = collect_june(q1, q2)
    print(f"  収集: {len(june)}R", flush=True)

    # ──── picks_history 実績（参考）────
    live = live_actual()
    print(f"\n{'='*90}")
    print("  ◆ 6月ライブ実績（picks_history 実績値・見送り除外）")
    print(f"{'='*90}")
    total_bet, total_pay = 0, 0
    for r in live:
        roi = 100.0 * r["pay"] / r["bet"] if r["bet"] else 0
        flag = "★" if roi >= 100 else " "
        print(f"  {r['rank']:<12}{r['n']:>5}R  ROI={roi:>6.1f}%{flag}  "
              f"的中{r['hits']:>3}  投資{r['bet']:>8,}円  払戻{r['pay']:>8,}円")
        if r["rank"] in ("7PLUS_S", "7PLUS_A"):
            total_bet += r["bet"]
            total_pay += r["pay"]
    if total_bet:
        print(f"  {'S+A 合計':<12}{'':<5}  ROI={100*total_pay/total_bet:>6.1f}%   "
              f"投資{total_bet:>8,}円  払戻{total_pay:>8,}円")

    # ──── バックテスト シミュレーション ────
    STRATS = [
        ("B0  全相手流し(現行)",   lambda r: True,       "B0"),
        ("B10 上位3&≥10倍",       lambda r: True,       "B10"),
        ("B12(全R) 3連単odds上3", lambda r: True,       "B12"),
        ("B12+R6 波乱ゲート",     lambda r: r["f_r6"],  "B12"),
        ("B12+R9 gap≥0.10&syn≥1.2",lambda r: r["f_r9"],"B12"),
        ("B12+R12 9車&gap≥0.10",  lambda r: r["f_r12"], "B12"),
    ]
    print(f"\n{'='*90}")
    print("  ◆ バックテストシミュレーション（6月・eval model・最終オッズ上限値）")
    print(f"  注: 現行ライブは朝オッズで gami チェック → このBTより対象R数が異なる場合あり")
    print(f"{'='*90}")
    print(f"  {'戦略':<28}{'件数 ROI  的中率  投資  払戻':>55}")
    print(f"  {'-'*85}")
    for label, ffn, bt in STRATS:
        s = roi_str(june, ffn, bt)
        print(f"  {label:<28}  {s}")

    # ──── 週別推移（B0 vs B10 vs B12+R6）────
    print(f"\n{'='*90}")
    print("  ◆ 週別ROI推移（B0 vs B10 vs B12+R6）")
    print(f"{'='*90}")
    from collections import defaultdict
    weeks = defaultdict(list)
    for r in june:
        d = str(r["race_date"])[:10]
        w = d[:8] + ("01" if int(d[8:]) <= 7 else
                     "08" if int(d[8:]) <= 14 else
                     "15" if int(d[8:]) <= 21 else "22")
        weeks[w].append(r)

    print(f"  {'週':<12}{'B0(現行)':>22}{'B10(上位3&≥10)':>22}{'B12+R6(波乱3連単)':>22}")
    for w in sorted(weeks):
        wr = weeks[w]
        b0  = roi_str(wr, lambda r: True,      "B0")
        b10 = roi_str(wr, lambda r: True,      "B10")
        b12 = roi_str(wr, lambda r: r["f_r6"], "B12")
        print(f"  {w:<12}{b0:>25}{b10:>25}{b12:>25}")


if __name__ == "__main__":
    main()
