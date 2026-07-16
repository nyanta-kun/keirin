"""7+車: レース厳選 × 買い目見直し の総合グリッドサーチ

[問題]  現行 S/A ランク全相手流し (gami≥5) は eval OOS で TEST 80% と赤字。
        前実験で syn_min≥1.5 が ★再現 (TE 102%) だが件数 287R・CI [70%, 140%]。

[本実験] 2軸 — レース厳選 × 買い目構造 — を全組み合わせでグリッドサーチし
         TR/TE 両方 100%超・件数 50R+ のセルを発見する。

────────────────────────────────────────────────────────────
◆ レース厳選軸 (R_*)
────────────────────────────────────────────────────────────
 R0  ベースライン (gap12≥0.07, gami≥5)
 R1  gap12 ≥ 0.10 のみ (Sランク相当)
 R2  gap12 ≥ 0.12
 R3  gap12 ≥ 0.15
 R4  7車のみ (n=7)
 R5  9車のみ (n=9)
 R6  top3_sum ≤ Q1_loose (upset gate)
 R7  gap12≥0.10 & top3_sum≤Q2 (S × 波乱)
 R8  gap12≥0.10 & syn_min≥1.0 (S × 最悪収支保証)
 R9  gap12≥0.10 & syn_min≥1.2
 R10 gap12≥0.15 & top3_sum≤Q2
 R11 7車 & gap12≥0.10
 R12 9車 & gap12≥0.10

────────────────────────────────────────────────────────────
◆ 買い目構造軸 (B_*)
────────────────────────────────────────────────────────────
 B0  全相手流し (現行 n-2 点)
 B1  上位3相手固定 (ランク3-5 · 常時3点)
 B2  上位4相手固定 (ランク3-6 · 常時4点)
 B3  オッズ上位3点 (オッズ降順で3点購入)
 B4  オッズ上位4点
 B5  中間オッズ帯 [8, 80] 絞り
 B6  中間オッズ帯 [10, 80] 絞り
 B7  オッズ ≥ 7 のみ (低配当カット)
 B8  オッズ ≥ 10 のみ
 B9  上位3相手 & オッズ≥7
 B10 上位3相手 & オッズ≥10
 B11 3連単 p1→p2→相手3-5 (3点)
 B12 3連単 p1→p2→上位3相手 (3点)

────────────────────────────────────────────────────────────
バイアス回避 (doc18)
  ① 全エントリーでランキング (欠車含む)
  ② 7+フィルタは出走表ベース
  ③ 軸欠車=無効 / 相手欠車=その点のみ除外
  モデル: lgbm_wt_eval (週次リークなし)
  払戻: wt_odds 最終オッズ (上限値)
────────────────────────────────────────────────────────────
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from roi_robustness_wt import roi_summary

MODEL     = "lgbm_wt_eval"
GAMI_MIN  = 5.0
TRAIN_END = "2026-02-28"
TEST_END  = "2026-06-25"


# ──────────────────────────────────────────
# データ収集
# ──────────────────────────────────────────

def collect(date_from: str, date_to: str) -> list[dict]:
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))

    # バイアス②: 出走表ベース 7+フィルタ (pred_prob 前)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()

    # バイアス①: DNS 含む全エントリーで pred_prob
    df = _apply_pred_prob_wt(model, df)

    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        n_entries = len(g)
        if n_entries < 7:
            continue

        # DNS セット (バイアス③)
        dns = frozenset(g[g["finish_order"] == 0]["frame_no"].astype(int))

        # 全エントリーでランキング
        g_s   = g.sort_values("pred_prob", ascending=False)
        probs = g_s["pred_prob"].tolist()
        frs   = g_s["frame_no"].astype(int).tolist()

        if len(probs) < 3:
            continue

        gap12    = probs[0] - probs[1]
        top3_sum = probs[0] + probs[1] + probs[2]
        p1, p2   = frs[0], frs[1]

        # 軸欠車 → 無効
        if p1 in dns or p2 in dns:
            continue

        # 相手: 全エントリー top2 以外 (DNS 除外)
        opponents = [f for f in frs[2:] if f not in dns]
        if not opponents:
            continue

        # trio オッズ収集 (全相手)
        rp = pm.get(rk, {})
        opp_with_odds = []
        for opp in opponents:
            o = rp.get(("trio", frozenset((p1, p2, opp))))
            if o is not None and o > 0:
                opp_with_odds.append((opp, o / 100.0))

        if not opp_with_odds:
            continue

        # 三連単オッズ収集 (p1→p2→相手)
        trifecta_with_odds = []
        for opp, _ in opp_with_odds:
            o = rp.get(("trifecta", (p1, p2, opp)))
            if o is not None and o > 0:
                trifecta_with_odds.append((opp, o / 100.0))

        gami = min(o for _, o in opp_with_odds)
        if gami < GAMI_MIN:
            continue

        n_bets    = len(opp_with_odds)
        syn_min   = gami / n_bets
        syn_harm  = 1.0 / sum(1.0 / o for _, o in opp_with_odds)

        # 的中確認
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3  = frozenset(fin["frame_no"].astype(int))
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int))

        rows.append({
            "race_key":          rk,
            "n_entries":         n_entries,
            "gap12":             gap12,
            "top3_sum":          top3_sum,
            "gami":              gami,
            "syn_min":           syn_min,
            "syn_harm":          syn_harm,
            "p1": p1, "p2": p2,
            "opp_with_odds":     opp_with_odds,   # [(frame, odds)]  trio
            "trifecta_with_odds": trifecta_with_odds,  # [(frame, odds)]  trifecta
            "top3":              top3,
            "order":             order,
            "rp":                rp,
        })

    return rows


# ──────────────────────────────────────────
# 買い目構造 → (pays, bets) 生成
# ──────────────────────────────────────────

def bet_outcomes(race: dict, bet_type: str) -> tuple[float, int] | None:
    """買い目戦略を適用し (pay, bet) を返す。レース不採用なら None。"""
    p1, p2        = race["p1"], race["p2"]
    top3          = race["top3"]
    order         = race["order"]
    owd           = race["opp_with_odds"]     # [(frame, odds)]
    twd           = race["trifecta_with_odds"]

    if not owd:
        return None

    # ──── trio 系 ────
    if bet_type == "B0":   # 全相手流し
        legs = owd
    elif bet_type == "B1": # ランク上位3相手
        legs = owd[:3]
    elif bet_type == "B2": # ランク上位4相手
        legs = owd[:4]
    elif bet_type == "B3": # オッズ上位3点
        legs = sorted(owd, key=lambda x: -x[1])[:3]
    elif bet_type == "B4": # オッズ上位4点
        legs = sorted(owd, key=lambda x: -x[1])[:4]
    elif bet_type == "B5": # 中間オッズ [8, 80]
        legs = [(f, o) for f, o in owd if 8 <= o < 80]
    elif bet_type == "B6": # 中間オッズ [10, 80]
        legs = [(f, o) for f, o in owd if 10 <= o < 80]
    elif bet_type == "B7": # オッズ≥7
        legs = [(f, o) for f, o in owd if o >= 7]
    elif bet_type == "B8": # オッズ≥10
        legs = [(f, o) for f, o in owd if o >= 10]
    elif bet_type == "B9": # 上位3相手 & オッズ≥7
        legs = [(f, o) for f, o in owd[:3] if o >= 7]
    elif bet_type == "B10": # 上位3相手 & オッズ≥10
        legs = [(f, o) for f, o in owd[:3] if o >= 10]

    if bet_type in ("B0","B1","B2","B3","B4","B5","B6","B7","B8","B9","B10"):
        if not legs:
            return None
        bet = len(legs) * 100
        combos = {frozenset((p1, p2, f)) for f, _ in legs}
        pay = next((o * 100 for f, o in legs if frozenset((p1, p2, f)) == top3), 0.0)
        return (pay, bet)

    # ──── 三連単 系 ────
    if bet_type == "B11": # p1→p2→ランク3-5相手 (3点)
        legs_t = twd[:3]
    elif bet_type == "B12": # p1→p2→オッズ上位3相手
        legs_t = sorted(twd, key=lambda x: -x[1])[:3]
    else:
        return None

    if not legs_t:
        return None
    bet = len(legs_t) * 100
    pay = next((o * 100 for f, o in legs_t if (p1, p2, f) == order), 0.0)
    return (pay, bet)


# ──────────────────────────────────────────
# レース厳選フィルター (Q値は TRAIN で計算)
# ──────────────────────────────────────────

def build_race_filters(train: list[dict]) -> dict[str, callable]:
    t3 = [r["top3_sum"] for r in train]
    q1 = float(np.percentile(t3, 25))
    q2 = float(np.percentile(t3, 50))

    return {
        "R0":  lambda r: True,
        "R1":  lambda r: r["gap12"] >= 0.10,
        "R2":  lambda r: r["gap12"] >= 0.12,
        "R3":  lambda r: r["gap12"] >= 0.15,
        "R4":  lambda r: r["n_entries"] == 7,
        "R5":  lambda r: r["n_entries"] == 9,
        "R6":  lambda r: r["top3_sum"] <= q1,
        "R7":  lambda r: r["gap12"] >= 0.10 and r["top3_sum"] <= q2,
        "R8":  lambda r: r["gap12"] >= 0.10 and r["syn_min"] >= 1.0,
        "R9":  lambda r: r["gap12"] >= 0.10 and r["syn_min"] >= 1.2,
        "R10": lambda r: r["gap12"] >= 0.15 and r["top3_sum"] <= q2,
        "R11": lambda r: r["n_entries"] == 7 and r["gap12"] >= 0.10,
        "R12": lambda r: r["n_entries"] == 9 and r["gap12"] >= 0.10,
    }


BET_TYPES = ["B0","B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12"]


# ──────────────────────────────────────────
# ROI 計算
# ──────────────────────────────────────────

def calc_roi(races: list[dict], race_filt, bet_type: str):
    pays, bets = [], []
    for r in races:
        if not race_filt(r):
            continue
        res = bet_outcomes(r, bet_type)
        if res is None:
            continue
        pay, bet = res
        pays.append(pay)
        bets.append(bet)
    if not pays:
        return None, 0
    return roi_summary(pays, bets), len(pays)


# ──────────────────────────────────────────
# 出力
# ──────────────────────────────────────────

def fmt(s, n):
    if s is None or n == 0:
        return f"{'0':>5}R    -- "
    flag = "★" if s["roi"] > 1.0 else " "
    return f"{n:>5}R {s['roi']:>6.0%}{flag} 除{s['roi_ex_max']:>5.0%}"


def flag_cell(str_, ntr, ste, nte):
    if str_ is None or ste is None or nte < 30:
        return "小標本" if nte < 30 else ""
    if str_["roi"] > 1.0 and ste["roi"] > 1.0:
        if ste["ci_lo"] > 1.0:
            return "★★再現+CI"
        return "★再現"
    return ""


def main():
    print("collecting TRAIN...", flush=True)
    tr = collect("2023-07-01", TRAIN_END)
    print(f"  TRAIN: {len(tr)}R", flush=True)
    print("collecting TEST...", flush=True)
    te = collect("2023-03-01", TEST_END)
    # TEST は 2026-03-01 以降のみ
    te_hold = [r for r in te if r["race_key"][:10] >= "2026-03-01"]
    print(f"  TEST (hold): {len(te_hold)}R", flush=True)

    filters = build_race_filters(tr)

    # ──────────────────────────────────────────
    # グリッドサーチ: ★再現 セルを抽出
    # ──────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  グリッドサーチ: TR/TE 両方 100%超 & TE ≥ 30R の全セル (払戻=最終オッズ上限値・eval OOS)")
    print(f"{'='*110}")
    print(f"  {'レース厳選':<8}{'買い目':<6}{'TR':>32}{'TE':>32}{'判定':>12}")
    print(f"  {'-'*108}")

    hits = []
    for rk, rfn in filters.items():
        for bt in BET_TYPES:
            str_, ntr = calc_roi(tr, rfn, bt)
            ste, nte  = calc_roi(te_hold, rfn, bt)
            flag = flag_cell(str_, ntr, ste, nte)
            if flag:
                hits.append((rk, bt, str_, ntr, ste, nte, flag))
                print(f"  {rk:<8}{bt:<6}{fmt(str_, ntr):>35}{fmt(ste, nte):>35}  {flag}")

    if not hits:
        print("  (★再現セル なし — 以下は TE ROI 降順 top20 を掲載)")

    # ──────────────────────────────────────────
    # TOP 20 (再現なしでも参考として TE ROI 降順)
    # ──────────────────────────────────────────
    all_cells = []
    for rk, rfn in filters.items():
        for bt in BET_TYPES:
            str_, ntr = calc_roi(tr, rfn, bt)
            ste, nte  = calc_roi(te_hold, rfn, bt)
            if ste and nte >= 30:
                all_cells.append((rk, bt, str_, ntr, ste, nte))

    all_cells.sort(key=lambda x: -x[4]["roi"])

    print(f"\n{'='*110}")
    print(f"  TE ROI 降順 TOP 20（TE ≥ 30R・★は TR/TE 両方 100%超）")
    print(f"{'='*110}")
    print(f"  {'#':>3}{'レース厳選':<10}{'買い目':<8}{'TR':>32}{'TE':>32}{'判定':>12}")
    print(f"  {'-'*108}")
    for i, (rk, bt, str_, ntr, ste, nte) in enumerate(all_cells[:20], 1):
        flag = flag_cell(str_, ntr, ste, nte)
        print(f"  {i:>3} {rk:<10}{bt:<8}{fmt(str_, ntr):>32}{fmt(ste, nte):>32}  {flag}")

    # ──────────────────────────────────────────
    # 買い目別サマリ (R0 ベースライン固定)
    # ──────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  買い目構造別サマリー（R0 ベースライン・全7+車・gami≥5）")
    print(f"{'='*110}")
    print(f"  {'買い目':<8}{'内容':<30}{'TR':>32}{'TE':>32}")
    BET_DESC = {
        "B0": "全相手流し (現行)",
        "B1": "ランク上位3点",
        "B2": "ランク上位4点",
        "B3": "オッズ上位3点",
        "B4": "オッズ上位4点",
        "B5": "中間オッズ [8,80]",
        "B6": "中間オッズ [10,80]",
        "B7": "オッズ≥7のみ",
        "B8": "オッズ≥10のみ",
        "B9": "上位3相手 & ≥7倍",
        "B10":"上位3相手 & ≥10倍",
        "B11":"3連単 p1→p2→rank3-5",
        "B12":"3連単 p1→p2→odds上位3",
    }
    rfn = filters["R0"]
    for bt in BET_TYPES:
        str_, ntr = calc_roi(tr, rfn, bt)
        ste, nte  = calc_roi(te_hold, rfn, bt)
        flag = "★" if (str_ and ste and str_["roi"] > 1.0 and ste["roi"] > 1.0) else ""
        print(f"  {bt:<8}{BET_DESC.get(bt,''):<30}{fmt(str_, ntr):>32}{fmt(ste, nte):>32}  {flag}")

    # ──────────────────────────────────────────
    # レース厳選別サマリ (B1=ランク上位3点 固定)
    # ──────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  レース厳選別サマリー（B1 ランク上位3点固定）")
    print(f"{'='*110}")
    FILTER_DESC = {
        "R0": "ベースライン",
        "R1": "gap12≥0.10",
        "R2": "gap12≥0.12",
        "R3": "gap12≥0.15",
        "R4": "7車のみ",
        "R5": "9車のみ",
        "R6": "top3_sum≤Q1",
        "R7": "gap12≥0.10 & ≤Q2",
        "R8": "gap12≥0.10 & syn_min≥1.0",
        "R9": "gap12≥0.10 & syn_min≥1.2",
        "R10":"gap12≥0.15 & ≤Q2",
        "R11":"7車 & gap12≥0.10",
        "R12":"9車 & gap12≥0.10",
    }
    for rk in filters:
        rfn = filters[rk]
        str_, ntr = calc_roi(tr, rfn, "B1")
        ste, nte  = calc_roi(te_hold, rfn, "B1")
        flag = flag_cell(str_, ntr, ste, nte)
        print(f"  {rk:<6}{FILTER_DESC.get(rk,''):<28}{fmt(str_, ntr):>32}{fmt(ste, nte):>32}  {flag}")

    print(f"\n{'='*110}")
    print("  ★★再現+CI = TR>100% & TE>100% & TE CI下限>100%  (最強根拠)")
    print("  ★再現    = TR>100% & TE>100% & n≥30  (有望)")
    print("  ※ 最終オッズ上限値。朝→確定ドリフトで実運用はさらに下振れ。")
    print("     採否は picks_history(live実測)で前向きに確認すること。")


if __name__ == "__main__":
    main()
