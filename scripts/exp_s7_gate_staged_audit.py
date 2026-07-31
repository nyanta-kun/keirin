"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

さらに本スクリプトは S7_AXIS_SUM_MAX = 1.3 をハードコードしているが、
現行 src/strategy_wt.py:328 の S7_AXIS_SUM_MAX は 1.5 に変更済みであり、
本スクリプトの閾値定義は現行仕様と食い違っている（この点でも数値は再現不可）。

S7ゲートの段階的honest再検証（2026-07-29・全ランクROI<100%発覚を受けた監査）。

axis_sum/entropy/mark3ゲートが導入された順に、全期間(2024-01-01〜)を通して
honestにROIを再検証する。build_rows()と同じcandidate生成ロジックを再利用し、
ゲート適用前の「全候補（hit/payout計算済み）」を一度だけ構築してから、
複数のゲート組み合わせを事後フィルタで比較する（モデル予測の再計算を避ける）。

変遷順:
  A) base       : wt_overlap_n in {0,1} のみ（2026-07-21時点の設計・axis_sum等なし）
  B) +axis_sum  : A + axis_sum<=1.3（2026-07-24導入）
  C) +entropy   : B + entropy<=1.8329（2026-07-26導入）
  D) +mark3     : C + mark3<=1（2026-07-27導入・現行S7の完全な定義）

日次S7_DAILY_CAPのトリムは含めない（honest全期間では「ほぼ発火しない安全網」と
記録されており、段階比較の単純化のため意図的に省略）。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    S7_STAKE, s7_field_entropy, s7_select_axis, s7_wt_mark3_overlap_n, s7_wt_overlap_n,
)

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
    ("2026-04-13", "2026-07-28", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

S7_AXIS_SUM_MAX = 1.3
S7_ENTROPY_MAX = 1.8329
S7_MARK3_OVERLAP_MAX = 1


def _load_trio_boards(race_keys):
    import re
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def build_scored_candidates(model_name, date_from, date_to, win_model_name):
    """ゲート適用前の全候補（hit/payout計算済み）を返す。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rt_map = dict(c.execute(
            "SELECT race_key, race_type FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    out = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue
        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s7_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = s7_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue
        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3 = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        combos = []
        combo_odds = {}
        for x in others:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combos.append(key)
                combo_odds[key] = trio[key]
        if not combos:
            continue
        hit = actual_top3 in combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        pay = trio_pay * S7_STAKE // 100 if hit else 0
        bet = len(combos) * S7_STAKE

        out.append({
            "race_key": rk, "axis_sum": axis_sum, "entropy": entropy,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3,
            "hit": int(hit), "payout": pay, "bet_amount": bet,
            "race_type": rt_map.get(rk), "actual_top3": actual_top3,
            "combo_odds": combo_odds, "trio_pay": trio_pay,
        })
    return out


def summarize(cands):
    n = len(cands)
    hits = sum(c["hit"] for c in cands)
    bet = sum(c["bet_amount"] for c in cands)
    pay = sum(c["payout"] for c in cands)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "exp_cache" / "s7_staged_audit_candidates.pkl"


def main():
    import pickle
    if CACHE_PATH.exists():
        print(f"[audit] キャッシュから読み込み: {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as f:
            all_cands = pickle.load(f)
    else:
        all_cands = []
        for date_from, date_to, eval_model, win_model in QUARTERS:
            print(f"[audit] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
            cands = build_scored_candidates(eval_model, date_from, date_to, win_model)
            print(f"[audit]   raw candidates(axis選定成功・盤面あり): {len(cands)}", flush=True)
            all_cands.extend(cands)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(all_cands, f)
        print(f"[audit] キャッシュ保存: {CACHE_PATH}", flush=True)

    print(f"\n[audit] ===== 全期間 raw candidates合計: {len(all_cands)} =====\n")

    variants = {
        "A) base(wt_overlap in {0,1}のみ)":
            lambda c: c["wt_overlap_n"] in (0, 1),
        "B) A+axis_sum<=1.3":
            lambda c: c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX,
        "C) B+entropy<=1.8329":
            lambda c: (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
                       and c["entropy"] <= S7_ENTROPY_MAX),
        "D) C+mark3<=1(現行S7の完全定義)":
            lambda c: (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
                       and c["entropy"] <= S7_ENTROPY_MAX
                       and (c["wt_mark3_overlap_n"] is not None
                            and c["wt_mark3_overlap_n"] <= S7_MARK3_OVERLAP_MAX)),
    }

    print(f"{'variant':<38}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    d_filt = variants["D) C+mark3<=1(現行S7の完全定義)"]
    d_sel = [c for c in all_cands if d_filt(c)]
    for name, filt in variants.items():
        sel = [c for c in all_cands if filt(c)]
        n, hits, hitrate, bet, pay, roi = summarize(sel)
        mark = " ★100%超" if roi > 100 else ""
        print(f"{name:<38}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    # ── 却下済み案の再検証: 最終オッズ下限カット（D構成に対して）──
    print("\n[audit] --- 却下案再検証1: D構成へのオッズ下限カット（最終オッズ・honest post-hoc）---")

    def _cut_summary(cut: float):
        n = hits = bet = pay = 0
        for c in d_sel:
            kept = {k: v for k, v in c["combo_odds"].items() if v > cut}
            if not kept:
                continue
            n += 1
            b = len(kept) * S7_STAKE
            bet += b
            if c["actual_top3"] in kept:
                hits += 1
                pay += c["trio_pay"] * S7_STAKE // 100
        roi = pay / bet * 100 if bet else 0.0
        hitrate = hits / n * 100 if n else 0.0
        return n, hits, hitrate, bet, pay, roi

    for cut in (0, 3, 5, 7, 10, 15, 20):
        n, hits, hitrate, bet, pay, roi = _cut_summary(cut)
        mark = " ★100%超" if roi > 100 else ""
        print(f"  cut>{cut:>3}倍{'':<25}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    # ── 却下済み案の再検証: race_type別（D構成に対して）──
    print("\n[audit] --- 却下案再検証2: race_type別ROI(D構成・n>=20のみ表示) ---")
    by_rt: dict = defaultdict(list)
    for c in d_sel:
        by_rt[c.get("race_type") or "(NULL)"].append(c)
    rt_rows = []
    for rt, cs in by_rt.items():
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        if n >= 20:
            rt_rows.append((rt, n, hitrate, bet, pay, roi))
    rt_rows.sort(key=lambda r: -r[5])
    for rt, n, hitrate, bet, pay, roi in rt_rows:
        mark = " ★100%超" if roi > 100 else ""
        print(f"  {str(rt):<20}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")


if __name__ == "__main__":
    main()
