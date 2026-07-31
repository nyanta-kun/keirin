"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

overlap=2（軸2車がWINTICKET公式◎◯と完全一致）帯に、entropyフィルタを
適用すると三連複2軸流しが黒字化するか検証する（2026-07-26「別条件検討」）。

背景: overlap=2は2026-07-21の旧検証（axis_sumベース選出のみ・entropy未使用）で
ROI75.7%の赤字区分と判明し除外された。しかしS4の主力シグナルであるentropy
（低いほど軸2車に確率集中＝残り5車拮抗）はaxis_sumとほぼ無相関(spearman≈-0.08)
と判明済みのため、overlap=2でも同じメカニズムで機能する可能性がある。

母集団: axis_sum<=S4_AXIS_SUM_MAX ∧ wt_overlap_n==2（完全一致・現状除外中）
買い目: 三連複 軸2車+残り5車のいずれか1車（5点・S4と同一構造）

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_overlap2_entropy_wt.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S4_AXIS_SUM_MAX, S4_ENTROPY_MAX, S4_STAKE, s4_select_axis, s4_wt_overlap_n
from backfill_s4_rank_wt import _load_trio_boards

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
    ("2026-04-13", "2026-07-25", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]


def build_overlap2(model_name, win_model_name, date_from, date_to):
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
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

    rows = []
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
        sel = s4_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        if axis_sum > S4_AXIS_SUM_MAX:
            continue
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        if wt_overlap_n != 2:
            continue  # 完全一致のみ対象（S4本体と非重複）

        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        combos = {frozenset({axis1, axis2, x}): trio[frozenset({axis1, axis2, x})]
                  for x in others if frozenset({axis1, axis2, x}) in trio}
        if not combos:
            continue
        hit = actual_top3 in combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        pay = trio_pay * S4_STAKE // 100 if hit else 0
        bet = len(combos) * S4_STAKE

        rows.append({
            "race_key": rk, "race_date": date_map.get(rk, ""), "entropy": ent,
            "hit": int(hit), "payout": pay, "bet_amount": bet, "trio_payout": trio_pay,
        })
    return rows


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"    {label}: n=0")
        return
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"    {label:<30} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%")


def main():
    all_rows = []
    for f, t, m, w in QUARTERS:
        rows = build_overlap2(m, w, f, t)
        all_rows.extend(rows)
        print(f"\n===== {f}〜{t}（overlap=2帯） =====")
        summarize(rows, "全体（entropyフィルタなし）")
        summarize([r for r in rows if r["entropy"] <= S4_ENTROPY_MAX], "entropy<=1.8329のみ")

    print("\n===== 全期間合算 =====")
    summarize(all_rows, "全体（entropyフィルタなし）")
    summarize([r for r in all_rows if r["entropy"] <= S4_ENTROPY_MAX], "entropy<=1.8329のみ")

    # entropy四分位（overlap=2内で独自に算出）
    ent = np.array([r["entropy"] for r in all_rows])
    qs = np.percentile(ent, [25, 50, 75])
    print(f"\noverlap=2内entropy四分位: {qs}")
    for i, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)], 1):
        sel = [r for r, v in zip(all_rows, ent) if (v > lo and (v <= hi if hi != np.inf else True))]
        summarize(sel, f"overlap=2内 entQ{i}")


if __name__ == "__main__":
    main()
