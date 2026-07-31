"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

S1(win軸1着固定×3着内モデル相手2車)のS1w_gate通過候補内で、フィールド全体の
指数エントロピー(s4_field_entropy)が高配当予測に使えるか検証する
（2026-07-26・「S1は高配当狙いモデル・オッズを見ずに低配当レースを除外したい」
というユーザー要望への対応）。

背景: S1は既にaxis_win_prob<=0.50・axis_class not in {S1,A1}という低配当除外
ゲートを持つ（2026-07-21/22導入・honest全期間で検証済み）。本スクリプトは
S4/S9で有効だった「フィールド全体のentropy」がS1w_gate通過後の母集団内でも
追加の高配当シグナルになるかを、既存ゲートとは独立に検証する。

母集団: s1w_gate通過候補（axis_win_prob<=0.50 ∧ axis_class not in {S1,A1} ∧
        top3_gap>=0.15）
目的変数: 的中時trifecta_payout>=3000円(30倍以上)
使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s1w_entropy_wt.py
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
from src.strategy_wt import (
    S1W_AXIS_WIN_PROB_MAX, S1W_DENY_AXIS_CLASS, S1W_STAKE, S1W_TOP3_GAP_MIN,
    s1w_gate, s1w_select, s4_field_entropy,
)

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401", "2024Q1"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404", "2024Q2"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407", "2024Q3"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410", "2024Q4"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501", "2025Q1"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504", "2025Q2"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507", "2025Q3"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3", "2025Q4"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2", "2026Qa"),
    ("2026-04-13", "2026-07-25", "lgbm_wt_eval", "lgbm_wt_win_eval", "2026Qb"),
]


def build_s1_gated(model_name, win_model_name, date_from, date_to):
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
        fins = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue
        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s1w_select(win_probs, top3_probs)
        if sel is None:
            continue
        axis, p1, p2, top3_gap = sel
        axis_win_prob = win_probs[axis]
        axis_class = class_map.get(axis)
        if not s1w_gate(top3_gap, axis_win_prob, axis_class):
            continue

        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())

        order3 = tuple(fno for _, fno in fin[:3])
        hit = order3[0] == axis and {order3[1], order3[2]} == {p1, p2}
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        pay = trifecta_pay * S1W_STAKE // 100 if hit else 0
        bet = 2 * S1W_STAKE

        rows.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "entropy": ent, "top3_gap": top3_gap, "axis_win_prob": axis_win_prob,
            "hit": int(hit), "payout": pay, "bet_amount": bet, "trifecta_payout": trifecta_pay,
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
    upset = sum(1 for r in rows if r["hit"] and r["trifecta_payout"] >= 3000)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"    {label:<28} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%  30倍+的中={upset}件")


def main():
    all_by_q = {}
    for f, t, m, w, label in QUARTERS:
        rows = build_s1_gated(m, w, f, t)
        all_by_q[label] = rows
        print(f"\n===== {label}（{f}〜{t}） S1w_gate通過: n={len(rows)} =====")
        summarize(rows, "全体")
        if len(rows) < 20:
            continue
        ent = np.array([r["entropy"] for r in rows])
        qs = np.percentile(ent, [25, 50, 75])
        for i, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)], 1):
            sel = [r for r, v in zip(rows, ent) if (v > lo and (v <= hi if hi != np.inf else True))]
            summarize(sel, f"  entropy Q{i}")

    all_rows = [r for rows in all_by_q.values() for r in rows]
    print(f"\n===== 全期間合算 n={len(all_rows)} =====")
    summarize(all_rows, "全体")

    print("\n===== 真のwalk-forward（2024Q1のみでentropyしきい値決定→ブラインド適用） =====")
    q1 = all_by_q["2024Q1"]
    if len(q1) >= 20:
        for pct, label_pct in [(25, "下位25%(低entropy)"), (75, "上位25%(高entropy)")]:
            thresh = np.percentile([r["entropy"] for r in q1], pct)
            print(f"\n--- 2024Q1(n={len(q1)})で決定したentropy{label_pct}しきい値: {thresh:.4f} ---")
            total_below, total_above = [], []
            for label, rows in all_by_q.items():
                if label == "2024Q1" or len(rows) < 5:
                    continue
                below = [r for r in rows if r["entropy"] <= thresh]
                above = [r for r in rows if r["entropy"] > thresh]
                total_below.extend(below)
                total_above.extend(above)
            summarize(total_below, f"  entropy<=しきい値（全期間合算）")
            summarize(total_above, f"  entropy>しきい値（全期間合算）")


if __name__ == "__main__":
    main()
