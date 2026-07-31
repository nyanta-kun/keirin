"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

9車立てレースへのS4相当ロジック拡張・キャリブレーション（2026-07-26）。

背景: 2026-08開催予定の「ドリームレース」（S級・毎年8月・過去3回とも9車立て）
をターゲットに含めるため、7車専用だったS4のロジック（軸選定・axis_sum・
entropy）を9車立てに拡張する。9車は全体の8.0%(約6.5件/日・7車に次ぐ規模)。

7車のS4_AXIS_SUM_MAX=1.3・S4_ENTROPY_MAX=1.8329は7車の分布から較正した値の
ため、9車にそのまま流用せず9車自身の分布で再較正する
（entropy理論上限: 7車=ln(7)=1.946 / 9車=ln(9)=2.197）。

買い目 = 三連複 軸2車 + 残り7車のいずれか1車（7点・S4の5点より高コスト）

段階:
  1. 9車の軸選定(s4_select_axis)対象候補の生分布（axis_sum・entropy）を確認
  2. axis_sum閾値なしでの素のROI（軸2車+残り7車流し）を四半期別に確認
  3. axis_sum・entropyそれぞれの四分位別ROIを確認（7車と同じ設計思想が通用するか）
  4. 2024Q1のみで閾値を決定→残り9四半期にブラインド適用する真のwalk-forward

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_9car_calibration_wt.py
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
from src.strategy_wt import s4_select_axis, s4_wt_overlap_n
from backfill_s4_rank_wt import _load_trio_boards

N_CAR = 9
STAKE = 100

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


def build_9car(model_name, win_model_name, date_from, date_to):
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
        rksN = [rk for rk, ne in ne_map.items() if ne and int(ne) == N_CAR]
        fins, marks = {}, {}
        for i in range(0, len(rksN), 900):
            chunk = rksN[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rksN))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != N_CAR or len(g) != N_CAR:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != N_CAR:
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
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != N_CAR - 2:
            continue

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)

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
        pay = trio_pay * STAKE // 100 if hit else 0
        bet = len(combos) * STAKE

        rows.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis_sum": axis_sum, "entropy": ent, "wt_overlap_n": wt_overlap_n,
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
    print(f"    {label:<28} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%")


def main():
    all_by_q = {}
    for f, t, m, w, label in QUARTERS:
        rows = build_9car(m, w, f, t)
        all_by_q[label] = rows
        n_days = (__import__("datetime").date.fromisoformat(t) -
                  __import__("datetime").date.fromisoformat(f)).days + 1
        print(f"\n===== {label}（{f}〜{t}） 9車軸選定候補: n={len(rows)} ({len(rows)/n_days:.2f}件/日) =====")
        summarize(rows, "全体（フィルタなし）")
        for ov in (0, 1, 2, None):
            summarize([r for r in rows if r["wt_overlap_n"] == ov], f"  wt_overlap_n={ov}")

    all_rows = [r for rows in all_by_q.values() for r in rows]
    print(f"\n===== 全期間合算（9車・フィルタなし） n={len(all_rows)} =====")
    summarize(all_rows, "全体")

    # axis_sum分布・四分位
    axis_sum_vals = np.array([r["axis_sum"] for r in all_rows])
    print(f"\naxis_sum分布: min={axis_sum_vals.min():.3f} median={np.median(axis_sum_vals):.3f} "
          f"max={axis_sum_vals.max():.3f}")
    qs_as = np.percentile(axis_sum_vals, [10, 25, 50, 75, 90])
    print(f"パーセンタイル(10/25/50/75/90): {qs_as}")
    print("axis_sum四分位別ROI:")
    for i, (lo, hi) in enumerate([(-np.inf, qs_as[1]), (qs_as[1], qs_as[2]), (qs_as[2], qs_as[3]), (qs_as[3], np.inf)], 1):
        sel = [r for r, v in zip(all_rows, axis_sum_vals) if (v > lo and (v <= hi if hi != np.inf else True))]
        summarize(sel, f"  axis_sum Q{i}")

    # entropy分布・四分位（wt_overlap in (0,1)のみ、7車S4と同条件）
    base = [r for r in all_rows if r["wt_overlap_n"] in (0, 1)]
    ent_vals = np.array([r["entropy"] for r in base])
    print(f"\nentropy分布(wt_overlap∈{{0,1}}のみ、n={len(base)}): "
          f"min={ent_vals.min():.3f} median={np.median(ent_vals):.3f} max={ent_vals.max():.3f} "
          f"(理論上限ln(9)={np.log(9):.3f})")
    qs_e = np.percentile(ent_vals, [10, 25, 50, 75, 90])
    print(f"パーセンタイル(10/25/50/75/90): {qs_e}")
    print("entropy四分位別ROI:")
    for i, (lo, hi) in enumerate([(-np.inf, qs_e[1]), (qs_e[1], qs_e[2]), (qs_e[2], qs_e[3]), (qs_e[3], np.inf)], 1):
        sel = [r for r, v in zip(base, ent_vals) if (v > lo and (v <= hi if hi != np.inf else True))]
        summarize(sel, f"  entropy Q{i}")

    # 真のwalk-forward: 2024Q1のみでentropy25%点のしきい値を決定→残り9四半期へブラインド適用
    print("\n===== 真のwalk-forward（2024Q1のみでentropyしきい値決定→ブラインド適用） =====")
    q1_base = [r for r in all_by_q["2024Q1"] if r["wt_overlap_n"] in (0, 1)]
    if len(q1_base) >= 20:
        thresh = np.percentile([r["entropy"] for r in q1_base], 25)
        print(f"2024Q1(n={len(q1_base)})で決定したentropyしきい値: {thresh:.4f}")
        total_below, total_above = [], []
        for label, rows in all_by_q.items():
            if label == "2024Q1":
                continue
            base_q = [r for r in rows if r["wt_overlap_n"] in (0, 1)]
            below = [r for r in base_q if r["entropy"] <= thresh]
            above = [r for r in base_q if r["entropy"] > thresh]
            print(f"  --- {label} ---")
            summarize(below, "entropy<=しきい値")
            summarize(above, "entropy>しきい値")
            total_below.extend(below)
            total_above.extend(above)
        print("\n  === 全期間合算（2024Q1除く・真のOOS） ===")
        summarize(total_below, "entropy<=しきい値")
        summarize(total_above, "entropy>しきい値")
    else:
        print("2024Q1のデータ不足のため実施不可")


if __name__ == "__main__":
    main()
