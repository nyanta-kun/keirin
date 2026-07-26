"""S4 entropyフィルターの再検証: 「1日6件（バッチ内）/ 10件（日次）」の件数capを
解除した生プールに対して、entropyフィルタが選定として機能するかを確認する（2026-07-26）。

背景: exp_s4_entropy_walkforward.py の検証は s4_daily_select(cap=S4_DAILY_TOP_N=10)
で「axis_sum昇順で日次上位10件」に絞った後の集合に対して entropy 四分位を見ていた。
これだと axis_sum の cap で先に弾かれた（entropy的には良いかもしれない）候補が
母集団に入らないため、「entropyがaxis_sum件数capより優れた選定基準か」を検証した
ことにならない。本スクリプトは件数capを外し、
  - tier0（wt_overlap_n==0, SS+/SS相当）: 元から無条件で全件（capなし、変更なし）
  - tier1（wt_overlap_n==1, S相当）: axis_sum<=S4_AXIS_SUM_MAX の業務要件ゲートは
    維持しつつ、件数cap（S4_HALF_CAP=6 / S4_DAILY_TOP_N=10）は外して全件を対象にする
とした生プールに対して entropy フィルタを適用し、2024Q1で固定したしきい値を
以降7四半期にブラインド適用する真のwalk-forwardで検証する。

entropy は g["pred_prob"]（モデルのtop3予測確率、出走表+選手属性+過去実績のみに
依存＝オッズ非依存）のみから計算されるため、当日朝でオッズが見えない時点でも
計算可能。

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_entropy_uncapped_wt.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S4_AXIS_SUM_MAX, S4_STAKE, s4_select_axis, s4_wt_overlap_n
from backfill_s4_rank_wt import _load_trio_boards

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401", "2024Q1"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404", "2024Q2"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407", "2024Q3"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410", "2024Q4"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501", "2025Q1"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504", "2025Q2"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507", "2025Q3"),
    ("2026-04-24", "2026-07-25", "lgbm_wt_eval", "lgbm_wt_win_eval", "2026Q2-3(直近)"),
]


def build_uncapped(model_name, win_model_name, date_from, date_to):
    """cap無し版: tier0全件 + tier1(axis_sum<=1.3のみ、件数cap無し)全件。"""
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

    candidates = []
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
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s4_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue
        if axis_sum > S4_AXIS_SUM_MAX:
            continue  # 業務要件ゲート（三連複5倍未満想定レース除外）は維持

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        if wt_overlap_n not in (0, 1):
            continue  # 完全一致(2)・マーク欠損(None)は業務要件で除外（変更なし）

        gate_label = "SS/SS+" if wt_overlap_n == 0 else "S"

        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())

        trio_combos = {}
        for x in others:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                trio_combos[key] = trio[key]
        if not trio_combos:
            continue
        hit = actual_top3 in trio_combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        pay = trio_pay * S4_STAKE // 100 if hit else 0
        bet = len(trio_combos) * S4_STAKE

        candidates.append({
            "race_date": date_map.get(rk, ""), "race_key": rk,
            "gate_label": gate_label, "wt_overlap_n": wt_overlap_n,
            "entropy": ent, "axis_sum": axis_sum,
            "hit": int(hit), "payout": pay, "bet_amount": bet, "trio_payout": trio_pay,
        })
    return candidates


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"    {label}: n=0")
        return
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    upset = sum(1 for r in rows if r["hit"] and r["trio_payout"] >= 3000)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"    {label:<28} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%  30倍+的中={upset:>3}件")


def main():
    quarter_rows = {}
    for f, t, model, win_model, label in QUARTERS:
        rows = build_uncapped(model, win_model, f, t)
        quarter_rows[label] = rows
        n = len(rows)
        by_day = defaultdict(int)
        for r in rows:
            by_day[r["race_date"]] += 1
        n_days = len(by_day)
        avg_per_day = n / n_days if n_days else 0
        print(f"\n===== {label}（{f}〜{t}） cap解除後の生プール =====")
        summarize(rows, f"全体（{n_days}日・平均{avg_per_day:.1f}件/日）")
        summarize([r for r in rows if r["gate_label"] == "SS/SS+"], "  内訳 SS/SS+(overlap0)")
        summarize([r for r in rows if r["gate_label"] == "S"], "  内訳 S(overlap1・cap解除)")

    print("\n\n========== cap解除後プールでの真のwalk-forward（2024Q1のみでしきい値決定） ==========")
    base_rows = quarter_rows.get("2024Q1", [])
    ent_base = np.array([r["entropy"] for r in base_rows])
    thresh = np.percentile(ent_base, 25)
    print(f"2024Q1(n={len(base_rows)}, cap解除後)で決定した固定entropyしきい値: {thresh:.4f}\n")

    total_below, total_above = [], []
    for f, t, model, win_model, label in QUARTERS:
        if label == "2024Q1":
            continue
        rows = quarter_rows[label]
        below = [r for r in rows if r["entropy"] <= thresh]
        above = [r for r in rows if r["entropy"] > thresh]
        below_days = len(set(r["race_date"] for r in below))
        print(f"  --- {label} ---")
        summarize(below, f"entropy<=固定しきい値（{below_days}日）")
        summarize(above, "entropy>固定しきい値")

        # 参考: 「axis_sum昇順で上位を採用」を同じ的中数まで拡大した場合との比較用に、
        # cap無しプールをaxis_sum昇順で先頭から entropy<=閾値と同じ件数だけ取った場合
        n_target = len(below)
        by_axis_sum = sorted(rows, key=lambda r: r["axis_sum"])[:n_target]
        summarize(by_axis_sum, f"[比較]axis_sum昇順上位{n_target}件（同数条件）")

        total_below.extend(below)
        total_above.extend(above)

    print("\n  === 2024Q2〜2026直近 全期間合算（cap解除・真のOOS） ===")
    summarize(total_below, "entropy<=固定しきい値")
    summarize(total_above, "entropy>固定しきい値")
    summarize(total_below + total_above, "cap解除後フィルタなし全体")

    n_below_total = len(total_below)
    n_days_total = len(set(r["race_date"] for r in (total_below + total_above)))
    print(f"\n  entropy<=しきい値の1日あたり平均採用件数（全期間平均）: "
          f"{n_below_total/n_days_total:.2f}件/日" if n_days_total else "")


if __name__ == "__main__":
    main()
