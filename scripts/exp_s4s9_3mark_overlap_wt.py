"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

S4/S9の軸2車がWINTICKET公式印◎◯△（mark1/2/3）のうち2つと一致する場合、
市場人気と重なり払戻が低くなりやすいのではという仮説を検証する（2026-07-26）。

現行のwt_overlap_n（s4_wt_overlap_n）は◎(mark1)/◯(mark2)のみを見ており、
△(mark3)は考慮していない。そのため現行「重なり1(S)」の中には、片方の軸が
◎か◯、もう片方が△という「実質2/3が公式印と一致」のレースが紛れている
可能性がある。本スクリプトはこの「軸2車のうち2車が◎◯△のいずれかと一致」
という条件（新定義）でROI・payoutを再集計する。

母集団: S4(7車)+S9(9車)の現行本番選出済み候補（axis_sum<=1.3 ∧ wt_overlap_n∈{0,1}
        ∧ entropy<=各S4/S9のENTROPY_MAX＝現行ライブの実採用条件と同一）
使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4s9_3mark_overlap_wt.py
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
    S4_AXIS_SUM_MAX, S4_ENTROPY_MAX, S4_STAKE, S9_ENTROPY_MAX, S9_STAKE,
    s4_field_entropy, s4_select_axis, s4_wt_overlap_n,
)
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


def build(model_name, win_model_name, date_from, date_to, n_car, entropy_max, stake):
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
        rksN = [rk for rk, ne in ne_map.items() if ne and int(ne) == n_car]
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
        if ne_map.get(rk) != n_car or len(g) != n_car:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != n_car:
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
        if len(others) != n_car - 2:
            continue

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)  # △(mark3)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        if wt_overlap_n not in (0, 1):
            continue  # 完全一致(2)・マーク欠損(None)は現行同様除外

        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        if ent > entropy_max:
            continue  # 現行ライブと同一のentropyゲート

        # 新定義: 軸2車のうち◎◯△(mark1/2/3)のいずれかと一致する車の数
        marked_set = {x for x in (wt_honmei, wt_taikou, wt_ana) if x is not None}
        overlap3 = len({axis1, axis2} & marked_set)

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        combos = {frozenset({axis1, axis2, x}): trio[frozenset({axis1, axis2, x})]
                  for x in others if frozenset({axis1, axis2, x}) in trio}
        if not combos:
            continue
        hit = actual_top3 in combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        pay = trio_pay * stake // 100 if hit else 0
        bet = len(combos) * stake

        rows.append({
            "race_key": rk, "race_date": date_map.get(rk, ""), "n_car": n_car,
            "wt_overlap_n": wt_overlap_n, "overlap3": overlap3,
            "hit": int(hit), "payout": pay, "bet_amount": bet, "trio_payout": trio_pay,
        })
    return rows


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label}: n=0")
        return
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"  {label:<40} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%")


def main():
    all_rows = []
    for f, t, m, w in QUARTERS:
        rows7 = build(m, w, f, t, n_car=7, entropy_max=S4_ENTROPY_MAX, stake=S4_STAKE)
        rows9 = build(m, w, f, t, n_car=9, entropy_max=S9_ENTROPY_MAX, stake=S9_STAKE)
        all_rows.extend(rows7)
        all_rows.extend(rows9)
        print(f"{f}~{t}: 7車n={len(rows7)} 9車n={len(rows9)}", flush=True)

    print(f"\n===== 全期間合算 n={len(all_rows)}（S4+S9・現行ライブ採用条件と同一母集団） =====")
    summarize(all_rows, "全体")
    summarize([r for r in all_rows if r["overlap3"] >= 2], "軸2車のうち2車が◎◯△のいずれかと一致(新定義)")
    summarize([r for r in all_rows if r["overlap3"] < 2], "上記以外")

    # 現行wt_overlap_n別のクロス集計（overlap3との関係を可視化）
    print("\n--- 現行wt_overlap_n × 新定義overlap3 クロス表 ---")
    for ov in (0, 1):
        sub = [r for r in all_rows if r["wt_overlap_n"] == ov]
        print(f" wt_overlap_n={ov} (現行{'SS/SS+' if ov==0 else 'S'}相当): n={len(sub)}")
        for ov3 in (0, 1, 2):
            s3 = [r for r in sub if r["overlap3"] == ov3]
            summarize(s3, f"   うちoverlap3={ov3}")

    # 高払戻トップ5
    hits_sorted = sorted([r for r in all_rows if r["hit"]], key=lambda r: -r["trio_payout"])
    print("\n--- 全体 払戻トップ5 ---")
    for r in hits_sorted[:5]:
        print(f"  {r['race_date']} {r['race_key']} {r['n_car']}車 wt_overlap_n={r['wt_overlap_n']} "
              f"overlap3={r['overlap3']} payout={r['trio_payout']:,}円")

    print("\n--- overlap3>=2（新定義・市場人気重なり）のみ 払戻トップ5 ---")
    sub_hits = sorted([r for r in all_rows if r["hit"] and r["overlap3"] >= 2],
                       key=lambda r: -r["trio_payout"])
    for r in sub_hits[:5]:
        print(f"  {r['race_date']} {r['race_key']} {r['n_car']}車 wt_overlap_n={r['wt_overlap_n']} "
              f"payout={r['trio_payout']:,}円")


if __name__ == "__main__":
    main()
