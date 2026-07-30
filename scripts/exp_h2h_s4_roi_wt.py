"""H2H(対戦表)特徴が S4(内部名S7)戦略の実ROIに効くか検証（netkeirin未活用データ調査・続き・2026-07-28）。

exp_h2h_wt.py で AUC/SS的中率(上位2車とも3着内)が改善したため、次段として実際の
S4戦略（win_probs上位3∩top3_probs上位3の重なりから軸2車選定→三連複軸2車流し5点、
wt_overlap_n/entropy/axis_sumゲート）でROIが動くかを見る。ロジックは
scripts/backfill_s7_rank_wt.py の build_rows() を、load_model(name) 済みモデルではなく
その場で学習した baseline / +h2h の2モデルへ差し替えて再現する（本番picks_historyへの
書き込みは一切行わない・読み取り専用の実験）。

win_model（軸選定に使う単勝指数側）は本番 lgbm_wt_win をそのまま両variant共通で使用する
（軸選定は win_probs と top3_probs の両方に依存するため、win_model 側に本番同様の
学習範囲があっても baseline/+h2h で共通なので delta比較としては妥当。差し替えるのは
top3(複勝)モデルのみ＝H2H特徴の効果を単離する）。

TRAIN <=2026-03-31 で学習 → EVAL 2026-04-01〜2026-07-10（TEST+FWD相当・7車のみ）で
S7選出→三連複ROIを計算。baseline vs +h2h を複数seedで比較。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (
    S7_STAKE, s7_evening_reselect, s7_field_entropy, s7_gate_label,
    s7_select_axis, s7_wt_mark3_overlap_n, s7_wt_overlap_n,
)
from exp_h2h_wt import H2H_COLS, compute_h2h

TR_TO = "2026-03-31"
EV_FROM, EV_TO = "2026-04-01", "2026-07-10"
SEEDS = [42, 7, 123]
PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)


def _load_trio_boards(race_keys: list[str]) -> dict:
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
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


def build_rows(df_eval: pd.DataFrame, top3_probs_col: str, win_probs: dict,
               date_map: dict, ne_map: dict) -> list[dict]:
    """backfill_s7_rank_wt.py::build_rows() を「学習済みモデルではなく既に
    pred_prob列を持つ df」を受け取る形へ再構成したもの（ロジックは同一）。"""
    with get_connection() as c:
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins: dict[str, list[tuple[int, int]]] = {}
        marks: dict[str, dict[int, int]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)

    df = df_eval[df_eval["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    candidates: list[dict] = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs_r = {int(r.frame_no): float(win_probs.get((rk, int(r.frame_no)), 0.0))
                       for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(getattr(r, top3_probs_col))
                      for r in g.itertuples(index=False)}
        class_map = {int(r.frame_no): r.player_class for r in g.itertuples(index=False)}
        sel = s7_select_axis(win_probs_r, top3_probs)
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
        wt_mark3_overlap_n = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n,
            "axis1_class": class_map.get(axis1), "axis2_class": class_map.get(axis2),
        })

    by_day: dict[str, list[dict]] = defaultdict(list)
    for c_ in candidates:
        by_day[c_["race_date"]].append(c_)

    rows: list[dict] = []
    for d, day_cands in by_day.items():
        for c_ in s7_evening_reselect(day_cands, [], set()):
            axis1, axis2 = c_["axis1"], c_["axis2"]
            trio = c_["trio"]
            combos = []
            for x in c_["others"]:
                key = frozenset({axis1, axis2, x})
                if key in trio:
                    combos.append(key)
            if not combos:
                continue
            rk = c_["race_key"]
            hit = c_["actual_top3"] in combos
            trio_pay = pm.get(rk, {}).get(("trio", c_["actual_top3"]), 0)
            pay = trio_pay * S7_STAKE // 100 if hit else 0
            bet = len(combos) * S7_STAKE
            gate_label = s7_gate_label(c_["wt_overlap_n"], c_.get("axis1_class"), c_.get("axis2_class"))
            rows.append({"race_date": d, "race_key": rk, "hit": int(hit),
                          "payout": pay, "bet_amount": bet, "gate_label": gate_label})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    bet = sum(r["bet_amount"] for r in rows)
    ret = sum(r["payout"] for r in rows)
    roi = ret / bet * 100 if bet else 0.0
    return dict(n=n, hits=hits, hit_rate=hits / n * 100 if n else 0.0,
                bet=bet, ret=ret, roi=roi)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=EV_TO)
    raw = compute_h2h(raw)
    df = build_features_wt(raw)
    df_fit = df[df["finish_order"] >= 1].copy()

    tr = df_fit[df_fit["race_date"] <= TR_TO].copy()
    ev = df[(df["race_date"] >= EV_FROM) & (df["race_date"] <= EV_TO)].copy()
    print(f"TRAIN {tr['race_key'].nunique()}R / EVAL {ev['race_key'].nunique()}R "
          f"({EV_FROM}〜{EV_TO})")

    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (EV_FROM, EV_TO)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (EV_FROM, EV_TO)))

    win_model = load_model("lgbm_wt_win")
    X_win = ev.reindex(columns=FEATURE_COLS_WT).fillna(0)
    ev = ev.copy()
    ev["predwin_x"] = win_model.predict_proba(X_win)[:, 1]
    win_probs = {(r.race_key, int(r.frame_no)): r.predwin_x
                 for r in ev.itertuples(index=False)}

    variants = {
        "baseline": list(FEATURE_COLS_WT),
        "+h2h": list(FEATURE_COLS_WT) + H2H_COLS,
    }

    results = {v: [] for v in variants}
    for seed in SEEDS:
        for vname, cols in variants.items():
            Xtr = tr[cols].fillna(0).values
            ytr = tr[TARGET_COL_WT].values
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(Xtr, ytr)
            ev_v = ev.copy()
            ev_v["predprob_x"] = m.predict_proba(ev_v[cols].fillna(0).values)[:, 1]
            rows = build_rows(ev_v, "predprob_x", win_probs, date_map, ne_map)
            s = summarize(rows)
            results[vname].append(s)
            print(f"  seed={seed} {vname:<10} n={s['n']:>5} 的中{s['hits']:>4} "
                  f"({s['hit_rate']:.1f}%) 投資{s['bet']:>9,} 回収{s['ret']:>9,} "
                  f"ROI {s['roi']:.1f}%")

    print("\n================ S4(S7)戦略 実ROI比較（seed平均, n_seeds=%d）================"
          % len(SEEDS))
    print(f"評価期間: {EV_FROM}〜{EV_TO}（TEST+FWD相当・7車のみ）")
    print(f"{'variant':<12}{'選出R数':>10}{'的中率':>10}{'投資額':>14}{'回収額':>14}{'ROI':>10}")
    for v in variants:
        rs = results[v]
        n_m = np.mean([r["n"] for r in rs])
        hr_m = np.mean([r["hit_rate"] for r in rs])
        bet_m = np.mean([r["bet"] for r in rs])
        ret_m = np.mean([r["ret"] for r in rs])
        roi_m = np.mean([r["roi"] for r in rs])
        roi_s = np.std([r["roi"] for r in rs])
        print(f"{v:<12}{n_m:>10.0f}{hr_m:>9.1f}%{bet_m:>14,.0f}{ret_m:>14,.0f}"
              f"{roi_m:>8.1f}%±{roi_s:.1f}pt")


if __name__ == "__main__":
    main()
