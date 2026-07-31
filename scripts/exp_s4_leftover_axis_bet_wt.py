"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

S4のentropyゲートで除外される「堅いレース」(entropy>S4_ENTROPY_MAX)を
別の買い目（二車複axis1-axis2 / ワイドaxis1-axis2）で拾えないか検証する
（2026-07-26・「重ならない新セグメント」調査）。

背景: S4(三連複2軸流し)はentropy<=1.8329(母集団下位25%)のみを対象とし、
残り75%（entropy>1.8329・「軸は堅いが三連複配当が薄いレース」帯）は完全に
捨てている。この帯では軸1・軸2自体の的中率は高いと推測されるため、
三連複より的中率の高い買い目（二車複・ワイド、いずれもaxis1-axis2の1点のみ・
5点流しより安い）に切り替えれば、entropy<=1.8329帯と重ならない別の
プラス収支セグメントになる可能性がある。

母集団: axis_sum<=S4_AXIS_SUM_MAX ∧ wt_overlap_n∈{0,1} ∧ entropy>S4_ENTROPY_MAX
       （S4本体が捨てている帯そのもの）
買い目: 二車複(quinella) axis1=axis2（1点100円） / ワイド(quinellaPlace) 同型（1点100円）

使い方:
  PYTHONPATH=. .venv/bin/python scripts/exp_s4_leftover_axis_bet_wt.py
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S4_AXIS_SUM_MAX, S4_ENTROPY_MAX, s4_select_axis, s4_wt_overlap_n

STAKE = 100  # 円/点（1点のみ）

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


def _load_pair_odds(race_keys: list[str]) -> dict:
    """{race_key: {bet_type: {frozenset(int,int): odds_value}}}"""
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('quinella','quinellaPlace') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, ov in c.execute(q, chunk):
                try:
                    fv = float(ov) if ov is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 2:
                    out.setdefault(rk, {}).setdefault(bt, {})[parts] = fv
    return out


def build_leftover(model_name, win_model_name, date_from, date_to):
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
    pair_odds = _load_pair_odds(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
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

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_overlap_n = s4_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        if wt_overlap_n not in (0, 1):
            continue

        p = g["pred_prob"].to_numpy(dtype=float)
        s = p / p.sum() if p.sum() > 0 else p
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        if ent <= S4_ENTROPY_MAX:
            continue  # S4本体の対象（現行採用帯）はスキップ＝完全に非重複

        top1 = fin[0][1]
        top2 = fin[1][1] if len(fin) > 1 else None
        top3_frames = frozenset(fno for _, fno in fin[:3])
        axis_pair = frozenset({axis1, axis2})

        quin_hit = axis_pair == frozenset({top1, top2}) if top2 is not None else False
        wide_hit = axis_pair <= top3_frames

        po = pair_odds.get(rk, {})
        quin_odds = po.get("quinella", {}).get(axis_pair)
        wide_odds = po.get("quinellaPlace", {}).get(axis_pair)

        rows.append({
            "race_key": rk, "race_date": date_map.get(rk, ""), "entropy": ent,
            "quin_hit": int(quin_hit) if quin_odds else None,
            "quin_pay": quin_odds * STAKE if (quin_odds and quin_hit) else 0,
            "quin_bet": STAKE if quin_odds else 0,
            "wide_hit": int(wide_hit) if wide_odds else None,
            "wide_pay": wide_odds * STAKE if (wide_odds and wide_hit) else 0,
            "wide_bet": STAKE if wide_odds else 0,
        })
    return rows


def summarize(rows, hit_key, pay_key, bet_key, label):
    valid = [r for r in rows if r[bet_key] > 0]
    n = len(valid)
    if n == 0:
        print(f"    {label}: n=0")
        return
    hits = sum(r[hit_key] for r in valid)
    bet = sum(r[bet_key] for r in valid)
    pay = sum(r[pay_key] for r in valid)
    roi = pay / bet * 100 if bet else float("nan")
    print(f"    {label:<24} n={n:>5} hit={hits:>4}({hits/n:.1%}) ROI={roi:>6.1f}%")


def main():
    all_rows = []
    for f, t, m, w in QUARTERS:
        rows = build_leftover(m, w, f, t)
        all_rows.extend(rows)
        print(f"\n===== {f}〜{t}（entropy>{S4_ENTROPY_MAX}帯・S4非対象） =====")
        summarize(rows, "quin_hit", "quin_pay", "quin_bet", "二車複(axis1=axis2)")
        summarize(rows, "wide_hit", "wide_pay", "wide_bet", "ワイド(axis1=axis2)")

    print("\n===== 全期間合算 =====")
    summarize(all_rows, "quin_hit", "quin_pay", "quin_bet", "二車複(axis1=axis2)")
    summarize(all_rows, "wide_hit", "wide_pay", "wide_bet", "ワイド(axis1=axis2)")


if __name__ == "__main__":
    main()
