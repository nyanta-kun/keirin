"""S1: th>=0.15を母集団に、20倍以上配当を狙いつつ的中率を上げるフィルターを検討する。

ユーザー依頼(2026-07-21): 「20倍以上を対象で取ることを目的とし、的中率向上する
ためのレースを絞るためのフィルター設定を検討して。万馬券の発生、取得している
条件は極力取りこぼさないよう検討して」

exp_s1_manshaken_analysis.py は的中レースのみを対象にmanと otherを比較したが、
実際のフィルターは発走前にしか使えない属性(軸勝率・グレード・軸の級班・
ライン構成等)でなければならない。本スクリプトは的中・非的中を問わずth>=0.15の
全候補レースを収集し、配当帯(20倍未満/20-100倍/100倍以上=万車券)別に
事前属性の分布を比較したうえで、いくつかの事前フィルター案を評価する。
評価軸: 選出後の母数・的中率・ROIに加え、現状捕捉している20倍以上・
30倍以上・50倍以上・万車券レースをどれだけ残せるか(再現率)。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_20x_filter_design.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S1W_STAKE, s1w_select

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
    ("2026-04-13", "2026-07-10", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

TH = 0.15
BUCKET_20X = 2000
BUCKET_30X = 3000
BUCKET_50X = 5000
BUCKET_100X = 10000


def _load_trifecta_boards(race_keys):
    tri = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trifecta' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    tri[rk][parts] = fv
    return tri


def collect(eval_model_name, win_model_name, date_from, date_to):
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        meta_map = dict(c.execute(
            "SELECT race_key, race_date || '|' || race_no || '|' || venue_id || '|' || "
            "COALESCE(grade,'') || '|' || COALESCE(distance,'') FROM wt_races "
            "WHERE race_date BETWEEN ? AND ?", (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins: dict[str, list[tuple[int, int]]] = {}
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
    tri_bd = _load_trifecta_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        tri = tri_bd.get(rk)
        if not tri:
            continue
        board: set[int] = set()
        for k in tri:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s1w_select(win_probs, top3_probs)
        if sel is None:
            continue
        axis, p1, p2, top3_gap = sel
        if top3_gap < TH:
            continue
        if axis not in board or p1 not in board or p2 not in board:
            continue
        combo_a, combo_b = (axis, p1, p2), (axis, p2, p1)
        buy = [c for c in (combo_a, combo_b) if tri.get(c) is not None]
        if not buy:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        hit = order3 in buy
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0) if hit else 0
        pay = trifecta_pay * S1W_STAKE // 100 if hit else 0
        bet = len(buy) * S1W_STAKE

        meta = meta_map.get(rk, "||||")
        race_date, race_no, venue_id, grade, distance = (meta.split("|") + [""] * 5)[:5]

        info = {int(r.frame_no): r for r in g.itertuples(index=False)}
        axis_row = info.get(axis)

        def _attr(row, name, default=None):
            return getattr(row, name, default) if row is not None else default

        races.append({
            "race_key": rk, "race_date": race_date, "race_no": race_no,
            "venue_id": venue_id, "grade": grade, "distance": distance,
            "top3_gap": top3_gap, "bet": bet, "pay": pay, "hit": hit,
            "trifecta_pay": trifecta_pay,
            "axis_style": _attr(axis_row, "style"),
            "axis_line_pos": _attr(axis_row, "line_pos"),
            "axis_line_size": _attr(axis_row, "line_size"),
            "axis_n_lines": _attr(axis_row, "n_lines"),
            "axis_player_class": _attr(axis_row, "player_class"),
            "axis_win_prob": _attr(axis_row, "pred_win"),
        })
    return races


def _stats(rows):
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    roi = pay / bet * 100 if bet else 0.0
    hit_rate = hits / n * 100 if n else 0.0
    return n, hits, hit_rate, bet, pay, roi


def _bucket_recall(rows, all_rows, thresh, label):
    base = [r for r in all_rows if r["trifecta_pay"] >= thresh]
    kept = [r for r in rows if r["trifecta_pay"] >= thresh]
    n_base = len(base)
    n_kept = len(kept)
    recall = n_kept / n_base * 100 if n_base else 0.0
    return f"{label}: {n_kept}/{n_base} 残存({recall:.1f}%)"


def main():
    all_races = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rs = collect(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(rs)}R収集(的中+非的中)", flush=True)
        all_races.extend(rs)

    print(f"\n収集完了: th>=0.15 全候補 {len(all_races)}R\n")

    n, hits, hit_rate, bet, pay, roi = _stats(all_races)
    print("=" * 100)
    print(f"[ベースライン(フィルターなし)] n={n}  的中={hits}({hit_rate:.1f}%)  "
          f"投資={bet:,}  回収={pay:,}  ROI={roi:.1f}%")
    for thresh, label in ((BUCKET_20X, "20倍以上"), (BUCKET_30X, "30倍以上"),
                           (BUCKET_50X, "50倍以上"), (BUCKET_100X, "万車券")):
        cnt = sum(1 for r in all_races if r["trifecta_pay"] >= thresh)
        print(f"  {label}: {cnt}R ({cnt/n*100:.1f}%)")

    # ------------------------------------------------------------------
    # フィルター案（すべて発走前にわかる事前属性のみ使用）
    # ------------------------------------------------------------------
    def f_win_prob(rows, max_prob):
        return [r for r in rows if r["axis_win_prob"] is not None and r["axis_win_prob"] <= max_prob]

    def f_grade_s(rows):
        return [r for r in rows if r["grade"] == "S級"]

    def f_class_s(rows):
        return [r for r in rows if r["axis_player_class"] in ("S1", "S2")]

    def f_3line(rows):
        return [r for r in rows if r["axis_n_lines"] == 3]

    def f_not_1man_line(rows):
        return [r for r in rows if r["axis_line_size"] and r["axis_line_size"] >= 2]

    filters = [
        ("軸勝率<=40%",                    lambda rs: f_win_prob(rs, 0.40)),
        ("軸勝率<=50%",                    lambda rs: f_win_prob(rs, 0.50)),
        ("S級グレード",                     f_grade_s),
        ("軸級班S1/S2",                     f_class_s),
        ("3ライン構成",                     f_3line),
        ("軸ライン2人以上(単騎除外)",         f_not_1man_line),
        ("軸勝率<=50% AND 3ライン",          lambda rs: f_3line(f_win_prob(rs, 0.50))),
        ("軸勝率<=50% AND 軸級班S1/S2",      lambda rs: f_class_s(f_win_prob(rs, 0.50))),
        ("軸勝率<=50% AND 単騎除外",          lambda rs: f_not_1man_line(f_win_prob(rs, 0.50))),
        ("(S級 OR 軸級班S1/S2) AND 軸勝率<=55%",
         lambda rs: f_win_prob(
             [r for r in rs if r["grade"] == "S級" or r["axis_player_class"] in ("S1", "S2")], 0.55)),
    ]

    print("\n" + "=" * 100)
    print("[フィルター案の評価]")
    print("=" * 100)
    for label, fn in filters:
        sub = fn(all_races)
        n2, hits2, hr2, bet2, pay2, roi2 = _stats(sub)
        if n2 == 0:
            print(f"\n■ {label}: 該当0件")
            continue
        print(f"\n■ {label}")
        print(f"  n={n2}({n2/n*100:.1f}%)  的中率={hr2:.1f}%(元{hit_rate:.1f}%)  "
              f"投資={bet2:,}  回収={pay2:,}  ROI={roi2:.1f}%(元{roi:.1f}%)")
        print(f"  再現率: "
              f"{_bucket_recall(sub, all_races, BUCKET_20X, '20倍+')} / "
              f"{_bucket_recall(sub, all_races, BUCKET_30X, '30倍+')} / "
              f"{_bucket_recall(sub, all_races, BUCKET_50X, '50倍+')} / "
              f"{_bucket_recall(sub, all_races, BUCKET_100X, '万車券')}")


if __name__ == "__main__":
    main()
