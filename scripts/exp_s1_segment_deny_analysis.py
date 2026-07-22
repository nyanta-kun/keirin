"""S1: 現行ゲート(top3_gap>=0.15 AND 軸勝率<=0.50)通過母集団を対象に、
セグメント別(場・グレード・距離・ライン構成等)の配当特性を分析し、
「低配当になりそうなレースを事前に除外」するdenyフィルター候補を検討する。

ユーザー要望(2026-07-22): 「高的中率を目指すが高配当は捨てない
（低配当になりそうなレースを省き、高配当の的中率を上げる）」

正規プロトコル: 学習+検証(〜2026-03-31)でセグメント傾向を探索・フィルター候補を
選定し、テスト(2026-04-01〜2026-07-22)で一度だけ評価する（多重比較回避）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_segment_deny_analysis.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_PATH = Path("/tmp/exp_s1_segment_deny_cache.pkl")

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S1W_STAKE, s1w_gate, s1w_select

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
    ("2026-04-13", "2026-07-22", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

TRAIN_VAL_END = "2026-03-31"  # これ以前=探索用（train+val統合）、以降=test（一度だけ評価）

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
            "COALESCE(grade,'') || '|' || COALESCE(CAST(distance AS TEXT),'') FROM wt_races "
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
        # 現行本番ゲート（top3_gap>=0.15 AND 軸勝率<=0.50）を母集団の基準にする
        if not s1w_gate(top3_gap, win_probs[axis]):
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
        p1_row = info.get(p1)
        p2_row = info.get(p2)

        def _attr(row, name, default=None):
            return getattr(row, name, default) if row is not None else default

        try:
            dist_i = int(float(distance)) if distance not in ("", None) else None
        except (TypeError, ValueError):
            dist_i = None

        races.append({
            "race_key": rk, "race_date": race_date, "race_no": race_no,
            "venue_id": venue_id, "grade": grade, "distance": dist_i,
            "top3_gap": top3_gap, "bet": bet, "pay": pay, "hit": hit,
            "trifecta_pay": trifecta_pay,
            "axis_win_prob": win_probs[axis],
            "axis_style": _attr(axis_row, "style"),
            "axis_line_pos": _attr(axis_row, "line_pos"),
            "axis_line_size": _attr(axis_row, "line_size"),
            "axis_n_lines": _attr(axis_row, "n_lines"),
            "axis_player_class": _attr(axis_row, "player_class"),
            "p1_line_size": _attr(p1_row, "line_size"),
            "p2_line_size": _attr(p2_row, "line_size"),
            "n_senko": _attr(axis_row, "n_senko"),
        })
    return races


def _stats(rows):
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    roi = pay / bet * 100 if bet else 0.0
    hit_rate = hits / n * 100 if n else 0.0
    avg_pay_on_hit = sum(r["trifecta_pay"] for r in rows if r["hit"]) / hits if hits else 0.0
    return n, hits, hit_rate, bet, pay, roi, avg_pay_on_hit


def _bucket_recall(rows, all_rows, thresh, label):
    base = [r for r in all_rows if r["trifecta_pay"] >= thresh]
    kept = [r for r in rows if r["trifecta_pay"] >= thresh]
    n_base = len(base)
    n_kept = len(kept)
    recall = n_kept / n_base * 100 if n_base else 0.0
    return f"{label}: {n_kept}/{n_base}残存({recall:.1f}%)"


def _segment_breakdown(rows, key_fn, label):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    print(f"\n--- セグメント別（{label}）train+val（〜{TRAIN_VAL_END}）---")
    print(f"{'segment':<20}{'n':>6}{'hit%':>8}{'avg_pay(全体)':>14}{'avg_pay(的中時)':>16}{'ROI':>8}")
    for seg in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        sub = groups[seg]
        n, hits, hr, bet, pay, roi, avg_hit = _stats(sub)
        avg_all = pay / n if n else 0.0
        if n < 15:
            continue
        print(f"{str(seg):<20}{n:>6}{hr:>7.1f}%{avg_all:>14,.0f}{avg_hit:>16,.0f}{roi:>7.1f}%")


def main():
    if CACHE_PATH.exists():
        print(f"[cache] {CACHE_PATH} からロード", flush=True)
        with open(CACHE_PATH, "rb") as f:
            all_races = pickle.load(f)
    else:
        all_races = []
        for date_from, date_to, eval_model, win_model in QUARTERS:
            print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
            rs = collect(eval_model, win_model, date_from, date_to)
            print(f"[collect]   {len(rs)}R収集(現行ゲート通過・的中+非的中)", flush=True)
            all_races.extend(rs)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(all_races, f)
        print(f"[cache] {CACHE_PATH} に保存", flush=True)

    train_val = [r for r in all_races if r["race_date"] <= TRAIN_VAL_END]
    test = [r for r in all_races if r["race_date"] > TRAIN_VAL_END]
    print(f"\n収集完了: 現行ゲート通過 全候補 {len(all_races)}R "
          f"(train+val={len(train_val)}R / test={len(test)}R)\n")

    n, hits, hr, bet, pay, roi, avg_hit = _stats(train_val)
    print("=" * 110)
    print(f"[train+val ベースライン] n={n} 的中={hits}({hr:.1f}%) 投資={bet:,} 回収={pay:,} "
          f"ROI={roi:.1f}%  的中時平均配当={avg_hit:,.0f}円")
    for thresh, label in ((BUCKET_20X, "20倍以上"), (BUCKET_30X, "30倍以上"),
                          (BUCKET_50X, "50倍以上"), (BUCKET_100X, "万車券")):
        cnt = sum(1 for r in train_val if r["trifecta_pay"] >= thresh)
        print(f"  {label}: {cnt}R ({cnt/n*100:.1f}%)")

    # ------------------------------------------------------------------
    # セグメント別breakdown（train+valのみで探索）
    # ------------------------------------------------------------------
    _segment_breakdown(train_val, lambda r: r["venue_id"], "場(venue_id)")
    _segment_breakdown(train_val, lambda r: r["grade"], "グレード")
    _segment_breakdown(train_val, lambda r: r["distance"], "距離")
    _segment_breakdown(train_val, lambda r: r["axis_n_lines"], "軸レースのライン数")
    _segment_breakdown(train_val, lambda r: r["axis_line_size"], "軸ライン人数")
    _segment_breakdown(train_val, lambda r: r["axis_player_class"], "軸級班")
    _segment_breakdown(train_val, lambda r: r["axis_style"], "軸脚質")
    _segment_breakdown(train_val, lambda r: r["n_senko"], "レース内逃げ人数")

    print(f"\n{'='*110}\n[フィルター候補の評価（train+valで選定 → testで一度だけ検証）]\n{'='*110}")

    def f_deny_top_class(rows):
        return [r for r in rows if r["axis_player_class"] not in ("S1", "A1")]

    def f_deny_nlines4(rows):
        return [r for r in rows if r["axis_n_lines"] != 4]

    filters = [
        ("軸級班 S1/A1 除外", f_deny_top_class),
        ("軸級班 S1/A1 除外 AND n_lines!=4", lambda rs: f_deny_nlines4(f_deny_top_class(rs))),
    ]

    def _report(label, base_rows, sub_rows, base_stats):
        n0, hits0, hr0, bet0, pay0, roi0, avghit0 = base_stats
        n, hits, hr, bet, pay, roi, avghit = _stats(sub_rows)
        if n == 0:
            print(f"\n■ {label}: 該当0件")
            return
        print(f"\n■ {label}")
        print(f"  n={n}({n/n0*100:.1f}%)  的中率={hr:.1f}%(元{hr0:.1f}%)  "
              f"投資={bet:,}  回収={pay:,}  ROI={roi:.1f}%(元{roi0:.1f}%)  "
              f"的中時平均配当={avghit:,.0f}円(元{avghit0:,.0f}円)")
        print(f"  再現率: "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_20X, '20倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_30X, '30倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_50X, '50倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_100X, '万車券')}")

    print("\n" + "-" * 60 + " train+val（選定用） " + "-" * 60)
    tv_stats = _stats(train_val)
    for label, fn in filters:
        _report(label, train_val, fn(train_val), tv_stats)

    print("\n" + "-" * 60 + " ★ test（一度だけ評価） " + "-" * 60)
    test_stats = _stats(test)
    n0, hits0, hr0, bet0, pay0, roi0, avghit0 = test_stats
    print(f"[testベースライン(フィルターなし)] n={n0} 的中={hits0}({hr0:.1f}%) "
          f"投資={bet0:,} 回収={pay0:,} ROI={roi0:.1f}% 的中時平均配当={avghit0:,.0f}円")
    for label, fn in filters:
        _report(label, test, fn(test), test_stats)


if __name__ == "__main__":
    main()
