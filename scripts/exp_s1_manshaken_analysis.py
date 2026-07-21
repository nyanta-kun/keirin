"""S1: 閾値0.15時点で「万車券」（三連単配当10,000円以上）的中したレースの条件分析。

ユーザー依頼: top3_gap閾値を0.15に変更した上で万車券（配当10,000円以上）が
発生するレースの購入に絞ることを検討したい。まずは実際に万車券的中した
レースの事前情報（グレード・距離・ライン構成・脚質等）に共通パターンが
ないか分析する。

honest全期間・四半期walk-forwardモデルで、th>=0.15の全的中レースを対象に
万車券（trifecta_pay>=10000）とそれ以外を比較する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_manshaken_analysis.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
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
MANSHAKEN_MIN = 10000  # 万車券: 三連単配当10,000円以上（オッズ100倍以上相当）


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
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0)
        pay = trifecta_pay * S1W_STAKE // 100 if hit else 0
        bet = len(buy) * S1W_STAKE
        if not hit:
            continue  # 万車券分析は的中レースのみ対象

        meta = meta_map.get(rk, "||||")
        race_date, race_no, venue_id, grade, distance = (meta.split("|") + [""] * 5)[:5]

        # 軸・p1・p2の属性
        info = {int(r.frame_no): r for r in g.itertuples(index=False)}
        axis_row = info.get(axis)
        p1_row = info.get(p1)
        p2_row = info.get(p2)

        def _attr(row, name, default=None):
            return getattr(row, name, default) if row is not None else default

        races.append({
            "race_key": rk, "race_date": race_date, "race_no": race_no,
            "venue_id": venue_id, "grade": grade, "distance": distance,
            "top3_gap": top3_gap, "pay": pay, "trifecta_pay": trifecta_pay,
            "axis": axis, "p1": p1, "p2": p2, "order3": order3,
            "axis_style": _attr(axis_row, "style"),
            "axis_line_pos": _attr(axis_row, "line_pos"),
            "axis_line_size": _attr(axis_row, "line_size"),
            "axis_n_lines": _attr(axis_row, "n_lines"),
            "axis_is_leader": _attr(axis_row, "is_line_leader"),
            "axis_player_class": _attr(axis_row, "player_class"),
            "axis_win_prob": _attr(axis_row, "pred_win"),
            "which_hit": "axis->p1->p2" if order3 == combo_a else "axis->p2->p1",
        })
    return races


def main():
    all_races = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rs = collect(eval_model, win_model, date_from, date_to)
        print(f"[collect]   {len(rs)}R的中(th>=0.15)収集", flush=True)
        all_races.extend(rs)

    man = [r for r in all_races if r["trifecta_pay"] >= MANSHAKEN_MIN]
    other = [r for r in all_races if r["trifecta_pay"] < MANSHAKEN_MIN]

    print("\n" + "=" * 90)
    print(f"th>=0.15 的中レース全体: {len(all_races)}R")
    print(f"うち万車券（配当≥{MANSHAKEN_MIN:,}円）: {len(man)}R "
          f"({len(man)/len(all_races)*100:.1f}%)")
    print(f"それ以外の的中: {len(other)}R\n")

    print("[配当倍率別内訳(単勝オッズ換算, 100円=配当円で概算)]")
    for odds_x, pay_min in ((10, 1000), (20, 2000), (30, 3000), (50, 5000), (100, 10000)):
        n = sum(1 for r in all_races if r["trifecta_pay"] >= pay_min)
        print(f"  {odds_x:3d}倍以上(配当≥{pay_min:,}円): {n:4d}R ({n/len(all_races)*100:5.1f}%)")
    print()

    print("[万車券レース一覧]")
    for r in sorted(man, key=lambda r: -r["trifecta_pay"]):
        print(f"  {r['race_date']} {r['venue_id']}場{r['race_no']}R  "
              f"配当{r['trifecta_pay']:,}円  top3_gap={r['top3_gap']:.3f}  "
              f"軸{r['axis']}(勝率{r['axis_win_prob']:.1%} 脚質{r['axis_style']} "
              f"ライン{r['axis_line_pos']}/{r['axis_line_size']}本 "
              f"級{r['axis_player_class']})  グレード={r['grade']} 距離={r['distance']}  "
              f"的中目={r['which_hit']}")

    def _dist(rows, key, label):
        c = Counter(str(r.get(key)) for r in rows)
        total = len(rows)
        print(f"  [{label}]")
        for k, v in c.most_common():
            print(f"    {k}: {v} ({v/total*100:.1f}%)")

    print("\n===== 万車券レース vs その他的中レース の属性比較 =====")
    for key, label in (
        ("grade", "グレード"),
        ("axis_style", "軸の脚質"),
        ("axis_player_class", "軸の級班"),
        ("axis_n_lines", "ライン本数(レース全体)"),
        ("axis_line_size", "軸のライン人数"),
        ("axis_line_pos", "軸のライン内位置(1=先頭)"),
        ("which_hit", "的中目(軸→p1→p2 or 軸→p2→p1)"),
    ):
        print(f"\n-- {label} --")
        print(" 万車券:")
        _dist(man, key, label)
        print(" その他的中:")
        _dist(other, key, label)

    def _avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n===== 数値属性の平均値比較 =====")
    for key, label in (("top3_gap", "top3_gap"), ("axis_win_prob", "軸の単勝確率")):
        m = _avg(man, key)
        o = _avg(other, key)
        print(f"  {label}: 万車券={m:.4f}  その他的中={o:.4f}" if m is not None and o is not None
              else f"  {label}: データ不足")


if __name__ == "__main__":
    main()
