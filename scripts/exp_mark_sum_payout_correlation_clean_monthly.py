"""mark_sum(◎◯複勝確率合算)と「両方3着内」率・三連複payoutの相関関係を、
月次凍結vintageモデル（2026-07-29構築・書き込み保護済み・唯一の正本
`src.wt_vintage_config`）で再検証する（[[keirin_s7_foundational_rethink_2026_07_29]]）。

【重要】以前の検証(`exp_honmei_taikou_both_top3_predict.py` /
`exp_wt_honmei_taikou_both_top3_payout_dist.py`)は`wt_entries.pred_top3_pct`
というDB格納値（四半期vintageモデルが2026-07-28に無断上書きされ再現不能と
判明した後のもの）を直接参照して計算していた。本スクリプトはDB格納値を一切
使わず、月次vintageモデルをその都度ロードして`predict_proba`し直すことで、
以下2点をクリーンな状態から再検証する:
  1. mark_sum区間別の「◎◯両方3着内」率（用量反応関係の再現性）
  2. 「両方3着内」vs「そうでない」場合の三連複配当分布
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.wt_vintage_config import monthly_windows

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20261231"
PCTS = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def _load_trio_win_odds(race_keys, entries_by_race, fins_by_race):
    """各レースの実際の勝ち三連複組み合わせのオッズを返す。"""
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            trio_bd = defaultdict(dict)
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
                    trio_bd[rk][parts] = fv
            for rk in chunk:
                fin = fins_by_race.get(rk)
                if not fin or len(fin) < 3:
                    continue
                winners = frozenset(fno for _, fno in sorted(fin)[:3])
                odds = trio_bd.get(rk, {}).get(winners)
                if odds is not None:
                    out[rk] = odds
    return out


def build_month(eval_model_name, date_from, date_to, win_model_name):
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
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
    df["pred_top3"] = model.predict_proba(X)[:, 1] * 100

    race_keys = df["race_key"].unique().tolist()
    trio_odds_map = _load_trio_win_odds(race_keys, None, fins)

    out = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        if wt_honmei is None or wt_taikou is None:
            continue
        row_by_frame = {int(r.frame_no): r for r in g.itertuples(index=False)}
        if wt_honmei not in row_by_frame or wt_taikou not in row_by_frame:
            continue
        fin = [(fo, fno) for fo, fno in fins.get(rk, []) if fno in row_by_frame]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = wt_honmei in winners
        t_hit = wt_taikou in winners
        both_top3 = h_hit and t_hit

        mark_sum = float(row_by_frame[wt_honmei].pred_top3) + float(row_by_frame[wt_taikou].pred_top3)
        odds = trio_odds_map.get(rk)

        out.append({
            "race_key": rk, "race_date": rk[:8], "mark_sum": mark_sum,
            "both_top3": both_top3, "trio_odds": odds,
        })
    return out


def pctile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
    return sorted_vals[idx]


def main():
    windows = monthly_windows()
    print(f"対象月数: {len(windows)}（{windows[0][0]}〜{windows[-1][1]}）")

    all_races = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        recs = build_month(eval_model, date_from, date_to, win_model)
        print(f"[build]   races: {len(recs)}", flush=True)
        all_races.extend(recs)

    print(f"\n[main] 全期間 races合計: {len(all_races)}")
    train = [r for r in all_races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in all_races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    print("\n" + "=" * 78)
    print("1. mark_sum区間別「◎◯両方3着内」率（用量反応関係の再現性確認）")
    print("=" * 78)
    edges = [(0, 60), (60, 80), (80, 100), (100, 110), (110, 120), (120, 130),
             (130, 140), (140, 150), (150, 160), (160, 170), (170, 200)]
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        print(f"{'mark_sum帯':<12}{'n':>8}{'both_top3率':>14}")
        for lo, hi in edges:
            sub = [r for r in data if lo <= r["mark_sum"] < hi]
            n = len(sub)
            if n == 0:
                continue
            rate = sum(r["both_top3"] for r in sub) / n * 100
            lr = f"{lo}-{hi}" if hi != 200 else f"{lo}+"
            print(f"{lr:<12}{n:>8}{rate:>13.1f}%")

    print("\n" + "=" * 78)
    print("2. 「両方3着内」vs「そうでない」の三連複配当分布")
    print("=" * 78)
    for label, data in (("全期間", all_races), ("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        for cat_label, cond in (("both_top3=True", True), ("both_top3=False", False)):
            sub = [r for r in data if r["both_top3"] == cond and r["trio_odds"] is not None]
            n = len(sub)
            if n == 0:
                continue
            vals = sorted(r["trio_odds"] for r in sub)
            mean = sum(vals) / n
            over5 = sum(1 for v in vals if v >= 5) / n * 100
            print(f"  {cat_label}: n={n} 平均={mean:.2f}倍 中央値={pctile(vals,50):.2f}倍 "
                  f"p90={pctile(vals,90):.2f}倍 5倍以上率={over5:.1f}%")


if __name__ == "__main__":
    main()
