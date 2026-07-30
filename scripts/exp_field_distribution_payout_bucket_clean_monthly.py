"""レース全体(7車)の1着率(pred_win_pct)・3着内率(pred_top3_pct)分布特性と、
実際の三連複配当帯(≤10倍/10-30倍/30倍+等)の関係を、月次凍結vintageモデル
（2026-07-29構築・書き込み保護済み・唯一の正本`src.wt_vintage_config`）で
検証する（[[keirin_s7_foundational_rethink_2026_07_29]]）。

DB格納値(wt_entries.pred_win_pct/pred_top3_pct)は一切参照せず、月次vintage
モデルをその都度ロードして`predict_proba`し直す。

計算する分布特性（7車全体）:
  - win_max/win_2nd/win_gap12/win_sum_top2/win_entropy（単勝側）
  - top3_max/top3_2nd/top3_gap12/top3_sum_top2/top3_entropy（複勝側、既存entropyと同一定義）

配当帯: 0-5/5-10/10-20/20-30/30-50/50-100/100+倍
"""
import math
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
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float("inf"))]


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def _load_trio_win_odds(race_keys, fins_by_race):
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
    df["pred_top3"] = model.predict_proba(X)[:, 1] * 100
    df["pred_win"] = win_model.predict_proba(X)[:, 1] * 100

    race_keys = df["race_key"].unique().tolist()
    trio_odds_map = _load_trio_win_odds(race_keys, fins)

    out = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        odds = trio_odds_map.get(rk)
        if odds is None:
            continue
        win_vals = sorted((float(r.pred_win) for r in g.itertuples(index=False)), reverse=True)
        top3_vals = sorted((float(r.pred_top3) for r in g.itertuples(index=False)), reverse=True)

        out.append({
            "race_key": rk, "race_date": rk[:8], "trio_odds": odds,
            "win_max": win_vals[0], "win_2nd": win_vals[1],
            "win_gap12": win_vals[0] - win_vals[1],
            "win_sum_top2": win_vals[0] + win_vals[1],
            "win_entropy": _entropy(win_vals),
            "top3_max": top3_vals[0], "top3_2nd": top3_vals[1],
            "top3_gap12": top3_vals[0] - top3_vals[1],
            "top3_sum_top2": top3_vals[0] + top3_vals[1],
            "top3_entropy": _entropy(top3_vals),
        })
    return out


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

    metrics = ["win_max", "win_2nd", "win_gap12", "win_sum_top2", "win_entropy",
               "top3_max", "top3_2nd", "top3_gap12", "top3_sum_top2", "top3_entropy"]

    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n{'='*90}\n[{label}] 配当帯別 分布特性平均値 (n={len(data)})\n{'='*90}")
        header = f"{'配当帯':<10}{'n':>7}" + "".join(f"{m:>13}" for m in metrics)
        print(header)
        for lo, hi in BUCKETS:
            sub = [r for r in data if lo <= r["trio_odds"] < hi]
            n = len(sub)
            if n == 0:
                continue
            lr = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
            row = f"{lr:<10}{n:>7}"
            for m in metrics:
                avg = sum(r[m] for r in sub) / n
                row += f"{avg:>13.2f}"
            print(row)

    print("\n" + "=" * 90)
    print("参考: 30倍以上 vs 10倍以下 の直接比較")
    print("=" * 90)
    for label, data in (("TRAIN", train), ("TEST", test)):
        low = [r for r in data if r["trio_odds"] <= 10]
        high = [r for r in data if r["trio_odds"] >= 30]
        print(f"\n[{label}] 10倍以下 n={len(low)} / 30倍以上 n={len(high)}")
        for m in metrics:
            avg_low = sum(r[m] for r in low) / len(low) if low else 0
            avg_high = sum(r[m] for r in high) / len(high) if high else 0
            diff = avg_high - avg_low
            print(f"  {m:<16} 10倍以下={avg_low:>8.2f}  30倍以上={avg_high:>8.2f}  差={diff:>+8.2f}")


if __name__ == "__main__":
    main()
