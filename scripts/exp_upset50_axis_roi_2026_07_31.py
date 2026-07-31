"""波乱予兆レース抽出(exp_upset50_pattern_train_test_2026_07_31.py)に軸選定を載せ、
実際の三連複2軸5点流しROIを算出する（2026-07-31）。

ユーザー要望: 「10%の的中率は許容。5点を10レースで当てれば、50倍以上的中で
ROIは100%を超える想定。穴狙いなので的中率30%にこだわらない」

軸選定（モデル非依存・公表値のみ）:
  axis_score(frame) = 0.5*first_rate + 0.5*third_rate
  axis1/axis2 = axis_scoreの上位2枠（S7と同じ「軸2車+残り5車流し」構造）
  的中条件 = 実際の3着内に axis1・axis2 の両方が入っている（第3の着はどれでもよい）

honest分割は前スクリプトと同一（TRAIN=2022〜2023・TEST=2024〜2026-07-30）。
複合「穴指数」もTRAINのみで平均・標準偏差・符号を決定しTESTへ固定適用（前スクリプトと同一ロジック）。
比較対象:
  (a) 全レースに軸選定のみ適用（レース絞り込みなし）
  (b) 穴指数 上位10/20/25%（閾値はTRAINで確定しTESTに固定適用）に絞った上で軸選定

DB書き込みなし・読み取り専用。
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2022-01-01", "2023-12-31"
TEST_FROM, TEST_TO = "2024-01-01", "2026-07-30"
UPSET_ODDS = 50.0
N_LEGS = 5          # 軸2車 + 残り5車流し(7車立てなので7-2=5)
STAKE_PER_PT = 100  # 円/点


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def load_races(date_from, date_to):
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (date_from, date_to)).fetchall()
    return {r["race_key"]: str(r["race_date"]) for r in rows}


def load_entries(race_keys):
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, race_point, line_group, line_size, n_lines, "
                 "       first_rate, third_rate, finish_order "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    return by_race


def load_trio_win_odds(race_keys, winners_by_race):
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            boards = defaultdict(dict)
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
                    boards[rk][parts] = fv
            for rk in chunk:
                w = winners_by_race.get(rk)
                if w is None:
                    continue
                odds = boards.get(rk, {}).get(w)
                if odds is not None:
                    out[rk] = odds
    return out


FEATURES = [
    "rp_max", "rp_std", "rp_gap12",
    "fr_max", "fr_std", "fr_gap12",
    "tr_max", "tr_std", "tr_gap12",
    "n_lines", "max_line_size", "n_solo", "line_entropy",
]


def build_rows(races, entries_by_race):
    winners_by_race = {}
    prelim = {}
    for rk, race_date in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["race_point"] is None or e["first_rate"] is None or e["third_rate"] is None
               for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        winners_by_race[rk] = winners

        rps = sorted((float(e["race_point"]) for e in ents), reverse=True)
        frs = sorted((float(e["first_rate"]) for e in ents), reverse=True)
        trs = sorted((float(e["third_rate"]) for e in ents), reverse=True)

        line_sizes = defaultdict(int)
        for e in ents:
            if e["line_group"] is not None:
                line_sizes[e["line_group"]] += 1
        n_lines = float(ents[0]["n_lines"] or len(line_sizes) or 0)
        max_line_size = max(line_sizes.values()) if line_sizes else 0
        n_solo = sum(1 for v in line_sizes.values() if v == 1)
        line_entropy = _entropy(list(line_sizes.values())) if line_sizes else 0.0

        # 軸選定: 1着率+3着内率の平均が高い順に2枠
        axis_ranked = sorted(
            ents, key=lambda e: 0.5 * float(e["first_rate"]) + 0.5 * float(e["third_rate"]),
            reverse=True)
        axis1 = int(axis_ranked[0]["frame_no"])
        axis2 = int(axis_ranked[1]["frame_no"])

        prelim[rk] = {
            "race_date": race_date,
            "rp_max": rps[0], "rp_std": float(np.std(rps)), "rp_gap12": rps[0] - rps[1],
            "fr_max": frs[0], "fr_std": float(np.std(frs)), "fr_gap12": frs[0] - frs[1],
            "tr_max": trs[0], "tr_std": float(np.std(trs)), "tr_gap12": trs[0] - trs[1],
            "n_lines": n_lines, "max_line_size": float(max_line_size),
            "n_solo": float(n_solo), "line_entropy": line_entropy,
            "axis1": axis1, "axis2": axis2,
        }
    return prelim, winners_by_race


def finalize(prelim, trio_win_odds):
    rows = []
    for rk, feat in prelim.items():
        odds = trio_win_odds.get(rk)
        if odds is None:
            continue
        row = dict(feat)
        row["race_key"] = rk
        row["trio_win_odds"] = odds
        row["is_upset"] = 1 if odds >= UPSET_ODDS else 0
        rows.append(row)
    return rows


def load_period(date_from, date_to, label):
    print(f"[{label}] loading races {date_from}..{date_to} ...", flush=True)
    races = load_races(date_from, date_to)
    entries = load_entries(list(races.keys()))
    prelim, winners = build_rows(races, entries)
    trio_odds = load_trio_win_odds(list(prelim.keys()), winners)
    rows = finalize(prelim, trio_odds)
    for r in rows:
        r["_winners"] = winners[r["race_key"]]
    print(f"[{label}]   final rows: {len(rows)}", flush=True)
    return rows


def roi_report(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label}: n=0")
        return
    n_hit = 0
    total_bet = 0
    total_return = 0
    for r in rows:
        bet = N_LEGS * STAKE_PER_PT
        total_bet += bet
        hit = {r["axis1"], r["axis2"]} <= r["_winners"]
        if hit:
            n_hit += 1
            total_return += STAKE_PER_PT * r["trio_win_odds"]
    hit_rate = 100.0 * n_hit / n
    roi = 100.0 * total_return / total_bet if total_bet else 0.0
    print(f"  {label}: n={n:6d}  的中率={hit_rate:5.2f}%  "
          f"投資={total_bet:9,d}円  回収={total_return:11,.0f}円  ROI={roi:6.2f}%")
    return {"n": n, "hit_rate": hit_rate, "roi": roi, "bet": total_bet, "ret": total_return}


def main():
    train_rows = load_period(TRAIN_FROM, TRAIN_TO, "TRAIN")
    test_rows = load_period(TEST_FROM, TEST_TO, "TEST")

    mu = {f: float(np.mean([r[f] for r in train_rows])) for f in FEATURES}
    sd = {f: float(np.std([r[f] for r in train_rows])) or 1.0 for f in FEATURES}
    corrs = {}
    for f in FEATURES:
        xs = np.array([r[f] for r in train_rows])
        ys = np.array([r["is_upset"] for r in train_rows], dtype=float)
        corrs[f] = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 0 else 0.0
    sign = {f: (1.0 if corrs[f] >= 0 else -1.0) for f in FEATURES}

    def score(row):
        return sum(sign[f] * (row[f] - mu[f]) / sd[f] for f in FEATURES)

    for r in train_rows:
        r["_score"] = score(r)
    for r in test_rows:
        r["_score"] = score(r)

    print("\n=== (a) 軸選定のみ（レース絞り込みなし・全レース） ===")
    roi_report(train_rows, "TRAIN 全レース")
    roi_report(test_rows, "TEST  全レース")

    print("\n=== (b) 穴指数 上位X% + 軸選定（閾値はTRAINで確定しTESTへ固定適用） ===")
    for pct in (10, 20, 25, 30, 40, 50):
        thr = float(np.percentile([r["_score"] for r in train_rows], 100 - pct))
        tr_flag = [r for r in train_rows if r["_score"] >= thr]
        te_flag = [r for r in test_rows if r["_score"] >= thr]
        print(f"\n [上位{pct}%]")
        roi_report(tr_flag, f"  TRAIN 上位{pct}%")
        roi_report(te_flag, f"  TEST  上位{pct}%")


if __name__ == "__main__":
    main()
