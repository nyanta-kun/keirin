"""軸2をWT印(◯/△/✕=prediction_mark 2/3/4)の中から、軸1と重ならない
3着内率(third_rate)最上位で選定する場合のROIを検証する（2026-07-31）。

ユーザー要望: 「2軸目をWINTICKETにおける印◯△✕の中で、1軸と重ならない
3着内率最上位を選んだ場合、検証して」

軸1は前回までの検証で最良だった race_point(競走得点) 単独top1を維持。
軸2 = prediction_mark in (2,3,4) のうち axis1以外で third_rate 最大の1頭。
　（◎=1は軸1候補としては別途race_pointで選ぶため、軸2の候補プールからは
　　意図的に除外していない＝◎馬がrace_point1位でなければ軸2候補にも入りうる
　　が、axis1と同一枠は重複除外する）
該当馬が存在しない（axis1以外に◯△✕馬がいない＝ほぼ無いはずだが欠損時）は
選定不能として除外。

exp_upset50_axis_rp_roi_2026_07_31.py（軸1/軸2ともrace_point独立top2）・
exp_upset50_axis_s7style_roi_2026_07_31.py（S7式連動選定）とROIを比較する。

DB書き込みなし・読み取り専用。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.exp_upset50_axis_roi_2026_07_31 import (
    FEATURES, TEST_FROM, TEST_TO, TRAIN_FROM, TRAIN_TO,
    load_races, load_trio_win_odds, roi_report,
)
from src.database import get_connection


def load_entries(race_keys):
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, race_point, line_group, line_size, n_lines, "
                 "       first_rate, third_rate, finish_order, prediction_mark "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    return by_race


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def select_axis(ents):
    """axis1=race_point top1 / axis2=WT印(2,3,4)のうちaxis1以外でthird_rate最大。

    returns (axis1, axis2) or None（候補なしで選定不能）。
    """
    by_frame = {int(e["frame_no"]): e for e in ents}
    axis1 = max(by_frame, key=lambda f: float(by_frame[f]["race_point"]))

    mark_frames = [f for f in by_frame
                   if by_frame[f]["prediction_mark"] in (2, 3, 4) and f != axis1]
    if not mark_frames:
        return None
    axis2 = max(mark_frames, key=lambda f: float(by_frame[f]["third_rate"]))
    return axis1, axis2


def build_rows(races, entries_by_race):
    winners_by_race = {}
    prelim = {}
    n_no_cand = 0
    for rk, race_date in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["race_point"] is None or e["first_rate"] is None or e["third_rate"] is None
               or e["prediction_mark"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])

        axis = select_axis(ents)
        if axis is None:
            n_no_cand += 1
            continue
        axis1, axis2 = axis
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

        prelim[rk] = {
            "race_date": race_date,
            "rp_max": rps[0], "rp_std": float(np.std(rps)), "rp_gap12": rps[0] - rps[1],
            "fr_max": frs[0], "fr_std": float(np.std(frs)), "fr_gap12": frs[0] - frs[1],
            "tr_max": trs[0], "tr_std": float(np.std(trs)), "tr_gap12": trs[0] - trs[1],
            "n_lines": n_lines, "max_line_size": float(max_line_size),
            "n_solo": float(n_solo), "line_entropy": line_entropy,
            "axis1": axis1, "axis2": axis2,
        }
    print(f"    (軸2候補なしで除外: {n_no_cand}件)", flush=True)
    return prelim, winners_by_race


def finalize(prelim, trio_win_odds, upset_odds=50.0):
    rows = []
    for rk, feat in prelim.items():
        odds = trio_win_odds.get(rk)
        if odds is None:
            continue
        row = dict(feat)
        row["race_key"] = rk
        row["trio_win_odds"] = odds
        row["is_upset"] = 1 if odds >= upset_odds else 0
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
    print(f"[{label}]   最終行: {len(rows)}", flush=True)
    return rows


def axis_recall(rows, label):
    n = len(rows)
    axis1_in = sum(1 for r in rows if r["axis1"] in r["_winners"])
    axis2_in = sum(1 for r in rows if r["axis2"] in r["_winners"])
    both = sum(1 for r in rows if {r["axis1"], r["axis2"]} <= r["_winners"])
    print(f"  {label}: n={n}  axis1 3着内率={100*axis1_in/n:.1f}%  "
          f"axis2 3着内率={100*axis2_in/n:.1f}%  両方3着内={100*both/n:.1f}%")


def main():
    train_rows = load_period(TRAIN_FROM, TRAIN_TO, "TRAIN")
    test_rows = load_period(TEST_FROM, TEST_TO, "TEST")

    print("\n=== 軸的中率(参考) ===")
    axis_recall(train_rows, "TRAIN")
    axis_recall(test_rows, "TEST")

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

    print("\n=== 軸1=race_point / 軸2=WT印(◯△✕)中third_rate最大 (a) 絞り込みなし ===")
    roi_report(train_rows, "TRAIN 全レース")
    roi_report(test_rows, "TEST  全レース")

    print("\n=== (b) 穴指数 上位X%（閾値TRAIN固定→TEST適用） ===")
    for pct in (10, 20, 25, 30, 40, 50):
        thr = float(np.percentile([r["_score"] for r in train_rows], 100 - pct))
        tr_flag = [r for r in train_rows if r["_score"] >= thr]
        te_flag = [r for r in test_rows if r["_score"] >= thr]
        print(f"\n [上位{pct}%]")
        roi_report(tr_flag, f"  TRAIN 上位{pct}%")
        roi_report(te_flag, f"  TEST  上位{pct}%")


if __name__ == "__main__":
    main()
