"""軸選定基準を「1着率+3着内率」→「競走得点(race_point)単独上位2頭」に差し替えて
ROIを再計算する（2026-07-31）。exp_upset_axis_alt_criteria_2026_07_31.py で
race_point単独が波乱予兆セグメントでも全体でも実力複合スコアより軸的中率が高く
TRAIN/TESTで安定していたため、軸2車ともrace_point基準に変更したときの実際の
ROIを exp_upset50_axis_roi_2026_07_31.py と同じ枠組みで比較する。

DB書き込みなし・読み取り専用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.exp_upset50_axis_roi_2026_07_31 import (
    FEATURES, TEST_FROM, TEST_TO, TRAIN_FROM, TRAIN_TO,
    load_entries, load_races, load_trio_win_odds, roi_report,
)
from src.database import get_connection  # noqa: F401  (loadersが暗黙に使用)


def build_rows_rp_axis(races, entries_by_race):
    import math
    from collections import defaultdict

    def _entropy(vals):
        total = sum(vals)
        if total <= 0:
            return 0.0
        ent = 0.0
        for v in vals:
            s = max(v / total, 1e-9)
            ent -= s * math.log(s)
        return ent

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

        # 軸選定: race_point(競走得点)単独上位2枠
        rp_ranked = sorted(ents, key=lambda e: float(e["race_point"]), reverse=True)
        axis1 = int(rp_ranked[0]["frame_no"])
        axis2 = int(rp_ranked[1]["frame_no"])

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
    prelim, winners = build_rows_rp_axis(races, entries)
    trio_odds = load_trio_win_odds(list(prelim.keys()), winners)
    rows = finalize(prelim, trio_odds)
    for r in rows:
        r["_winners"] = winners[r["race_key"]]
    print(f"[{label}]   final rows: {len(rows)}", flush=True)
    return rows


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

    print("\n=== 軸=race_point単独上位2頭 (a) 絞り込みなし・全レース ===")
    roi_report(train_rows, "TRAIN 全レース")
    roi_report(test_rows, "TEST  全レース")

    print("\n=== 軸=race_point単独上位2頭 (b) 穴指数 上位X%（閾値はTRAIN固定→TEST適用） ===")
    for pct in (10, 20, 25, 30, 40, 50):
        thr = float(np.percentile([r["_score"] for r in train_rows], 100 - pct))
        tr_flag = [r for r in train_rows if r["_score"] >= thr]
        te_flag = [r for r in test_rows if r["_score"] >= thr]
        print(f"\n [上位{pct}%]")
        roi_report(tr_flag, f"  TRAIN 上位{pct}%")
        roi_report(te_flag, f"  TEST  上位{pct}%")


if __name__ == "__main__":
    main()
