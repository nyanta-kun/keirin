#!/usr/bin/env python3
"""S9/9A axis_sum再設計案の honest全期間検証（2024Q1で閾値較正→残り9四半期へ
blind適用。S9のentropy閾値(1.9938)較正と同一方法論・DB書き込みなし）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.exp_s9_9a_axis_sum_recalibration import (
    build_candidates, current_9a, current_s9, new_9a, new_s9, score,
)
from src.strategy_wt import S9_ENTROPY_MAX
from src.wt_vintage_config import monthly_windows

QUARTERS = [
    ("2024-04-01", "2024-06-30"), ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"), ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-07-31"),
]


def build_range(date_from: str, date_to: str) -> tuple[list[dict], dict, dict]:
    all_c, all_t, all_p = [], {}, {}
    for wf, wt, em, wm in monthly_windows():
        if wt < date_from or wf > date_to:
            continue
        seg_from = max(wf, date_from)
        seg_to = min(wt, date_to)
        c, t, p = build_candidates(em, seg_from, seg_to, wm)
        all_c.extend(c)
        all_t.update(t)
        all_p.update(p)
    return all_c, all_t, all_p


def main() -> None:
    print("[calib] 2024Q1で axis_sum 閾値較正中...", flush=True)
    train_c, _, _ = build_range("2024-01-01", "2024-03-31")
    train_pool = [c for c in train_c if c.get("wt_overlap_n") in (0, 1)]
    axis_sums = sorted(c["axis_sum"] for c in train_pool)
    p25_idx = int(len(axis_sums) * 0.25)
    axis_sum_thresh = axis_sums[p25_idx]
    print(f"[calib] 2024Q1 base pool n={len(train_pool)} axis_sum下位25%点={axis_sum_thresh:.4f}", flush=True)
    print(f"[calib] entropy閾値は既存S9_ENTROPY_MAX={S9_ENTROPY_MAX}をそのまま使用（再較正なし）", flush=True)

    cur_s9_tot = {"n": 0, "hit": 0, "bet": 0, "pay": 0}
    cur_9a_tot = {"n": 0, "hit": 0, "bet": 0, "pay": 0}
    new_s9_tot = {"n": 0, "hit": 0, "bet": 0, "pay": 0}
    new_9a_tot = {"n": 0, "hit": 0, "bet": 0, "pay": 0}

    def acc(tot, r):
        tot["n"] += r["n"]; tot["hit"] += r["hit"]; tot["bet"] += r["bet"]; tot["pay"] += r["pay"]

    print(f"\n{'=' * 110}")
    print(f"blind適用（2024Q2〜2026Q3・9四半期・axis_sum<={axis_sum_thresh:.4f}固定）")
    print(f"{'=' * 110}")

    for qf, qt in QUARTERS:
        cands, trio, pm = build_range(qf, qt)
        r_cs9 = score(current_s9(cands), trio, pm)
        r_c9a = score(current_9a(cands), trio, pm)
        r_ns9 = score(new_s9(cands, axis_sum_thresh, S9_ENTROPY_MAX), trio, pm)
        r_n9a = score(new_9a(cands, axis_sum_thresh, S9_ENTROPY_MAX), trio, pm)
        acc(cur_s9_tot, r_cs9); acc(cur_9a_tot, r_c9a)
        acc(new_s9_tot, r_ns9); acc(new_9a_tot, r_n9a)
        print(f"\n[{qf}〜{qt}] 候補{len(cands)}件")
        print(f"  現行S9: n={r_cs9['n']:3d} 的中{r_cs9['hit']:3d} ({r_cs9['hit']/r_cs9['n']*100 if r_cs9['n'] else 0:5.1f}%) ROI={r_cs9['roi']:6.1f}%"
              f"   |  現行9A: n={r_c9a['n']:3d} 的中{r_c9a['hit']:3d} ({r_c9a['hit']/r_c9a['n']*100 if r_c9a['n'] else 0:5.1f}%) ROI={r_c9a['roi']:6.1f}%")
        print(f"  新S9  : n={r_ns9['n']:3d} 的中{r_ns9['hit']:3d} ({r_ns9['hit']/r_ns9['n']*100 if r_ns9['n'] else 0:5.1f}%) ROI={r_ns9['roi']:6.1f}%"
              f"   |  新9A  : n={r_n9a['n']:3d} 的中{r_n9a['hit']:3d} ({r_n9a['hit']/r_n9a['n']*100 if r_n9a['n'] else 0:5.1f}%) ROI={r_n9a['roi']:6.1f}%")

    def roi(t): return t["pay"] / t["bet"] * 100 if t["bet"] else 0.0

    print(f"\n{'=' * 110}")
    print("全期間合計（2024Q2〜2026Q3・9四半期）")
    print(f"{'=' * 110}")
    print(f"現行S9: n={cur_s9_tot['n']} 的中{cur_s9_tot['hit']} ({cur_s9_tot['hit']/cur_s9_tot['n']*100 if cur_s9_tot['n'] else 0:.1f}%) ROI={roi(cur_s9_tot):.1f}%")
    print(f"現行9A: n={cur_9a_tot['n']} 的中{cur_9a_tot['hit']} ({cur_9a_tot['hit']/cur_9a_tot['n']*100 if cur_9a_tot['n'] else 0:.1f}%) ROI={roi(cur_9a_tot):.1f}%")
    print(f"新S9  : n={new_s9_tot['n']} 的中{new_s9_tot['hit']} ({new_s9_tot['hit']/new_s9_tot['n']*100 if new_s9_tot['n'] else 0:.1f}%) ROI={roi(new_s9_tot):.1f}%")
    print(f"新9A  : n={new_9a_tot['n']} 的中{new_9a_tot['hit']} ({new_9a_tot['hit']/new_9a_tot['n']*100 if new_9a_tot['n'] else 0:.1f}%) ROI={roi(new_9a_tot):.1f}%")


if __name__ == "__main__":
    main()
