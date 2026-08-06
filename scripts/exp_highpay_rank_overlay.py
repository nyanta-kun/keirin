#!/usr/bin/env python3
"""既存ランク（7S/7A/7SS/7B）のレース選別が「高額払い戻し」に転用できるかを測る。

## なぜこれが最後のレバーなのか

`exp_highpay_race_selection.py` で次が確定した:

    P(高額) = 帯ROI / 30       （点数・券種・配分に依存しない恒等式）

つまり **「高配当レースの選別モデル」＝「そのレース群で ROI を上げること」** であり、
数学的に同じ問題。新しい問題ではない。したがって既に ROI 80〜84% を達成している
**既存ランクのレース選別がそのまま使える**なら、高額率は
75.7/30 = 2.52% → 84/30 = **2.80%** まで上がるはず。これが唯一の残ったレバー。

⚠️ ただし既存ランクの 80〜84% は「三連複5点・軸2車総流し」という**買い目込み**の
数字である。レース選別だけを抜き出して別の買い目（30倍張り付き）に移したとき
同じ水準が出る保証はない。それを測るのが本スクリプト。

キャッシュ済み候補（`exp_wt_candidate_cache`）を使うので候補生成のやり直しは不要。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_highpay_race_selection import (  # noqa: E402
    CACHE_DIR, HIGHPAY, STAKE, WINDOWS, load_odds, load_race_meta, structures,
)
from scripts.exp_wt_candidate_cache import iter_months  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
)


def classify(c: dict, race_type: str | None) -> str | None:
    """被覆マップ（memory keirin_7car_coverage_gaps_2026_08_05）に従って分類する。"""
    ov = c.get("wt_overlap_n")
    if ov is None:
        return None
    axis_ok = c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
    ent_ok = c["entropy"] <= RANK_7S_ENTROPY_MAX
    if ov in (0, 1):
        if axis_ok and ent_ok:
            return "7S"
        if not axis_ok and ent_ok:
            return "7A"
        if axis_ok and not ent_ok:
            return "7SS" if c.get("same_line") else "空白1"
        return "空白2"
    if ov == 2:
        return "7B/空白3" if race_type == "準決勝" else "overlap2他"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="掃引窓,確認窓")
    args = ap.parse_args()

    meta = load_race_meta(7).set_index("race_key")
    keys = sorted(meta.index)
    trio_all = load_odds(keys, "trio", 7, 25.0, 1e6)
    tf_all = load_odds(keys, "trifecta", 7, 25.0, 1e6)

    for wname in args.windows.split(","):
        d_from, d_to = WINDOWS[wname]
        print(f"\n{'=' * 86}\n=== {wname} {d_from}〜{d_to} ===", flush=True)
        acc = defaultdict(lambda: defaultdict(
            lambda: {"n": 0, "big": 0, "ret": 0.0, "cost": 0.0}))

        for _month, cands in iter_months(d_from, d_to, verbose=False):
            for c in cands:
                rk = c["race_key"]
                if rk not in meta.index:
                    continue
                row = meta.loc[rk]
                if not (d_from <= row["race_date"] <= d_to):
                    continue
                grp = classify(c, row["race_type"])
                if grp is None:
                    continue
                trio_od = trio_all.get(rk) or {}
                tf_od = tf_all.get(rk) or {}
                if not trio_od and not tf_od:
                    continue
                f1, f2, f3 = int(row["f1"]), int(row["f2"]), int(row["f3"])
                win_tf, win_tr = f"{f1}-{f2}-{f3}", frozenset((f1, f2, f3))
                for name, sel in structures(trio_od, tf_od).items():
                    if not sel:
                        continue
                    win = win_tr if name.startswith("trio") else win_tf
                    ret = 0.0
                    big = 0
                    for k, o, s in sel:
                        if k == win:
                            pay = s * o
                            ret += pay
                            if pay >= HIGHPAY - 1:
                                big += 1
                    for tgt in (acc[name][grp], acc[name]["ALL"]):
                        tgt["n"] += 1
                        tgt["big"] += big
                        tgt["ret"] += ret
                        tgt["cost"] += sum(s for _, _, s in sel)

        for name in ("trio1", "tf_budget", "tf10"):
            if name not in acc:
                continue
            base = acc[name]["ALL"]
            br = base["big"] / base["n"] * 100 if base["n"] else 0.0
            print(f"\n  --- {name}（全候補 高額 {br:.2f}% / ROI "
                  f"{base['ret'] / base['cost'] * 100:.1f}%）---")
            print("    群           レース   高額数  高額%   基準比   ROI%")
            for grp, a in sorted(acc[name].items(),
                                 key=lambda x: -(x[1]["big"] / max(x[1]["n"], 1))):
                if grp == "ALL" or a["n"] < 200:
                    continue
                r = a["big"] / a["n"] * 100
                print(f"    {grp:<12} {a['n']:6} {a['big']:7}  {r:5.2f}  "
                      f"{r - br:+6.2f}  {a['ret'] / a['cost'] * 100:6.1f}")


if __name__ == "__main__":
    main()
