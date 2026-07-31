"""軸2車選定を本番S7の s7_select_axis() と同じ「連動選定」ロジックに変える（2026-07-31）。

ユーザー要望: 「2軸目の選定を7Sと同様に、1軸目と連動して3着以内に来やすい選手を
選定し2軸目とできるようにし検証して」

本番 s7_select_axis()（src/strategy_wt.py）:
  win_top3  = win_probs（単勝確率・モデル予測）上位3
  place_top3 = top3_probs（複勝確率・モデル予測）上位3
  overlap = win_top3 ∩ place_top3
  overlap無し             → 選定不能（除外）
  overlap>=2車            → overlapの中でtop3_probs上位2車を axis1/axis2
  overlap==1車            → その1車をaxis1、残り(axis1を除く全体)からtop3_probs
                             最上位を axis2（＝1軸目と連動した2軸目選定）

pre-2024データにモデル予測(win_probs/top3_probs)は無いため、意味的に最も近い
公表値で代替する:
  win_probs  代替 → race_point（前回検証で単独軸として最良・TRAIN/TESTで安定）
  top3_probs 代替 → third_rate（3着内率・意味的にtop3_probsと直接対応）

overlap==0の場合は本番同様「選定不能＝除外」として母集団から外す（S7本番の
daily_selectにおける除外ルールと同じ扱い）。

exp_upset50_axis_rp_roi_2026_07_31.py（軸1/軸2ともrace_point独立top2）と
ROIを比較する。DB書き込みなし・読み取り専用。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.exp_upset50_axis_roi_2026_07_31 import (
    FEATURES, TEST_FROM, TEST_TO, TRAIN_FROM, TRAIN_TO,
    load_entries, load_races, load_trio_win_odds, roi_report,
)


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def s7style_select_axis(ents):
    """本番s7_select_axisと同じロジック。win=race_point / top3=third_rate で代替。

    returns (axis1, axis2) or None（overlap無し＝選定不能）。
    """
    by_frame = {int(e["frame_no"]): e for e in ents}
    win_probs = {f: float(by_frame[f]["race_point"]) for f in by_frame}
    top3_probs = {f: float(by_frame[f]["third_rate"]) for f in by_frame}

    win_top3 = {f for f, _ in sorted(win_probs.items(), key=lambda kv: -kv[1])[:3]}
    place_top3 = {f for f, _ in sorted(top3_probs.items(), key=lambda kv: -kv[1])[:3]}
    overlap = win_top3 & place_top3
    if not overlap:
        return None
    if len(overlap) >= 2:
        cands = sorted(overlap, key=lambda f: -top3_probs[f])
        return cands[0], cands[1]
    axis1 = next(iter(overlap))
    rest = sorted((f for f in top3_probs if f != axis1), key=lambda f: -top3_probs[f])
    if not rest:
        return None
    return axis1, rest[0]


def build_rows(races, entries_by_race):
    winners_by_race = {}
    prelim = {}
    n_no_overlap = 0
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

        axis = s7style_select_axis(ents)
        if axis is None:
            n_no_overlap += 1
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
    print(f"    (overlap無し・選定不能で除外: {n_no_overlap}件)", flush=True)
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
    print(f"[{label}]   races(母集団): {len(races)}", flush=True)
    entries = load_entries(list(races.keys()))
    prelim, winners = build_rows(races, entries)
    trio_odds = load_trio_win_odds(list(prelim.keys()), winners)
    rows = finalize(prelim, trio_odds)
    for r in rows:
        r["_winners"] = winners[r["race_key"]]
    print(f"[{label}]   最終行(軸選定成功&オッズ取得できたレース): {len(rows)}", flush=True)
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

    print("\n=== S7式・連動選定軸（win=race_point / top3=third_rate） (a) 絞り込みなし ===")
    roi_report(train_rows, "TRAIN 全レース")
    roi_report(test_rows, "TEST  全レース")

    print("\n=== S7式・連動選定軸 (b) 穴指数 上位X%（閾値TRAIN固定→TEST適用） ===")
    for pct in (10, 20, 25, 30, 40, 50):
        thr = float(np.percentile([r["_score"] for r in train_rows], 100 - pct))
        tr_flag = [r for r in train_rows if r["_score"] >= thr]
        te_flag = [r for r in test_rows if r["_score"] >= thr]
        print(f"\n [上位{pct}%]")
        roi_report(tr_flag, f"  TRAIN 上位{pct}%")
        roi_report(te_flag, f"  TEST  上位{pct}%")


if __name__ == "__main__":
    main()
