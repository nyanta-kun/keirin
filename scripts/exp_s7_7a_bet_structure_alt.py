"""7S/7A 買い目構造の代替案検証（2026-07-31）。

## 検証する2案

1. **ワイド1点化**: 現行は「軸2車＋残り5車流し」の三連複5点均等買い（1点2,000円）。
   軸2車の的中条件（軸2車がともに3着内）は、この5点trioと数学的に完全に同一
   （軸2車以外のどの車が3着でも5点のいずれかが的中する設計のため）。よって
   「軸2車ワイド(quinellaPlace)1点買い」は的中条件を変えずに payout 構造だけを
   比較する純粋な検証になる。

2. **三連複オッズ傾斜配分**: 5点それぞれの実オッズを使い、低オッズ点(＝市場の
   的中確率が高い点)に多く、高オッズ点に少なく配分する。前回検証
   （exp_s7_stake_tilt_by_3rd_car_confidence.py）はオッズ非依存の制約付き
   だったが、今回は明示的にオッズを使用する（ユーザー確認済み）。

対象: S7 + 7A（2026-07-31改定後の現行2ゲート版）の合算母集団。
月次凍結vintageモデルによるhonest walk-forward。読み取り専用・DB書き込みなし。

頑健性診断（前回検証で確立した手法を踏襲）:
  - 配分加重平均オッズ順位・実際の的中のオッズ順位分布
  - 上位1件・3件除外時のROI変化（少数依存でないかの確認）
"""
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.strategy_wt import s7_evening_reselect, s7a_daily_select
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

TOTAL_STAKE = 10000.0
UNIT = 100.0

TILT_SCHEMES = ["U", "InvOdds", "InvOddsSqrt", "R5", "R3"]
RANK_TABLE = {
    "R5": [3.0, 2.5, 2.0, 1.5, 1.0],
    "R3": [2.5, 2.0, 1.5, 1.0, 1.0],
}


def _load_quinella_place_odds(race_keys: list[str]) -> dict:
    """wt_odds(bet_type='quinellaPlace'=ワイド) → {race_key: {frozenset({a,b}): odds}}"""
    out = defaultdict(dict)
    if not race_keys:
        return out
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='quinellaPlace' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 2:
                    out[rk][parts] = fv
    return out


def allocate_by_odds(scheme: str, odds_map: dict) -> dict:
    """odds_map: {x: odds}。返り値は合計1に正規化した配分比率 {x: ratio}。"""
    if scheme == "U":
        w = {x: 1.0 for x in odds_map}
    elif scheme == "InvOdds":
        w = {x: 1.0 / o for x, o in odds_map.items()}
    elif scheme == "InvOddsSqrt":
        w = {x: math.sqrt(1.0 / o) for x, o in odds_map.items()}
    elif scheme in RANK_TABLE:
        order = sorted(odds_map, key=lambda x: odds_map[x])  # オッズ昇順=1位が本命
        table = RANK_TABLE[scheme]
        w = {x: table[min(i, len(table) - 1)] for i, x in enumerate(order)}
    else:
        raise ValueError(scheme)
    tot = sum(w.values())
    return {x: v / tot for x, v in w.items()} if tot > 0 else {x: 1.0 / len(odds_map) for x in odds_map}


def stake_units(alloc: dict, total: float = TOTAL_STAKE, unit: float = UNIT) -> dict:
    n_units = round(total / unit)
    raw = {x: v * n_units for x, v in alloc.items()}
    floor_u = {x: int(u) for x, u in raw.items()}
    remainder = n_units - sum(floor_u.values())
    order = sorted(alloc, key=lambda x: -alloc[x])
    for i in range(remainder):
        floor_u[order[i % len(order)]] += 1
    return {x: u * unit for x, u in floor_u.items()}


def main(date_from_filter: str | None = None, date_to_filter: str | None = None, label: str = "全期間") -> None:
    windows = monthly_windows()
    if date_from_filter:
        windows = [w for w in windows if w[1] >= date_from_filter and w[0] <= date_to_filter]
    print(f"[main] {label}: 月次窓数={len(windows)}", flush=True)

    wide_totals = {"trio5": {"bet": 0.0, "ret": 0.0, "hit": 0},
                   "wide1": {"bet": 0.0, "ret": 0.0, "hit": 0}}
    wide_monthly = {"trio5": [], "wide1": []}
    n_total_selected = 0
    no_wide_odds = 0

    tilt_totals = {s: {"bet": 0.0, "ret": 0.0} for s in TILT_SCHEMES}
    tilt_monthly = {s: [] for s in TILT_SCHEMES}
    diag_weighted_rank = {s: [] for s in TILT_SCHEMES}
    hit_odds_rank = []
    hit_log = {s: [] for s in TILT_SCHEMES}

    for date_from, date_to, eval_model, win_model in windows:
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            continue
        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        selected = []
        for _d, day_cands in by_day.items():
            selected.extend(s7_evening_reselect(day_cands, [], set()))
            selected.extend(s7a_daily_select(day_cands))
        if not selected:
            continue
        n_total_selected += len(selected)

        wide_odds_by_rk = _load_quinella_place_odds([c_["race_key"] for c_ in selected])

        m_wide = {"trio5": {"bet": 0.0, "ret": 0.0}, "wide1": {"bet": 0.0, "ret": 0.0}}
        m_tilt = {s: {"bet": 0.0, "ret": 0.0} for s in TILT_SCHEMES}

        for c_ in selected:
            a, b = c_["axis1"], c_["axis2"]
            trio = c_["trio"]
            others = c_["others"]
            combos = {x: frozenset({a, b, x}) for x in others if frozenset({a, b, x}) in trio}
            if len(combos) < 2:
                continue
            odds_map = {x: trio[key] for x, key in combos.items()}
            actual_top3 = c_["actual_top3"]
            hit_x = next((x for x, key in combos.items() if key == actual_top3), None)
            hit_axis_pair = (a in actual_top3) and (b in actual_top3)  # ワイド的中条件

            # --- 1. 現行5点trio均等 vs 軸2車ワイド1点 ---
            bet5 = 2000.0 * len(combos)
            ret5 = odds_map[hit_x] * 2000.0 if hit_x is not None else 0.0
            m_wide["trio5"]["bet"] += bet5
            m_wide["trio5"]["ret"] += ret5
            if hit_x is not None:
                wide_totals["trio5"]["hit"] += 1

            wide_pair_odds = wide_odds_by_rk.get(c_["race_key"], {}).get(frozenset({a, b}))
            if wide_pair_odds is not None:
                ret_w = wide_pair_odds * bet5 if hit_axis_pair else 0.0
                m_wide["wide1"]["bet"] += bet5
                m_wide["wide1"]["ret"] += ret_w
                if hit_axis_pair:
                    wide_totals["wide1"]["hit"] += 1
            else:
                no_wide_odds += 1

            # --- 2. オッズ傾斜配分（5点trio・現行=Uが基準） ---
            order_odds = sorted(odds_map, key=lambda x: odds_map[x])
            rank_of = {x: i + 1 for i, x in enumerate(order_odds)}
            if hit_x is not None:
                hit_odds_rank.append(rank_of[hit_x])
            for scheme in TILT_SCHEMES:
                alloc = allocate_by_odds(scheme, odds_map)
                stakes = stake_units(alloc)
                bet = sum(stakes.values())
                ret = 0.0
                if hit_x is not None and hit_x in stakes:
                    ret = stakes[hit_x] * odds_map[hit_x]
                    hit_log[scheme].append((c_["race_key"], c_["race_date"],
                                             stakes[hit_x], odds_map[hit_x], ret))
                m_tilt[scheme]["bet"] += bet
                m_tilt[scheme]["ret"] += ret
                wavg_rank = sum(stakes[x] / bet * rank_of[x] for x in stakes) if bet > 0 else 0.0
                diag_weighted_rank[scheme].append(wavg_rank)

        for key in ("trio5", "wide1"):
            b_, r_ = m_wide[key]["bet"], m_wide[key]["ret"]
            wide_monthly[key].append((r_ / b_ * 100) if b_ else None)
            wide_totals[key]["bet"] += b_
            wide_totals[key]["ret"] += r_
        for s in TILT_SCHEMES:
            b_, r_ = m_tilt[s]["bet"], m_tilt[s]["ret"]
            tilt_monthly[s].append((r_ / b_ * 100) if b_ else None)
            tilt_totals[s]["bet"] += b_
            tilt_totals[s]["ret"] += r_

        t5 = m_wide["trio5"]
        w1 = m_wide["wide1"]
        print(f"[{date_from}〜{date_to}] n={len(selected)} "
              f"trio5={t5['ret']/t5['bet']*100 if t5['bet'] else 0:.1f}% "
              f"wide1={w1['ret']/w1['bet']*100 if w1['bet'] else 0:.1f}% | "
              + " ".join(f"{s}={m_tilt[s]['ret']/m_tilt[s]['bet']*100 if m_tilt[s]['bet'] else 0:.1f}%"
                         for s in TILT_SCHEMES), flush=True)

    print(f"\n[main] 選出候補合計={n_total_selected}件・ワイドオッズ欠損={no_wide_odds}件")

    print("\n" + "=" * 110)
    print("【1. ワイド1点化】現行5点trio均等 vs 軸2車ワイド1点")
    print("=" * 110)
    for key, jp_label in (("trio5", "現行5点trio均等"), ("wide1", "軸2車ワイド1点")):
        t = wide_totals[key]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in wide_monthly[key] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        n_hit = t["hit"]
        n_bet_races = int(t["bet"] / (2000.0 if key == "trio5" else 1))  # 参考値。厳密なR数は下記nで代用
        print(f"  {jp_label:<18} 的中{n_hit:>5}件  投資{t['bet']:>13,.0f}  回収{t['ret']:>13,.0f}  "
              f"ROI={roi:>7.1f}%  月次標準偏差={sd:>6.1f}")

    print("\n" + "=" * 110)
    print("【2. 三連複オッズ傾斜配分】U(現行均等)を基準に比較")
    print("=" * 110)
    print(f"{'方式':<12}{'投資':>13}{'回収':>13}{'ROI':>9}{'月次標準偏差':>14}{'配分加重平均オッズ順位':>22}")
    for s in TILT_SCHEMES:
        t = tilt_totals[s]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in tilt_monthly[s] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        avg_wrank = statistics.mean(diag_weighted_rank[s]) if diag_weighted_rank[s] else 0.0
        print(f"{s:<12}{t['bet']:>13,.0f}{t['ret']:>13,.0f}{roi:>8.1f}%{sd:>14.1f}{avg_wrank:>22.2f}")

    print("\n" + "=" * 110)
    print("【頑健性診断】実際の的中のオッズ順位分布")
    print("=" * 110)
    if hit_odds_rank:
        cnt = Counter(hit_odds_rank)
        n = len(hit_odds_rank)
        for r in sorted(cnt):
            print(f"  順位{r}: {cnt[r]:>5}件 ({cnt[r]/n*100:.1f}%)")
        print(f"  平均オッズ順位: {statistics.mean(hit_odds_rank):.2f} / n={n}")

    print("\n" + "=" * 110)
    print("【頑健性診断】各方式の少数的中依存チェック（上位1件・3件を除いた場合のROI）")
    print("=" * 110)
    for s in TILT_SCHEMES:
        log = sorted(hit_log[s], key=lambda x: -x[4])
        if not log:
            continue
        total_ret = sum(x[4] for x in log)
        t = tilt_totals[s]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        ret_wo1 = total_ret - log[0][4]
        ret_wo3 = total_ret - sum(x[4] for x in log[:3])
        roi_wo1 = ret_wo1 / t["bet"] * 100 if t["bet"] else 0.0
        roi_wo3 = ret_wo3 / t["bet"] * 100 if t["bet"] else 0.0
        print(f"  {s}: 全体ROI={roi:.1f}% / 上位1件除外={roi_wo1:.1f}% / "
              f"上位3件除外={roi_wo3:.1f}%  (的中件数={len(log)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--label", default="全期間")
    args = ap.parse_args()
    main(args.date_from, args.date_to, args.label)
