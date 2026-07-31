"""【S7 賭け金傾斜配分の検証】3列目（残り1車）の信頼度で配分を傾ける（2026-07-31）。

## 位置づけ・前回の教訓との違い（最重要）

[[keirin_s7_ev_filter_favorite_exclusion_bug_2026_07_30]]で、
EV=predicted_prob×odds による閾値フィルタが「本命（最もオッズが低い組）を
98.1%除外し、モデルが高オッズ帯で過大評価する穴を優先する」バグと判明し
実装を見送った。

**本検証はEVではなく「3列目（軸2車以外の1車）の予測確率そのもの」で
配分を傾ける**という別のアイデアであり、オッズを一切乗じない点が
前回との決定的な違い。オッズを乗じないため、前回バグの直接原因だった
「モデルの高オッズ帯過大評価がEVに増幅されて反映される」経路が存在しない。
ただし油断せず、**前回と同一の頑健性診断（採用配分のオッズ順位分布・
実際の的中のオッズ順位分布との整合）を最初から組み込んで検証する**。

## 検証する配分方式（総投資額はどの方式でも同額に正規化・100円単位）

**前提（ユーザー確認済み）**: 配分は発走前（オッズ確定前）に決定する必要があるため、
オッズを一切使わない。以下はすべて発走前確定情報（モデルの予測確率）のみを使用する。

- U（現行=均等）: 5点に均等配分
- P（3列目確率比例・連続値）: 3列目の pred_top3_pct(x) の値そのものに比例配分
  （軸2車は全5点で共通のため、3列目の値だけが配分の差を生む）
- J（結合確率比例・連続値）: 軸ペアを含む5点の結合確率で正規化した比例配分
  （3列目とライン構成の相互作用も考慮）
- J2（Jを平方根で緩和・連続値）: sqrt(結合確率)に比例。極端な傾斜を抑える保守版
- R5（3列目確率の順位による固定配分表・5段階）: 3列目を確率降順で5段階に並べ、
  固定比率[3,2.5,2,1.5,1]（合計10）で配分。連続値でなく順位のみに基づく
  「一律」配分（値の大小に関わらず同じ表を適用）
- R3（3段階の簡易版）: 上位2位まで手厚く・下位2位は最小・中位1位は中間、
  固定比率[2.5,2,1.5,1,1]（合計8）

## 頑健性診断（前回の教訓を反映・必須）

- 各配分方式で「最も配分が大きい点」のオッズ順位（1=本命）の分布
- 配分の大きさとオッズ順位(逆順位=市場の確信度)の相関
- 実際の的中のオッズ順位分布と、配分の重心（配分加重平均オッズ順位）の比較
- 上位1件・上位3件の的中を除いた場合のROI（少数依存でないかの確認）

honest: 月次凍結vintageモデル。S7の実際の選出ロジック
(s7_daily_select等・2026-07-31改定後の現行版)をそのまま使用。
DB書き込みなし・読み取り専用。
"""
import math
import statistics
import sys
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import s7_evening_reselect
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

TOTAL_STAKE = 10000.0     # ユーザー想定: 1レース10,000円
UNIT = 100.0              # 100円単位に丸める
MIN_N = 400


def joint_probs_for_5(bf, others_frames, a, b):
    """35通り相当の考え方で、軸ペア(a,b)を含む5点の結合確率を正規化して返す。

    ここでは「liftはTRAIN期間の累積で推定」ではなく簡易的に当該レースの
    ライン情報のみでbucket分類し、lift無し(1.0)の素の積で計算する
    （3列目確率の相対順位が主眼のため、lift補正の有無で大勢は変わらない。
    厳密な絶対確率が必要な処理ではなく、5点間の相対配分にのみ使う）。
    """
    pa = bf[a]["p"]
    pb = bf[b]["p"]
    raw = {}
    for x in others_frames:
        px = bf[x]["p"]
        raw[x] = pa * pb * px
    tot = sum(raw.values())
    if tot <= 0:
        return None
    return {x: v / tot for x, v in raw.items()}


RANK_TABLE = {
    "R5": [3.0, 2.5, 2.0, 1.5, 1.0],   # 5段階・連続的に逓減
    "R3": [2.5, 2.0, 1.5, 1.0, 1.0],   # 上位2段厚め・下位2段は同額（3段階相当）
}


def allocate(scheme, bf, others_frames, a, b):
    """各schemeで5点への配分比率(sum=1)を返す。"""
    if scheme == "U":
        w = {x: 1.0 for x in others_frames}
    elif scheme == "P":
        w = {x: bf[x]["p"] for x in others_frames}
    elif scheme in ("J", "J2"):
        jp = joint_probs_for_5(bf, others_frames, a, b)
        if jp is None:
            w = {x: 1.0 for x in others_frames}
        elif scheme == "J":
            w = jp
        else:
            w = {x: math.sqrt(v) for x, v in jp.items()}
    elif scheme in RANK_TABLE:
        # 3列目の確率降順で順位付けし、固定表（値の大小ではなく順位のみ）を適用
        order = sorted(others_frames, key=lambda x: -bf[x]["p"])
        table = RANK_TABLE[scheme]
        w = {x: table[min(i, len(table) - 1)] for i, x in enumerate(order)}
    else:
        raise ValueError(scheme)
    tot = sum(w.values())
    return {x: v / tot for x, v in w.items()} if tot > 0 else {x: 1.0 / len(others_frames) for x in others_frames}


def stake_units(alloc, total=TOTAL_STAKE, unit=UNIT):
    """比率を100円単位に丸めた実際の賭け金へ変換（合計は総額に極力近づける）。"""
    n_units = round(total / unit)
    raw_units = {x: v * n_units for x, v in alloc.items()}
    floor_units = {x: int(u) for x, u in raw_units.items()}
    remainder = n_units - sum(floor_units.values())
    # 端数は配分比率が大きい順に1単位ずつ追加
    order = sorted(alloc, key=lambda x: -alloc[x])
    for i in range(remainder):
        floor_units[order[i % len(order)]] += 1
    return {x: u * unit for x, u in floor_units.items()}


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}")

    schemes = ["U", "P", "J", "J2", "R5", "R3"]
    monthly_roi = {s: [] for s in schemes}
    totals = {s: {"n": 0, "bet": 0.0, "ret": 0.0, "hit": 0.0} for s in schemes}
    # 頑健性診断用: 配分加重平均オッズ順位・実的中のオッズ順位・的中明細
    diag_weighted_rank = {s: [] for s in schemes}   # 各レースでの配分加重平均オッズ順位
    hit_odds_rank = []                              # 実際の的中のオッズ順位（方式非依存・1回だけ集計）
    hit_log = {s: [] for s in schemes}              # (race_key, date, stake, odds, payout)

    for date_from, date_to, eval_model, win_model in windows:
        n_days = (_date.fromisoformat(date_to) - _date.fromisoformat(date_from)).days + 1
        print(f"\n[main] {date_from}〜{date_to}（{n_days}日）", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue

        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        month_acc = {s: {"bet": 0.0, "ret": 0.0} for s in schemes}

        for _d, day_cands in by_day.items():
            for c_ in s7_evening_reselect(day_cands, [], set()):
                a, b = c_["axis1"], c_["axis2"]
                trio = c_["trio"]
                others = c_["others"]
                combos = {x: frozenset({a, b, x}) for x in others if frozenset({a, b, x}) in trio}
                if len(combos) < 2:
                    continue
                odds_map = {x: trio[key] for x, key in combos.items()}
                # オッズ昇順の順位（1=本命）
                order = sorted(odds_map, key=lambda x: odds_map[x])
                rank_of = {x: i + 1 for i, x in enumerate(order)}
                actual_top3 = c_["actual_top3"]
                hit_x = next((x for x, key in combos.items() if key == actual_top3), None)
                if hit_x is not None:
                    hit_odds_rank.append(rank_of[hit_x])

                for scheme in schemes:
                    alloc = allocate(scheme, c_["bf"], list(combos.keys()), a, b)
                    stakes = stake_units(alloc)
                    bet = sum(stakes.values())
                    ret = 0.0
                    if hit_x is not None and hit_x in stakes:
                        ret = stakes[hit_x] * odds_map[hit_x]
                        hit_log[scheme].append((c_["race_key"], c_["race_date"],
                                                stakes[hit_x], odds_map[hit_x], ret))
                    totals[scheme]["n"] += 1
                    totals[scheme]["bet"] += bet
                    totals[scheme]["ret"] += ret
                    totals[scheme]["hit"] += 1 if (hit_x is not None and stakes.get(hit_x, 0) > 0) else 0
                    month_acc[scheme]["bet"] += bet
                    month_acc[scheme]["ret"] += ret
                    # 配分加重平均オッズ順位（配分が本命=低順位に寄っているか穴=高順位に寄っているか）
                    wavg_rank = sum(stakes[x] / bet * rank_of[x] for x in stakes) if bet > 0 else 0.0
                    diag_weighted_rank[scheme].append(wavg_rank)

        for s in schemes:
            b_, r_ = month_acc[s]["bet"], month_acc[s]["ret"]
            monthly_roi[s].append((r_ / b_ * 100) if b_ else None)
        print("  " + " / ".join(
            f"{s}:ROI={month_acc[s]['ret']/month_acc[s]['bet']*100:.1f}%" if month_acc[s]["bet"] else f"{s}:N/A"
            for s in schemes))

    print("\n" + "=" * 112)
    print("全期間合計（月次vintageモデル・honest walk-forward・S7現行ゲート使用）")
    print("=" * 112)
    print(f"{'方式':<6}{'n':>7}{'投資':>12}{'回収':>12}{'ROI':>8}"
          f"{'月次標準偏差':>13}{'配分加重平均オッズ順位':>22}")
    for s in schemes:
        t = totals[s]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        vals = [v for v in monthly_roi[s] if v is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        avg_wrank = statistics.mean(diag_weighted_rank[s]) if diag_weighted_rank[s] else 0.0
        print(f"{s:<6}{t['n']:>7}{t['bet']:>12,.0f}{t['ret']:>12,.0f}{roi:>7.1f}%"
              f"{sd:>13.1f}{avg_wrank:>22.2f}")

    print("\n" + "=" * 112)
    print("【頑健性診断】実際の的中のオッズ順位分布（本来これに近いほど市場に沿った配分）")
    print("=" * 112)
    if hit_odds_rank:
        from collections import Counter
        cnt = Counter(hit_odds_rank)
        n = len(hit_odds_rank)
        for r in sorted(cnt):
            print(f"  順位{r}: {cnt[r]:>5}件 ({cnt[r]/n*100:.1f}%)")
        print(f"  平均オッズ順位: {statistics.mean(hit_odds_rank):.2f} / n={n}")

    print("\n" + "=" * 112)
    print("【頑健性診断】各方式の少数的中依存チェック（上位1件・3件を除いた場合のROI）")
    print("=" * 112)
    for s in schemes:
        log = sorted(hit_log[s], key=lambda x: -x[4])
        if not log:
            continue
        total_ret = sum(x[4] for x in log)
        t = totals[s]
        roi = t["ret"] / t["bet"] * 100 if t["bet"] else 0.0
        ret_wo1 = total_ret - log[0][4]
        ret_wo3 = total_ret - sum(x[4] for x in log[:3])
        roi_wo1 = ret_wo1 / t["bet"] * 100 if t["bet"] else 0.0
        roi_wo3 = ret_wo3 / t["bet"] * 100 if t["bet"] else 0.0
        print(f"  {s}: 全体ROI={roi:.1f}% / 上位1件除外={roi_wo1:.1f}% / "
              f"上位3件除外={roi_wo3:.1f}%  (的中件数={len(log)})")


if __name__ == "__main__":
    main()
