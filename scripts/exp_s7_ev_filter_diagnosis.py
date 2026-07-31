"""【診断】ev_threshold_filterが「最も可能性が高い(低オッズ)組」を除外していないか（2026-07-30）。

ユーザー指摘: 「的中率が低すぎる。月次0%もだいぶ悪く、対象レースを十分に
拾えていないように思う」への回答調査。

## 仮説

[[keirin_clean_baseline_market_efficiency_2026_07_30]]で判明済みの
「モデルは高オッズ帯で実確率を1.3〜2.2倍過大評価する」バイアスにより、
EV = our_prob × odds のフィルタが**高オッズ（人気薄）の組を優先的に通し、
最も的中しやすい低オッズ（本命）の組を除外している**可能性が高い。
市場は efficient なので、5点のうち最もオッズが低い組=最も可能性が高い組
のはずだが、これがフィルタで弾かれていれば「拾えていない」の直接の原因になる。

## 検証方法

S7候補（本番ゲート通過後）の各レースについて:
  - 5点をオッズ昇順で並べ、各点の順位（1=最低オッズ=本命側）を記録
  - ev_threshold_filter(EV>=1.0)で「採用された点」と「除外された点」に分け、
    それぞれの平均オッズ順位を比較
  - 最もオッズが低い点（1位）が採用されたか除外されたかの比率を集計
  - 実際に的中した組の的中時点でのオッズ順位分布も見る

軽量診断のため、liftはTRAIN全期間相当の1回推定のみで固定し、直近半年程度の
月をサンプルして傾向を見る（全期間honestの精緻な数値化はexp_s7_ev_threshold_
staking_validation.pyが別途担当）。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import s7_evening_reselect
from src.wt_vintage_config import monthly_windows

from scripts.exp_s7_ev_threshold_staking_validation import (
    build_candidates_with_lineinfo, estimate_lifts, race_our_probs,
)

SAMPLE_MONTHS = 6   # 直近から数えてこの数の月を診断対象にする


def main():
    windows = monthly_windows()
    # lift推定用に全体からサンプリング（診断目的なので簡易に前半4ヶ月を使う）
    lift_windows = windows[:4]
    hist_races = []
    for date_from, date_to, eval_model, win_model in lift_windows:
        print(f"[lift-build] {date_from}〜{date_to}", flush=True)
        candidates, _pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        for c_ in candidates:
            hist_races.append((c_["bf"], c_["actual_top3"]))
    lifts = estimate_lifts(hist_races)
    print("[lift]", {k: round(v, 3) for k, v in lifts.items()})

    sample_windows = windows[-SAMPLE_MONTHS:]

    rank_included = []
    rank_excluded = []
    rank1_included_count = rank1_total_count = 0
    hit_rank_dist = []
    n_zero_qualify = n_races = 0

    for date_from, date_to, eval_model, win_model in sample_windows:
        print(f"\n[diag] {date_from}〜{date_to}", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        for _d, day_cands in by_day.items():
            for c_ in s7_evening_reselect(day_cands, [], set()):
                axis1, axis2 = c_["axis1"], c_["axis2"]
                trio = c_["trio"]
                combos = []
                for x in c_["others"]:
                    key = frozenset({axis1, axis2, x})
                    if key in trio:
                        combos.append((key, trio[key]))
                if len(combos) < 2:
                    continue
                n_races += 1
                # オッズ昇順に順位付け（1=最低オッズ=本命側）
                combos_sorted = sorted(combos, key=lambda kv: kv[1])
                rank_of = {key: i + 1 for i, (key, _o) in enumerate(combos_sorted)}

                probs = race_our_probs(c_["bf"], lifts)
                if probs is None:
                    continue
                qualify_keys = set()
                for key, odds in combos:
                    p = probs.get(key, 0.0)
                    if p * odds >= 1.0:
                        qualify_keys.add(key)

                if not qualify_keys:
                    n_zero_qualify += 1
                else:
                    rank1_total_count += 1
                    rank1_key = combos_sorted[0][0]
                    if rank1_key in qualify_keys:
                        rank1_included_count += 1

                for key, _odds in combos:
                    r = rank_of[key]
                    if key in qualify_keys:
                        rank_included.append(r)
                    else:
                        rank_excluded.append(r)

                actual_top3 = c_["actual_top3"]
                for key, _odds in combos:
                    if key == actual_top3:
                        hit_rank_dist.append(rank_of[key])

    print("\n" + "=" * 100)
    print("【診断結果】")
    print("=" * 100)
    print(f"診断対象レース数: {n_races}")
    print(f"全点EV<1.0で見送り: {n_zero_qualify} ({n_zero_qualify/n_races*100:.1f}%)")
    print(f"\n最もオッズが低い点(1位=本命側)が採用された率: "
          f"{rank1_included_count}/{rank1_total_count} "
          f"({rank1_included_count/rank1_total_count*100 if rank1_total_count else 0:.1f}%)")
    if rank_included:
        print(f"\n採用された点の平均オッズ順位: {sum(rank_included)/len(rank_included):.2f} "
              f"(n={len(rank_included)}, 1に近いほど本命寄り)")
    if rank_excluded:
        print(f"除外された点の平均オッズ順位: {sum(rank_excluded)/len(rank_excluded):.2f} "
              f"(n={len(rank_excluded)})")
    if hit_rank_dist:
        from collections import Counter
        cnt = Counter(hit_rank_dist)
        print(f"\n実際に的中した組のオッズ順位分布 (n={len(hit_rank_dist)}):")
        for r in sorted(cnt):
            print(f"  順位{r}: {cnt[r]}件 ({cnt[r]/len(hit_rank_dist)*100:.1f}%)")
        print(f"  → 本来は順位1(最低オッズ)が最頻のはず（市場が効率的なら）。"
              f"採用点の平均順位({sum(rank_included)/len(rank_included):.2f})がこれより"
              f"大きいほど「本命を除外し穴を拾っている」ことを意味する。")


if __name__ == "__main__":
    main()
