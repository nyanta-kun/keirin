"""現行推奨のランク別再評価 — リーク無し・本番忠実セマンティクス（docs/analysis/18 の続き）。

ランク = 本番 _assign_tier（SS/S/A）。SSは本番仕様のBOX6点と旧直線3点を併記。
ガミ帯 = 使用レッグの最安オッズ: 推奨(≥5) / B(3-5) / 見送り(<3) / 全件。
arm A（TRAIN期間限定学習）= live予測と同じ「過去のみ学習→未来を予測」の誠実な推定値。
払戻=最終オッズ=上限値。期間: TRAIN 2023-07〜2025-06 / VAL 〜2026-02 / HOLDOUT 2026-03〜06-12。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp_leakfree_rescore_wt import collect, TRAIN, VAL
from roi_robustness_wt import roi_summary

INF = float("inf")
BANDS = [("推奨>=5", 5, INF), ("B 3-5", 3, 5), ("見送り<3", 0, 3), ("全件", 0, INF)]
ROWS = [("SS BOX6(本番)", "SS", "tf6"), ("SS 直線3点(旧)", "SS", "tf3"),
        ("S  trio3点", "S", "trio3"), ("A  trio3点", "A", "trio3")]


def cell(races, arm, tier, key, lo, hi):
    pays, bets = [], []
    for r in races:
        a = r[arm]
        if a["tier"] != tier:
            continue
        legs = a[key]
        if not legs:
            continue
        m = min(o for o, _ in legs)
        if not (lo <= m < hi):
            continue
        pays.append(sum(o * 100 for o, hit in legs if hit))
        bets.append(len(legs) * 100)
    return roi_summary(pays, bets), len(pays)


def fmt(s, n):
    if n == 0:
        return f"{0:>4}R  --"
    return (f"{n:>4}R {s['roi']:>5.0%} 的{s['hit_rate']:>4.0%} "
            f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] 除{s['roi_ex_max']:>4.0%}")


def main():
    races = collect()
    by = {"TRAIN": [r for r in races if r["date"] <= TRAIN[1]],
          "VAL":   [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
          "HOLD":  [r for r in races if r["date"] > VAL[1]]}
    print(f"  races: TR {len(by['TRAIN'])} / VA {len(by['VAL'])} / HO {len(by['HOLD'])}")
    for arm, albl in (("free", "armA リーク無し（live相当の誠実推定）"),
                      ("prod", "armB 本番lgbm_wt（参考・全期間学習済）")):
        print(f"\n{'='*118}\n  ◇ {albl}\n{'='*118}")
        for rlbl, tier, key in ROWS:
            print(f"\n  ◆ ランク {rlbl}")
            print(f"    {'ガミ帯':<10}{'TRAIN':<42}{'VAL':<42}{'HOLDOUT'}")
            for blbl, lo, hi in BANDS:
                cols = [fmt(*cell(by[p], arm, tier, key, lo, hi))
                        for p in ("TRAIN", "VAL", "HOLD")]
                print(f"    {blbl:<10}{cols[0]:<42}{cols[1]:<42}{cols[2]}")


if __name__ == "__main__":
    main()
