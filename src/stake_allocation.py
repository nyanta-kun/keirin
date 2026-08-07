"""買い目ごとの賭け金配分（netkeirin 入稿用・2026-08-07 新設）。

## なぜ均等割りをやめるのか

netkeirin の的中率は **ガミ（払戻 < 投資）を不的中として扱う**。1レース10,000円を
点数で均等割りすると「ガミにならない境界＝的中オッズが点数を超えること」になり、
5点買いなら 5.0倍未満の的中はすべて不的中表示になる。実測（2026-06-08〜08-06・
7車1,061レース）で **的中 52.0% に対し表示されるのは 25.1%＝的中の 51.8% がガミ**。

そこで**想定着地オッズに応じて配分**する。三連複・軸2車固定なら買う点の差は
3列目の車だけなので、点ごとの相対的な当たりやすさが分かれば全点の払戻をそろえられる。

    実質的中率 25.07% → 34.97%（+9.90pt / 95%CI [+7.63, +12.35] / P(差>0)=100%）
    ROI 81.8% → 79.6%（−2.27pt / CI [−8.08, +2.94] ＝ 有意でない）

## 想定着地オッズの作り方

朝8:00 の板は薄く、最終オッズとの中央値の比は 0.85（15%下振れ）で個別誤差も大きい。
一方で**一律の割引は比率を変えないので無意味**（配分に効くのは相対値だけ）。
使えるのは次の2つで、実測ではどちらも単独で同程度に効く:

| 重み | 実質的中% | 相対確率のL1誤差 |
|---|---|---|
| 均等 | 25.07 | 0.601 |
| モデルの3着内率 p3 のみ | 33.55 | 0.376 |
| 1/朝オッズ のみ | 34.78 | 0.339 |
| **相乗平均（λ=0.5）** | **35.25** | **0.330** |
| 最終オッズ（実装不能・上限） | 48.54 | 0 |

⚠️ **相乗平均が単独より有意に良いわけではない**（blend−morning は +0.19pt
[−0.94, +1.23]・P=60.4%）。両方を使うのは**どちらかが欠けても劣化しないため**で、
実際 朝オッズが買う点すべてに揃うのは **65%** しかない（残りは p3 単独で走る）。

⚠️ **レース単位のゲート（期待値の下限・保証下限）は入れない**。測ったが全部不採用。
   買う点の合成ブックは約0.65なので想定上の「ガミ無し保証」は96.4%のレースで成立し、
   それでも実ガミ率は32.8%。ガミの原因はレース水準ではなく**点ごとの相対誤差**で、
   下限はそれを見ていない。詳細は memory `keirin_netkeirin_gami_allocation_2026_08_07`。
"""
from __future__ import annotations

BUDGET_DEFAULT = 10_000
UNIT_DEFAULT = 100

# 朝オッズ側の指数。0=モデルのみ / 1=朝オッズのみ。
# 両窓（6月・7-8月）で最も安定していた 0.5（相乗平均）を採る。
# 差は有意でないので**単一レースや単月の結果でこの値を動かさないこと**。
LANDING_LAMBDA = 0.5

# 重みの出どころ。ログと dry-run 表示に使う（どの経路で走ったか分からないと
# 「朝オッズが取れていない」ことに気づけない）。
SOURCE_BLEND = "blend"
SOURCE_ODDS = "odds"
SOURCE_MODEL = "model"
SOURCE_EQUAL = "equal"


def _usable_odds(legs: list[int], odds: dict[int, float] | None) -> bool:
    """買う点**すべて**にオッズがあるときだけ使う。

    一部だけ使うと、欠けた点の重みを別の尺度で決めることになり比率が壊れる。
    """
    if not odds:
        return False
    return all(isinstance(odds.get(t), (int, float)) and odds.get(t, 0) > 0 for t in legs)


def _usable_probs(legs: list[int], probs: dict[int, float] | None) -> bool:
    if not probs:
        return False
    return all(isinstance(probs.get(t), (int, float)) and probs.get(t, 0) > 0 for t in legs)


def landing_weights(
    legs: list[int],
    morning_odds: dict[int, float] | None,
    top3_probs: dict[int, float] | None,
    lam: float = LANDING_LAMBDA,
) -> tuple[dict[int, float], str]:
    """3列目の車ごとの重み（賭け金はこれに比例させる）と、その出どころを返す。

    legs:         買う相手（3列目）の車番
    morning_odds: {3列目の車番: その点の朝の三連複オッズ}。欠けていてよい
    top3_probs:   {車番: モデルの3着内率 0-1}。欠けていてよい

    重みは「その点の当たりやすさ」に比例する。賭け金をこれに比例させると
    全点の払戻がそろい、どの点で決まっても元返し以上になりやすくなる。
    """
    if not legs:
        raise ValueError("legs が空です")
    has_o, has_p = _usable_odds(legs, morning_odds), _usable_probs(legs, top3_probs)
    if has_o and has_p:
        return ({t: (1.0 / morning_odds[t]) ** lam * top3_probs[t] ** (1.0 - lam)
                 for t in legs}, SOURCE_BLEND)
    if has_o:
        return {t: 1.0 / morning_odds[t] for t in legs}, SOURCE_ODDS
    if has_p:
        return {t: float(top3_probs[t]) for t in legs}, SOURCE_MODEL
    return {t: 1.0 for t in legs}, SOURCE_EQUAL


def allocate_budget(
    weights: dict[int, float],
    budget: int = BUDGET_DEFAULT,
    unit: int = UNIT_DEFAULT,
) -> dict[int, int]:
    """重み → 1点あたりの賭け金（unit の倍数・合計は必ず budget）。

    ⚠️ **どの点も必ず1単位以上を持つ**。比例配分だけだと極端に人気薄の点が
       0円になり、**買い目の集合が黙って変わってしまう**（点数が減る）。
       買う目を決めるのは配分の役目ではない。

    ⚠️ 端数の寄せ先は「想定払戻が最小の点」＝ **口数 / 重み が最小の点**。
       重みが 1/想定オッズ に比例するので、これはオッズを参照せずに決まる。
       最終オッズで決めると先読みになる（検証スクリプトで実際にやらかし、
       効果を 2pt 過大評価していた）。
    """
    if not weights:
        raise ValueError("weights が空です")
    n_units = budget // unit
    if n_units < len(weights):
        raise ValueError(f"予算 {budget} 円では {len(weights)} 点に配分できません")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("重みの合計が0以下です")

    # 全点に1口ずつ配ってから、残りを比例配分する
    units = {k: 1 for k in weights}
    rest = n_units - len(weights)
    share = {k: rest * w / total for k, w in weights.items()}
    for k, s in share.items():
        units[k] += int(s)
    for _ in range(n_units - sum(units.values())):
        units[min(units, key=lambda k: units[k] / max(weights[k], 1e-12))] += 1
    return {k: v * unit for k, v in units.items()}


def tilted_stakes(
    legs: list[int],
    morning_odds: dict[int, float] | None,
    top3_probs: dict[int, float] | None,
    budget: int = BUDGET_DEFAULT,
    unit: int = UNIT_DEFAULT,
    lam: float = LANDING_LAMBDA,
) -> tuple[dict[int, int], str]:
    """買う相手ごとの賭け金と、重みの出どころを返す。"""
    weights, source = landing_weights(legs, morning_odds, top3_probs, lam)
    return allocate_budget(weights, budget, unit), source


def group_by_stake(stakes: dict[int, int]) -> list[tuple[int, list[int]]]:
    """同額の相手をまとめる。[(賭け金, [車番,…]), …] を賭け金の降順で返す。

    netkeirin の `kaime` は1行につき1つの `bet_money` しか持てないので、
    同額の相手は1行にまとめられる（行数が減って買い目が読みやすくなる）。
    """
    by_stake: dict[int, list[int]] = {}
    for car, stake in stakes.items():
        by_stake.setdefault(stake, []).append(car)
    return [(s, sorted(cars)) for s, cars in
            sorted(by_stake.items(), key=lambda kv: -kv[0])]
