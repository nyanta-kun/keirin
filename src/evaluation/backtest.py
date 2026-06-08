"""
バックテスト・買い目戦略比較

【実装している買い目戦略】

■ 3連複
  box_top3    : 上位3頭ボックス         C(3,3)=1点
  box_top4    : 上位4頭ボックス         C(4,3)=4点
  box_top5    : 上位5頭ボックス         C(5,3)=10点
  jiku1_3     : 軸1頭×残り上位3頭流し   C(3,2)=3点
  jiku1_4     : 軸1頭×残り上位4頭流し   C(4,2)=6点
  jiku2_3     : 軸2頭×残り上位3頭流し   3点

■ 3連単（標準）
  str_top3    : 上位3頭マルチ(全順列)   3!=6点
  str_jiku1_3 : 1着固定×2着3着上位3頭マルチ  P(3,2)=6点
  str_jiku1_4 : 1着固定×2着3着上位4頭マルチ  P(4,2)=12点
  str_jiku12_3: 1-2着固定×3着上位3頭   3点
  str_top2_1ch: 上位2頭を各々1着固定×2着3着上位4頭マルチ  2×P(4,2)=24点

■ 3連単（穴狙い）
  str_chuana_2nd1st : #2を1着固定×上位5頭の残りマルチ  P(4,2)=12点
  str_chuana_3rd1st : #3を1着固定×上位5頭の残りマルチ  P(4,2)=12点
  str_oana_4th1st   : #4を1着固定×上位4頭の残りマルチ  P(3,2)=6点
  str_oana_top3_4th : #1〜#3を各1着×#4含む4頭マルチ    3×P(3,2)=18点

■ 3連単（的中率重視）
  str_top3_each_1st  : 上位3頭を各々1着×残り上位3頭マルチ  3×P(3,2)=18点
  str_jiku1_cover2nd : #1を1着or2着に固定             1着12点+2着6点=18点
  str_top2_both_ord  : #1-#2を両順序1-2着×3着上位4頭  2×4=8点

■ 2車複（quinella / 競輪表記）
  quinella_top2 : 上位2車BOX(1点)
  quinella_23   : 2-3位BOX(1点)

■ 2車単（exacta / 競輪表記）
  exacta_12 : 1位→2位(1点)
  exacta_21 : 2位→1位(1点)

■ ワイド（wide）
  wide_12 : 1-2位ワイド(1点)
  wide_23 : 2-3位ワイド(1点)
  wide_13 : 1-3位ワイド(1点)
"""
import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..preprocessing.feature_engineer import FEATURE_COLS, TARGET_COL
from ..database import get_connection


# ---------------------------------------------------------------------------
# 買い目生成関数
# ---------------------------------------------------------------------------

def _combos_box(ranked: list[int], take: int) -> list[frozenset]:
    """上位take頭ボックス: C(take,3)点"""
    return [frozenset(c) for c in itertools.combinations(ranked[:take], 3)]


def _combos_jiku1_n(ranked: list[int], n: int) -> list[frozenset]:
    """軸1頭(1位) × 残り上位n頭 流し"""
    pivot = ranked[0]
    rest = ranked[1:n + 1]
    return [frozenset([pivot, a, b]) for a, b in itertools.combinations(rest, 2)]


def _combos_jiku2_n(ranked: list[int], n: int) -> list[frozenset]:
    """軸2頭(1-2位) × 残り上位n頭 流し"""
    p1, p2 = ranked[0], ranked[1]
    rest = ranked[2:n + 2]
    return [frozenset([p1, p2, r]) for r in rest]


def _combos_str_multi(ranked: list[int], take: int) -> list[tuple]:
    """上位take頭マルチ(全順列)"""
    return list(itertools.permutations(ranked[:take], 3))


def _combos_str_jiku1_n(ranked: list[int], n: int) -> list[tuple]:
    """1着固定(1位) × 2着3着は上位n頭の残りマルチ"""
    pivot = ranked[0]
    rest = ranked[1:n + 1]
    return [(pivot, a, b) for a, b in itertools.permutations(rest, 2)]


def _combos_str_jiku12_n(ranked: list[int], n: int) -> list[tuple]:
    """1-2着固定(1-2位) × 3着は上位n頭の残り"""
    p1, p2 = ranked[0], ranked[1]
    rest = ranked[2:n + 2]
    return [(p1, p2, r) for r in rest]


def _combos_str_top2_1ch(ranked: list[int], n: int) -> list[tuple]:
    """上位2頭を各々1着 × 2着3着は上位n頭の残りマルチ"""
    combos = []
    for pivot in ranked[:2]:
        rest = [r for r in ranked[:n + 1] if r != pivot][:n]
        for a, b in itertools.permutations(rest, 2):
            combos.append((pivot, a, b))
    return list(set(combos))


def _combos_str_jiku_nth_1st(ranked: list[int], nth: int, rest_n: int) -> list[tuple]:
    """nth位を1着固定 × 2着3着は上位rest_n頭の残りからマルチ"""
    if len(ranked) < nth:
        return []
    pivot = ranked[nth - 1]
    candidates = [r for r in ranked[:rest_n + 1] if r != pivot][:rest_n]
    return [(pivot, a, b) for a, b in itertools.permutations(candidates, 2)]


def _combos_str_top3_each_1st(ranked: list[int], rest_n: int) -> list[tuple]:
    """上位3頭を各々1着 × 2着3着は残り上位rest_n頭マルチ (3×P(rest_n,2)点)"""
    combos = []
    for i in range(3):
        pivot = ranked[i]
        rest = [r for r in ranked[:rest_n + 2] if r != pivot][:rest_n]
        for a, b in itertools.permutations(rest, 2):
            combos.append((pivot, a, b))
    return list(set(combos))


def _combos_str_jiku1_cover2nd(ranked: list[int], rest_n: int) -> list[tuple]:
    """#1を1着または2着に固定
    - 1着: (axis, a, b) P(rest_n,2)点
    - 2着: (#2か#3が1着, axis, 残り) 2×(rest_n-1)点
    """
    axis = ranked[0]
    combos = []
    rest = [r for r in ranked[:rest_n + 1] if r != axis][:rest_n]
    for a, b in itertools.permutations(rest, 2):
        combos.append((axis, a, b))
    for winner in ranked[1:3]:
        thirds = [r for r in ranked[:rest_n + 1] if r not in (axis, winner)][:rest_n - 1]
        for t in thirds:
            combos.append((winner, axis, t))
    return list(set(combos))


def _combos_str_top2_both_orders(ranked: list[int], rest_n: int) -> list[tuple]:
    """#1-#2を両順序1-2着固定 × 3着は残り上位rest_n頭 (2×rest_n点)"""
    p1, p2 = ranked[0], ranked[1]
    rest = [r for r in ranked if r not in (p1, p2)][:rest_n]
    combos = []
    for r in rest:
        combos.append((p1, p2, r))
        combos.append((p2, p1, r))
    return combos


def _combos_str_top3_with_4th(ranked: list[int], rest_n: int) -> list[tuple]:
    """上位3頭を各々1着 × 2着3着に#4を必ず含む流し"""
    if len(ranked) < 4:
        return []
    anchor = ranked[3]  # 4位
    combos = []
    for pivot in ranked[:3]:
        rest = [r for r in ranked[:rest_n + 1] if r != pivot][:rest_n]
        for a, b in itertools.permutations(rest, 2):
            if anchor in (a, b):
                combos.append((pivot, a, b))
    return list(set(combos))


def _combos_pair_set(ranked: list[int], r1: int, r2: int) -> list[frozenset]:
    """指定ランク2頭のBOXペア: 1点（馬連/ワイド共用）"""
    if len(ranked) < max(r1, r2):
        return []
    return [frozenset([ranked[r1 - 1], ranked[r2 - 1]])]


def _combos_pair_ordered(ranked: list[int], r_first: int, r_second: int) -> list[tuple]:
    """指定ランク2頭の順序付きペア: 1点（2連単用）"""
    if len(ranked) < max(r_first, r_second):
        return []
    return [(ranked[r_first - 1], ranked[r_second - 1])]


def _combos_box2(ranked: list[int], take: int) -> list[frozenset]:
    """上位take頭の2頭BOX: C(take,2)点（馬連/ワイド共用）"""
    return [frozenset(c) for c in itertools.combinations(ranked[:take], 2)]


# ---------------------------------------------------------------------------
# 戦略定義
# ---------------------------------------------------------------------------

@dataclass
class BetStrategy:
    name: str
    label: str
    bet_type: str          # "trifecta_box" or "trifecta"
    combo_fn: callable
    combo_fn_kwargs: dict = field(default_factory=dict)

    def generate(self, ranked_frames: list[int]) -> list:
        return self.combo_fn(ranked_frames, **self.combo_fn_kwargs)

    @property
    def is_box(self) -> bool:
        return self.bet_type == "trifecta_box"

    @property
    def is_pair_set(self) -> bool:
        return self.bet_type in ("quinella", "wide")


STRATEGIES: list[BetStrategy] = [
    # --- 3連複 ---
    BetStrategy("box_top3",  "3連複: 上位3頭BOX(1点)",      "trifecta_box", _combos_box,    {"take": 3}),
    BetStrategy("box_top4",  "3連複: 上位4頭BOX(4点)",      "trifecta_box", _combos_box,    {"take": 4}),
    BetStrategy("box_top5",  "3連複: 上位5頭BOX(10点)",     "trifecta_box", _combos_box,    {"take": 5}),
    BetStrategy("jiku1_3",   "3連複: 軸1頭×残り3頭流し(3点)","trifecta_box", _combos_jiku1_n,{"n": 3}),
    BetStrategy("jiku1_4",   "3連複: 軸1頭×残り4頭流し(6点)","trifecta_box", _combos_jiku1_n,{"n": 4}),
    BetStrategy("jiku2_3",   "3連複: 軸2頭×残り3頭流し(3点)","trifecta_box", _combos_jiku2_n,{"n": 3}),
    # --- 3連単（標準） ---
    BetStrategy("str_top3",    "3連単: 上位3頭マルチ(6点)",         "trifecta", _combos_str_multi,    {"take": 3}),
    BetStrategy("str_jiku1_3", "3連単: 1着固定×残り3頭マルチ(6点)", "trifecta", _combos_str_jiku1_n,  {"n": 3}),
    BetStrategy("str_jiku1_4", "3連単: 1着固定×残り4頭マルチ(12点)","trifecta", _combos_str_jiku1_n,  {"n": 4}),
    BetStrategy("str_jiku12_3","3連単: 1-2着固定×残り3頭(3点)",    "trifecta", _combos_str_jiku12_n, {"n": 3}),
    BetStrategy("str_top2_1ch","3連単: 上位2頭1着×残り4頭マルチ(24点)","trifecta",_combos_str_top2_1ch,{"n": 4}),
]

# 穴狙い専用戦略
ANA_STRATEGIES: list[BetStrategy] = [
    # 中穴: モデル2位が1着に来るパターン (番手差しなど)
    BetStrategy("str_chuana_2nd1st", "3連単(中穴): #2を1着×上位5頭マルチ(12点)", "trifecta",
                _combos_str_jiku_nth_1st, {"nth": 2, "rest_n": 4}),
    # 中穴: モデル3位が1着に来るパターン (穴人気)
    BetStrategy("str_chuana_3rd1st", "3連単(中穴): #3を1着×上位5頭マルチ(12点)", "trifecta",
                _combos_str_jiku_nth_1st, {"nth": 3, "rest_n": 4}),
    # 大穴: モデル4位が1着に来るパターン
    BetStrategy("str_oana_4th1st",   "3連単(大穴): #4を1着×上位4頭マルチ(6点)",  "trifecta",
                _combos_str_jiku_nth_1st, {"nth": 4, "rest_n": 3}),
    # 中穴+: 上位3頭1着固定だが2着3着に#4を必ず含む組み合わせ
    BetStrategy("str_chuana_top3_4th","3連単(中穴+): 上位3頭1着×#4含む流し(~9点)", "trifecta",
                _combos_str_top3_with_4th, {"rest_n": 4}),
]

# 的中率重視戦略（買い目を広げ / 軸の2着まで拾う）
HITRATE_STRATEGIES: list[BetStrategy] = [
    # 上位3頭すべてを1着軸として展開 (18点)
    BetStrategy("str_top3_each_1st",  "3連単(的中率): 上位3頭各1着×残り3頭マルチ(18点)", "trifecta",
                _combos_str_top3_each_1st, {"rest_n": 3}),
    # 軸馬(#1)が1着or2着どちらでも的中 (18点)
    BetStrategy("str_jiku1_cover2nd", "3連単(的中率): #1を1着or2着に固定(18点)", "trifecta",
                _combos_str_jiku1_cover2nd, {"rest_n": 4}),
    # #1-#2の着順両対応×3着上位4頭 (8点)
    BetStrategy("str_top2_both_ord",  "3連単(的中率): #1-#2両順序×3着4頭(8点)", "trifecta",
                _combos_str_top2_both_orders, {"rest_n": 4}),
]

QUINELLA_STRATEGIES: list[BetStrategy] = [
    BetStrategy("quinella_top2", "2車複: 上位2車BOX(1点)",  "quinella", _combos_pair_set,     {"r1": 1, "r2": 2}),
    BetStrategy("quinella_23",   "2車複: 2-3位BOX(1点)",    "quinella", _combos_pair_set,     {"r1": 2, "r2": 3}),
]

EXACTA_STRATEGIES: list[BetStrategy] = [
    BetStrategy("exacta_12", "2車単: 1位→2位(1点)", "exacta", _combos_pair_ordered, {"r_first": 1, "r_second": 2}),
    BetStrategy("exacta_21", "2車単: 2位→1位(1点)", "exacta", _combos_pair_ordered, {"r_first": 2, "r_second": 1}),
]

WIDE_STRATEGIES: list[BetStrategy] = [
    BetStrategy("wide_12", "ワイド: 1-2位(1点)", "wide", _combos_pair_set, {"r1": 1, "r2": 2}),
    BetStrategy("wide_23", "ワイド: 2-3位(1点)", "wide", _combos_pair_set, {"r1": 2, "r2": 3}),
    BetStrategy("wide_13", "ワイド: 1-3位(1点)", "wide", _combos_pair_set, {"r1": 1, "r2": 3}),
]

ALL_STRATEGIES = STRATEGIES + ANA_STRATEGIES + HITRATE_STRATEGIES + QUINELLA_STRATEGIES + EXACTA_STRATEGIES + WIDE_STRATEGIES


# ---------------------------------------------------------------------------
# 場×戦略フィルター（テスト期間実績ベース、回収率が著しく低い組み合わせを除外）
# ---------------------------------------------------------------------------
# 場×戦略フィルターは廃止（統計的根拠不十分 + テストデータ使用による過学習）
# データ量が十分になった段階で訓練期間のみで再評価すること
VENUE_STRATEGY_FILTER: dict[str, frozenset] = {}


# ---------------------------------------------------------------------------
# コア計算ヘルパー
# ---------------------------------------------------------------------------

def _apply_pred_prob(model, df: pd.DataFrame) -> pd.DataFrame:
    """pred_probを計算してdfに追加（DNS選手・FEATURE_COLS欠損行は除去）"""
    # finish_positionがNULLのDNS選手を先に除去（ファントムベット防止）
    if "finish_position" in df.columns:
        df = df[df["finish_position"].notna()].copy()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = pd.DataFrame(df[FEATURE_COLS].values, columns=FEATURE_COLS)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    return df


def _filter_by_top1(df: pd.DataFrame, max_top1_prob: float) -> pd.DataFrame:
    """レース内top1_probが閾値を超えるレースを除外"""
    top1 = df.groupby("race_key")["pred_prob"].max()
    valid = top1[top1 <= max_top1_prob].index
    return df[df["race_key"].isin(valid)]


def _filter_by_n_riders(df: pd.DataFrame, max_riders: int) -> pd.DataFrame:
    """出走頭数が max_riders を超えるレースを除外"""
    race_sizes = df.groupby("race_key")["frame_no"].count()
    valid = race_sizes[race_sizes <= max_riders].index
    return df[df["race_key"].isin(valid)]


def _evaluate_combos(s: "BetStrategy", combos, actual_order: tuple,
                     top3_set: frozenset, race_payouts: dict) -> tuple[bool, int]:
    """bet_type に応じて的中判定とペイアウト合算を返す"""
    hit = False
    payout = 0
    if s.bet_type == "trifecta_box":
        pk = "=".join(map(str, sorted(top3_set)))
        for combo in combos:
            if combo == top3_set:
                payout = race_payouts.get(("trifecta_box", pk), 0)
                hit = True
                break
    elif s.bet_type == "trifecta":
        pk = "-".join(map(str, actual_order))
        for combo in combos:
            if combo == actual_order:
                payout = race_payouts.get(("trifecta", pk), 0)
                hit = True
                break
    elif s.bet_type == "quinella":
        actual_q = frozenset([actual_order[0], actual_order[1]])
        for combo in combos:
            if combo == actual_q:
                pk = "=".join(map(str, sorted(combo)))
                payout = race_payouts.get(("quinella", pk), 0)
                hit = True
                break
    elif s.bet_type == "wide":
        for combo in combos:
            if frozenset(combo).issubset(top3_set):
                pk = "=".join(map(str, sorted(combo)))
                payout += race_payouts.get(("wide", pk), 0)
                hit = True
    elif s.bet_type == "exacta":
        actual_e = (actual_order[0], actual_order[1])
        for combo in combos:
            if combo == actual_e:
                pk = f"{combo[0]}-{combo[1]}"
                payout = race_payouts.get(("exacta", pk), 0)
                hit = True
                break
    return hit, payout


def _compute_accum(df: pd.DataFrame, strategies: list[BetStrategy],
                   payout_map: dict) -> dict[str, dict]:
    """pred_prob計算済みdfで戦略ごとの集計を行う"""
    accum = {s.name: {"bets": 0, "returns": 0, "hits": 0} for s in strategies}

    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        ranked = grp["frame_no"].tolist()

        actual_top3_set = frozenset(
            grp[grp["finish_position"] <= 3]["frame_no"].tolist()
        )
        if len(actual_top3_set) < 3:
            continue

        actual_order = tuple(
            grp[grp["finish_position"].isin([1, 2, 3])]
            .sort_values("finish_position")["frame_no"].tolist()
        )
        race_payouts = payout_map.get(race_key, {})

        for s in strategies:
            combos = s.generate(ranked)
            n_combos = len(combos)
            if n_combos == 0:
                continue
            accum[s.name]["bets"] += n_combos * 100
            hit, payout = _evaluate_combos(s, combos, actual_order, actual_top3_set, race_payouts)
            if hit:
                accum[s.name]["returns"] += payout
                accum[s.name]["hits"] += 1

    return accum


def _accum_to_df(accum: dict, strategies: list[BetStrategy],
                 total_races: int) -> pd.DataFrame:
    rows = []
    for s in strategies:
        a = accum[s.name]
        bpr = a["bets"] / total_races / 100 if total_races else 0
        hit_rate = a["hits"] / total_races if total_races else 0
        roi = a["returns"] / a["bets"] if a["bets"] > 0 else 0
        rows.append({
            "戦略": s.label,
            "戦略名": s.name,
            "1Rあたり点数": f"{bpr:.0f}点",
            "的中率": f"{hit_rate:.1%}",
            "的中数": a["hits"],
            "回収率": f"{roi:.1%}",
            "回収率_raw": roi,
            "総投資(円)": a["bets"],
            "総回収(円)": a["returns"],
            "対象レース数": total_races,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# バックテスト本体
# ---------------------------------------------------------------------------

def run_backtest(model, df: pd.DataFrame,
                 strategies: list[BetStrategy] = None,
                 max_top1_prob: float | None = None,
                 max_riders: int | None = None) -> pd.DataFrame:
    """
    複数の買い目戦略でバックテストを実行し、結果DataFrameを返す。

    max_top1_prob: レース内の最高pred_probがこの値を超えるレースをスキップ。
                  None=全レース対象。
    max_riders:   出走頭数がこの値を超えるレースをスキップ。
                  6を指定すると6車立て以下のみ（実運用と同じ母集団）。
                  None=全レース対象。
    """
    if strategies is None:
        strategies = STRATEGIES

    df = _apply_pred_prob(model, df)
    if max_riders is not None:
        df = _filter_by_n_riders(df, max_riders)
    if max_top1_prob is not None:
        df = _filter_by_top1(df, max_top1_prob)

    payout_map = _load_payouts(df["race_key"].unique().tolist())
    accum = _compute_accum(df, strategies, payout_map)
    return _accum_to_df(accum, strategies, df["race_key"].nunique())


# ---------------------------------------------------------------------------
# 1日シミュレーション
# ---------------------------------------------------------------------------

# ラベル別 top1_prob 閾値（day-sim 用）
# 穴: <0.65 / 通常: 0.65〜0.70 / 安定: 0.70〜0.80 / SKIP: >=0.80
_DAY_SIM_TIERS: list[tuple[float, str]] = [
    (0.65, "穴"),
    (0.70, "通常"),
    (0.80, "安定"),
]
_DAY_SIM_SKIP_THRESHOLD = 0.80


def _get_race_tier(top1_prob: float) -> str:
    for threshold, label in _DAY_SIM_TIERS:
        if top1_prob < threshold:
            return label
    return "SKIP"


def run_day_simulation(model, df: pd.DataFrame,
                       strategies: list[BetStrategy] = None,
                       max_top1_prob: float | None = _DAY_SIM_SKIP_THRESHOLD
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1日分のレースに対して戦略を適用し、レース詳細と集計を返す。

    tier ラベル（race_rows の "tier" 列）:
      穴  … top1_prob < 0.65  （高回収・少数絞り込み）
      通常 … 0.65 <= top1 < 0.70  （標準BET）
      安定 … 0.70 <= top1 < 0.80  （的中率重視）
      SKIP … top1 >= 0.80

    Returns:
        (df_races, df_summary)
        df_races: レースごとの購入判定・的中・払戻・tier
        df_summary: 戦略ごとの集計（tier 別内訳付き）
    """
    if strategies is None:
        key_names = {"box_top4", "str_jiku1_4", "str_chuana_2nd1st"}
        strategies = [s for s in ALL_STRATEGIES if s.name in key_names]

    df = _apply_pred_prob(model, df)
    top1_per_race = df.groupby("race_key")["pred_prob"].max()
    payout_map = _load_payouts(df["race_key"].unique().tolist())

    race_rows = []
    tier_labels = [label for _, label in _DAY_SIM_TIERS] + ["SKIP"]
    # tier × strategy ごとの集計
    accum = {
        tier: {s.name: {"bets": 0, "returns": 0, "hits": 0} for s in strategies}
        for tier in tier_labels
    }

    skip_threshold = max_top1_prob if max_top1_prob is not None else float("inf")

    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        ranked = grp["frame_no"].tolist()
        top1_prob = top1_per_race[race_key]
        skipped = top1_prob >= skip_threshold
        tier = "SKIP" if skipped else _get_race_tier(top1_prob)

        top3 = frozenset(grp[grp["finish_position"] <= 3]["frame_no"].tolist())
        has_result = len(top3) >= 3
        actual_order = ()
        if has_result:
            actual_order = tuple(
                grp[grp["finish_position"].isin([1, 2, 3])]
                .sort_values("finish_position")["frame_no"].tolist()
            )

        race_payouts = payout_map.get(race_key, {})
        row = {
            "race_key": race_key,
            "venue_code": grp["venue_code"].iloc[0],
            "race_no": int(race_key.split("_")[-1]),
            "top1_prob": round(top1_prob, 3),
            "tier": tier,
            "skip": skipped,
        }

        for s in strategies:
            if skipped or not has_result:
                row[f"{s.name}_hit"] = None
                row[f"{s.name}_payout"] = 0
                continue

            combos = s.generate(ranked)
            n_combos = len(combos)
            if n_combos == 0:
                row[f"{s.name}_hit"] = False
                row[f"{s.name}_payout"] = 0
                continue

            accum[tier][s.name]["bets"] += n_combos * 100
            hit, payout = _evaluate_combos(s, combos, actual_order, top3, race_payouts)
            if hit:
                accum[tier][s.name]["returns"] += payout
                accum[tier][s.name]["hits"] += 1
            row[f"{s.name}_hit"] = hit
            row[f"{s.name}_payout"] = payout

        race_rows.append(row)

    df_races = pd.DataFrame(race_rows)

    # 集計: tier × strategy
    summary_rows = []
    for tier in tier_labels:
        for s in strategies:
            a = accum[tier][s.name]
            roi = a["returns"] / a["bets"] if a["bets"] > 0 else 0
            n_tier = len(df_races[df_races["tier"] == tier])
            summary_rows.append({
                "tier": tier,
                "戦略": s.label,
                "戦略名": s.name,
                "全レース数": len(df_races),
                "tier_レース数": n_tier,
                "的中数": a["hits"],
                "総投資(円)": a["bets"],
                "総回収(円)": a["returns"],
                "回収率_raw": roi,
            })
    df_summary = pd.DataFrame(summary_rows)

    return df_races, df_summary


# ---------------------------------------------------------------------------
# 会場別分析
# ---------------------------------------------------------------------------

def run_venue_analysis(model, df: pd.DataFrame,
                       strategies: list[BetStrategy] = None,
                       max_top1_prob: float | None = 0.70,
                       min_races: int = 50) -> pd.DataFrame:
    """
    会場ごとにバックテストを実行し比較する。

    min_races: 対象レース数がこれ未満の会場は除外
    """
    if strategies is None:
        key_names = {"box_top4", "str_jiku1_4", "str_chuana_2nd1st"}
        strategies = [s for s in ALL_STRATEGIES if s.name in key_names]

    df = _apply_pred_prob(model, df)
    if max_top1_prob is not None:
        df_filtered = _filter_by_top1(df, max_top1_prob)
    else:
        df_filtered = df

    all_race_keys = df_filtered["race_key"].unique().tolist()
    payout_map = _load_payouts(all_race_keys)

    # 会場コード → 名前マッピング
    venue_name_map: dict = {}
    try:
        from ..database import get_connection
        with get_connection() as conn:
            venues = conn.execute("SELECT venue_code, name FROM venue_info").fetchall()
            venue_name_map = {v[0]: v[1] for v in venues}
    except Exception:
        venue_name_map = {vc: vc for vc in df_filtered["venue_code"].unique()}

    rows = []
    for venue_code, venue_df in df_filtered.groupby("venue_code"):
        n_races = venue_df["race_key"].nunique()
        if n_races < min_races:
            continue

        accum = _compute_accum(venue_df, strategies, payout_map)

        row = {
            "会場": venue_name_map.get(venue_code, venue_code),
            "会場コード": venue_code,
            "対象レース数": n_races,
        }
        for s in strategies:
            a = accum[s.name]
            roi = a["returns"] / a["bets"] if a["bets"] > 0 else 0
            hit_rate = a["hits"] / n_races if n_races else 0
            row[f"{s.name}_的中率"] = hit_rate
            row[f"{s.name}_回収率"] = roi
            row[f"{s.name}_的中数"] = a["hits"]
            row[f"{s.name}_投資"] = a["bets"]
            row[f"{s.name}_回収"] = a["returns"]
        rows.append(row)

    return pd.DataFrame(rows)


def run_daily_venue_summary(model, df: pd.DataFrame,
                            strategies: list[BetStrategy] = None,
                            max_top1_prob: float | None = 0.70,
                            venue_filter: dict[str, frozenset] | None = VENUE_STRATEGY_FILTER) -> pd.DataFrame:
    """
    日付×会場ごとに両戦略の結果を集計して返す。

    venue_filter: 場×戦略フィルター。Noneで無効化。
                  デフォルトはVENUE_STRATEGY_FILTERを使用。
    Returns:
        行: (race_date, venue_name), 列: 戦略ごとの的中・投資・回収
    """
    if strategies is None:
        key_names = {"box_top4", "str_jiku1_4", "str_chuana_2nd1st"}
        strategies = [s for s in ALL_STRATEGIES if s.name in key_names]

    df = _apply_pred_prob(model, df)
    if max_top1_prob is not None:
        df = _filter_by_top1(df, max_top1_prob)

    all_race_keys = df["race_key"].unique().tolist()
    payout_map = _load_payouts(all_race_keys)

    # 会場名マップ
    venue_name_map: dict = {}
    try:
        with get_connection() as conn:
            for v in conn.execute("SELECT venue_code, name FROM venue_info").fetchall():
                venue_name_map[v[0]] = v[1]
    except Exception:
        pass

    vf = venue_filter or {}

    rows = []
    for (race_date, venue_code), grp_df in df.groupby(["race_date", "venue_code"]):
        n_races = grp_df["race_key"].nunique()
        if n_races == 0:
            continue

        venue_name = venue_name_map.get(venue_code, venue_code)
        skip_strategies = vf.get(venue_name, frozenset())

        # フィルター対象外の戦略だけで集計
        active_strategies = [s for s in strategies if s.name not in skip_strategies]
        accum = _compute_accum(grp_df, active_strategies, payout_map) if active_strategies else {}

        row = {
            "日付": race_date,
            "会場": venue_name,
            "購入R": n_races,
        }
        total_invest = 0
        total_return = 0
        for s in strategies:
            if s.name in skip_strategies:
                row[f"{s.name}_的中"] = -1   # フィルター除外フラグ
                row[f"{s.name}_投資"] = 0
                row[f"{s.name}_回収"] = 0
                row[f"{s.name}_roi"] = -1.0
            else:
                a = accum[s.name]
                roi = a["returns"] / a["bets"] if a["bets"] > 0 else 0
                row[f"{s.name}_的中"] = a["hits"]
                row[f"{s.name}_投資"] = a["bets"]
                row[f"{s.name}_回収"] = a["returns"]
                row[f"{s.name}_roi"] = roi
                total_invest += a["bets"]
                total_return += a["returns"]

        row["合計_投資"] = total_invest
        row["合計_回収"] = total_return
        row["合計_roi"] = total_return / total_invest if total_invest > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


def print_daily_venue_summary(df: pd.DataFrame, strategies: list[BetStrategy] = None):
    """日付×会場集計を表形式で表示"""
    if df.empty:
        print("データなし")
        return

    if strategies is None:
        key_names = {"box_top4", "str_jiku1_4", "str_chuana_2nd1st"}
        from src.evaluation.backtest import ALL_STRATEGIES
        strategies = [s for s in ALL_STRATEGIES if s.name in key_names]

    s_names = [s.name for s in strategies] if strategies else []
    # カラムから推定
    if not s_names:
        s_names = [c[:-4] for c in df.columns if c.endswith("_roi") and c != "合計_roi"]

    short = {"box_top4": "3連複4頭BOX(4pt)", "str_jiku1_4": "jiku1(12pt)", "str_chuana_2nd1st": "#2-1st(12pt)"}

    W = 92
    print(f"\n{'='*W}")
    print(f" 日別・場別集計（top1<70%フィルター適用後）")
    print(f"{'='*W}")
    hdr = f"{'日付':<12} {'会場':<7} {'購入R':>5}"
    for sn in s_names:
        lbl = short.get(sn, sn[:10])
        hdr += f"  {'的中':>4}/{lbl:<11}  {'回収率':>7}  {'損益':>9}"
    hdr += f"  {'合計損益':>10}"
    print(hdr)
    print("-" * W)

    prev_date = None
    date_totals: dict = {}

    for _, row in df.sort_values(["日付", "会場"]).iterrows():
        d = row["日付"]
        if prev_date and prev_date != d:
            # 日計を出力
            dt = date_totals[prev_date]
            _print_date_total(dt, s_names, short)
            print()
        prev_date = d

        if d not in date_totals:
            date_totals[d] = {sn: {"hits": 0, "invest": 0, "ret": 0} for sn in s_names}
            date_totals[d]["total_invest"] = 0
            date_totals[d]["total_ret"] = 0
            date_totals[d]["races"] = 0

        date_totals[d]["races"] += row["購入R"]
        date_totals[d]["total_invest"] += row["合計_投資"]
        date_totals[d]["total_ret"] += row["合計_回収"]

        line = f"  {d:<10} {row['会場']:<7} {int(row['購入R']):>5}"
        for sn in s_names:
            hits_raw = row.get(f"{sn}_的中", 0)
            filtered = (hits_raw == -1)
            invest = int(row.get(f"{sn}_投資", 0))
            ret = int(row.get(f"{sn}_回収", 0))
            roi = row.get(f"{sn}_roi", 0.0)
            if filtered:
                line += f"  {'---':>4}   {'除外':>7}   {'---':>9}"
            else:
                hits = int(hits_raw)
                profit = ret - invest
                marker = "*" if roi >= 1.10 else " "
                line += f"  {hits:>4}回  {roi:>6.1%}{marker}  {profit:>+9,}"
                date_totals[d][sn]["hits"] += hits
                date_totals[d][sn]["invest"] += invest
                date_totals[d][sn]["ret"] += ret
        profit_total = int(row["合計_回収"]) - int(row["合計_投資"])
        line += f"  {profit_total:>+10,}"
        print(line)

    # 最終日の日計
    if prev_date and prev_date in date_totals:
        _print_date_total(date_totals[prev_date], s_names, short)

    # 全体合計
    print()
    print("=" * W)
    total_invest = df["合計_投資"].sum()
    total_ret = df["合計_回収"].sum()
    overall_roi = total_ret / total_invest if total_invest > 0 else 0
    total_profit = total_ret - total_invest
    summary = f"  {'【週間合計】':<18} {int(df['購入R'].sum()):>5}"
    for sn in s_names:
        inv = df[f"{sn}_投資"].sum()
        ret = df[f"{sn}_回収"].sum()
        # -1 はフィルター除外フラグなので除いて合算
        hits = df[df[f"{sn}_的中"] >= 0][f"{sn}_的中"].sum()
        roi = ret / inv if inv > 0 else 0
        profit = ret - inv
        marker = "*" if roi >= 1.10 else " "
        summary += f"  {int(hits):>4}回  {roi:>6.1%}{marker}  {profit:>+9,}"
    summary += f"  {int(total_profit):>+10,}"
    print(summary)
    print(f"  全体回収率: {overall_roi:.1%}  投資 {int(total_invest):,}円  回収 {int(total_ret):,}円")
    print("=" * W)


def _print_date_total(dt: dict, s_names: list, short: dict):
    """日計行を出力"""
    line = f"  {'  ↳ 日計':<19} {int(dt['races']):>5}"
    for sn in s_names:
        inv = dt[sn]["invest"]
        ret = dt[sn]["ret"]
        hits = dt[sn]["hits"]
        roi = ret / inv if inv > 0 else 0
        profit = ret - inv
        marker = "*" if roi >= 1.10 else " "
        line += f"  {int(hits):>4}回  {roi:>6.1%}{marker}  {profit:>+9,}"
    total_profit = dt["total_ret"] - dt["total_invest"]
    line += f"  {int(total_profit):>+10,}"
    print(line)


def run_threshold_analysis(model, df: pd.DataFrame,
                           strategies: list[BetStrategy] = None,
                           thresholds: list[float | None] = None) -> dict:
    """
    top1_prob閾値を変えながらバックテストを比較する。

    Returns:
        {label: df_result} のdict。labelは "全レース" や "top1<30%" 形式。
    """
    if strategies is None:
        # 代表的な3連単戦略に絞る
        key_names = {"str_top3", "str_jiku1_4", "str_top2_1ch",
                     "str_chuana_2nd1st", "str_chuana_3rd1st", "str_oana_4th1st",
                     "str_chuana_top3_4th",
                     "str_top3_each_1st", "str_jiku1_cover2nd", "str_top2_both_ord"}
        strategies = [s for s in ALL_STRATEGIES if s.name in key_names]

    if thresholds is None:
        thresholds = [None, 0.90, 0.85, 0.80, 0.75]

    # pred_probを一度だけ計算し、閾値ループで再利用する
    df = _apply_pred_prob(model, df)
    payout_map = _load_payouts(df["race_key"].unique().tolist())

    results = {}
    for threshold in thresholds:
        label = "全レース" if threshold is None else f"top1<{threshold:.0%}"
        df_t = _filter_by_top1(df, threshold) if threshold is not None else df
        accum = _compute_accum(df_t, strategies, payout_map)
        results[label] = _accum_to_df(accum, strategies, df_t["race_key"].nunique())

    return results


def _load_payouts(race_keys: list[str]) -> dict:
    """払戻金を一括ロード"""
    if not race_keys:
        return {}
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        rows = conn.execute(f"""
            SELECT race_key, bet_type, combination, payout
            FROM odds
            WHERE race_key IN ({placeholders})
              AND payout IS NOT NULL
        """, race_keys).fetchall()

    result = defaultdict(dict)
    for row in rows:
        result[row["race_key"]][(row["bet_type"], row["combination"])] = row["payout"]
    return result


def print_backtest(df_result: pd.DataFrame, total_races: int = None):
    """バックテスト結果を表形式で表示"""
    if total_races is None and "対象レース数" in df_result.columns:
        total_races = df_result["対象レース数"].iloc[0] if len(df_result) > 0 else None

    print(f"\n{'='*80}")
    print(f" バックテスト結果{'  (' + str(total_races) + 'レース)' if total_races else ''}")
    print(f"{'='*80}")
    print(f"{'戦略':<38} {'点数':>5} {'的中率':>7} {'的中数':>6} {'回収率':>7}")
    print("-" * 80)

    prev_type = None
    for _, row in df_result.iterrows():
        label = row["戦略"]
        if "3連複" in label:
            cur_type = "3連複"
        elif "3連単(穴" in label or "3連単(大穴" in label:
            cur_type = "穴狙い"
        elif "3連単(的中" in label:
            cur_type = "的中率"
        elif "3連単" in label:
            cur_type = "3連単"
        elif "2車複" in label:
            cur_type = "2車複"
        elif "2車単" in label:
            cur_type = "2車単"
        elif "ワイド" in label:
            cur_type = "ワイド"
        else:
            cur_type = "その他"
        if prev_type and prev_type != cur_type:
            print()
        prev_type = cur_type
        print(f"  {label:<36} {row['1Rあたり点数']:>5} "
              f"{row['的中率']:>7} {row['的中数']:>6} {row['回収率']:>7}")
    print("=" * 80)


def print_threshold_analysis(analysis: dict):
    """フィルター閾値別の回収率比較を表示"""
    if not analysis:
        return

    # 戦略名の順序を最初のラベルから取得
    first_df = next(iter(analysis.values()))
    strategy_labels = first_df["戦略"].tolist()
    strategy_names = first_df["戦略名"].tolist()

    header_width = 18
    col_width = 9

    print(f"\n{'='*90}")
    print(" 人気フィルター × 回収率分析（3連単）")
    print(f"{'='*90}")

    # ヘッダー
    header = f"{'フィルター':<{header_width}} {'レース数':>7}"
    for name in strategy_names:
        short = _short_strategy_name(name)
        header += f"  {short:>{col_width}}"
    print(header)
    print("-" * 90)

    for label, df_r in analysis.items():
        n_races = df_r["対象レース数"].iloc[0] if len(df_r) > 0 else 0
        row_str = f"{label:<{header_width}} {n_races:>7,}"
        for name in strategy_names:
            match = df_r[df_r["戦略名"] == name]
            if len(match) == 0:
                row_str += f"  {'N/A':>{col_width}}"
            else:
                roi_str = match.iloc[0]["回収率"]
                raw = match.iloc[0]["回収率_raw"]
                marker = " *" if raw >= 1.05 else ("  " if raw >= 1.00 else "  ")
                row_str += f"  {roi_str:>{col_width - 2}}{marker}"
        print(row_str)

    print("=" * 90)
    print("  * = 回収率 105%以上")
    print()

    # 穴狙い戦略の内訳も表示（全レース分のみ）
    first_label = next(iter(analysis))
    df_all = analysis[first_label]
    ana_rows = df_all[df_all["戦略名"].str.contains("chuana|oana")]
    if len(ana_rows) > 0:
        print(f"{'='*80}")
        print(" 穴狙い戦略詳細（全レース）")
        print(f"{'='*80}")
        print(f"{'戦略':<40} {'点数':>5} {'的中率':>7} {'的中数':>6} {'回収率':>7} {'総投資':>10} {'総回収':>10}")
        print("-" * 80)
        for _, row in ana_rows.iterrows():
            print(f"  {row['戦略']:<38} {row['1Rあたり点数']:>5} "
                  f"{row['的中率']:>7} {row['的中数']:>6} {row['回収率']:>7} "
                  f"{row['総投資(円)']:>10,} {row['総回収(円)']:>10,}")
        print("=" * 80)


def _short_strategy_name(name: str) -> str:
    mapping = {
        "str_top3":            "top3(6pt)",
        "str_jiku1_4":         "jiku1(12p)",
        "str_top2_1ch":        "top2x(24p)",
        "str_chuana_2nd1st":   "#2-1st(12)",
        "str_chuana_3rd1st":   "#3-1st(12)",
        "str_oana_4th1st":     "#4-1st(6)",
        "str_chuana_top3_4th": "3x+4th(~9)",
        "str_top3_each_1st":   "3x1st(18p)",
        "str_jiku1_cover2nd":  "ax1or2(18)",
        "str_top2_both_ord":   "12both(8p)",
        "quinella_top2":       "Q12(1pt)",
        "quinella_23":         "Q23(1pt)",
        "exacta_12":           "E12(1pt)",
        "exacta_21":           "E21(1pt)",
        "wide_12":             "W12(1pt)",
        "wide_23":             "W23(1pt)",
        "wide_13":             "W13(1pt)",
    }
    return mapping.get(name, name[:10])


def print_day_simulation(df_races: pd.DataFrame, df_summary: pd.DataFrame,
                         target_date: str, max_top1_prob: float | None):
    """1日シミュレーション結果を表示（tier ラベル付き）"""
    if df_summary.empty:
        print("データなし")
        return

    strategy_names = df_summary["戦略名"].unique().tolist()
    strategy_labels = {r["戦略名"]: r["戦略"] for _, r in df_summary.iterrows()}

    W = 88 + 16 * len(strategy_names)
    print(f"\n{'='*W}")
    print(f" {target_date} シミュレーション  [穴:<65% / 通常:<70% / 安定:<80% / SKIP:≥80%]")
    print(f"{'='*W}")

    col_w = 14
    header = f"{'会場':<6} {'R':>2}  {'top1':>6}  {'ラベル':>4}"
    for sname in strategy_names:
        lbl = strategy_labels.get(sname, sname)
        short = (lbl.split(":")[1].strip().split("(")[0].strip() if ":" in lbl else lbl)[:col_w]
        header += f"  {short:>{col_w}}"
    print(header)
    print("-" * W)

    venue_name_map: dict = {}
    try:
        from ..database import get_connection
        with get_connection() as conn:
            for v in conn.execute("SELECT venue_code, name FROM venue_info").fetchall():
                venue_name_map[v[0]] = v[1]
    except Exception:
        pass

    tier_order = {"穴": 0, "通常": 1, "安定": 2, "SKIP": 3}
    sorted_races = df_races.copy()
    sorted_races["_tier_ord"] = sorted_races["tier"].map(tier_order).fillna(9)
    sorted_races = sorted_races.sort_values(["venue_code", "race_no"])

    prev_tier = None
    for _, row in sorted_races.iterrows():
        tier = row.get("tier", "SKIP")
        if tier != prev_tier and tier != "SKIP":
            if prev_tier is not None:
                print()
            prev_tier = tier

        vname = venue_name_map.get(row["venue_code"], row["venue_code"])[:5]
        line = f"{vname:<6} {int(row['race_no']):>2}  {row['top1_prob']:>6.3f}  {tier:<4}"
        for sname in strategy_names:
            hit = row.get(f"{sname}_hit")
            payout = row.get(f"{sname}_payout", 0)
            if hit is None:
                cell = "-"
            elif hit:
                cell = f"○ ¥{int(payout):,}"
            else:
                cell = "×"
            line += f"  {cell:>{col_w}}"
        print(line)

    print("=" * W)
    print(f" 集計")
    print("-" * W)

    n_total = len(df_races)
    tier_labels = [label for _, label in _DAY_SIM_TIERS]

    for tier in tier_labels:
        tier_df = df_summary[df_summary["tier"] == tier]
        if tier_df.empty:
            continue
        n_tier = int(tier_df["tier_レース数"].iloc[0])
        if n_tier == 0:
            continue

        print(f"\n  【{tier}】{n_tier}レース  (top1 {'<' + str(int([t for t,l in _DAY_SIM_TIERS if l==tier][0]*100))+'%'})")
        t_invest = t_return = 0
        for _, srow in tier_df.iterrows():
            inv = srow["総投資(円)"]
            ret = srow["総回収(円)"]
            t_invest += inv
            t_return += ret
            roi = srow["回収率_raw"]
            hits = int(srow["的中数"])
            profit = ret - inv
            print(f"    [{srow['戦略名']}] 的中:{hits}回  投資:{inv:,}円  "
                  f"回収:{ret:,}円  回収率:{roi:.1%}  損益:{profit:+,}円")
        if len(tier_df) > 1 and t_invest > 0:
            combined_roi = t_return / t_invest
            print(f"    → {tier}合計: 投資{t_invest:,}円  回収{t_return:,}円  "
                  f"回収率{combined_roi:.1%}  損益{t_return-t_invest:+,}円")

    # 全BET合計
    bet_df = df_summary[df_summary["tier"].isin(tier_labels)]
    all_invest = bet_df["総投資(円)"].sum()
    all_return = bet_df["総回収(円)"].sum()
    if all_invest > 0:
        combined_roi = all_return / all_invest
        n_bet = sum(
            int(df_summary[df_summary["tier"] == t]["tier_レース数"].iloc[0])
            for t in tier_labels
            if not df_summary[df_summary["tier"] == t].empty
        )
        print(f"\n  全BET合計({n_bet}R / 全{n_total}R): "
              f"投資{all_invest:,}円  回収{all_return:,}円  "
              f"回収率{combined_roi:.1%}  損益{all_return-all_invest:+,}円")
    print("=" * W)


def print_venue_analysis(df_venue: pd.DataFrame, strategies: list[BetStrategy] = None,
                         max_top1_prob: float | None = None):
    """会場別分析結果を表示"""
    if df_venue.empty:
        print("データなし")
        return

    # 使用された戦略を推測
    s_names = []
    for col in df_venue.columns:
        if col.endswith("_回収率"):
            s_names.append(col[:-4])

    filter_str = f"top1<{max_top1_prob:.0%}" if max_top1_prob is not None else "フィルター適用後"
    print(f"\n{'='*90}")
    print(f" 会場別バックテスト（{filter_str}）")
    print(f"{'='*90}")

    header = f"{'会場':<7} {'対象R':>6}"
    for sn in s_names:
        short = _short_strategy_name(sn)
        header += f"  {'的中率':>7}  {'回収率':>7}  {'損益':>9}"
    print(header)
    print(f"  {'':7}       " + "  " + "  ".join([f"[{_short_strategy_name(sn)}]" + " " * 15 for sn in s_names]))
    print("-" * 90)

    # 最初の戦略の回収率でソート
    if s_names:
        sort_col = f"{s_names[0]}_回収率"
        df_sorted = df_venue.sort_values(sort_col, ascending=False)
    else:
        df_sorted = df_venue

    for _, row in df_sorted.iterrows():
        line = f"{row['会場']:<7} {int(row['対象レース数']):>6}"
        for sn in s_names:
            hit_rate = row.get(f"{sn}_的中率", 0)
            roi = row.get(f"{sn}_回収率", 0)
            inv = row.get(f"{sn}_投資", 0)
            ret = row.get(f"{sn}_回収", 0)
            profit = ret - inv
            marker = "*" if roi >= 1.10 else (" " if roi >= 1.00 else " ")
            line += f"  {hit_rate:>6.1%}  {roi:>6.1%}{marker}  {profit:>+9,}"
        print(line)

    print("=" * 90)
    print("  * = 回収率 110%以上")

    # 全会場合計
    print()
    print("  [全会場合計]")
    for sn in s_names:
        total_inv = df_venue[f"{sn}_投資"].sum()
        total_ret = df_venue[f"{sn}_回収"].sum()
        total_hits = df_venue[f"{sn}_的中数"].sum()
        total_races = df_venue["対象レース数"].sum()
        roi = total_ret / total_inv if total_inv > 0 else 0
        print(f"    {_short_strategy_name(sn)}: "
              f"的中率 {total_hits/total_races:.1%}  回収率 {roi:.1%}  "
              f"投資 {total_inv:,}円  回収 {total_ret:,}円  損益 {total_ret-total_inv:+,}円")
