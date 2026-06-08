"""winticket ルート用バックテスト

keirin-station 版 (backtest.py) と買い目戦略・combo生成関数を共有しつつ、
以下の差分を吸収する:

- 特徴量列: FEATURE_COLS_WT
- 着順列:  finish_order（ks版は finish_position）
- 払戻:    wt_odds.odds_value（小数オッズ）から payout = odds_value * 100 を算出
           ks版の odds.payout（実払戻金）に相当
- 市場対応: ks戦略の bet_type → winticket 市場名
    trifecta_box (上位3頭BOX等, 順不同3車) → trio        (三連複)
    trifecta     (順序付き3車)             → trifecta    (三連単)
    quinella     (順不同2車)               → quinella    (二車複)
    wide         (順不同2車)               → quinellaPlace(ワイド)
    exacta       (順序付き2車)             → exacta      (二車単)

オッズはレース確定前の最終オッズ（wt_odds に保存された値）を使用するため、
実運用と同じ「AI予想 → オッズ参照 → 購入」のフローを再現する。
"""
import re
import pandas as pd

from ..database import get_connection
from ..preprocessing.feature_wt import FEATURE_COLS_WT, TARGET_COL_WT
from .backtest import (
    BetStrategy,
    STRATEGIES, ANA_STRATEGIES, HITRATE_STRATEGIES,
    QUINELLA_STRATEGIES, EXACTA_STRATEGIES, WIDE_STRATEGIES,
)

# ks bet_type → winticket 市場名
_MARKET_MAP = {
    "trifecta_box": "trio",
    "trifecta":     "trifecta",
    "quinella":     "quinella",
    "wide":         "quinellaPlace",
    "exacta":       "exacta",
}
_ORDERED_BETS = {"trifecta", "exacta"}  # 順序を保持する市場


# ---------------------------------------------------------------------------
# 予測確率 / オッズロード
# ---------------------------------------------------------------------------

def _apply_pred_prob_wt(model, df: pd.DataFrame) -> pd.DataFrame:
    """pred_prob を計算して付与（finish_order 欠損/0=DNS/欠車/失格行を除去）"""
    df = df[df["finish_order"].notna() & (df["finish_order"] >= 1)].copy()
    df = df.dropna(subset=FEATURE_COLS_WT).copy()
    X = pd.DataFrame(df[FEATURE_COLS_WT].values, columns=FEATURE_COLS_WT)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    return df


def _load_payouts_wt(race_keys: list[str]) -> dict[str, dict]:
    """wt_odds から {race_key: {(market, key): payout}} を構築。

    payout = odds_value * 100（100円賭けたときの払戻金）。
    key は順序市場なら tuple(int) / 順不同市場なら frozenset(int)。
    """
    payout_map: dict[str, dict] = {}
    if not race_keys:
        return payout_map

    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, bet_type, combination, odds_value "
                f"FROM wt_odds WHERE race_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for race_key, market, combo, odds_value in rows:
                if odds_value is None:
                    continue
                parts = [p for p in re.split(r"[-=→]", str(combo)) if p != ""]
                try:
                    nums = [int(p) for p in parts]
                except ValueError:
                    continue
                key = tuple(nums) if market in _ORDERED_BETS else frozenset(nums)
                payout = int(round(odds_value * 100))
                payout_map.setdefault(race_key, {})[(market, key)] = payout
    return payout_map


# ---------------------------------------------------------------------------
# 的中判定
# ---------------------------------------------------------------------------

def _evaluate_combos_wt(s: BetStrategy, combos, actual_order: tuple,
                        top3_set: frozenset, race_payouts: dict) -> tuple[bool, int]:
    """winticket 市場の的中判定とペイアウト合算"""
    market = _MARKET_MAP.get(s.bet_type)
    hit = False
    payout = 0

    if s.bet_type == "trifecta_box":            # 三連複: 順不同3車
        for combo in combos:
            if combo == top3_set:
                payout = race_payouts.get((market, frozenset(combo)), 0)
                hit = True
                break
    elif s.bet_type == "trifecta":              # 三連単: 順序付き3車
        for combo in combos:
            if combo == actual_order:
                payout = race_payouts.get((market, tuple(combo)), 0)
                hit = True
                break
    elif s.bet_type == "quinella":              # 二車複: 順不同2車
        actual_q = frozenset([actual_order[0], actual_order[1]])
        for combo in combos:
            if combo == actual_q:
                payout = race_payouts.get((market, frozenset(combo)), 0)
                hit = True
                break
    elif s.bet_type == "wide":                  # ワイド: 順不同2車が共に3着以内
        for combo in combos:
            if frozenset(combo).issubset(top3_set):
                payout += race_payouts.get((market, frozenset(combo)), 0)
                hit = True
    elif s.bet_type == "exacta":                # 二車単: 順序付き2車
        actual_e = (actual_order[0], actual_order[1])
        for combo in combos:
            if combo == actual_e:
                payout = race_payouts.get((market, tuple(combo)), 0)
                hit = True
                break
    return hit, payout


# ---------------------------------------------------------------------------
# フィルター
# ---------------------------------------------------------------------------

def _filter_by_n_riders(df: pd.DataFrame, max_riders: int) -> pd.DataFrame:
    sizes = df.groupby("race_key")["frame_no"].count()
    valid = sizes[sizes <= max_riders].index
    return df[df["race_key"].isin(valid)]


def _filter_by_gap12(df: pd.DataFrame, min_gap: float) -> pd.DataFrame:
    """レース内 top1_prob - top2_prob が min_gap 未満のレースを除外
    （wave-picks-wt の本命確度フィルターを再現）"""
    def _gap(s):
        p = s.sort_values(ascending=False).tolist()
        return (p[0] - p[1]) if len(p) >= 2 else 0.0
    gaps = df.groupby("race_key")["pred_prob"].apply(_gap)
    valid = gaps[gaps >= min_gap].index
    return df[df["race_key"].isin(valid)]


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------

def _compute_accum_wt(df: pd.DataFrame, strategies: list[BetStrategy],
                      payout_map: dict) -> dict[str, dict]:
    accum = {s.name: {"bets": 0, "returns": 0, "hits": 0} for s in strategies}

    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        ranked = grp["frame_no"].astype(int).tolist()

        top3 = grp[grp["finish_order"].between(1, 3)]
        actual_top3_set = frozenset(top3["frame_no"].astype(int).tolist())
        if len(actual_top3_set) < 3:
            continue

        actual_order = tuple(
            top3.sort_values("finish_order")["frame_no"].astype(int).tolist()
        )
        race_payouts = payout_map.get(race_key, {})

        for s in strategies:
            combos = s.generate(ranked)
            if not combos:
                continue
            accum[s.name]["bets"] += len(combos) * 100
            hit, payout = _evaluate_combos_wt(
                s, combos, actual_order, actual_top3_set, race_payouts
            )
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

# winticket は trio/trifecta/quinella/quinellaPlace/exacta 全市場のオッズを持つため
# ks の全戦略を評価可能
WT_STRATEGIES = (
    STRATEGIES + ANA_STRATEGIES + HITRATE_STRATEGIES
    + QUINELLA_STRATEGIES + EXACTA_STRATEGIES + WIDE_STRATEGIES
)


def run_backtest_wt(model, df: pd.DataFrame,
                    strategies: list[BetStrategy] | None = None,
                    max_riders: int | None = None,
                    min_gap12: float | None = None) -> pd.DataFrame:
    """winticket データで複数戦略のバックテストを実行。

    max_riders: 出走頭数フィルター（実運用は ≤6 車）
    min_gap12:  top1-top2 pred_prob 差フィルター（wave-picks-wt は 0.06）
    """
    if strategies is None:
        strategies = WT_STRATEGIES

    df = _apply_pred_prob_wt(model, df)
    if max_riders is not None:
        df = _filter_by_n_riders(df, max_riders)
    if min_gap12 is not None:
        df = _filter_by_gap12(df, min_gap12)

    if df.empty:
        return _accum_to_df(
            {s.name: {"bets": 0, "returns": 0, "hits": 0} for s in strategies},
            strategies, 0,
        )

    payout_map = _load_payouts_wt(df["race_key"].unique().tolist())
    accum = _compute_accum_wt(df, strategies, payout_map)
    return _accum_to_df(accum, strategies, df["race_key"].nunique())


# ---------------------------------------------------------------------------
# SS/S/A 層別バックテスト（wave-picks-wt の本番戦略を完全再現）
# ---------------------------------------------------------------------------
# SS: gap12≥0.15 & ratio<1.3        → 3連単 pivot1→pivot2→各third（3点・順序）  trifecta市場
# S : gap12≥0.15 & ratio∈[1.3,1.6)  → 3連複 {pivot1,pivot2,third}（3点）        trio市場
# A : gap12∈[0.06,0.15)             → 3連複 {pivot1,pivot2,third}（3点）        trio市場
# (gap12≥0.15 & ratio≥1.6 はスキップ / 全て6車以下・gap12≥0.06)

def _assign_tier(gap12: float, ratio: float) -> str | None:
    if gap12 < 0.06:
        return None
    if gap12 >= 0.15:
        if ratio < 1.3:
            return "SS"
        if ratio < 1.6:
            return "S"
        return None            # ratio≥1.6 は低配当リスクでスキップ
    return "A"                 # gap12 ∈ [0.06, 0.15)


def run_tiered_backtest_wt(model, df: pd.DataFrame,
                           max_riders: int = 6) -> pd.DataFrame:
    """wave-picks-wt と同条件の SS/S/A 層別バックテスト。

    各レースで上位確率順に pivot1/pivot2/thirds(上位3〜5位) を取り、
    層に応じて 3連単(SS) / 3連複(S・A) を 3点購入する。
    payout は wt_odds の実オッズ ×100。
    """
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, max_riders)
    if df.empty:
        return pd.DataFrame()

    payout_map = _load_payouts_wt(df["race_key"].unique().tolist())
    tiers = {t: {"races": 0, "bets": 0, "returns": 0, "hits": 0}
             for t in ("SS", "S", "A")}

    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue
        probs = grp["pred_prob"].tolist()
        gap12 = probs[0] - probs[1]
        ratio = probs[0] / (3.0 / n)
        tier = _assign_tier(gap12, ratio)
        if tier is None:
            continue

        frames = grp["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:5]
        if not thirds:
            continue

        finished = grp[grp["finish_order"].between(1, 3)]
        top3_set = frozenset(finished["frame_no"].astype(int).tolist())
        if len(top3_set) < 3:
            continue
        actual_order = tuple(
            finished.sort_values("finish_order")["frame_no"].astype(int).tolist()
        )
        race_payouts = payout_map.get(race_key, {})

        tiers[tier]["races"] += 1
        if tier == "SS":   # 3連単（順序）
            for t in thirds:
                tiers[tier]["bets"] += 100
                if actual_order == (pivot1, pivot2, t):
                    tiers[tier]["returns"] += race_payouts.get(("trifecta", (pivot1, pivot2, t)), 0)
                    tiers[tier]["hits"] += 1
        else:              # 3連複（順不同）
            for t in thirds:
                tiers[tier]["bets"] += 100
                combo = frozenset((pivot1, pivot2, t))
                if combo == top3_set:
                    tiers[tier]["returns"] += race_payouts.get(("trio", combo), 0)
                    tiers[tier]["hits"] += 1

    rows = []
    label = {"SS": "SS: gap12≥0.15&ratio<1.3 (3連単)",
             "S":  "S : gap12≥0.15&ratio[1.3,1.6) (3連複)",
             "A":  "A : gap12[0.06,0.15) (3連複)"}
    for t in ("SS", "S", "A"):
        a = tiers[t]
        roi = a["returns"] / a["bets"] if a["bets"] > 0 else 0
        hit_rate = a["hits"] / a["races"] if a["races"] else 0
        rows.append({
            "層": label[t],
            "対象R数": a["races"],
            "的中率": f"{hit_rate:.1%}",
            "的中数": a["hits"],
            "投資(円)": a["bets"],
            "回収(円)": a["returns"],
            "回収率": f"{roi:.1%}",
            "回収率_raw": roi,
        })
    # 合計行
    tot_bets = sum(tiers[t]["bets"] for t in tiers)
    tot_ret = sum(tiers[t]["returns"] for t in tiers)
    tot_races = sum(tiers[t]["races"] for t in tiers)
    tot_hits = sum(tiers[t]["hits"] for t in tiers)
    rows.append({
        "層": "合計", "対象R数": tot_races,
        "的中率": f"{(tot_hits/tot_races if tot_races else 0):.1%}",
        "的中数": tot_hits, "投資(円)": tot_bets, "回収(円)": tot_ret,
        "回収率": f"{(tot_ret/tot_bets if tot_bets else 0):.1%}",
        "回収率_raw": (tot_ret/tot_bets if tot_bets else 0),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# バリュー(期待値)ベース・バックテスト
# ---------------------------------------------------------------------------
# ユーザー戦略: モデルの「正当評価」(top3確率)に対し、市場オッズが過小評価
# している買い目（EV = モデル確率 × オッズ > 1）だけを買う。
# 特に実力拮抗レース（ratioが低い=本命が割れる）で、市場と逆転した高配当側を拾う。
#
# 三連複の組み合わせ確率 P({a,b,c} が上位3着) を選手別top3確率から推定する。
# 近似: combo_score = p_a * p_b * p_c をレース内 C(n,3) 全通りで正規化（合計1）。
#       厳密な順序依存は無視するが、value比較のランキングには十分機能する。

import itertools


def _trio_combo_probs(frame_probs: dict[int, float]) -> dict[frozenset, float]:
    """選手別top3確率から、各三連複組合せの確率（正規化積）を推定"""
    frames = list(frame_probs.keys())
    scores = {}
    total = 0.0
    for a, b, c in itertools.combinations(frames, 3):
        s = frame_probs[a] * frame_probs[b] * frame_probs[c]
        scores[frozenset((a, b, c))] = s
        total += s
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}


def run_value_backtest_wt(model, df: pd.DataFrame,
                          ev_min: float = 1.0,
                          max_per_race: int = 5,
                          max_riders: int = 9,
                          max_ratio: float | None = None) -> dict:
    """三連複のEVベース・バリューバックテスト。

    各レースで C(n,3) 全組合せの EV = combo_prob × trio_odds を計算し、
    EV ≥ ev_min の組合せを EV 降順に最大 max_per_race 点購入する。

    ev_min:       購入する最低EV（1.0=損益分岐、市場と互角。>1.0でモデル優位分のみ）
    max_per_race: 1レースあたり最大購入点数
    max_riders:   出走頭数上限
    max_ratio:    top1_prob/(3/n) がこの値未満のレースのみ（実力拮抗フィルター）。None=無効
    """
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, max_riders)
    if df.empty:
        return {"races": 0, "bets": 0, "returns": 0, "hits": 0, "roi": 0.0,
                "hit_rate": 0.0, "n_bet_races": 0, "avg_ev": 0.0, "avg_payout": 0.0}

    payout_map = _load_payouts_wt(df["race_key"].unique().tolist())
    races = bets = returns = hits = n_bet_races = 0
    ev_sum = 0.0
    hit_payouts = []

    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue
        races += 1

        if max_ratio is not None:
            ratio = grp["pred_prob"].iloc[0] / (3.0 / n)
            if ratio >= max_ratio:
                continue

        frame_probs = dict(zip(grp["frame_no"].astype(int), grp["pred_prob"]))
        combo_probs = _trio_combo_probs(frame_probs)
        if not combo_probs:
            continue

        race_payouts = payout_map.get(race_key, {})
        # 各組合せの EV を計算（オッズが存在するもののみ）
        candidates = []
        for combo, p in combo_probs.items():
            odds = race_payouts.get(("trio", combo))
            if not odds:
                continue
            ev = p * (odds / 100.0)   # odds はpayout(円)なので /100 で倍率
            if ev >= ev_min:
                candidates.append((ev, combo, odds))
        if not candidates:
            continue

        candidates.sort(reverse=True)
        selected = candidates[:max_per_race]

        fin = grp[grp["finish_order"].between(1, 3)]
        top3_set = frozenset(fin["frame_no"].astype(int).tolist())
        valid_result = len(top3_set) == 3

        n_bet_races += 1
        for ev, combo, odds in selected:
            bets += 100
            ev_sum += ev
            if valid_result and combo == top3_set:
                returns += odds
                hits += 1
                hit_payouts.append(odds)

    roi = returns / bets if bets > 0 else 0.0
    n_sel = bets / 100 if bets else 0
    return {
        "races": races,
        "n_bet_races": n_bet_races,
        "bets": bets,
        "returns": returns,
        "hits": hits,
        "roi": roi,
        "hit_rate": hits / n_bet_races if n_bet_races else 0.0,
        "avg_ev": ev_sum / n_sel if n_sel else 0.0,
        "avg_bets_per_race": n_sel / n_bet_races if n_bet_races else 0.0,
        "avg_payout": (sum(hit_payouts) / len(hit_payouts)) if hit_payouts else 0.0,
    }


def print_value_backtest_wt(result: dict, params: str = "") -> None:
    import click
    click.echo(f"\n{'='*72}")
    click.echo(f"winticket バリュー(EV)バックテスト  {params}")
    click.echo(f"{'='*72}")
    click.echo(f"  全レース:        {result['races']:,}")
    click.echo(f"  購入レース数:    {result['n_bet_races']:,}  "
               f"(1Rあたり平均 {result.get('avg_bets_per_race',0):.1f}点)")
    click.echo(f"  総購入点数:      {result['bets']//100:,}点 / {result['bets']:,}円")
    click.echo(f"  的中数:          {result['hits']:,}  "
               f"(購入レース的中率 {result['hit_rate']:.1%})")
    click.echo(f"  的中平均払戻:    {result['avg_payout']:.0f}円")
    click.echo(f"  総回収:          {result['returns']:,}円")
    click.echo(f"  平均EV(選択時):  {result['avg_ev']:.3f}")
    click.echo(f"  {'─'*40}")
    click.echo(f"  回収率(ROI):     {result['roi']:.1%}")


def print_tiered_backtest_wt(df_result: pd.DataFrame) -> None:
    import click
    click.echo(f"\n{'='*72}")
    click.echo("winticket SS/S/A 層別バックテスト（wave-picks-wt 本番戦略・3点300円/R）")
    click.echo(f"{'='*72}")
    if df_result.empty:
        click.echo("対象レースがありません。")
        return
    cols = ["層", "対象R数", "的中率", "的中数", "投資(円)", "回収(円)", "回収率"]
    click.echo(df_result[cols].to_string(index=False))


def print_backtest_wt(df_result: pd.DataFrame, total_races: int) -> None:
    import click
    click.echo(f"\n{'='*72}")
    click.echo(f"winticket バックテスト結果  (対象レース数: {total_races:,})")
    click.echo(f"{'='*72}")
    if df_result.empty:
        click.echo("対象レースがありません。")
        return
    show = df_result.sort_values("回収率_raw", ascending=False)
    cols = ["戦略", "1Rあたり点数", "的中率", "的中数", "回収率"]
    click.echo(show[cols].to_string(index=False))
    best = show.iloc[0]
    click.echo(f"\n最高回収率: {best['戦略']}  →  {best['回収率']} "
               f"(的中 {best['的中数']}/{total_races})")
