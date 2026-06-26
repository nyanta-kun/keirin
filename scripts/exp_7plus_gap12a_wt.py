"""7+車 A ランク gap12 閾値実験（0.07 / 0.08 / 0.09）

本番戦略（doc48）の A ランク条件 gap12 ∈ [min_gap12, 0.10) について、
下限 min_gap12 を 0.07 / 0.08 / 0.09 で比較する。

設計（本番忠実・バイアスなし）:
  ① 予測はエントリー表全車でランキング（欠車を事前に知らない）
  ② 7+車フィルタはエントリー数基準
  ③ gami = min trio odds {p1,p2,t} for t in thirds (ranks 3+)、< 5.0 はスキップ
  ④ gap12 ∈ [threshold, 0.10) → A ランク評価
  ⑤ 欠車処理: 軸欠車=レース無効 / 相手欠車=その目のみ除外

モデル:
  VAL  2025-07-01〜2026-02-28  lgbm_wt_train_only（リーク無し）
  HOLD 2026-03-01〜present     lgbm_wt_eval（リーク無し）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from src.evaluation.void_rules import void_by_dns

THRESHOLDS = [0.07, 0.08, 0.09]
S_GAP12    = 0.10
GAMI_MIN   = 5.0


def _entry_size(df):
    return df.groupby("race_key")["frame_no"].count()


def collect(date_from, date_to, model):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = _apply_pred_prob_wt(model, df)

    # エントリー数基準で 7+ 車フィルタ（バイアス②修正）
    sizes = _entry_size(df)
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()

    rks = df["race_key"].unique().tolist()
    payouts = _load_payouts_wt(rks)

    races = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 7:
            continue

        # 結果確定チェック
        fin3 = g[g["finish_order"].between(1, 3)]
        if len(fin3) < 3:
            continue

        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 >= S_GAP12:
            continue  # Sランク → A評価対象外

        frames = g["frame_no"].astype(int).tolist()
        p1, p2 = frames[0], frames[1]
        thirds = frames[2:]

        # trio payout map for this race
        rp = payouts.get(rk, {})

        # gami = min trio odds {p1,p2,t}
        trio_odds_list = []
        for t in thirds:
            key = frozenset((p1, p2, t))
            o = rp.get(("trio", key))
            if o and o > 0:
                trio_odds_list.append(o)
        gami = min(trio_odds_list) / 100 if trio_odds_list else 0.0  # payout→odds
        if gami < GAMI_MIN:
            continue

        # 欠車情報
        runners = set(g[g["finish_order"] >= 1]["frame_no"].astype(int).tolist())
        void, valid_thirds = void_by_dns(p1, p2, thirds, runners)
        if void:
            continue

        # 的中判定
        top3 = frozenset(fin3["frame_no"].astype(int).tolist())
        hit = top3 in [frozenset((p1, p2, t)) for t in valid_thirds]
        payout_raw = 0
        if hit:
            winning_combo = top3
            payout_raw = rp.get(("trio", winning_combo), 0)
        stake = len(valid_thirds) * 100

        races.append({
            "gap12":  gap12,
            "gami":   gami,
            "n_pts":  len(valid_thirds),
            "stake":  stake,
            "hit":    hit,
            "payout": payout_raw,
            "date":   g["race_date"].iloc[0],
        })

    return races


def roi_stats(races):
    if not races:
        return {"n": 0, "stake": 0, "payout": 0, "roi": 0.0, "hit_rate": 0.0}
    n = len(races)
    stake = sum(r["stake"] for r in races)
    payout = sum(r["payout"] for r in races)
    hits = sum(r["hit"] for r in races)
    roi = payout / stake if stake > 0 else 0.0
    return {"n": n, "stake": stake, "payout": payout, "roi": roi, "hit_rate": hits / n}


def main():
    print("Loading models...", flush=True)
    m_val  = load_model("lgbm_wt_train_only")
    m_hold = load_model("lgbm_wt_eval")

    print("Collecting VAL 2025-07-01〜2026-02-28...", flush=True)
    val_races = collect("2025-07-01", "2026-02-28", m_val)
    print(f"  VAL raw A-range races (gap12<{S_GAP12}+gami≥{GAMI_MIN}): {len(val_races)}", flush=True)

    print("Collecting HOLD 2026-03-01〜2026-06-25...", flush=True)
    hold_races = collect("2026-03-01", "2026-06-25", m_hold)
    print(f"  HOLD raw A-range races (gap12<{S_GAP12}+gami≥{GAMI_MIN}): {len(hold_races)}", flush=True)

    print()
    hdr = f"  {'閾値':<10}{'VAL_R':>7}{'VAL_ROI':>9}{'VAL_的中':>9}{'HOLD_R':>8}{'HOLD_ROI':>10}{'HOLD_的中':>10}"
    print("=" * len(hdr))
    print("  7+車 A ランク gap12 閾値別 ROI（gami≥5.0・リーク無しモデル）")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for th in THRESHOLDS:
        v = roi_stats([r for r in val_races  if r["gap12"] >= th])
        h = roi_stats([r for r in hold_races if r["gap12"] >= th])
        label = f"gap12≥{th:.2f}"
        print(f"  {label:<10}{v['n']:>7}{v['roi']:>9.1%}{v['hit_rate']:>9.1%}"
              f"{h['n']:>8}{h['roi']:>10.1%}{h['hit_rate']:>10.1%}")

    print()
    print("  ※ gap12≥0.10 = Sランク（参考）")
    # S rank reference (gap12 >= 0.10) would need separate collect; show note only
    print()

    # gap12 分布確認（VAL）
    print("  【VAL gap12 分布 for A-range (gap12<0.10)】")
    buckets = [(0.07, 0.08), (0.08, 0.09), (0.09, 0.10)]
    for lo, hi in buckets:
        sub = [r for r in val_races if lo <= r["gap12"] < hi]
        st = roi_stats(sub)
        print(f"    [{lo:.2f},{hi:.2f})  {st['n']:>5}R  ROI {st['roi']:>6.1%}  的中 {st['hit_rate']:>5.1%}")

    print()
    print("  【HOLD gap12 分布 for A-range (gap12<0.10)】")
    for lo, hi in buckets:
        sub = [r for r in hold_races if lo <= r["gap12"] < hi]
        st = roi_stats(sub)
        print(f"    [{lo:.2f},{hi:.2f})  {st['n']:>5}R  ROI {st['roi']:>6.1%}  的中 {st['hit_rate']:>5.1%}")


if __name__ == "__main__":
    main()
