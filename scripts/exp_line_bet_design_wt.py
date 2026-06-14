"""ライン構造×指数差を使った買い目設計検証

ユーザー仮説:
  「指数差が大きい → 逆らわず・低配当なら見送り（現行ガミ足切りで対応済み）」
  「指数差が狭い  → 逆転候補を LINE 情報で特定し別の買い目を構成する」
  「強番手（ライン内得点差大）→ 先頭でなく番手を軸に切り替える」

テスト戦略（全て ≤6車・ガミ≥5倍・リーク無し・最終オッズ上限値）:
  S0: 現行 C0 baseline（pred1+pred2→thirds 3点流し）
  S1: 強番手軸シフト（within_line_rp_gap > 5 のレースで強番手を軸の1本に）
  S2: 拮抗×最強ライン軸（gap12 < 0.06 のレースで最強ライン先頭+番手を軸に）
  S3: 拮抗×強番手（gap12 < 0.06 AND within_line_rp_gap > 5 の交差）

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-14
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.database import get_connection
from src.evaluation.backtest_wt import _assign_tier
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS
from exp_line_features_wt import add_line_features, LINE_FEATURE_COLS

GAMI_THRESHOLD = 5.0   # 最安目オッズ足切り（倍）
STRONG_FOLLOWER_GAP = 5.0  # 強番手の閾値（番手が先頭より何点以上強いか）
TIGHT_GAP12 = 0.06     # 拮抗判定の閾値


# ─── ヘルパー ────────────────────────────────────────────────────────────────

def load_all_data():
    """データ読み込み・特徴量構築・モデル学習まで一括。"""
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries FROM wt_races", conn)
        odds_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'",
            conn,
        )

    df = df.merge(races_info, on="race_key", how="left")
    df = add_line_features(df)

    # ライン特徴量込みの TRAIN 専用モデル
    print("  リーク無しモデル学習中 (TRAIN 期間のみ)...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    X_tr = pd.concat([prepare_X(fit).reset_index(drop=True),
                      fit[LINE_FEATURE_COLS].reset_index(drop=True)], axis=1)
    y_tr = fit["top3_flag"].reset_index(drop=True).values
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)

    # 全行に pred_prob 付与
    X_all = pd.concat([prepare_X(df).reset_index(drop=True),
                       df[LINE_FEATURE_COLS].reset_index(drop=True)], axis=1)
    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(X_all)[:, 1]
    print(f"  モデル学習完了 ({len(fit):,} 行)")

    # trio オッズ辞書
    import re
    def _parse(s):
        parts = re.split(r"[-=]", str(s))
        try:
            return frozenset(int(p) for p in parts)
        except Exception:
            return None
    odds_df["combo_key"] = odds_df["combination"].apply(_parse)
    odds_df = odds_df.dropna(subset=["combo_key"])
    odds_map = {}
    for row in odds_df.itertuples(index=False):
        odds_map.setdefault(row.race_key, {})[row.combo_key] = row.odds_value * 100

    # 実際の trio 結果辞書
    actual_result = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    return df, odds_map, actual_result


def _min_leg_odds(combos, odds_map, race_key):
    """指定組み合わせリストの最安オッズ（円）を返す。"""
    race_odds = odds_map.get(race_key, {})
    vals = [race_odds.get(k, 0) for k in combos]
    return min((v for v in vals if v > 0), default=0)


def roi_for_strategy(df, odds_map, actual_result, strategy_fn):
    """
    strategy_fn(grp, race_key) → list[frozenset] | None
      レースの全エントリー DataFrame を受け取り、買い目フロゼンセットのリストを返す。
      None または空リスト = 見送り。
    ガミ足切り（最安目オッズ < GAMI_THRESHOLD 倍）はここで統一適用。
    """
    results = {}
    for period, lo, hi in [
        ("TRAIN", TRAIN[0], TRAIN[1]),
        ("VAL",   TRAIN[1], VAL[1]),
        ("HOLD",  VAL[1],   HOLD[1]),
    ]:
        sub = df[(df["race_date"] > lo) & (df["race_date"] <= hi)
                 & (df["finish_order"] >= 0)]
        total_bet = total_pay = hits = bets = n_races = 0

        for race_key, grp in sub.groupby("race_key"):
            if grp["n_entries"].iloc[0] > 6:
                continue
            if race_key not in actual_result:
                continue
            n_races += 1

            combos = strategy_fn(grp, race_key)
            if not combos:
                continue

            # ガミ足切り
            min_pay = _min_leg_odds(combos, odds_map, race_key)
            if min_pay > 0 and min_pay / 100 < GAMI_THRESHOLD:
                continue

            actual = actual_result[race_key]
            race_odds = odds_map.get(race_key, {})
            n_pts = len(combos)
            hit_pay = race_odds.get(actual, 0) if actual in combos else 0

            bets += 1
            total_bet += n_pts * 100
            total_pay += hit_pay
            if hit_pay > 0:
                hits += 1

        roi = total_pay / total_bet * 100 if total_bet > 0 else 0
        results[period] = {
            "n_races": n_races, "bets": bets, "hits": hits, "roi": roi,
            "total_bet": total_bet, "total_pay": total_pay,
        }
    return results


# ─── 戦略定義 ────────────────────────────────────────────────────────────────

def s0_baseline(grp, race_key):
    """S0: 現行 C0 (pred1+pred2→thirds 2軸流し)。"""
    grp_s = grp.sort_values("pred_prob", ascending=False)
    frames = grp_s["frame_no"].astype(int).tolist()
    if len(frames) < 3:
        return None
    p1, p2 = frames[0], frames[1]
    thirds = frames[2:]
    probs = grp_s["pred_prob"].tolist()
    gap12 = probs[0] - probs[1]
    n = len(grp)
    ratio = probs[0] / (3 / n)
    tier = _assign_tier(gap12, ratio)
    if tier not in ("SS", "S", "A"):
        return None
    return [frozenset([p1, p2, t]) for t in thirds]


def s1_strong_follower(grp, race_key):
    """S1: 強番手軸シフト。
    within_line_rp_gap > STRONG_FOLLOWER_GAP のレースで
    最も強い番手(SF)を pred2 と組み合わせて軸にする。
    SF が pred1 なら通常通り（番手が既に1位予測）= S0 と同じ処理。
    """
    grp_s = grp.sort_values("pred_prob", ascending=False)
    frames = grp_s["frame_no"].astype(int).tolist()
    probs = grp_s["pred_prob"].tolist()
    if len(frames) < 3:
        return None

    # 強番手の特定（番手で最大 within_line_rp_gap）
    followers = grp[(grp["is_line_leader"] == 0) & (grp["line_size"] >= 2)].copy()
    if followers.empty or followers["within_line_rp_gap"].max() < STRONG_FOLLOWER_GAP:
        return None  # 強番手がいないレースは見送り

    sf_row = followers.loc[followers["within_line_rp_gap"].idxmax()]
    sf_frame = int(sf_row["frame_no"])

    # SF がすでに pred1 なら通常軸と同じ = S0 を呼ぶ
    if sf_frame == frames[0]:
        return s0_baseline(grp, race_key)

    # SF の属するラインを除く中で最高 pred_prob の選手を対軸に
    sf_line = int(sf_row["line_group"])
    other_line = grp[grp["line_group"] != sf_line]
    if other_line.empty:
        other_frame = frames[1] if frames[1] != sf_frame else frames[0]
    else:
        other_frame = int(other_line.sort_values("pred_prob", ascending=False)["frame_no"].iloc[0])

    # 残り選手（thirds）
    axis_set = {sf_frame, other_frame}
    thirds = [f for f in [int(r) for r in grp["frame_no"].astype(int)] if f not in axis_set]
    if not thirds:
        return None

    return [frozenset([sf_frame, other_frame, t]) for t in thirds]


def s2_tight_best_line(grp, race_key):
    """S2: 拮抗レース×最強ライン軸。
    gap12 < TIGHT_GAP12 のレースで、
    最強ライン（先頭の race_point 最大）の先頭+番手を軸に。
    """
    grp_s = grp.sort_values("pred_prob", ascending=False)
    probs = grp_s["pred_prob"].tolist()
    if len(probs) < 2:
        return None
    gap12 = probs[0] - probs[1]
    if gap12 >= TIGHT_GAP12:
        return None  # 拮抗でないレースはスキップ

    # 最強ライン先頭を特定
    leaders = grp[grp["is_line_leader"] == 1]
    if leaders.empty:
        return None
    best_leader = leaders.loc[leaders["race_point"].idxmax()]
    bl_frame = int(best_leader["frame_no"])
    bl_line = int(best_leader["line_group"])

    # その先頭のラインメンバー（番手）
    line_members = grp[(grp["line_group"] == bl_line) & (grp["frame_no"] != best_leader["frame_no"])]

    if not line_members.empty:
        # 番手の中で最高 pred_prob を対軸に
        bf_frame = int(line_members.sort_values("pred_prob", ascending=False)["frame_no"].iloc[0])
    else:
        # 単騎の場合は pred2（モデルの2番手）を対軸に
        bf_frame = int(grp_s.iloc[1]["frame_no"])

    axis_set = {bl_frame, bf_frame}
    thirds = [int(f) for f in grp["frame_no"].astype(int) if f not in axis_set]
    if not thirds:
        return None

    return [frozenset([bl_frame, bf_frame, t]) for t in thirds]


def s3_tight_strong_follower(grp, race_key):
    """S3: 拮抗×強番手（S1 と S2 の交差）。
    gap12 < TIGHT_GAP12 AND within_line_rp_gap > STRONG_FOLLOWER_GAP の両条件。
    """
    grp_s = grp.sort_values("pred_prob", ascending=False)
    probs = grp_s["pred_prob"].tolist()
    if len(probs) < 2:
        return None
    gap12 = probs[0] - probs[1]
    if gap12 >= TIGHT_GAP12:
        return None

    # 強番手チェック
    followers = grp[(grp["is_line_leader"] == 0) & (grp["line_size"] >= 2)]
    if followers.empty or followers["within_line_rp_gap"].max() < STRONG_FOLLOWER_GAP:
        return None

    # 以降は S1 と同じ軸選択
    return s1_strong_follower(grp, race_key)


# ─── オッズ帯別内訳（詳細分析）────────────────────────────────────────────────

def breakdown_by_gap12(df, odds_map, actual_result, strategy_fn, label):
    """gap12 帯別の ROI 内訳。"""
    bins = [0, 0.03, 0.06, 0.10, 0.15, 1.0]
    bin_labels = ["<0.03", "0.03-0.06", "0.06-0.10", "0.10-0.15", "0.15+"]

    sub = df[(df["race_date"] > TRAIN[1]) & (df["race_date"] <= HOLD[1])
             & (df["finish_order"] >= 0)]

    rows = []
    for race_key, grp in sub.groupby("race_key"):
        if grp["n_entries"].iloc[0] > 6:
            continue
        if race_key not in actual_result:
            continue
        probs = grp.sort_values("pred_prob", ascending=False)["pred_prob"].tolist()
        if len(probs) < 2:
            continue
        gap12 = probs[0] - probs[1]

        combos = strategy_fn(grp, race_key)
        if not combos:
            continue
        min_pay = _min_leg_odds(combos, odds_map, race_key)
        if min_pay > 0 and min_pay / 100 < GAMI_THRESHOLD:
            continue

        actual = actual_result[race_key]
        race_odds = odds_map.get(race_key, {})
        hit_pay = race_odds.get(actual, 0) if actual in combos else 0
        rows.append({"gap12": gap12, "bet": len(combos) * 100,
                     "pay": hit_pay, "hit": int(hit_pay > 0)})

    if not rows:
        print(f"\n  {label}: データなし")
        return

    rdf = pd.DataFrame(rows)
    rdf["gap_bin"] = pd.cut(rdf["gap12"], bins=bins, labels=bin_labels)
    g = rdf.groupby("gap_bin").agg(
        n=("bet", "count"),
        total_bet=("bet", "sum"),
        total_pay=("pay", "sum"),
        hits=("hit", "sum"),
    )
    g["roi"] = (g["total_pay"] / g["total_bet"] * 100).round(1)
    g["hit_pct"] = (g["hits"] / g["n"] * 100).round(1)
    print(f"\n  {label} × gap12 帯 (VAL+HOLD):")
    print(f"  {'gap12':<12} {'n':>5} {'hits':>5} {'hit%':>6} {'ROI':>8}")
    for idx, row in g.iterrows():
        v = "★" if row["roi"] >= 100 else " "
        print(f"  {str(idx):<12} {int(row['n']):>5} {int(row['hits']):>5} "
              f"{row['hit_pct']:>5.1f}% {row['roi']:>7.1f}%{v}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    print("ライン買い目設計検証スクリプト")
    print(f"  期間: TRAIN {TRAIN[0]}〜{TRAIN[1]} / VAL {VAL[0]}〜{VAL[1]} / HOLD {VAL[1]}〜{HOLD[1]}")
    print(f"  パラメータ: 強番手閾値={STRONG_FOLLOWER_GAP}点 / 拮抗閾値 gap12<{TIGHT_GAP12} / ガミ≥{GAMI_THRESHOLD}倍")

    print("\nデータ準備中...", flush=True)
    df, odds_map, actual_result = load_all_data()

    strategies = [
        ("S0: 現行 C0 baseline        ", s0_baseline),
        ("S1: 強番手軸シフト           ", s1_strong_follower),
        ("S2: 拮抗×最強ライン軸        ", s2_tight_best_line),
        ("S3: 拮抗×強番手（S1∩S2）     ", s3_tight_strong_follower),
    ]

    print("\n" + "=" * 70)
    print("戦略別 ROI（全3期間）")
    print("=" * 70)
    hdr = f"  {'戦略':<28} {'期間':<6} {'対象R':>6} {'買R':>5} {'的中':>5} {'ROI':>8}"
    print(hdr)
    print("  " + "-" * 62)

    all_results = {}
    for label, fn in strategies:
        res = roi_for_strategy(df, odds_map, actual_result, fn)
        all_results[label] = res
        for period in ["TRAIN", "VAL", "HOLD"]:
            r = res[period]
            v = "★" if r["roi"] >= 100 else " "
            print(f"  {label:<28} {period:<6} {r['n_races']:>6,} {r['bets']:>5,} "
                  f"{r['hits']:>5} {r['roi']:>7.1f}%{v}")
        print()

    # gap12 帯別内訳（VAL+HOLD）
    print("=" * 70)
    print("gap12 帯別 ROI 内訳（VAL+HOLD 合算）")
    print("=" * 70)
    for label, fn in strategies:
        breakdown_by_gap12(df, odds_map, actual_result, fn, label.strip())

    print("\n" + "=" * 70)
    print("サマリ・解釈")
    print("=" * 70)
    print("""
  評価基準（doc18 セマンティクス）:
    ★ = ROI ≥ 100%（全3期間一致が最も頑健）
    最終オッズ上限値・live実測が採否の唯一の根拠

  戦略の狙い:
    S1: ライン構造の読み（番手が先頭より格上 → 番手が主役になりうる）
    S2: 拮抗レースでモデル指数でなくライン先頭強度で軸選択
    S3: 両条件の交差（数が減る分、精度を高める）
""")


if __name__ == "__main__":
    main()
