"""fav_mismatch リーク無し単独検証

定義: fav_mismatch = モデル1位(pred1) ≠ 市場本命(trio implied P(top3)最大の車番)
市場本命計算: q_i = Σ_{iを含むtrio組} 1/odds  (main.py _market_fav_frame と同一ロジック)

検証内容:
  1. fav_mismatch の発生頻度・分布確認
  2. fav_mismatch=True/False 別 ROI（現行C0戦略: pred1+pred2→thirds・ガミ≥5倍）
  3. 全期間(TRAIN/VAL/HOLD)・gap12 帯別 内訳

doc13 の 1168%/576% は doc18 以前のバイアス込み。本検証がリーク無し初回。

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-14
"""

import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_THRESHOLD = 5.0


# ─── 市場本命計算（main.py と同一ロジック）──────────────────────────────────

def market_fav_frame(race_trio_odds: dict) -> int | None:
    """trio盤面から市場本命の車番を返す。q_i = Σ 1/odds for all combos containing i."""
    q: dict[int, float] = {}
    n_combo = 0
    for combo_key, ov in race_trio_odds.items():
        if ov <= 0 or ov >= 9000 * 100:
            continue
        n_combo += 1
        for fno in combo_key:
            q[fno] = q.get(fno, 0.0) + 100.0 / ov   # ov は円建て
    if n_combo < 4 or not q:
        return None
    return max(q, key=q.get)


# ─── データ準備 ──────────────────────────────────────────────────────────────

def load_all():
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        races_info = pd.read_sql("SELECT race_key, n_entries FROM wt_races", conn)
        trio_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'",
            conn,
        )

    df = df.merge(races_info, on="race_key", how="left")

    print("  リーク無しモデル学習中 (TRAIN 期間のみ)...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    X_tr = prepare_X(fit).reset_index(drop=True)
    y_tr = fit["top3_flag"].reset_index(drop=True).values
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)
    print(f"  学習完了 ({len(fit):,} 行)")

    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # trio オッズ辞書（frozenset → 円）
    def _parse(s):
        try:
            return frozenset(int(p) for p in re.split(r"[-=]", str(s)))
        except Exception:
            return None

    trio_df["combo_key"] = trio_df["combination"].apply(_parse)
    trio_df = trio_df.dropna(subset=["combo_key"])
    trio_odds_map: dict[str, dict] = {}
    for row in trio_df.itertuples(index=False):
        trio_odds_map.setdefault(row.race_key, {})[row.combo_key] = row.odds_value * 100

    # 実際の trio 結果
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    return df, trio_odds_map, actual_trio


# ─── メイン分析 ──────────────────────────────────────────────────────────────

def get_period(race_date):
    if TRAIN[0] <= race_date <= TRAIN[1]: return "TRAIN"
    if VAL[0] <= race_date <= VAL[1]: return "VAL"
    if HOLD[0] <= race_date <= HOLD[1]: return "HOLD"
    return None


def main():
    print("fav_mismatch リーク無し単独検証")
    print(f"  doc13 の 1168%/576% はバイアス込み → 本検証がリーク無し初回")
    print()

    print("データ準備中...", flush=True)
    df, trio_odds_map, actual_trio = load_all()

    records = []

    for race_key, grp in df.groupby("race_key"):
        period = get_period(grp["race_date"].iloc[0])
        if period is None:
            continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6:
            continue

        grp_s = grp.sort_values("pred_prob", ascending=False)
        rows = grp_s.reset_index(drop=True)
        if len(rows) < 3:
            continue

        p1_prob, p2_prob = rows.iloc[0]["pred_prob"], rows.iloc[1]["pred_prob"]
        gap12 = p1_prob - p2_prob
        n = int(rows.iloc[0].get("n_entries", len(rows)))

        pred1_frame = int(rows.iloc[0]["frame_no"])
        pred2_frame = int(rows.iloc[1]["frame_no"])
        thirds = [int(r["frame_no"]) for _, r in rows.iloc[2:].iterrows()]

        race_trio = trio_odds_map.get(race_key, {})

        # ガミ足切り（pred1+pred2→thirds の最安目）
        combos = [frozenset({pred1_frame, pred2_frame, t}) for t in thirds]
        min_odds = min(
            (race_trio.get(k, 0) for k in combos if race_trio.get(k, 0) > 0),
            default=0,
        )
        if min_odds < GAMI_THRESHOLD * 100:
            continue

        # 市場本命
        mkt_fav = market_fav_frame(race_trio)
        fav_mm = (mkt_fav is not None and mkt_fav != pred1_frame)

        # 的中・払戻
        actual = actual_trio.get(race_key, frozenset())
        pay = 0
        for t in thirds:
            k = frozenset({pred1_frame, pred2_frame, t})
            if actual == k:
                pay = race_trio.get(k, 0)
                break

        # gap12 帯
        if gap12 < 0.06:
            band = "<0.06"
        elif gap12 < 0.10:
            band = "0.06-0.10"
        elif gap12 < 0.15:
            band = "0.10-0.15"
        else:
            band = "0.15+"

        records.append({
            "period": period,
            "race_key": race_key,
            "fav_mismatch": fav_mm,
            "mkt_fav_is_none": mkt_fav is None,
            "gap12_band": band,
            "gap12": gap12,
            "pay": pay,
            "cost": len(thirds) * 100,
            "hit": int(pay > 0),
        })

    recs = pd.DataFrame(records)

    # ── 発生頻度 ──────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("発生頻度（≤6車・ガミ≥5倍 対象レース）")
    print(f"{'='*70}")
    for period in ["TRAIN", "VAL", "HOLD"]:
        sub = recs[recs["period"] == period]
        if len(sub) == 0: continue
        mm = sub["fav_mismatch"].sum()
        total = len(sub)
        print(f"  {period}: {total}R 中 fav_mismatch={mm}R ({mm/total*100:.1f}%) "
              f"/ non-mismatch={total-mm}R ({(total-mm)/total*100:.1f}%)")
    print()

    # ── 全期間 ROI ────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("全期間 ROI 比較（現行C0戦略: pred1+pred2→thirds・ガミ≥5倍）")
    print(f"{'='*70}")
    print(f"  {'グループ':<20} {'期間':<7} {'対象R':>7} {'的中':>6} {'ROI':>9}")
    print("  " + "-"*52)

    for mm_flag, label in [(None, "全レース"), (True, "fav_mismatch=True"), (False, "fav_mismatch=False")]:
        for period in ["TRAIN", "VAL", "HOLD"]:
            if mm_flag is None:
                sub = recs[recs["period"] == period]
            else:
                sub = recs[(recs["period"] == period) & (recs["fav_mismatch"] == mm_flag)]
            if len(sub) == 0: continue
            roi = sub["pay"].sum() / sub["cost"].sum() * 100
            mark = "★" if roi >= 100 else ""
            print(f"  {label:<20} {period:<7} {len(sub):>7} {sub['hit'].sum():>6} {roi:>8.1f}%{mark}")
        print()

    # ── gap12 帯 × fav_mismatch ───────────────────────────────────────────────
    print(f"{'='*70}")
    print("gap12 帯別 ROI（VAL+HOLD 合算）")
    print(f"{'='*70}")
    bands_order = ["<0.06", "0.06-0.10", "0.10-0.15", "0.15+"]
    sub_oos = recs[recs["period"].isin(["VAL", "HOLD"])]

    print(f"  {'gap12帯':<12} {'全体':>9} {'mismatch':>10} {'non-mismatch':>13}  {'n(mm)':>7} {'n(non)':>7}")
    print("  " + "-"*62)
    for band in bands_order:
        sub_b = sub_oos[sub_oos["gap12_band"] == band]
        if len(sub_b) == 0: continue
        roi_all = sub_b["pay"].sum() / sub_b["cost"].sum() * 100

        sub_mm = sub_b[sub_b["fav_mismatch"]]
        roi_mm = sub_mm["pay"].sum() / sub_mm["cost"].sum() * 100 if len(sub_mm) > 0 else float("nan")
        sub_nm = sub_b[~sub_b["fav_mismatch"]]
        roi_nm = sub_nm["pay"].sum() / sub_nm["cost"].sum() * 100 if len(sub_nm) > 0 else float("nan")

        mark_mm = "★" if roi_mm >= 100 else " "
        mark_nm = "★" if roi_nm >= 100 else " "
        print(f"  {band:<12} {roi_all:>8.1f}%  {roi_mm:>8.1f}%{mark_mm}  {roi_nm:>11.1f}%{mark_nm}  "
              f"{len(sub_mm):>7} {len(sub_nm):>7}")

    # ── fav_mismatch 限定の gap12 帯別詳細 ───────────────────────────────────
    print(f"\n{'='*70}")
    print("fav_mismatch=True 限定 / gap12 帯別 全3期間")
    print(f"{'='*70}")
    sub_mm_all = recs[recs["fav_mismatch"]]
    print(f"  {'gap12帯':<12} {'TRAIN':>10} {'VAL':>10} {'HOLD':>10}  n(TRN/VAL/HLD)")
    print("  " + "-"*62)
    for band in bands_order:
        sub_b = sub_mm_all[sub_mm_all["gap12_band"] == band]
        row = f"  {band:<12}"
        ns = []
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub_p = sub_b[sub_b["period"] == period]
            ns.append(len(sub_p))
            if len(sub_p) == 0:
                row += f"{'  -':>10}"
                continue
            roi = sub_p["pay"].sum() / sub_p["cost"].sum() * 100
            mark = "★" if roi >= 100 else ""
            row += f"{roi:>9.1f}%{mark}"
        row += f"  {ns[0]:>4}/{ns[1]:>3}/{ns[2]:>3}"
        print(row)

    # ── 結論サマリ ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("サマリ（doc13 との比較）")
    print(f"{'='*70}")

    oos_mm = recs[recs["period"].isin(["VAL", "HOLD"]) & recs["fav_mismatch"]]
    oos_nm = recs[recs["period"].isin(["VAL", "HOLD"]) & ~recs["fav_mismatch"]]
    roi_mm_oos = oos_mm["pay"].sum() / oos_mm["cost"].sum() * 100 if len(oos_mm) > 0 else 0
    roi_nm_oos = oos_nm["pay"].sum() / oos_nm["cost"].sum() * 100 if len(oos_nm) > 0 else 0

    print(f"  doc13 の数字:       fav_mismatch ROI 1168%(TRAIN) / 576%(OOS) ← バイアス込み")
    print(f"  本検証（リーク無し）:")

    for mm_flag, label in [(True, "fav_mismatch=True "), (False, "fav_mismatch=False")]:
        for period in ["TRAIN", "VAL", "HOLD"]:
            sub = recs[(recs["period"] == period) & (recs["fav_mismatch"] == mm_flag)]
            if len(sub) == 0:
                continue
            roi = sub["pay"].sum() / sub["cost"].sum() * 100
            mark = "★" if roi >= 100 else ""
            print(f"    {label} {period}: {len(sub)}R / ROI {roi:.1f}%{mark}")
        print()

    print("  採否判断: live実測(picks_history)のみ。backtestは最終オッズ上限値。")


if __name__ == "__main__":
    main()
