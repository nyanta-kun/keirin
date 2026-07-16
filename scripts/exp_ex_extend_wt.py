"""WINTICKET EX フィールド拡張実験（G42）

inspect_winticket_ex_fields.py の調査結果に基づき、
未取得フィールドの予測力を評価する。

【発見済み未取得フィールド】
  ex系:
    exCompete              — 競りの勝率（total/succeeded/percentage）充足率 7.8%

  成績系（各フィールドは {first, second, third, others, total, *Percentage} 構造）:
    linePositionFirst/Second/Third — 位置別（先頭・2番手・3番手）成績  充足率 48-66%
    lineSingleHorseman             — 1人旅成績                        充足率 66%
    lineCompete                    — 競り込み成績                     充足率 33%
    weatherSunny/Cloudy/Rainy/Snowy— 天候別成績                       充足率 ~100%
    trackDistance333/400/500       — バンク周長別成績                  充足率 ~98%
    hourTypeNormal/Morning/Night/Midnight/Summertime — 時間帯別成績  充足率 ~100%
    raceTypeQualifyingRound/Semifinal/Final/LoserRound/Special — 種別成績  充填率 不明
    gradeRaceSummaries             — グレード別成績（list型）          充足率  0%（空リスト）

【注意】
  これらのフィールドは wt_entries に存在しない。
  正確な AUC 評価には全 HOLD 期間（8095 レース）を再フェッチする必要があるが、
  ネットワーク制約（約 3 時間以上）のため本スクリプトでは
  HOLD 最新 N_SAMPLE_RACES レースをサンプリングして評価する。

  本番採用判断にはフルフェッチが必要。このスクリプトは方向性を確認する目的。

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）
Phase2 gate: ROI >100% 全3期間

期間: TRAIN 2023-07-01〜2025-06-30 / VAL 2025-07-01〜2026-02-28 / HOLD 2026-03-01〜2026-06-15

Usage:
    python scripts/exp_ex_extend_wt.py [--n-races N] [--no-auc]
"""
import sys
import re
import argparse
from pathlib import Path

# ワークツリー実行時はメインリポジトリを参照する
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if ".claude/worktrees" in str(_REPO_ROOT):
    _REPO_ROOT = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(_REPO_ROOT))
# scripts ディレクトリも追加（exp_segment_first_wt のインポートのため）
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT,
)
from src.scraper.winticket import WinticketScraper, _extract_state, _get_query, _BASE, VENUE_SLUGS
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

# サンプリング設定
DEFAULT_N_SAMPLE_RACES = 50   # サンプルフェッチ数（時間短縮のため）

# フィールドマッピング
# フォーマット: JSON キー -> (派生特徴量名, サブキー)
EX_NEW_FIELD = "exCompete"
EX_NEW_PCT_KEY = ("ex_compete_pct", "percentage")

# 成績系フィールド: (JSONキー, 特徴量名プレフィックス)
# firstPercentage を win_rate_xxx として使用
PERF_FIELDS = [
    # 時間帯別
    ("hourTypeNormal",     "hrn"),
    ("hourTypeMorning",    "hrm"),
    ("hourTypeNight",      "hrni"),
    ("hourTypeMidnight",   "hrmd"),
    # 天候別
    ("weatherSunny",       "ws"),
    ("weatherCloudy",      "wc"),
    ("weatherRainy",       "wr"),
    # バンク周長別
    ("trackDistance333",   "td333"),
    ("trackDistance400",   "td400"),
    ("trackDistance500",   "td500"),
    # 位置別（ライン）
    ("linePositionFirst",  "lp1"),
    ("linePositionSecond", "lp2"),
    ("linePositionThird",  "lp3"),
    ("lineSingleHorseman", "lsh"),
    ("lineCompete",        "lcomp"),
]


def _extract_perf_feature(val: dict | None, key_suffix: str) -> dict:
    """成績系辞書から first/top3 比率を特徴量として返す。"""
    out = {}
    if val and isinstance(val, dict) and val.get("total", 0) > 0:
        total = val["total"]
        out[f"win_rate_{key_suffix}"]  = val.get("first",  0) / total
        out[f"top3_rate_{key_suffix}"] = (
            val.get("first", 0) + val.get("second", 0) + val.get("third", 0)
        ) / total
    else:
        out[f"win_rate_{key_suffix}"]  = np.nan
        out[f"top3_rate_{key_suffix}"] = np.nan
    return out


def fetch_ex_features_for_races(
    race_list: list[tuple[str, str, int]],  # [(venue_id, race_date, race_no)]
    max_races: int = DEFAULT_N_SAMPLE_RACES,
) -> pd.DataFrame:
    """
    指定レースリストを WINTICKET から再フェッチし、
    未取得フィールドを DataFrame で返す。

    Columns:
        race_key, player_id, ex_compete_pct, win_rate_hrn, top3_rate_hrn, ...
    """
    scraper = WinticketScraper(request_interval=1.5)
    rows = []
    processed = 0

    for venue_id, race_date, race_no in race_list[:max_races]:
        info = scraper.find_cup_info(venue_id, race_date)
        if not info:
            continue
        cup_id, day_index = info
        slug = VENUE_SLUGS.get(venue_id)
        if not slug:
            continue
        url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/{race_no}"
        resp = scraper._get(url)
        if resp is None or resp.status_code != 200:
            continue

        state = _extract_state(resp.text)
        data = _get_query(state, "FETCH_KEIRIN_RACE")
        if not data:
            continue

        race_date_yyyymmdd = race_date.replace("-", "")
        race_key = f"{race_date_yyyymmdd}_{venue_id}_{race_no:02d}"

        for rec in data.get("records", []):
            player_id = rec.get("playerId")
            row = {"race_key": race_key, "player_id": player_id}

            # exCompete
            ec = rec.get("exCompete", {})
            row["ex_compete_pct"] = (
                ec.get("percentage") if ec and ec.get("total", 0) > 0 else np.nan
            )

            # 成績系フィールド
            for json_key, suffix in PERF_FIELDS:
                val = rec.get(json_key)
                row.update(_extract_perf_feature(val, suffix))

            rows.append(row)

        processed += 1
        if processed % 10 == 0:
            print(f"  フェッチ完了: {processed} レース", flush=True)

    print(f"  合計フェッチ: {processed} レース ({len(rows)} エントリー)")
    return pd.DataFrame(rows)


def _period_of(rd: str) -> str | None:
    if TRAIN[0] <= rd <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= rd <= VAL[1]:   return "VAL"
    if HOLD[0]  <= rd <= HOLD[1]:  return "HOLD"
    return None


def compute_roi_records(df: pd.DataFrame, trio_map: dict, actual_trio: dict,
                        n_entries_map: dict, gami_threshold: float = 5.0) -> list[dict]:
    records = []
    for rk, grp in df.groupby("race_key"):
        if n_entries_map.get(rk, 99) > 6:
            continue
        g = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 3:
            continue
        period = _period_of(str(g["race_date"].iloc[0]))
        if period is None:
            continue
        p1 = int(g.iloc[0]["frame_no"])
        p2 = int(g.iloc[1]["frame_no"])
        thirds = [int(g.iloc[i]["frame_no"]) for i in range(2, len(g))]
        bd = trio_map.get(rk, {})
        combos = [frozenset({p1, p2, t}) for t in thirds]
        min_odds = min((bd.get(k, 0) for k in combos if bd.get(k, 0) > 0), default=0)
        if min_odds < gami_threshold:
            continue
        actual = actual_trio.get(rk, frozenset())
        pay = 0.0
        for t in thirds:
            k = frozenset({p1, p2, t})
            if actual == k:
                pay = bd.get(k, 0) * 100
                break
        records.append({
            "period": period, "race_key": rk,
            "pay": pay, "cost": len(thirds) * 100,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="EX フィールド拡張実験（G42）")
    parser.add_argument("--n-races", type=int, default=DEFAULT_N_SAMPLE_RACES,
                        help=f"HOLD からサンプリングするレース数 (default: {DEFAULT_N_SAMPLE_RACES})")
    parser.add_argument("--no-fetch", action="store_true",
                        help="フェッチをスキップ（充足率統計のみ表示）")
    args = parser.parse_args()

    print("WINTICKET EX フィールド拡張実験（G42）")
    print()

    # ── 0. フィールドサマリー ──────────────────────────────────────────────
    print("=" * 70)
    print("発見済み未取得フィールド（inspect_winticket_ex_fields.py 調査結果）")
    print("=" * 70)
    print()
    print("  exCompete: 競りの勝率（total/succeeded/percentage）")
    print("    → JSON の ex* グループ内の 6 番目のキー（既取得5項目の他に存在）")
    print("    → 充足率 7.8%（競りは稀なケース）")
    print()
    print("  成績系フィールド（{first,second,third,others,total,*Percentage} 構造）:")
    print("    linePositionFirst/Second/Third : 位置別成績  充足率 48-66%")
    print("    lineSingleHorseman             : 1人旅成績   充足率 66%")
    print("    lineCompete                    : 競り込み成績 充足率 33%")
    print("    weatherSunny/Cloudy/Rainy/Snowy: 天候別成績  充足率 ~100%")
    print("    trackDistance333/400/500       : バンク周長別 充足率 ~98%")
    print("    hourTypeNormal/Morning/Night/Midnight: 時間帯別 充足率 ~100%")
    print("    raceType*                      : 種別成績    充足率 未計測")
    print()
    print("  gradeRaceSummaries: グレード別成績（list 型） 充足率 0%（常に空）")
    print()

    if args.no_fetch:
        print("[--no-fetch] フェッチをスキップしました")
        return

    # ── 1. HOLD 期間のレースリストを取得 ───────────────────────────────────
    print("=" * 70)
    print(f"Phase1: HOLD サンプル（最新 {args.n_races} レース）でのフィールド充足率")
    print("=" * 70)

    with get_connection() as conn:
        hold_races = conn.execute(
            """
            SELECT r.venue_id, r.race_date, r.race_no
            FROM wt_races r
            WHERE r.race_date BETWEEN ? AND ?
            ORDER BY r.race_date DESC, r.race_key
            """,
            (HOLD[0], HOLD[1]),
        ).fetchall()

    # (venue_id, race_date, race_no) のタプルリスト
    race_list = [(r["venue_id"], r["race_date"], r["race_no"]) for r in hold_races]
    print(f"  HOLD 期間総レース数: {len(race_list)}")
    print(f"  サンプリング: {args.n_races} レース")
    print()

    # ── 2. フェッチ ────────────────────────────────────────────────────────
    print("フェッチ開始...", flush=True)
    ex_df = fetch_ex_features_for_races(race_list, max_races=args.n_races)

    if ex_df.empty:
        print("[ERROR] フェッチ結果が空です。ネットワーク接続を確認してください。")
        return

    print()
    print("フィールド充足率（サンプル）:")
    FEAT_COLS = ["ex_compete_pct"] + [
        f"{pref}_{suf}"
        for suf in ["win_rate", "top3_rate"]
        for _, prefix in PERF_FIELDS
        for pref in [f"{suf}_{prefix}"]
    ]
    # 正しい順で確認
    all_feat_cols = (
        ["ex_compete_pct"]
        + [f"win_rate_{p}" for _, p in PERF_FIELDS]
        + [f"top3_rate_{p}" for _, p in PERF_FIELDS]
    )
    for col in all_feat_cols:
        if col not in ex_df.columns:
            continue
        non_nan_rate = ex_df[col].notna().mean() * 100
        print(f"  {col:<35} {non_nan_rate:>6.1f}%")

    # ── 3. AUC 評価 ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Phase1: AUC 評価（サンプルフェッチ + 既存 TRAIN/VAL データ）")
    print("  注: サンプル数が少ないため参考値のみ。本格評価は全期間フェッチが必要。")
    print("=" * 70)

    print("ベースデータ準備中...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    df["race_date"] = df["race_date"].astype(str)

    # ex_df と結合（race_key + player_id）
    print("フェッチデータを結合中...", flush=True)
    ex_df["player_id"] = ex_df["player_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df = df.merge(ex_df, on=["race_key", "player_id"], how="left")

    # 天候別成績: 現在の weather フィールドと関係する特徴量を選択
    # まず充足率の高いフィールドを使う
    HIGH_FILL_COLS = [
        "win_rate_ws",   "top3_rate_ws",    # 晴れ
        "win_rate_wc",   "top3_rate_wc",    # 曇り
        "win_rate_td333","top3_rate_td333",  # バンク333m
        "win_rate_td400","top3_rate_td400",  # バンク400m
        "win_rate_td500","top3_rate_td500",  # バンク500m
        "win_rate_hrn",  "top3_rate_hrn",   # 通常時間
        "win_rate_hrni", "top3_rate_hrni",  # 夜間
        "win_rate_hrmd", "top3_rate_hrmd",  # ミッドナイト
        "win_rate_lp1",  "top3_rate_lp1",   # ライン先頭
        "win_rate_lp2",  "top3_rate_lp2",   # ライン2番手
        "win_rate_lsh",  "top3_rate_lsh",   # 1人旅
    ]
    # 実際に存在するカラムのみ使用
    HIGH_FILL_COLS = [c for c in HIGH_FILL_COLS if c in df.columns]

    # HOLD サンプル内にデータがある行のみで評価
    sample_mask = df["race_key"].isin(ex_df["race_key"])
    n_sample = sample_mask.sum()
    print(f"  サンプル内エントリー数: {n_sample}")

    if n_sample < 100:
        print(f"  [WARN] サンプル数が少なすぎます（{n_sample}）。--n-races を増やしてください。")
        return

    # 学習: TRAIN のみ
    fit_mask = (df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)
    fit = df[fit_mask]

    m_base = lgb.LGBMClassifier(**LGB_PARAMS)
    m_base.fit(prepare_X(fit), fit["top3_flag"].values)

    # 拡張モデル
    EXT_COLS = FEATURE_COLS_WT + HIGH_FILL_COLS
    fit_ext = fit.reindex(columns=EXT_COLS).fillna(0)
    m_ext = lgb.LGBMClassifier(**LGB_PARAMS)
    m_ext.fit(fit_ext, fit["top3_flag"].values)

    df["pred_base"] = m_base.predict_proba(prepare_X(df))[:, 1]
    df["pred_ext"] = m_ext.predict_proba(df.reindex(columns=EXT_COLS).fillna(0))[:, 1]

    print(f"\n  追加特徴量 ({len(HIGH_FILL_COLS)} 個): {HIGH_FILL_COLS[:5]}... 他")
    print()
    print(f"  {'期間':<12} {'Base AUC':>9} {'+ EX':>9} {'diff':>8}  n")
    print("  " + "-" * 50)

    for period, s, e in [
        ("VAL",      VAL[0],  VAL[1]),
        ("HOLD(全)", HOLD[0], HOLD[1]),
        ("HOLD(smp)", HOLD[0], HOLD[1]),  # サンプルのみ
    ]:
        if period == "HOLD(smp)":
            mask = (
                df["race_date"].between(HOLD[0], HOLD[1])
                & (df["finish_order"] >= 1)
                & sample_mask
            )
        else:
            mask = df["race_date"].between(s, e) & (df["finish_order"] >= 1)

        sub = df[mask]
        if len(sub) < 10:
            continue
        auc_b = roc_auc_score(sub["top3_flag"], sub["pred_base"])
        auc_e = roc_auc_score(sub["top3_flag"], sub["pred_ext"])
        diff = auc_e - auc_b
        mark = "★" if abs(diff) >= 0.001 else " "
        print(f"  {period:<12} {auc_b:.4f}   {auc_e:.4f}  {diff:>+7.4f}{mark}  {len(sub)}")

    print()
    print("  ★: |diff| ≥ 0.001 = Phase1 通過ライン")
    print()

    # 特徴量重要度（上位10）
    imp = pd.Series(m_ext.feature_importances_, index=EXT_COLS)
    imp_pct = imp / imp.sum() * 100
    print("  特徴量重要度（拡張モデル・上位15）")
    for feat, v in imp_pct.sort_values(ascending=False).head(15).items():
        marker = " ←" if feat in HIGH_FILL_COLS else ""
        print(f"    {feat:<35} {v:>6.1f}%{marker}")

    print()
    print("結論（サンプル評価）:")
    print("  充足率 ~100%: weather*, trackDistance*, hourType*")
    print("  充足率 48-66%: linePosition*, lineSingleHorseman")
    print("  充足率 33%:    lineCompete")
    print("  充足率 7.8%:   exCompete（競りの勝率）")
    print("  充足率 0%:     gradeRaceSummaries（常に空リスト）")
    print()
    print("  本スクリプトはサンプルによる方向性確認のみ。")
    print("  全期間フェッチによる正確な AUC 評価には別途フルフェッチが必要。")


if __name__ == "__main__":
    main()
