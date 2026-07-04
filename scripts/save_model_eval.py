#!/usr/bin/env python3
"""バックテスト結果を PostgreSQL keirin.model_evaluation に保存する。

7+車 SS/S/A 本番戦略（wave-picks-wt と同条件）で VAL / HOLD を評価し、
kiseki フロントエンドの「モデル精度」表示用に保存する。

事前条件:
  - KEIRIN_DB_URL 環境変数が設定されていること
  - lgbm_wt_train_only モデルが data/models/ に存在すること
    （週次再学習リークを避けるため TRAIN 期間のみで学習したモデルを使用）
  - keirin.model_evaluation テーブルが存在すること
    （kiseki Alembic migration e1f2g3h4i5j6 を適用済みであること）

実行例:
  export KEIRIN_DB_URL="postgresql://user:pass@host:5432/dbname"
  python3 scripts/save_model_eval.py
  python3 scripts/save_model_eval.py --dry-run   # DB書き込みなし（数値確認のみ）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import lightgbm as lgb
import pandas as pd

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)
from src.models.trainer import load_model

# ── バックテスト対象期間 ──────────────────────────────────────────────
VAL  = ("2025-07-01", "2026-02-28")
HOLD = ("2026-03-01", "2026-06-16")

# ── 期間別評価モデル（汚染なし設計） ─────────────────────────────────
# VAL評価:  lgbm_wt_train_only（TRAIN 2022-12〜2025-06-30のみ学習・VALを汚染していない）
# HOLD評価: lgbm_wt          （TRAIN+VAL 2022-12〜2026-02-28学習・HOLDを汚染していない）
VAL_MODEL_NAME  = "lgbm_wt_train_only"
HOLD_MODEL_NAME = "lgbm_wt"

# ── 戦略パラメータ（wave-picks-wt と同条件） ─────────────────────────
MIN_RIDERS  = 7       # 7車以上対象
MIN_GAP12   = 0.07    # gap12 最低閾値
S_GAP12     = 0.10    # S/A 分岐閾値
GAMI_MIN    = 5.0     # ガミ足切りオッズ


def _load_trio_odds(race_keys: list[str]) -> dict[str, dict[frozenset, float]]:
    """wt_odds から {race_key: {combo_frozenset: odds}} を返す。"""
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, combination, odds_value FROM wt_odds "
            f"WHERE bet_type='trio' AND race_key IN ({placeholders})",
            race_keys,
        ).fetchall()

    odds_map: dict[str, dict] = {}
    combo_re = re.compile(r"[\d]+")
    for rk, combo_str, ov in rows:
        if ov is None or float(ov) <= 0:
            continue
        parts = combo_re.findall(str(combo_str))
        if len(parts) == 3:
            fs = frozenset(int(p) for p in parts)
            odds_map.setdefault(str(rk), {})[fs] = float(ov)
    return odds_map


def _load_n_entries(race_keys: list[str]) -> dict[str, int]:
    """wt_races から {race_key: n_entries} を返す。"""
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, n_entries FROM wt_races WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {str(rk): (int(ne) if ne else 0) for rk, ne in rows}


def run_7plus_backtest(
    df: pd.DataFrame,
    model,
    period_from: str,
    period_to: str,
) -> dict:
    """7+車 SS/S/A 本番戦略のバックテストを実行して集計結果を返す。

    戦略 (wave-picks-wt と同条件):
      - n_entries ≥ 7 かつ gap12 ≥ MIN_GAP12
      - SS: ガミ目カット後 残り1〜3目（残存目のみ購入）
      - S : 全目 gami ≥ 5倍 + gap12 ≥ S_GAP12 → 全3相手(3点)
      - A : 全目 gami ≥ 5倍 + gap12 ∈ [MIN_GAP12, S_GAP12) → 全3相手(3点)
    """
    df_period = df[
        (df["race_date"] >= period_from) &
        (df["race_date"] <= period_to) &
        (df["finish_order"].notna())
    ].copy()

    if df_period.empty:
        print(f"  [{period_from}〜{period_to}] データなし", flush=True)
        return {"n_picks": 0, "n_hits": 0, "total_bet": 0, "total_payout": 0, "roi": None}

    # モデル予測確率を付与（実走者のみで予測し、欠車選手は pred_prob=0 でランク末尾に）
    # この方式は exp_7plus_conditional_wt.py 等の実験ハーネスと同一の条件であり、
    # doc48/doc49 の 137-138% HOLD 結果と整合する。
    df_runners = df_period[df_period["finish_order"] >= 1].copy()
    X = prepare_X(df_runners)
    df_runners["pred_prob"] = model.predict_proba(X)[:, 1]
    df_period = df_period.merge(
        df_runners[["race_key", "frame_no", "pred_prob"]],
        on=["race_key", "frame_no"],
        how="left",
    )
    df_period["pred_prob"] = df_period["pred_prob"].fillna(0.0)

    all_race_keys = df_period["race_key"].unique().tolist()
    n_entries_map = _load_n_entries(all_race_keys)
    trio_odds_map = _load_trio_odds(all_race_keys)

    # 7+車のレースのみ
    target_keys = {rk for rk in all_race_keys if n_entries_map.get(rk, 0) >= MIN_RIDERS}
    df_7plus = df_period[df_period["race_key"].isin(target_keys)].copy()

    total_n_bet_races = total_bets = total_returns = total_hits = 0
    ss_bets = ss_returns = ss_hits = ss_races = 0
    s_bets  = s_returns  = s_hits  = s_races  = 0
    a_bets  = a_returns  = a_hits  = a_races  = 0

    for race_key, grp in df_7plus.groupby("race_key"):
        # 欠車(pred_prob=0)は自然にランク末尾へ → pivot1/pivot2 は常に実走者
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue

        probs = grp["pred_prob"].tolist()
        gap12 = probs[0] - probs[1]
        if gap12 < MIN_GAP12:
            continue

        frames = grp["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds_all = frames[2:]
        if not thirds_all:
            continue

        runners = set(grp[grp["finish_order"] >= 1]["frame_no"].astype(int).tolist())
        if pivot1 not in runners or pivot2 not in runners:
            continue

        fin = grp[grp["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3_set = frozenset(fin["frame_no"].astype(int).tolist())

        rk_odds = trio_odds_map.get(race_key, {})

        combo_odds: dict[int, float] = {}
        for t in thirds_all:
            if t not in runners:
                continue
            cs = frozenset({pivot1, pivot2, t})
            ov = rk_odds.get(cs)
            if ov and ov > 0:
                combo_odds[t] = ov

        if not combo_odds:
            continue

        race_bet = False

        # SSランク: gami≥5倍の目が 1〜3点
        valid_ss = [t for t in thirds_all if combo_odds.get(t, 0) >= GAMI_MIN]
        if 1 <= len(valid_ss) <= 3:
            for t in valid_ss:
                cs = frozenset({pivot1, pivot2, t})
                ov = combo_odds[t]
                total_bets += 100
                ss_bets += 100
                if cs == top3_set:
                    # 公式払戻金は10円単位に切り捨て
                    payout = round(ov * 100) // 10 * 10
                    total_returns += payout
                    total_hits += 1
                    ss_returns += payout
                    ss_hits += 1
            race_bet = True
            ss_races += 1

        # S/Aランク: 全相手がgami≥5倍。gap12 で S / A を分岐
        all_thirds_runner = [t for t in thirds_all if t in runners]
        if all_thirds_runner and all(combo_odds.get(t, 0) >= GAMI_MIN for t in all_thirds_runner):
            is_s_rank = (gap12 >= S_GAP12)
            for t in all_thirds_runner:
                ov = combo_odds.get(t, 0)
                if ov <= 0:
                    continue
                cs = frozenset({pivot1, pivot2, t})
                total_bets += 100
                hit_this = (cs == top3_set)
                pay = (round(ov * 100) // 10 * 10) if hit_this else 0
                if is_s_rank:
                    s_bets += 100
                    if hit_this:
                        s_returns += pay; s_hits += 1
                else:
                    a_bets += 100
                    if hit_this:
                        a_returns += pay; a_hits += 1
                if hit_this:
                    total_returns += pay
                    total_hits += 1
            race_bet = True
            if is_s_rank:
                s_races += 1
            else:
                a_races += 1

        if race_bet:
            total_n_bet_races += 1

    roi    = round(total_returns / total_bets, 3) if total_bets > 0 else None
    ss_roi = round(ss_returns / ss_bets, 3)       if ss_bets  > 0 else None
    s_roi  = round(s_returns  / s_bets,  3)       if s_bets   > 0 else None
    a_roi  = round(a_returns  / a_bets,  3)       if a_bets   > 0 else None

    result = {
        "n_picks":      total_n_bet_races,
        "n_hits":       total_hits,
        "total_bet":    total_bets,
        "total_payout": total_returns,
        "roi":          roi,
        "by_rank": {
            "SS": {"n_picks": ss_races, "n_hits": ss_hits, "total_bet": ss_bets,
                   "total_payout": ss_returns, "roi": ss_roi},
            "S":  {"n_picks": s_races,  "n_hits": s_hits,  "total_bet": s_bets,
                   "total_payout": s_returns,  "roi": s_roi},
            "A":  {"n_picks": a_races,  "n_hits": a_hits,  "total_bet": a_bets,
                   "total_payout": a_returns,  "roi": a_roi},
        },
    }

    def _fmt_roi(r):
        return f"{r:.1%}" if r is not None else "—"

    print(
        f"  [{period_from}〜{period_to}] "
        f"n_picks={total_n_bet_races:,}R  的中={total_hits:,}  "
        f"投資={total_bets:,}円  回収={total_returns:,}円  ROI={_fmt_roi(roi)}",
        flush=True,
    )
    print(f"    SS: {ss_races:,}R  投資={ss_bets:,}  回収={ss_returns:,}  ROI={_fmt_roi(ss_roi)}", flush=True)
    print(f"    S:  {s_races:,}R  投資={s_bets:,}  回収={s_returns:,}  ROI={_fmt_roi(s_roi)}",  flush=True)
    print(f"    A:  {a_races:,}R  投資={a_bets:,}  回収={a_returns:,}  ROI={_fmt_roi(a_roi)}",  flush=True)
    return result


def save_to_db(
    model_name: str,
    period_type: str,
    period_from: str,
    period_to: str,
    result: dict,
) -> None:
    """model_evaluation テーブルに UPSERT する（全体 + ランク別）。"""
    rows = [
        (model_name, period_from, period_to, period_type,
         result["n_picks"], result["n_hits"], result["total_bet"],
         result["total_payout"], result["roi"]),
    ]
    for rank_key, rd in result.get("by_rank", {}).items():
        rank_model = f"{model_name}#7{rank_key}"
        rows.append((
            rank_model, period_from, period_to, period_type,
            rd["n_picks"], rd["n_hits"], rd["total_bet"],
            rd["total_payout"], rd["roi"],
        ))

    with get_connection() as conn:
        for row in rows:
            conn.execute(
                "INSERT OR REPLACE INTO model_evaluation "
                "(model_name, period_from, period_to, period_type, "
                " n_picks, n_hits, total_bet, total_payout, roi) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
    print(
        f"  → DB保存完了 ({period_type}: {period_from}〜{period_to}, "
        f"{len(rows)}行)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="7+車バックテスト結果をDBに保存")
    parser.add_argument("--dry-run", action="store_true", help="DB書き込みなし（数値確認のみ）")
    args = parser.parse_args()

    # 期間別にモデルを読み込む（VAL: train_only / HOLD: lgbm_wt）
    models = {}
    for period_type, model_name in [("VAL", VAL_MODEL_NAME), ("HOLD", HOLD_MODEL_NAME)]:
        print(f"モデル読み込み [{period_type}]: {model_name}", flush=True)
        try:
            models[period_type] = (model_name, load_model(model_name))
        except FileNotFoundError:
            print(f"ERROR: モデル '{model_name}' が見つかりません。train-wt を先に実行してください。")
            sys.exit(1)

    # データ読み込みはローカル SQLite から（KEIRIN_DB_URL は保存専用）
    save_db_url = os.environ.pop("KEIRIN_DB_URL", None)
    print("データ読み込み中 ...", flush=True)
    df_raw = load_raw_data_wt(min_date=VAL[0], max_date=HOLD[1])
    if df_raw.empty:
        print("ERROR: データがありません。collect-wt を先に実行してください。")
        sys.exit(1)
    df = build_features_wt(df_raw)

    # バックテスト（wt_odds / wt_races 参照もローカル SQLite）
    # 保存フェーズで KEIRIN_DB_URL を復元
    results = {}
    for period_type, period in [("VAL", VAL), ("HOLD", HOLD)]:
        pfrom, pto = period
        model_name, model = models[period_type]
        print(f"\n--- {period_type}: {pfrom}〜{pto}  [{model_name}] ---", flush=True)
        results[period_type] = (model_name, run_7plus_backtest(df, model, pfrom, pto))

    # 保存フェーズ: KEIRIN_DB_URL を復元して VPS に書き込む
    if save_db_url:
        os.environ["KEIRIN_DB_URL"] = save_db_url
    for period_type, period in [("VAL", VAL), ("HOLD", HOLD)]:
        pfrom, pto = period
        model_name, result = results[period_type]
        if not args.dry_run and result["n_picks"] > 0:
            save_to_db(model_name, period_type, pfrom, pto, result)
        elif args.dry_run:
            print(f"  (dry-run: {period_type} DB書き込みスキップ)", flush=True)
        else:
            print(f"  ({period_type} n_picks=0: スキップ)", flush=True)

    print("\n完了", flush=True)


if __name__ == "__main__":
    main()
