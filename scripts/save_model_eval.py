#!/usr/bin/env python3
"""バックテスト結果を PostgreSQL keirin.model_evaluation に保存する。

7車ちょうど限定の本番戦略（2026-07-10〜 新ランク体系・notify_prerace_wt.py と同条件）で
VAL / HOLD を評価し、kiseki フロントエンドの「モデル精度」表示用に保存する。

ランク体系（notify_prerace_wt.py の `_determine_live_rank` / `_determine_st_rank` と同一）:
  R  （表示 SS・三連複・レース単位）: min(全目オッズ)≥GAMI_THRESHOLD ∧ gap12≥SEVEN_PLUS_S_GAP12
       ∧ gap23≥GAP23_MIN → 全目購入 100円/点。的中条件=軸2車(pivot1/pivot2)が3着内。
  ST （表示 S・三連単1着固定フォーメーション）: gap12≥ST_GAP12 ∧ min(全目オッズ)≥ST_GAMI
       → 100円/点。的中条件=1着=pivot1 ∧ 2着∈{pivot2, 指数3位}。
  STP（表示 S+・同上 + 増額）: ST条件 + gap12≥STP_GAP12 ∧ gap34≥STP_GAP34 → 200円/点。

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
from src.strategy_wt import line_score_features, ss_policy, st_normal_allowed

# ── バックテスト対象期間 ──────────────────────────────────────────────
VAL  = ("2025-07-01", "2026-02-28")
HOLD = ("2026-03-01", "2026-06-16")

# ── 期間別評価モデル（汚染なし設計） ─────────────────────────────────
# VAL評価:  lgbm_wt_train_only（TRAIN 2022-12〜2025-06-30のみ学習・VALを汚染していない）
# HOLD評価: lgbm_wt          （TRAIN+VAL 2022-12〜2026-02-28学習・HOLDを汚染していない）
VAL_MODEL_NAME  = "lgbm_wt_train_only"
HOLD_MODEL_NAME = "lgbm_wt"

# ── 戦略パラメータ（notify_prerace_wt.py と同値・2026-07-10〜 新ランク体系） ──────
N_ENTRIES_TARGET   = 7      # 7車ちょうど限定（8/9車は対象外。write_candidates_wt.py/main.py と同一基準）

# R（表示SS・三連複・レース単位）
GAMI_THRESHOLD     = 7.0    # min(全目三連複オッズ) 下限
SEVEN_PLUS_S_GAP12 = 0.10   # gap12 下限
GAP23_MIN          = 1.0    # gap23（2位-3位予測確率差, pt）下限

# ST/STP（表示S/S+・三連単1着固定フォーメーション）
ST_GAP12    = 0.15   # S: gap12 下限
ST_GAMI     = 10.0   # S: min(全目三連単オッズ) 下限
STP_GAP12   = 0.25   # S+: gap12 下限（Sの条件に追加）
STP_GAP34   = 0.04   # S+: gap34（3位-4位予測確率差）下限
ST_STAKE    = 100    # S 1点あたり金額（円）
STP_STAKE   = 200    # S+ 1点あたり金額（円）


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


def _load_trifecta_odds(race_keys: list[str]) -> dict[str, dict[tuple[int, int, int], float]]:
    """wt_odds(bet_type='trifecta') から {race_key: {(着順タプル): odds}} を返す。

    combination は "1-2-3"（着順そのまま）形式で保存されている（"=" 区切りの trio とは別形式）。
    ST/STP ランク（三連単1着固定フォーメーション）の判定・採点に使う。
    """
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, combination, odds_value FROM wt_odds "
            f"WHERE bet_type='trifecta' AND race_key IN ({placeholders})",
            race_keys,
        ).fetchall()

    odds_map: dict[str, dict] = {}
    for rk, combo_str, ov in rows:
        if ov is None or float(ov) <= 0:
            continue
        parts = re.split(r"[-=]", str(combo_str))
        try:
            nums = tuple(int(p) for p in parts)
        except ValueError:
            continue
        if len(nums) == 3:
            odds_map.setdefault(str(rk), {})[nums] = float(ov)
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


def _load_race_types(race_keys: list[str]) -> dict[str, str | None]:
    """wt_races から {race_key: race_type} を返す（doc53 選抜カット用）。"""
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, race_type FROM wt_races WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {str(rk): rt for rk, rt in rows}


def run_7plus_backtest(
    df: pd.DataFrame,
    model,
    period_from: str,
    period_to: str,
) -> dict:
    """7車ちょうど限定・現行R/ST/STPランク本番戦略のバックテストを実行して集計結果を返す。

    判定条件は notify_prerace_wt.py の `_determine_live_rank` / `_determine_st_rank` と同値
    （閾値定数は本ファイル冒頭に集約・値は notify_prerace_wt.py の同名定数と一致させること）。

    戦略:
      - n_entries == N_ENTRIES_TARGET(7) のレースのみ対象（8/9車は対象外）
      - R  (表示SS・三連複レース単位): min(全目オッズ)≥GAMI_THRESHOLD ∧ gap12≥SEVEN_PLUS_S_GAP12
        ∧ gap23≥GAP23_MIN → 全目購入 100円/点。的中=軸2車(pivot1/pivot2)が3着内。
      - ST (表示S・三連単1着固定F): gap12≥ST_GAP12 ∧ min(全目オッズ)≥ST_GAMI → 100円/点。
        的中=1着=pivot1 ∧ 2着∈{pivot2, 指数3位}（3着は全通り）。
      - STP(表示S+・同上+増額): ST条件 + gap12≥STP_GAP12 ∧ gap34≥STP_GAP34 → 200円/点。
      R と ST/STP は独立判定（同一レースで両方成立しうる。notify_prerace_wt.py と同様）。

    欠車(finish_order=0)の返還処理は notify_results_wt.py の `_void_by_dns` と同一規則:
      軸(R: pivot1/pivot2、ST: pivot1)が欠車 → 賭け不成立（除外）。
      相手（third・ST の pivot2/指数3位/3着候補）が欠車 → その目のみ除外。
    ランキングは全エントリー（欠車含む・pred_prob=0でランク末尾）で行い、車数判定は
    wt_races.n_entries（出走表基準）で行う — exp_leakfree_rescore_wt.py と同じバイアス回避。
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
    # ランキングは全エントリー（欠車含む）で行う — 本番 main.py / exp_leakfree_rescore_wt.py と同条件。
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
    race_type_map = _load_race_types(all_race_keys)
    trio_odds_map = _load_trio_odds(all_race_keys)
    trifecta_odds_map = _load_trifecta_odds(all_race_keys)

    # 7車ちょうどのレースのみ（出走表基準・write_candidates_wt.py/main.py と同一基準）
    target_keys = {rk for rk in all_race_keys if n_entries_map.get(rk, 0) == N_ENTRIES_TARGET}
    df_7 = df_period[df_period["race_key"].isin(target_keys)].copy()

    total_n_bet_races = total_bets = total_returns = total_hits = 0
    r_bets   = r_returns   = r_hits   = r_races   = 0   # R（表示SS）
    st_bets  = st_returns  = st_hits  = st_races  = 0   # ST（表示S）
    stp_bets = stp_returns = stp_hits = stp_races = 0   # STP（表示S+）

    for race_key, grp in df_7.groupby("race_key"):
        # 欠車(pred_prob=0)は自然にランク末尾へ → pivot1/pivot2 は原則実走者
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue

        probs = grp["pred_prob"].tolist()
        frames = grp["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:]
        if not thirds:
            continue

        fin = grp[grp["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue  # 結果未確定レース
        top3_set = frozenset(fin["frame_no"].astype(int).tolist())
        actual_order = tuple(
            fin.sort_values("finish_order")["frame_no"].astype(int).tolist()[:3]
        )

        runners = set(grp[grp["finish_order"] >= 1]["frame_no"].astype(int).tolist())

        gap12 = probs[0] - probs[1]
        race_bet = False

        # doc53 統合ポリシーのコンテキスト（選抜/4分戦/ライン格差増額）
        race_type = race_type_map.get(race_key)
        _line_pairs = [
            (None if pd.isna(_r.line_group) else int(_r.line_group),
             None if pd.isna(_r.race_point) else float(_r.race_point))
            for _r in grp.itertuples(index=False)
        ]
        avg_gap, n_lines, all_solo = line_score_features(_line_pairs)

        # ── Rランク（表示SS・三連複・レース単位ガミ） ──────────────────────
        if pivot1 in runners and pivot2 in runners:
            rk_trio_odds = trio_odds_map.get(race_key, {})
            valid_thirds = [t for t in thirds if t in runners]  # 相手欠車はその目のみ除外
            combo_odds: dict[int, float] = {}
            for t in valid_thirds:
                ov = rk_trio_odds.get(frozenset({pivot1, pivot2, t}))
                if ov and ov > 0:
                    combo_odds[t] = ov

            if combo_odds:
                gami_r = min(combo_odds.values())
                gap23 = (probs[1] - probs[2]) * 100.0 if len(probs) >= 3 else 0.0
                # doc53: 選抜/4分戦は見送り・ライン格差>=1.5は増額（200円/点）
                _skip_r, _stake_r = ss_policy(race_type, avg_gap, n_lines, all_solo)
                if (gami_r >= GAMI_THRESHOLD and gap12 >= SEVEN_PLUS_S_GAP12
                        and gap23 >= GAP23_MIN and not _skip_r):
                    race_bet = True
                    r_races += 1
                    for t, ov in combo_odds.items():
                        total_bets += _stake_r
                        r_bets += _stake_r
                        if frozenset({pivot1, pivot2, t}) == top3_set:
                            # 公式払戻金は10円単位に切り捨て
                            pay = (round(ov * 100) // 10 * 10) * (_stake_r // 100)
                            total_returns += pay
                            total_hits += 1
                            r_returns += pay
                            r_hits += 1

        # ── ST/STPランク（表示S/S+・三連単1着固定フォーメーション） ─────────
        # 1着=pivot1固定 / 2着=pivot2または指数3位(r3) / 3着=全通り（軸欠車は不成立・相手欠車はその半分/その目のみ除外）
        if gap12 >= ST_GAP12 and thirds and pivot1 in runners:
            rk_tri_odds = trifecta_odds_map.get(race_key, {})
            r3 = thirds[0]
            st_combos: dict[tuple[int, int, int], float] = {}
            for s in (pivot2, r3):
                if s not in runners:
                    continue  # 2着候補が欠車 → その半分は購入不成立
                for t in frames:
                    if t in (pivot1, s) or t not in runners:
                        continue
                    ov = rk_tri_odds.get((pivot1, s, t))
                    if ov and ov > 0:
                        st_combos[(pivot1, s, t)] = ov

            if st_combos:
                gami_st = min(st_combos.values())
                gap34 = (probs[2] - probs[3]) if len(probs) >= 4 else 0.0
                is_plus = (gap12 >= STP_GAP12 and gap34 >= STP_GAP34)
                # doc53: S通常帯は min>=ST_BASE_GAMI(15) ∧ 非選抜（S+帯は現行のまま）
                if gami_st >= ST_GAMI and (
                        is_plus or st_normal_allowed(race_type, gami_st)):
                    stake = STP_STAKE if is_plus else ST_STAKE
                    race_bet = True
                    if is_plus:
                        stp_races += 1
                    else:
                        st_races += 1
                    for combo, ov in st_combos.items():
                        total_bets += stake
                        hit_this = (combo == actual_order)
                        pay = (round(ov * 100) // 10 * 10) * (stake // 100) if hit_this else 0
                        if is_plus:
                            stp_bets += stake
                            if hit_this:
                                stp_returns += pay
                                stp_hits += 1
                        else:
                            st_bets += stake
                            if hit_this:
                                st_returns += pay
                                st_hits += 1
                        if hit_this:
                            total_returns += pay
                            total_hits += 1

        if race_bet:
            total_n_bet_races += 1

    roi     = round(total_returns / total_bets, 3) if total_bets > 0 else None
    r_roi   = round(r_returns   / r_bets,   3) if r_bets   > 0 else None
    st_roi  = round(st_returns  / st_bets,  3) if st_bets  > 0 else None
    stp_roi = round(stp_returns / stp_bets, 3) if stp_bets > 0 else None

    result = {
        "n_picks":      total_n_bet_races,
        "n_hits":       total_hits,
        "total_bet":    total_bets,
        "total_payout": total_returns,
        "roi":          roi,
        "by_rank": {
            "R":   {"n_picks": r_races,   "n_hits": r_hits,   "total_bet": r_bets,
                    "total_payout": r_returns,   "roi": r_roi},
            "ST":  {"n_picks": st_races,  "n_hits": st_hits,  "total_bet": st_bets,
                    "total_payout": st_returns,  "roi": st_roi},
            "STP": {"n_picks": stp_races, "n_hits": stp_hits, "total_bet": stp_bets,
                    "total_payout": stp_returns, "roi": stp_roi},
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
    print(f"    SS(R):  {r_races:,}R  投資={r_bets:,}  回収={r_returns:,}  ROI={_fmt_roi(r_roi)}", flush=True)
    print(f"    S(ST):  {st_races:,}R  投資={st_bets:,}  回収={st_returns:,}  ROI={_fmt_roi(st_roi)}", flush=True)
    print(f"    S+(STP):{stp_races:,}R  投資={stp_bets:,}  回収={stp_returns:,}  ROI={_fmt_roi(stp_roi)}", flush=True)
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
            # evaluated_at を明示更新（PG の ON CONFLICT DO UPDATE は列リストに
            # ある列しか SET しないため、省略すると既存行の評価日時が残る）
            conn.execute(
                "INSERT OR REPLACE INTO model_evaluation "
                "(model_name, period_from, period_to, period_type, "
                " n_picks, n_hits, total_bet, total_payout, roi, evaluated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                row,
            )
    print(
        f"  → DB保存完了 ({period_type}: {period_from}〜{period_to}, "
        f"{len(rows)}行)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="7車限定R/ST/STPランクバックテスト結果をDBに保存")
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
