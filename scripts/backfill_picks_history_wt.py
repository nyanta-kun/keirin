#!/usr/bin/env python3
"""7+車 picks_history バックフィルスクリプト

VAL/HOLD 期間の 7+車 SS/S/A picks を lgbm_wt 系モデルで遡及再現し、
実結果(wt_entries)・オッズ(wt_odds)で採点して picks_history に書き込む。

kiseki フロントエンドの「当月」で過去月の ROI サマリーを
現行 7+車戦略と同じ条件で確認できるようにするための一回限りの操作。

モデル割り当て（デフォルト）:
  VAL  期間 (2025-07-01〜2026-02-28): lgbm_wt_train_only
  HOLD 期間 (2026-03-01〜):            lgbm_wt
  ※ --eval-model を指定すると HOLD 期間のモデルを差し替え可能
    例: --eval-model lgbm_wt_june_eval で 2022-12-01〜2026-05-31 学習モデルを使用
    → 2026-06-01 以降が真の OOS 検証になる

実行例:
  python3 scripts/backfill_picks_history_wt.py --dry-run
  python3 scripts/backfill_picks_history_wt.py
  python3 scripts/backfill_picks_history_wt.py --from 2026-06-01 --eval-model lgbm_wt_june_eval --overwrite
  python3 scripts/backfill_picks_history_wt.py --from 2026-03-01 --to 2026-06-15
  python3 scripts/backfill_picks_history_wt.py --overwrite  # 既存 7PLUS picks も上書き

実行後:
  python3 scripts/migrate_sqlite_to_pg.py  # PostgreSQL へ反映
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model

# ── デフォルト期間 ─────────────────────────────────────────────────────────
DEFAULT_FROM = "2025-07-01"
DEFAULT_TO   = "2026-06-15"   # live pick 開始(2026-06-16)の前日

# ── VAL/HOLD の境界 ────────────────────────────────────────────────────────
_HOLD_START = "2026-03-01"   # これ以降は lgbm_wt を使用

# ── 7+車戦略パラメータ (wave-picks-wt と同一) ──────────────────────────────
MIN_RIDERS = 7
MIN_GAP12  = 0.07
S_GAP12    = 0.10
GAMI_MIN   = 5.0


# ── データ読み込みヘルパー ──────────────────────────────────────────────────

def _load_trio_odds(race_keys: list[str]) -> dict[str, dict[frozenset, float]]:
    """wt_odds から {race_key: {combo_frozenset: odds_value}} を返す。"""
    if not race_keys:
        return {}
    odds_map: dict[str, dict] = {}
    combo_re = re.compile(r"\d+")
    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i : i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                rk = str(row["race_key"])
                ov = row["odds_value"]
                if ov is None or float(ov) <= 0:
                    continue
                parts = combo_re.findall(str(row["combination"]))
                if len(parts) == 3:
                    fs = frozenset(int(p) for p in parts)
                    odds_map.setdefault(rk, {})[fs] = float(ov)
    return odds_map


def _load_n_entries(race_keys: list[str]) -> dict[str, int]:
    """wt_races から {race_key: n_entries} を返す。"""
    if not race_keys:
        return {}
    n_map: dict[str, int] = {}
    CHUNK = 900
    with get_connection() as conn:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i : i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, n_entries FROM wt_races WHERE race_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                n_map[str(row["race_key"])] = int(row["n_entries"] or 0)
    return n_map


def _ensure_columns(conn) -> None:
    """picks_history に route/miwokuri 列がなければ追加（後方互換）。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(picks_history)").fetchall()}
    if "route" not in cols:
        conn.execute("ALTER TABLE picks_history ADD COLUMN route TEXT DEFAULT 'ks'")
    if "miwokuri" not in cols:
        conn.execute("ALTER TABLE picks_history ADD COLUMN miwokuri INTEGER DEFAULT 0")


# ── バックフィル本体 ────────────────────────────────────────────────────────

def _race_date_str(race_key: str) -> str:
    """race_key (20260301_55_01) から YYYY-MM-DD を返す。"""
    d = race_key[:8]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def backfill_period(
    df: pd.DataFrame,
    model,
    period_from: str,
    period_to: str,
    dry_run: bool,
    overwrite: bool,
    model_label: str = "",
) -> tuple[int, dict]:
    """指定期間の 7+車 SS/S/A picks を picks_history に書き込む。

    Returns (n_written, monthly_stats_dict).
    """
    df_period = df[
        (df["race_date"] >= period_from) &
        (df["race_date"] <= period_to) &
        (df["finish_order"].notna())
    ].copy()

    if df_period.empty:
        print(f"  [{period_from}〜{period_to}] データなし", flush=True)
        return 0, {}

    # モデル予測（実走者のみ・欠車は pred_prob=0 でランク末尾へ）
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

    # 既存の 7PLUS picks を確認（overwrite でなければスキップ）
    existing_base_keys: set[str] = set()
    if not overwrite:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT race_key FROM picks_history "
                "WHERE race_date BETWEEN ? AND ? AND rank LIKE '7PLUS%'",
                (period_from, period_to),
            ).fetchall()
            existing_base_keys = {str(r["race_key"]).split("#")[0] for r in rows}

    history: list[tuple] = []
    monthly_stats: dict[str, dict] = {}

    for race_key, grp in df_7plus.groupby("race_key"):
        race_key = str(race_key)
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

        race_date = _race_date_str(race_key)
        month = race_date[:7]
        ms = monthly_stats.setdefault(month, {"races": 0, "bets": 0, "returns": 0, "hits": 0})

        def _add_pick(suffix: str, rank: str, combo_frames: list[int]) -> None:
            """picks_history 行を history リストに追加する。"""
            store_key = f"{race_key}{suffix}"
            if not overwrite and race_key in existing_base_keys:
                return
            pred = f"{pivot1}-{pivot2}-" + ",".join(map(str, combo_frames))
            n_combos = len(combo_frames)
            bet = n_combos * 100
            hit = False
            payout = 0
            for t in combo_frames:
                cs = frozenset({pivot1, pivot2, t})
                if cs == top3_set:
                    # 公式払戻金は10円単位に切り捨て
                    payout = round(combo_odds.get(t, 0) * 100) // 10 * 10
                    hit = True
                    break
            history.append((race_date, store_key, rank, pred, n_combos,
                             int(hit), payout, bet))
            ms["races"] += 1
            ms["bets"] += bet
            if hit:
                ms["returns"] += payout
                ms["hits"] += 1

        # SS: gami≥5倍の目が 1〜3点
        valid_ss = [t for t in thirds_all if combo_odds.get(t, 0) >= GAMI_MIN]
        if 1 <= len(valid_ss) <= 3:
            _add_pick("#7SS", "7PLUS_SS", valid_ss)

        # S / A: 全相手が gami≥5倍
        all_thirds_runner = [t for t in thirds_all if t in runners]
        if all_thirds_runner and all(combo_odds.get(t, 0) >= GAMI_MIN for t in all_thirds_runner):
            if gap12 >= S_GAP12:
                _add_pick("#7S", "7PLUS_S", all_thirds_runner)
            else:
                _add_pick("#7A", "7PLUS_A", all_thirds_runner)

    # DB 書き込み
    n_written = len(history)
    if not dry_run and history:
        with get_connection() as conn:
            _ensure_columns(conn)
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date, race_key, rank, pred_combo, n_combos, "
                " hit, payout, bet_amount, route, miwokuri) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'wt', False)",
                history,
            )
    label = f" [{model_label}]" if model_label else ""
    status = "(dry-run)" if dry_run else "書き込み"
    print(f"  [{period_from}〜{period_to}]{label} {n_written}件 {status}", flush=True)

    # 月別サマリー表示
    for month in sorted(monthly_stats):
        ms = monthly_stats[month]
        roi = ms["returns"] / ms["bets"] * 100 if ms["bets"] > 0 else 0
        print(f"    {month}: {ms['races']:4d}R 的中{ms['hits']:3d} "
              f"投{ms['bets']:>8,}→回{ms['returns']:>10,}  ROI {roi:.1f}%", flush=True)

    return n_written, monthly_stats


# ── メイン ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="7+車 picks_history バックフィル")
    parser.add_argument("--from", dest="date_from", default=DEFAULT_FROM,
                        help=f"開始日 (デフォルト: {DEFAULT_FROM})")
    parser.add_argument("--to",   dest="date_to",   default=DEFAULT_TO,
                        help=f"終了日 (デフォルト: {DEFAULT_TO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="DB書き込みなし（数値確認のみ）")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存の 7PLUS picks も上書きする")
    parser.add_argument("--eval-model", dest="eval_model", default=None,
                        help="HOLD期間に使うモデル名を上書き（例: lgbm_wt_june_eval）。"
                             "汚染なしOOS検証用。省略時は lgbm_wt を使用")
    args = parser.parse_args()

    date_from = args.date_from
    date_to   = args.date_to

    # モデル読み込み
    print("モデル読み込み ...", flush=True)
    try:
        val_model = load_model("lgbm_wt_train_only")
        print("  lgbm_wt_train_only: OK", flush=True)
    except FileNotFoundError:
        print("  lgbm_wt_train_only: 見つからず → lgbm_wt で代替（VAL数値が若干楽観的）", flush=True)
        val_model = None

    hold_model_name = args.eval_model if args.eval_model else "lgbm_wt"
    try:
        hold_model = load_model(hold_model_name)
        print(f"  {hold_model_name}: OK", flush=True)
    except FileNotFoundError:
        print(f"ERROR: {hold_model_name} が見つかりません。train-wt を実行してください。")
        sys.exit(1)

    if val_model is None:
        val_model = hold_model

    if args.eval_model:
        print(f"  [INFO] HOLD期間モデルを {args.eval_model} に差し替え（OOS検証モード）", flush=True)

    # データ読み込み
    print(f"\nデータ読み込み: {date_from}〜{date_to} ...", flush=True)
    df_raw = load_raw_data_wt(min_date=date_from, max_date=date_to)
    if df_raw.empty:
        print("ERROR: データがありません。collect-wt を先に実行してください。")
        sys.exit(1)
    df = build_features_wt(df_raw)
    print(f"  {len(df):,} エントリー読み込み完了", flush=True)

    total_written = 0

    # VAL 期間分
    val_from_eff = date_from
    val_to_eff   = min(date_to, "2026-02-28")
    if val_from_eff <= val_to_eff:
        print(f"\n--- VAL 期間 [{val_from_eff}〜{val_to_eff}] (lgbm_wt_train_only) ---",
              flush=True)
        n, _ = backfill_period(df, val_model, val_from_eff, val_to_eff,
                                args.dry_run, args.overwrite,
                                model_label="lgbm_wt_train_only")
        total_written += n

    # HOLD 期間分
    hold_from_eff = max(date_from, _HOLD_START)
    hold_to_eff   = date_to
    if hold_from_eff <= hold_to_eff:
        print(f"\n--- HOLD 期間 [{hold_from_eff}〜{hold_to_eff}] ({hold_model_name}) ---",
              flush=True)
        n, _ = backfill_period(df, hold_model, hold_from_eff, hold_to_eff,
                                args.dry_run, args.overwrite,
                                model_label=hold_model_name)
        total_written += n

    print(f"\n合計 {total_written:,} 件{'(dry-run)' if args.dry_run else '書き込み完了'}", flush=True)

    if not args.dry_run:
        print("\n次のステップ:")
        print("  python3 scripts/migrate_sqlite_to_pg.py")
        print("  → PostgreSQL の keirin.picks_history に反映されます。")
    print("", flush=True)


if __name__ == "__main__":
    main()
