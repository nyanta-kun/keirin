#!/usr/bin/env python3
"""バックテスト結果を PostgreSQL keirin.model_evaluation に保存する。

7車ちょうど限定の本番戦略（2026-07-10〜 新ランク体系・notify_prerace_wt.py と同条件）で
VAL / HOLD を評価し、kiseki フロントエンドの「モデル精度」表示用に保存する。

ランク体系（notify_prerace_wt.py の `_determine_live_rank` と同一）:
  R  （表示 SS・三連複・レース単位）: min(全目オッズ)≥GAMI_THRESHOLD ∧ gap12≥SEVEN_PLUS_S_GAP12
       ∧ gap23≥GAP23_MIN → 全目購入 100円/点。的中条件=軸2車(pivot1/pivot2)が3着内。

※ ST/STP（表示 S/S+・三連単1着固定F）は優位性なしのため 2026-07-15 に全廃。

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
from src.strategy_wt import line_score_features, ss_policy

# ── バックテスト対象期間 ──────────────────────────────────────────────
# HOLD = 検証期間（kiseki Web「検証期間」サマリーの表示対象）。
# 2026-07-16〜: 検証期間は 2026-06-30 以前で固定（2026-07〜 は本番フォワード=
# 当日/当月/当年サマリー側で表示するため HOLD には含めない）。
VAL  = ("2025-07-01", "2026-02-28")
HOLD = ("2026-03-01", "2026-06-30")

# ペーパーランク（S2/S3）の検証期間集計対象（picks_history バックフィル済み範囲。
# lgbm_wt_eval の OOS 開始 2026-04-13 〜 検証期間末 2026-06-30）
# 2026-07-16: 旧S1（7PLUS_R・実賭け）全廃 → 全ランクがペーパー。
# 2026-07-17: S1(SIX_S1)/A(7PLUS_A) 全廃・S3(7PLUS_M) は新定義（不一致×gap12≥0.10）。
PAPER_HOLD = ("2026-04-13", "2026-06-30")
PAPER_RANKS = [("U", "7PLUS_U", "#7U"), ("M", "7PLUS_M", "#7M")]

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
    """7車ちょうど限定・現行Rランク本番戦略のバックテストを実行して集計結果を返す。

    判定条件は notify_prerace_wt.py の `_determine_live_rank` と同値
    （閾値定数は本ファイル冒頭に集約・値は notify_prerace_wt.py の同名定数と一致させること）。

    戦略:
      - n_entries == N_ENTRIES_TARGET(7) のレースのみ対象（8/9車は対象外）
      - R  (表示SS・三連複レース単位): min(全目オッズ)≥GAMI_THRESHOLD ∧ gap12≥SEVEN_PLUS_S_GAP12
        ∧ gap23≥GAP23_MIN → 全目購入 100円/点。的中=軸2車(pivot1/pivot2)が3着内。

    実精算方式（2026-07-15・欠車バイアス排除）:
      - 指数ランキング・買い目は **発走前のオッズ盤面掲載車**（=購入可能だった車）で作成する。
        落車・失格・棄権（発走前に不可知）の選手もランキング・買い目に含める。
      - 落車・失格絡みの買い目は購入扱いのまま外れ計上（返還しない＝実際の精算と同じ）。
      - 欠車（盤面から除外済み＝発走前に判明・実際も返還）はランキング・買い目に入らない。
      - 完走者だけでランキングを組み直す旧方式は、落車した指数上位が最初から居なかった
        ことになる未来情報リークで ROI を約4倍過大評価していた（keirin-survivor-bias-inflation）。
    """
    df_period = df[
        (df["race_date"] >= period_from) &
        (df["race_date"] <= period_to)
    ].copy()

    if df_period.empty:
        print(f"  [{period_from}〜{period_to}] データなし", flush=True)
        return {"n_picks": 0, "n_hits": 0, "total_bet": 0, "total_payout": 0, "roi": None}

    # モデル予測確率は全エントリーに付与（発走前情報のみ。着順での絞り込みはしない）
    X = prepare_X(df_period)
    df_period["pred_prob"] = model.predict_proba(X)[:, 1]

    all_race_keys = df_period["race_key"].unique().tolist()
    n_entries_map = _load_n_entries(all_race_keys)
    race_type_map = _load_race_types(all_race_keys)
    trio_odds_map = _load_trio_odds(all_race_keys)

    # 7車ちょうどのレースのみ（出走表基準・write_candidates_wt.py/main.py と同一基準）
    target_keys = {rk for rk in all_race_keys if n_entries_map.get(rk, 0) == N_ENTRIES_TARGET}
    df_7 = df_period[df_period["race_key"].isin(target_keys)].copy()

    total_n_bet_races = total_bets = total_returns = total_hits = 0
    r_bets   = r_returns   = r_hits   = r_races   = 0   # R（表示SS）

    for race_key, grp in df_7.groupby("race_key"):
        rk_trio_odds = trio_odds_map.get(race_key, {})
        if not rk_trio_odds:
            continue  # オッズなし（中止等）は対象外

        # 発走前のオッズ盤面掲載車 = 購入可能だった車（欠車は盤面から除外済み）。
        # 落車・失格・棄権の選手は盤面に残っているためランキング・買い目に含まれる。
        board: set[int] = set()
        for _combo in rk_trio_odds:
            board |= {int(x) for x in _combo}

        grp = grp[grp["frame_no"].astype(int).isin(board)]
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

        gap12 = probs[0] - probs[1]
        race_bet = False

        # ポリシーコンテキスト（2026-07-16〜: 選抜カットのみ・ライン特徴は互換引数）
        race_type = race_type_map.get(race_key)
        _line_pairs = [
            (None if pd.isna(_r.line_group) else int(_r.line_group),
             None if pd.isna(_r.race_point) else float(_r.race_point))
            for _r in grp.itertuples(index=False)
        ]
        avg_gap, n_lines, all_solo = line_score_features(_line_pairs)

        # ── Rランク（表示SS・三連複・レース単位ガミ） ──────────────────────
        # 実精算: 盤面掲載車の買い目は全て購入。落車・失格絡みは外れ計上（返還しない）。
        combo_odds: dict[int, float] = {}
        for t in thirds:
            ov = rk_trio_odds.get(frozenset({pivot1, pivot2, t}))
            if ov and ov > 0:
                combo_odds[t] = ov

        if combo_odds:
            gami_r = min(combo_odds.values())
            gap23 = (probs[1] - probs[2]) * 100.0 if len(probs) >= 3 else 0.0
            # ポリシー: 選抜のみ見送り（4分戦カット・格差増額は2026-07-16廃止）
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

        if race_bet:
            total_n_bet_races += 1

    roi     = round(total_returns / total_bets, 3) if total_bets > 0 else None
    r_roi   = round(r_returns   / r_bets,   3) if r_bets   > 0 else None

    result = {
        "n_picks":      total_n_bet_races,
        "n_hits":       total_hits,
        "total_bet":    total_bets,
        "total_payout": total_returns,
        "roi":          roi,
        "by_rank": {
            "R":   {"n_picks": r_races,   "n_hits": r_hits,   "total_bet": r_bets,
                    "total_payout": r_returns,   "roi": r_roi},
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
    return result


def paper_rank_stats() -> dict:
    """picks_history から S2/S3（ペーパー）の検証期間集計を返す。

    バックフィル済みの picks_history（#7U/#7M・実精算）を PAPER_HOLD 期間で
    集計する。候補行（bet_amount=0）・見送り行は含めない。
    """
    out: dict[str, dict] = {}
    pfrom, pto = PAPER_HOLD
    with get_connection() as conn:
        for rank_key, rank_val, suffix in PAPER_RANKS:
            # 集約列は必ずエイリアスを付ける（PG RealDict は無名集約列名が重複し
            # row[i] 位置アクセスが崩れるため）
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(hit),0) AS h, "
                "COALESCE(SUM(bet_amount),0) AS b, "
                "COALESCE(SUM(CASE WHEN hit=1 THEN payout ELSE 0 END),0) AS p "
                "FROM picks_history WHERE rank = ? AND route='wt' "
                "AND race_date BETWEEN ? AND ? AND bet_amount > 0 AND NOT miwokuri "
                "AND race_key LIKE ?",
                (rank_val, pfrom, pto, f"%{suffix}"),
            ).fetchone()
            n, h, b, p = (int(row["n"] or 0), int(row["h"] or 0),
                          int(row["b"] or 0), int(row["p"] or 0))
            if n == 0:
                continue
            out[rank_key] = {
                "n_picks": n, "n_hits": h, "total_bet": b, "total_payout": p,
                "roi": round(p / b, 4) if b else None,
            }
    return out


def save_to_db(
    model_name: str,
    period_type: str,
    period_from: str,
    period_to: str,
    result: dict,
) -> None:
    """model_evaluation テーブルに UPSERT する（全体 + ランク別）。

    MIRROR_PG_URL 環境変数が設定されている場合は VPS PG にも同内容をミラーする
    （Mac＝完全オッズデータで計算し PG＝Web 表示用に反映する運用。
    KEIRIN_DB_URL を使うと get_connection がデータ読みごと PG に切り替わり
    PG のオッズ保持期間（2026-06〜）しか評価できないため別変数にしている）。
    """
    rows = [
        (model_name, period_from, period_to, period_type,
         result["n_picks"], result["n_hits"], result["total_bet"],
         result["total_payout"], result["roi"]),
    ]
    for rank_key, rd in result.get("by_rank", {}).items():
        # suffix 規約: {model}#7{key}（例 lgbm_wt#7U / lgbm_wt#7M）
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

    mirror_url = os.environ.get("MIRROR_PG_URL")
    if mirror_url and not os.environ.get("KEIRIN_DB_URL"):
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(mirror_url) as pg:
                with pg.cursor() as cur:
                    for row in rows:
                        cur.execute(
                            "INSERT INTO keirin.model_evaluation "
                            "(model_name, period_from, period_to, period_type, "
                            " n_picks, n_hits, total_bet, total_payout, roi, evaluated_at) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW()) "
                            "ON CONFLICT (model_name, period_type) DO UPDATE SET "
                            "period_from=EXCLUDED.period_from, period_to=EXCLUDED.period_to, "
                            "n_picks=EXCLUDED.n_picks, n_hits=EXCLUDED.n_hits, "
                            "total_bet=EXCLUDED.total_bet, total_payout=EXCLUDED.total_payout, "
                            "roi=EXCLUDED.roi, evaluated_at=NOW()",
                            row,
                        )
            print(f"  → VPS PG ミラー完了 ({len(rows)}行)", flush=True)
        except Exception as e:
            print(f"  → VPS PG ミラー失敗（継続）: {e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ペーパーランク（S2/S3）集計を model_evaluation に保存")
    parser.add_argument("--dry-run", action="store_true", help="DB書き込みなし（数値確認のみ）")
    args = parser.parse_args()

    # 2026-07-16〜: 旧S1（7PLUS_R・実賭け）全廃に伴い、モデル読み込み・
    # run_7plus_backtest（VAL/HOLD の R 戦略バックテスト）は廃止した。
    # 集計元は picks_history（バックフィル済み実精算行）のみ。
    # KEIRIN_DB_URL 設定時は get_connection が PG 直結（Web 表示と同一ソース）。

    # 2026-07-16〜: 旧S1（7PLUS_R）全廃により R バックテスト結果は保存しない。
    # HOLD = ペーパーランク（S2/S3）の picks_history 集計に一本化する。
    # メイン行 = 2ランクのプール合算（VAL は旧S1専用だったため廃止・行も削除）。
    try:
        paper = paper_rank_stats()
    except Exception as e:
        print(f"  ペーパーランク集計失敗: {e}", flush=True)
        sys.exit(1)
    pooled = {
        "n_picks": sum(v["n_picks"] for v in paper.values()),
        "n_hits": sum(v["n_hits"] for v in paper.values()),
        "total_bet": sum(v["total_bet"] for v in paper.values()),
        "total_payout": sum(v["total_payout"] for v in paper.values()),
        "by_rank": paper,
    }
    pooled["roi"] = (round(pooled["total_payout"] / pooled["total_bet"], 4)
                     if pooled["total_bet"] else None)
    for k, v in paper.items():
        roi_disp = f"{v['roi']:.1%}" if v["roi"] is not None else "—"
        print(f"    {k}(paper): {v['n_picks']:,}R  的中={v['n_hits']:,}  "
              f"ROI={roi_disp}  [{PAPER_HOLD[0]}〜{PAPER_HOLD[1]}]", flush=True)

    if not args.dry_run:
        # 廃止ランク行の掃除（表示に古い体系が混ざらないように）:
        # 旧S1(#7R)・旧VAL、2026-07-17 全廃の S1(#6S1)/A(#7A)
        with get_connection() as conn:
            conn.execute("DELETE FROM model_evaluation WHERE model_name LIKE '%#7R'")
            conn.execute("DELETE FROM model_evaluation WHERE model_name LIKE '%#6S1'")
            conn.execute("DELETE FROM model_evaluation WHERE model_name LIKE '%#7A'")
            conn.execute("DELETE FROM model_evaluation WHERE period_type = 'VAL'")
        save_to_db("lgbm_wt", "HOLD", PAPER_HOLD[0], PAPER_HOLD[1], pooled)
    else:
        print("  (dry-run: DB書き込みスキップ)", flush=True)

    print("\n完了", flush=True)


if __name__ == "__main__":
    main()
