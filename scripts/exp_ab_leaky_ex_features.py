#!/usr/bin/env python3
"""B-8: ex_spurt_pct / ex_thrust_pct の train/serve skew 影響 A/B 測定（読み取り専用実験）。

背景（PMタスク B-8）:
  `src/preprocessing/feature_wt.py` の `ex_spurt_pct`（捲り実行率）・
  `ex_thrust_pct`（差し実行率）は開催期間中に値が更新される実装になっており
  （`_get_collected_keys` が finish_order>=1 の行のみ収集済み扱いにするため、
   未確定レースは日中に何度も再収集され、そのたびに `INSERT OR REPLACE` で
   `wt_entries` の行全体が上書きされる）、学習データ（結果確定後に収集）には
   「そのレース自身の結果を反映した値」が混入し、ライブ推論時（結果確定前）
   とは分布が異なる train/serve skew を抱えている（sb_dyn バグと同型）。

  本スクリプトはこれが「見かけの性能を水増ししていたリークだったのか、
  除外しても実害がない程度の寄与だったのか」をデータで判定するための
  A/B 測定を行う。

設計（月次凍結vintageモデル体系に準拠・honest walk-forward）:
  arm A（現行48特徴）: 既存の凍結vintageモデル `lgbm_wt_eval_mYYMM` /
    `lgbm_wt_win_mYYMM`（`src/wt_vintage_config.py` 契約通り、月Mのレースは
    必ず前月末までのデータで学習したモデルでスコアする）をそのまま読み込む。
    **絶対に再学習・上書きしない**。
  arm B（ex_spurt_pct/ex_thrust_pct を除いた46特徴）: 同じ月次ウィンドウ
    （`src.wt_vintage_config.monthly_windows()`）・同じ学習データ範囲
    （BASE_FROM=2022-12-01〜月M前月末）で `train_lgbm(..., feature_cols=FEATURES_B)`
    を呼び新規に学習する。保存名は vintage命名規則（`_(q\\d{4}|w\\d+|m\\d{4})$`）に
    一致しない `ab46_eval_YYYYMM` / `ab46_win_YYYYMM` 形式にし、
    `vintage_manifest.json` 保護に一切触れないようにする。

  `src/preprocessing/feature_wt.py` は編集しない。特徴量の絞り込みは
  `train_lgbm(feature_cols=...)` 引数と、本スクリプト内のローカル
  `_prepare_X(df, cols)`（`prepare_X()` と同じ reindex+fillna(0) ロジックを
  任意の列リストに対して行う版）のみで実現する。

評価:
  1. AUC（3着内モデル=eval・1着モデル=win）を月ごとのホールドアウトで比較
  2. S7 / 7A ランクの honest ROI（本番選定ロジック
     `src.strategy_wt.s7_daily_select`/`s7_evening_reselect`/`s7a_daily_select`
     をそのまま使用。買い目・void処理は
     `scripts/backfill_s7_rank_wt.py` / `backfill_s7a_rank_wt.py`
     （閲覧のみ・編集禁止ファイル）と同一ロジックをこのファイル内に再実装
     — 対象ファイルの実行を避けるため import はしていない）
  3. arm A モデルの `ex_spurt_pct` / `ex_thrust_pct` の feature importance
     （LightGBM booster の gain・split 両方）

対象期間: 既定で直近6ヶ月（monthly_windows() の末尾6件）。
DB へは読み取り専用（SELECT のみ）。picks_history・data/models/ の既存ファイル
への書き込み・削除は一切行わない（本スクリプトが新規に保存する ab46_* モデルの
削除はこのスクリプト自身ではなく別途手動で行う）。

使い方:
    cd /Users/ysuzuki/GitHub/keirin
    .venv/bin/python scripts/exp_ab_leaky_ex_features.py --months 6
    .venv/bin/python scripts/exp_ab_leaky_ex_features.py --months 6 --skip-train  # 学習済みab46_*を再利用
    .venv/bin/python scripts/exp_ab_leaky_ex_features.py --cleanup  # ab46_*一時モデルを削除するのみ
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.models.trainer import MODEL_DIR, load_model, save_model, train_lgbm
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT,
    TARGET_COL_WT,
    WIN_TARGET_COL_WT,
)
from src.strategy_wt import (
    S7_STAKE,
    S7A_STAKE,
    s7_evening_reselect,
    s7_field_entropy,
    s7_select_axis,
    s7_wt_mark3_overlap_n,
    s7_wt_overlap_n,
    s7a_daily_select,
)
from src.wt_vintage_config import BASE_FROM, monthly_windows

CACHE_PATH = Path(
    "/private/tmp/claude-501/-Users-ysuzuki-GitHub-kiseki/"
    "01e642d0-9beb-4947-bb17-276473fd6b35/scratchpad/wt_features_full.pkl"
)

LEAKY_COLS = ["ex_spurt_pct", "ex_thrust_pct"]
FEATURES_A = list(FEATURE_COLS_WT)                                  # 48（現行）
FEATURES_B = [c for c in FEATURE_COLS_WT if c not in LEAKY_COLS]     # 46（除外案）
assert len(FEATURES_A) == 48 and len(FEATURES_B) == 46

# 一時モデル命名: vintage正規表現 _(q\d{4}|w\d+|m\d{4})$ に絶対一致しないよう
# 6桁YYYYMM (先頭にm/w/qを付けない) を使う。
_VINTAGE_RE = re.compile(r"_(q\d{4}|w\d+|m\d{4})$")


def ab46_names(tag6: str) -> tuple[str, str]:
    eval_name = f"ab46_eval_{tag6}"
    win_name = f"ab46_win_{tag6}"
    assert not _VINTAGE_RE.search(eval_name), eval_name
    assert not _VINTAGE_RE.search(win_name), win_name
    return eval_name, win_name


def _prepare_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """prepare_X() と同一ロジック（reindex+fillna(0)）を任意の列リストに適用する。
    feature_wt.py の prepare_X() は FEATURE_COLS_WT 固定なので、46列版はこちらを使う。
    """
    return df.reindex(columns=cols).fillna(0)


# ---------------------------------------------------------------------------
# データロード（scratchpadキャッシュ。build_cache.py で事前生成済み）
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"キャッシュが見つかりません: {CACHE_PATH}\n"
            "先に build_cache.py（scratchpad）で全期間の特徴量を構築してください。"
        )
    print(f"[load] {CACHE_PATH} を読み込み中 ...", flush=True)
    df = pd.read_pickle(CACHE_PATH)
    print(f"  rows={len(df):,}  race_date range: {df['race_date'].min()}..{df['race_date'].max()}",
          flush=True)
    return df


# ---------------------------------------------------------------------------
# 1. 月次walk-forward学習（arm B のみ新規学習・arm A は既存凍結vintageを読むだけ）
# ---------------------------------------------------------------------------

def train_arm_b_for_month(df_train: pd.DataFrame, test_from: str, test_to: str,
                           tag6: str, force: bool) -> tuple[str, str]:
    """指定月について46特徴のeval/winモデルを学習・保存する。既存があればスキップ。"""
    eval_name, win_name = ab46_names(tag6)
    eval_path = MODEL_DIR / f"{eval_name}.pkl"
    win_path = MODEL_DIR / f"{win_name}.pkl"

    df_tr = df_train[df_train["race_date"] < test_from]

    if eval_path.exists() and not force:
        print(f"  [skip] {eval_name} 既存", flush=True)
    else:
        print(f"  [train] {eval_name}  train_rows={len(df_tr):,} (race_date<{test_from})", flush=True)
        t0 = time.time()
        model = train_lgbm(df_tr, feature_cols=FEATURES_B, target_col=TARGET_COL_WT)
        save_model(model, eval_name, force=force)
        print(f"    -> {time.time()-t0:.1f}s", flush=True)

    if win_path.exists() and not force:
        print(f"  [skip] {win_name} 既存", flush=True)
    else:
        print(f"  [train] {win_name}  train_rows={len(df_tr):,} (race_date<{test_from})", flush=True)
        t0 = time.time()
        model = train_lgbm(df_tr, feature_cols=FEATURES_B, target_col=WIN_TARGET_COL_WT)
        save_model(model, win_name, force=force)
        print(f"    -> {time.time()-t0:.1f}s", flush=True)

    return eval_name, win_name


# ---------------------------------------------------------------------------
# 2. AUC比較
# ---------------------------------------------------------------------------

def month_auc(df_train: pd.DataFrame, test_from: str, test_to: str,
              model, feature_cols: list[str], target_col: str) -> tuple[float, int]:
    df_te = df_train[(df_train["race_date"] >= test_from) & (df_train["race_date"] <= test_to)]
    if df_te.empty:
        return float("nan"), 0
    X_te = _prepare_X(df_te, feature_cols)
    y_te = df_te[target_col].values
    if len(np.unique(y_te)) < 2:
        return float("nan"), len(df_te)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return auc, len(df_te)


# ---------------------------------------------------------------------------
# 3. S7/7A honest ROI（backfill_s7_rank_wt.py / backfill_s7a_rank_wt.py と同一ロジックを
#    このファイル内に再実装。対象ファイルは編集禁止のため import せず複製する）
# ---------------------------------------------------------------------------

def _load_trio_boards(race_keys: list[str]) -> dict:
    trio: dict[str, dict] = defaultdict(dict)
    if not race_keys:
        return trio
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def _load_trio_payouts(race_keys: list[str]) -> dict[str, dict[frozenset, int]]:
    """trio(三連複)払戻のみを {race_key: {frozenset(3 frame_no): payout}} で返す。
    _load_payouts_wt(src/evaluation/backtest_wt.py) と同一ロジック（trioのみに限定した縮小版）。
    """
    pay: dict[str, dict[frozenset, int]] = defaultdict(dict)
    if not race_keys:
        return pay
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is None:
                    continue
                parts = [p for p in re.split(r"[-=→]", str(comb)) if p != ""]
                try:
                    nums = [int(p) for p in parts]
                except ValueError:
                    continue
                if len(nums) != 3:
                    continue
                key = frozenset(nums)
                payout = round(float(od) * 100) // 10 * 10
                pay[rk][key] = payout
    return pay


def build_s7_7a_candidates(df_scored: pd.DataFrame, date_from: str, date_to: str) -> list[dict]:
    """7車ちょうどのレースについてS7/7A共通の候補リストを構築する
    （backfill_s7_rank_wt.build_rows / backfill_s7a_rank_wt.build_rows の候補構築部と同一ロジック）。

    df_scored: race_key, frame_no, pred_prob(top3), pred_win 列を持つDataFrame
               （対象期間の全レース・全出走馬。7車以外も含んでよい）。
    """
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins: dict[str, list[tuple[int, int]]] = {}
        marks: dict[str, dict[int, int]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)

    df7 = df_scored[df_scored["race_key"].isin(set(rks7))].copy()
    if df7.empty:
        return []
    trio_bd = _load_trio_boards(df7["race_key"].unique().tolist())

    candidates: list[dict] = []
    for rk, g in df7.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board: set[int] = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s7_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = s7_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue

        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)

        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3_overlap_n = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others, "trio": trio, "actual_top3": actual_top3,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n,
        })
    return candidates


def score_s7(candidates: list[dict], pay_trio: dict[str, dict[frozenset, int]]) -> list[dict]:
    """S7選出(s7_evening_reselect, day単位・lockedなし)→採点行を返す。"""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for c_ in candidates:
        by_day[c_["race_date"]].append(c_)

    rows: list[dict] = []
    for _d, day_cands in by_day.items():
        for c_ in s7_evening_reselect(day_cands, [], set()):
            rows.append(_score_row(c_, pay_trio, S7_STAKE))
    return rows


def score_7a(candidates: list[dict], pay_trio: dict[str, dict[frozenset, int]]) -> list[dict]:
    rows: list[dict] = []
    for c_ in s7a_daily_select(candidates):
        rows.append(_score_row(c_, pay_trio, S7A_STAKE))
    return rows


def _score_row(c_: dict, pay_trio: dict, stake: int) -> dict | None:
    axis1, axis2 = c_["axis1"], c_["axis2"]
    trio = c_["trio"]
    combos = []
    for x in c_["others"]:
        key = frozenset({axis1, axis2, x})
        if key in trio:
            combos.append(key)
    if not combos:
        return None
    rk = c_["race_key"]
    hit = c_["actual_top3"] in combos
    trio_pay = pay_trio.get(rk, {}).get(c_["actual_top3"], 0)
    pay = trio_pay * stake // 100 if hit else 0
    bet = len(combos) * stake
    return {"race_date": c_["race_date"], "race_key": rk, "hit": int(hit),
            "payout": pay, "bet_amount": bet}


def bootstrap_roi_ci(rows: list[dict], n_boot: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    """race単位ブートストラップでROIの95%CIを返す (roi, ci_lo, ci_hi)。"""
    rows = [r for r in rows if r]
    if not rows:
        return float("nan"), float("nan"), float("nan")
    bets = np.array([r["bet_amount"] for r in rows], dtype=float)
    pays = np.array([r["payout"] for r in rows], dtype=float)
    roi = pays.sum() / bets.sum() if bets.sum() else float("nan")
    rng = np.random.default_rng(seed)
    n = len(rows)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b = bets[idx].sum()
        boot[i] = pays[idx].sum() / b if b else np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return roi, float(lo), float(hi)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6, help="評価する直近月数")
    ap.add_argument("--skip-train", action="store_true", help="学習済みab46_*があれば再利用のみ")
    ap.add_argument("--force-retrain", action="store_true", help="ab46_*を強制再学習")
    ap.add_argument("--cleanup", action="store_true", help="ab46_*一時モデルを削除して終了")
    args = ap.parse_args()

    if args.cleanup:
        n = 0
        for p in MODEL_DIR.glob("ab46_*"):
            p.unlink()
            n += 1
        print(f"[cleanup] ab46_* を{n}件削除しました")
        return

    windows = monthly_windows()[-args.months:]
    print(f"対象月: {[w[0][:7] for w in windows]}")

    df = load_df()
    df_train = df[df["finish_order"].notna()].copy()
    print(f"df_train rows={len(df_train):,} (finish_order確定)")

    # ---- 1. 月次学習（arm B） + AUC集計 ----
    auc_rows = []
    b_models: dict[str, tuple] = {}
    a_models: dict[str, tuple] = {}
    for test_from, test_to, a_eval_name, a_win_name in windows:
        tag6 = test_from[:7].replace("-", "")  # e.g. 202602
        print(f"\n=== {test_from}〜{test_to} ===", flush=True)

        if not args.skip_train:
            b_eval_name, b_win_name = train_arm_b_for_month(
                df_train, test_from, test_to, tag6, force=args.force_retrain)
        else:
            b_eval_name, b_win_name = ab46_names(tag6)

        a_eval = load_model(a_eval_name)
        a_win = load_model(a_win_name)
        b_eval = load_model(b_eval_name)
        b_win = load_model(b_win_name)
        a_models[test_from] = (a_eval, a_win)
        b_models[test_from] = (b_eval, b_win)

        for arm, (m_eval, m_win), fcols in (
            ("A(48)", (a_eval, a_win), FEATURES_A),
            ("B(46)", (b_eval, b_win), FEATURES_B),
        ):
            auc_eval, n_eval = month_auc(df_train, test_from, test_to, m_eval, fcols, TARGET_COL_WT)
            auc_win, n_win = month_auc(df_train, test_from, test_to, m_win, fcols, WIN_TARGET_COL_WT)
            print(f"  {arm}: eval(top3) AUC={auc_eval:.4f} (n={n_eval:,})  "
                  f"win AUC={auc_win:.4f} (n={n_win:,})", flush=True)
            auc_rows.append({"month": test_from[:7], "arm": arm,
                              "auc_eval": auc_eval, "n_eval": n_eval,
                              "auc_win": auc_win, "n_win": n_win})

    auc_df = pd.DataFrame(auc_rows)
    print("\n" + "=" * 70)
    print("AUC まとめ（月別）")
    print(auc_df.to_string(index=False))

    def weighted_mean(sub, col, wcol):
        w = sub[wcol].values
        v = sub[col].values
        mask = ~np.isnan(v)
        if not mask.any() or w[mask].sum() == 0:
            return float("nan")
        return float(np.average(v[mask], weights=w[mask]))

    print("\n加重平均AUC（サンプル数重み）:")
    for arm in ("A(48)", "B(46)"):
        sub = auc_df[auc_df["arm"] == arm]
        print(f"  {arm}: eval(top3)={weighted_mean(sub, 'auc_eval', 'n_eval'):.4f}  "
              f"win={weighted_mean(sub, 'auc_win', 'n_win'):.4f}")

    # ---- 2. S7/7A honest ROI ----
    print("\n" + "=" * 70)
    print("S7 / 7A honest ROI（月次vintageモデルでスコア・本番選定ロジック同一）")
    all_s7 = {"A(48)": [], "B(46)": []}
    all_7a = {"A(48)": [], "B(46)": []}

    for test_from, test_to, _a_eval_name, _a_win_name in windows:
        a_eval, a_win = a_models[test_from]
        b_eval, b_win = b_models[test_from]
        df_te = df_train[(df_train["race_date"] >= test_from) & (df_train["race_date"] <= test_to)].copy()
        if df_te.empty:
            continue

        for arm, (m_eval, m_win), fcols, bucket_s7, bucket_7a in (
            ("A(48)", (a_eval, a_win), FEATURES_A, all_s7, all_7a),
            ("B(46)", (b_eval, b_win), FEATURES_B, all_s7, all_7a),
        ):
            X = _prepare_X(df_te, fcols)
            df_te["pred_prob"] = m_eval.predict_proba(X)[:, 1]
            df_te["pred_win"] = m_win.predict_proba(X)[:, 1]
            cands = build_s7_7a_candidates(df_te, test_from, test_to)
            pay_trio = _load_trio_payouts([c["race_key"] for c in cands])
            s7_rows = score_s7(cands, pay_trio)
            a7_rows = score_7a(cands, pay_trio)
            bucket_s7[arm].extend(r for r in s7_rows if r)
            bucket_7a[arm].extend(r for r in a7_rows if r)
            n7 = sum(1 for r in s7_rows if r)
            na = sum(1 for r in a7_rows if r)
            print(f"  {test_from[:7]} {arm}: S7 n={n7}  7A n={na}", flush=True)

    print("\n" + "=" * 70)
    print("集計（対象全期間合算）")
    for label, bucket in (("S7", all_s7), ("7A", all_7a)):
        print(f"\n◆ {label}")
        for arm in ("A(48)", "B(46)"):
            rows = bucket[arm]
            n = len(rows)
            hits = sum(r["hit"] for r in rows)
            bet = sum(r["bet_amount"] for r in rows)
            pay = sum(r["payout"] for r in rows)
            roi, lo, hi = bootstrap_roi_ci(rows)
            hit_rate = hits / n * 100 if n else float("nan")
            print(f"  {arm}: n={n}  的中={hits} ({hit_rate:.1f}%)  "
                  f"投資={bet:,}  回収={pay:,}  ROI={roi*100:.1f}%  "
                  f"[95%CI {lo*100:.1f}%, {hi*100:.1f}%]")

    print("\n完了。一時モデル(ab46_*)を削除する場合は --cleanup を実行してください。")


if __name__ == "__main__":
    main()
