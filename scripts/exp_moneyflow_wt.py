"""money-flow 検証ハーネス — 朝→直前オッズ変動の「賢い金の流れ」検証。

目的 (G04: docs/goals/G04-moneyflow-harness.md):
  公開オッズを超える唯一の市場内シグナル候補として、
  朝オッズ→確定オッズのドリフト（money-flow）が有意な情報を持つか検証する。

データ現実の留意:
  wt_odds_snapshot の morning データは 2026-06-08〜 収集開始。
  現時点では統計的結論を出せる標本数に**全く達していない**。
  本スクリプトの目的は「データが溜まり次第すぐ回せる再実行可能なハーネス」と
  「初期の記述統計」であり、現時点の数字はあくまで暫定値として参照すること。

事前登録セル（これ以外の探索的数字は「参考・追試しない」）:
  a) 「モデル上位目のうちオッズ短縮した目」vs「伸長した目」の確定ROI差
  b) 現行C0戦略（3連複2軸3点・最安≥5倍） × 「推奨目が短縮」ゲートの有無
  c) fav_mismatch（モデル1位≠市場本命） × 「市場本命が朝→直前で交代」の有無

doc18 セマンティクス:
  - ランキングは全エントリー（出走表基準）
  - ≤6車は出走表 n_entries≤6 で判定（完走者基準不可）
  - モデルは評価期間外で学習（TRAIN 2023-07〜2025-06 限定学習・リーク無し）
  - 欠車void: 軸欠車=レース無効、相手欠車=その点除外
  - 払戻=wt_odds（最終オッズ=上限値）

使い方:
  python3 scripts/exp_moneyflow_wt.py [--from 2026-06-08] [--to 2026-06-12]
  python3 scripts/exp_moneyflow_wt.py --report  # docs/analysis/23-moneyflow-initial.md 生成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import json
from collections import defaultdict
from datetime import date, timedelta
import math
import numpy as np
import lightgbm as lgb

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)
from src.evaluation.backtest_wt import _assign_tier
from roi_robustness_wt import roi_summary

# ── 期間定義（doc18標準）────────────────────────────────────────────────
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-12")

LGB_PARAMS = dict(
    objective="binary", n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)

# 最小結論判定標本数（事前設定・80%検出力でROI差20%pp・1サイドt近似）
MIN_RACES_FOR_CONCLUSION = 300  # ≤6車かつsnapshotありの対象レース数


# ── DB ロード ──────────────────────────────────────────────────────────

def _load_snapshot_boards(race_keys: list[str], snap_type: str) -> dict[str, dict]:
    """snapshot type='morning'/'evening' のtrio盤面を race_key→{frozenset:odds} で返す。"""
    boards: dict[str, dict] = defaultdict(dict)
    CH = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CH):
            chunk = race_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds_snapshot "
                f"WHERE bet_type='trio' AND snapshot_type=? AND race_key IN ({ph})",
                [snap_type] + chunk,
            ).fetchall()
            for rk, comb, ov in rows:
                if ov is None or ov <= 0:
                    continue
                try:
                    fr = [int(x) for x in re.split(r"[-=]", str(comb))]
                except ValueError:
                    continue
                if len(fr) == 3:
                    boards[rk][frozenset(fr)] = float(ov)
    return dict(boards)


def _load_final_boards(race_keys: list[str]) -> dict[str, dict]:
    """wt_odds から確定trio盤面を race_key→{frozenset:odds} で返す。"""
    boards: dict[str, dict] = defaultdict(dict)
    CH = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CH):
            chunk = race_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ({ph})",
                chunk,
            ).fetchall()
            for rk, comb, ov in rows:
                if ov is None or ov <= 0:
                    continue
                try:
                    fr = [int(x) for x in re.split(r"[-=]", str(comb))]
                except ValueError:
                    continue
                if len(fr) == 3:
                    boards[rk][frozenset(fr)] = float(ov)
    return dict(boards)


def _load_n_entries(race_keys: list[str]) -> dict[str, int]:
    """wt_races.n_entries を race_key→int で返す。"""
    out: dict[str, int] = {}
    CH = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CH):
            chunk = race_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            for rk, ne in c.execute(
                f"SELECT race_key, n_entries FROM wt_races WHERE race_key IN ({ph})",
                chunk,
            ):
                out[rk] = ne
    return out


def _races_with_morning_snapshot(date_from: str | None, date_to: str | None) -> list[str]:
    """morning snapshot が存在するレースキーを返す（全ベット種対象）。"""
    with get_connection() as c:
        conds = ["snapshot_type='morning'"]
        params: list = []
        if date_from:
            conds.append("race_key >= ?")
            params.append(date_from.replace("-", "") + "_00_00")
        if date_to:
            conds.append("race_key <= ?")
            params.append(date_to.replace("-", "") + "_99_99")
        sql = ("SELECT DISTINCT race_key FROM wt_odds_snapshot WHERE "
               + " AND ".join(conds))
        return [r[0] for r in c.execute(sql, params)]


def _market_fav_from_board(trio_board: dict) -> int | None:
    """trio盤面から市場本命frameを逆算（exp_segment_first_wtと同一ロジック）。"""
    q: dict[int, float] = {}
    n_combo = 0
    for combo, ov in trio_board.items():
        if ov >= 9000:
            continue
        n_combo += 1
        for f in combo:
            q[f] = q.get(f, 0.0) + 1.0 / ov
    if n_combo < 4 or not q:
        return None
    return max(q.items(), key=lambda x: x[1])[0]


# ── データ収集 ─────────────────────────────────────────────────────────

def collect(date_from: str | None = None, date_to: str | None = None) -> dict:
    """morning snapshot のある全レース + ≤6車サブセット について
    記述統計・エッジ検定用構造体を構築して返す。

    戻り値:
      "desc_all"  : 全ベット種ドリフト記述統計 (all races with morning snapshot)
      "edge_le6"  : エッジ検定用レース構造体リスト (≤6車・結果確定・モデル予測済み)
    """
    # ── 1. Morning snapshot があるレースキー ──────────────────────────
    all_keys = _races_with_morning_snapshot(date_from, date_to)
    if not all_keys:
        print("[WARN] morning snapshot データなし。期間を確認してください。", flush=True)
        return {"desc_all": {}, "edge_le6": []}

    print(f"  morning snapshot 対象レース: {len(all_keys)}R", flush=True)

    # ── 2. 記述統計用: 全bet_type の朝→確定ドリフト ───────────────────
    desc = _build_desc_stats(all_keys)

    # ── 3. エッジ検定用: 特徴量ビルド + リーク無しモデル ───────────────
    print("  loading & building features (TRAIN〜HOLDOUT)...", flush=True)
    df_full = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    fit = df_full[
        (df_full["race_date"] <= TRAIN[1]) & (df_full["finish_order"] >= 1)
    ]
    print(f"  training leakfree LGBM on TRAIN only ({len(fit):,} rows)...", flush=True)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"])

    # snapshot 期間に絞る
    if date_from:
        df_snap = df_full[df_full["race_date"] >= date_from].copy()
    elif date_to:
        df_snap = df_full[df_full["race_date"] <= date_to].copy()
    else:
        df_snap = df_full.copy()

    df_snap["pred_prob"] = m.predict_proba(prepare_X(df_snap))[:, 1]

    # 出走表基準 ≤6車
    n_entries = _load_n_entries(all_keys)
    le6_keys = {rk for rk in all_keys if n_entries.get(rk, 99) <= 6}
    df_le6 = df_snap[df_snap["race_key"].isin(le6_keys)].copy()
    print(f"  ≤6車（出走表基準）: {len(le6_keys)}R", flush=True)

    # 結果確定レースのみ
    done = df_le6.groupby("race_key")["finish_order"].apply(
        lambda s: (s.between(1, 3)).sum() >= 3
    )
    df_le6 = df_le6[df_le6["race_key"].isin(done[done].index)]
    print(f"  うち結果確定: {df_le6['race_key'].nunique()}R", flush=True)

    # 盤面ロード
    snap_rks = df_le6["race_key"].unique().tolist()
    morning_boards = _load_snapshot_boards(snap_rks, "morning")
    final_boards = _load_final_boards(snap_rks)

    # レース構造体ビルド
    edge_races = _build_edge_races(df_le6, morning_boards, final_boards)
    print(f"  エッジ検定用レース構造体: {len(edge_races)}R", flush=True)

    return {"desc_all": desc, "edge_le6": edge_races}


def _build_desc_stats(all_keys: list[str]) -> dict:
    """bet_type別・オッズ帯別の朝→確定ドリフト記述統計 + hit/non-hit比較。"""
    # morning snapshot と確定オッズを全bet_typeで結合
    rows_by_type: dict[str, list] = defaultdict(list)

    CH = 900
    with get_connection() as c:
        # 結果確定情報を取得
        all_keys_set = set(all_keys)
        hit_combos: dict[str, frozenset] = {}  # race_key → 的中trio組み合わせ
        for i in range(0, len(all_keys), CH):
            chunk = all_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            fin_rows = c.execute(
                f"SELECT race_key, frame_no, finish_order FROM wt_entries "
                f"WHERE race_key IN ({ph}) AND finish_order BETWEEN 1 AND 3",
                chunk,
            ).fetchall()
            race_fin: dict[str, list] = defaultdict(list)
            for rk, fn, fo in fin_rows:
                race_fin[rk].append((fo, fn))
            for rk, fins in race_fin.items():
                if len(fins) >= 3:
                    hit_combos[rk] = frozenset(fn for _, fn in fins[:3])

        # morning snapshot × 確定オッズ JOIN
        for i in range(0, len(all_keys), CH):
            chunk = all_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            # morning
            m_data = c.execute(
                f"SELECT race_key, bet_type, combination, odds_value FROM wt_odds_snapshot "
                f"WHERE snapshot_type='morning' AND race_key IN ({ph})",
                chunk,
            ).fetchall()
            # final
            f_data = c.execute(
                f"SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                f"WHERE race_key IN ({ph})",
                chunk,
            ).fetchall()
            final_map: dict[tuple, float] = {}
            for rk, bt, comb, ov in f_data:
                if ov is not None:
                    final_map[(rk, bt, comb)] = float(ov)

            for rk, bt, comb, m_ov in m_data:
                if m_ov is None or m_ov <= 0:
                    continue
                f_ov = final_map.get((rk, bt, comb))
                if f_ov is None or f_ov <= 0:
                    continue
                # 的中判定（trio のみ）
                is_hit = False
                if bt == "trio" and rk in hit_combos:
                    try:
                        fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                        is_hit = (fr == hit_combos[rk])
                    except ValueError:
                        pass

                rows_by_type[bt].append({
                    "morning": float(m_ov),
                    "final": float(f_ov),
                    "drift": float(f_ov) / float(m_ov),
                    "shortened": float(f_ov) < float(m_ov),
                    "is_hit": is_hit,
                })

    # 集計
    result: dict[str, dict] = {}
    for bt, rows in rows_by_type.items():
        drifts = [r["drift"] for r in rows]
        shortened = [r["shortened"] for r in rows]
        # 帯別
        by_band: dict[str, list] = defaultdict(list)
        for r in rows:
            m = r["morning"]
            band = "<3" if m < 3 else "3-5" if m < 5 else "5-10" if m < 10 else "10-30" if m < 30 else ">=30"
            by_band[band].append(r["drift"])
        # ガミ帯反転率
        n_flip = sum(
            1 for r in rows
            if _gami_band(r["morning"]) != _gami_band(r["final"])
        )
        # hit vs non-hit（trio のみ）
        hit_rows = [r for r in rows if r["is_hit"]]
        nonhit_rows = [r for r in rows if not r["is_hit"]]

        result[bt] = {
            "n": len(rows),
            "drift_median": float(np.median(drifts)) if drifts else 0.0,
            "drift_q25": float(np.percentile(drifts, 25)) if drifts else 0.0,
            "drift_q75": float(np.percentile(drifts, 75)) if drifts else 0.0,
            "pct_shortened": float(np.mean(shortened)) if shortened else 0.0,
            "gami_flip_rate": n_flip / len(rows) if rows else 0.0,
            "by_band": {
                band: {
                    "n": len(vs),
                    "median_drift": float(np.median(vs)),
                    "pct_shortened": float(np.mean([v < 1.0 for v in vs])),
                }
                for band, vs in sorted(by_band.items())
            },
            "hit_vs_nonhit": _hit_vs_nonhit(hit_rows, nonhit_rows),
        }
    return result


def _gami_band(odds: float) -> str:
    if odds < 3.0:
        return "<3"
    if odds < 5.0:
        return "3-5"
    return ">=5"


def _hit_vs_nonhit(
    hit_rows: list[dict], nonhit_rows: list[dict]
) -> dict:
    """的中目 vs 非的中目のドリフト差（smart money 仮説検定）。"""
    if not hit_rows or not nonhit_rows:
        return {
            "n_hit": len(hit_rows), "n_nonhit": len(nonhit_rows),
            "hit_pct_shortened": None, "nonhit_pct_shortened": None,
            "diff_pct_shortened": None,
            "hit_drift_median": None, "nonhit_drift_median": None,
            "note": "標本不足",
        }
    hs = [r["shortened"] for r in hit_rows]
    ns_ = [r["shortened"] for r in nonhit_rows]
    hd = [r["drift"] for r in hit_rows]
    nd = [r["drift"] for r in nonhit_rows]
    return {
        "n_hit": len(hit_rows),
        "n_nonhit": len(nonhit_rows),
        "hit_pct_shortened": float(np.mean(hs)),
        "nonhit_pct_shortened": float(np.mean(ns_)),
        "diff_pct_shortened": float(np.mean(hs)) - float(np.mean(ns_)),
        "hit_drift_median": float(np.median(hd)),
        "nonhit_drift_median": float(np.median(nd)),
        "note": "trio 的中組み合わせと非的中の朝→確定短縮率の差（smart money仮説）",
    }


def _build_edge_races(df, morning_boards: dict, final_boards: dict) -> list[dict]:
    """エッジ検定用レース構造体を構築する（≤6車・結果確定・全エントリーでランク）。"""
    races = []
    for rk, g0 in df.groupby("race_key"):
        n = len(g0)
        if n < 4:
            continue
        # morning snapshot がないレースはスキップ
        mb = morning_boards.get(rk, {})
        fb = final_boards.get(rk, {})
        if not mb or not fb:
            continue
        # 結果
        fin = g0[g0["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        dns = set(g0[g0["finish_order"] == 0]["frame_no"].astype(int).tolist())

        # 全エントリーでランキング（doc18セマンティクス）
        g = g0.sort_values("pred_prob", ascending=False)
        probs = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]

        tier = _assign_tier(probs[0] - probs[1], probs[0] / (3.0 / n))

        # 市場本命の朝/確定から計算
        mf_morning = _market_fav_from_board(mb)
        mf_final = _market_fav_from_board(fb)
        fav_swapped = (mf_morning is not None and mf_final is not None
                       and mf_morning != mf_final)
        fav_mismatch = (mf_final is not None and mf_final != p1)

        # 欠車void（notify_results_wt._void_by_dns と同一規則）
        axis_void = (p1 in dns) or (p2 in dns)

        # C0 用: 3連複2軸3点・最安≥5倍 + ドリフトゲート
        trio3_final = []
        trio3_morning = []
        if not axis_void:
            for x in thirds:
                if x in dns:
                    continue
                c = frozenset((p1, p2, x))
                f_ov = fb.get(c)
                m_ov = mb.get(c)
                if f_ov and m_ov and f_ov > 0 and m_ov > 0:
                    trio3_final.append((f_ov, c == top3))
                    trio3_morning.append((m_ov, c == top3))

        min3_final = min((o for o, _ in trio3_final), default=None)

        races.append({
            "race_key": rk,
            "date": g0["race_date"].iloc[0],
            "n": n,
            "p1": p1, "p2": p2,
            "top3": top3, "dns": dns,
            "tier": tier,
            "t3s": probs[0] + probs[1] + probs[2],
            "mf_morning": mf_morning,
            "mf_final": mf_final,
            "fav_mismatch": fav_mismatch,
            "fav_swapped": fav_swapped,
            "axis_void": axis_void,
            "trio3_final": trio3_final,
            "trio3_morning": trio3_morning,
            "min3_final": min3_final,
        })
    return races


# ── エッジ検定 ─────────────────────────────────────────────────────────

def _is_shortened(m_odds: float, f_odds: float) -> bool:
    return f_odds < m_odds


def cell_a(races: list[dict]) -> dict:
    """事前登録セル a:
    「モデル上位目（trio3）のうち朝→確定で短縮した目」vs「伸長した目」の確定ROI差。
    """
    pays_short, bets_short = [], []
    pays_long, bets_long = [], []
    for r in races:
        if r["axis_void"]:
            continue
        if not r["trio3_final"] or not r["trio3_morning"]:
            continue
        sf = r["trio3_final"]
        sm = r["trio3_morning"]
        sh_legs = [(fo, hit) for (fo, hit), (mo, _) in zip(sf, sm)
                   if _is_shortened(mo, fo)]
        lg_legs = [(fo, hit) for (fo, hit), (mo, _) in zip(sf, sm)
                   if not _is_shortened(mo, fo)]
        if sh_legs:
            pays_short.append(sum(fo * 100 for fo, hit in sh_legs if hit))
            bets_short.append(len(sh_legs) * 100)
        if lg_legs:
            pays_long.append(sum(fo * 100 for fo, hit in lg_legs if hit))
            bets_long.append(len(lg_legs) * 100)
    s_s = roi_summary(pays_short, bets_short)
    s_l = roi_summary(pays_long, bets_long)
    return {
        "shortened": {"roi": s_s, "n_races": len(pays_short)},
        "elongated": {"roi": s_l, "n_races": len(pays_long)},
        "roi_diff": s_s["roi"] - s_l["roi"],
    }


def cell_b(races: list[dict]) -> dict:
    """事前登録セル b:
    現行C0戦略（3連複・最安≥5倍） × 「推奨目が短縮」ゲートの有無によるROI差。
    """
    pays_all, bets_all = [], []
    pays_gate, bets_gate = [], []
    for r in races:
        if r["axis_void"]:
            continue
        legs_f = r["trio3_final"]
        legs_m = r["trio3_morning"]
        if not legs_f or r["min3_final"] is None or r["min3_final"] < 5.0:
            continue
        if r["tier"] is None:
            continue

        # C0全体
        pays_all.append(sum(fo * 100 for fo, hit in legs_f if hit))
        bets_all.append(len(legs_f) * 100)

        # ゲート条件: 全推奨目の過半数（3点中2点以上）が短縮
        n_short = sum(
            1 for (fo, _), (mo, _) in zip(legs_f, legs_m)
            if _is_shortened(mo, fo)
        )
        if n_short >= math.ceil(len(legs_f) / 2):
            pays_gate.append(sum(fo * 100 for fo, hit in legs_f if hit))
            bets_gate.append(len(legs_f) * 100)

    s_all = roi_summary(pays_all, bets_all)
    s_gate = roi_summary(pays_gate, bets_gate)
    return {
        "c0_all": {"roi": s_all, "n_races": len(pays_all)},
        "c0_gate": {"roi": s_gate, "n_races": len(pays_gate)},
        "gate_pct": len(pays_gate) / len(pays_all) if pays_all else 0.0,
    }


def cell_c(races: list[dict]) -> dict:
    """事前登録セル c:
    fav_mismatch（モデル1位≠市場本命） × 「市場本命が朝→直前で交代」の有無によるROI差。
    """
    pays_swap, bets_swap = [], []
    pays_noswap, bets_noswap = [], []
    for r in races:
        if r["axis_void"]:
            continue
        legs_f = r["trio3_final"]
        if not legs_f or r["min3_final"] is None or r["min3_final"] < 5.0:
            continue
        if not r["fav_mismatch"]:
            continue
        if r["tier"] is None:
            continue

        pay = sum(fo * 100 for fo, hit in legs_f if hit)
        bet = len(legs_f) * 100
        if r["fav_swapped"]:
            pays_swap.append(pay)
            bets_swap.append(bet)
        else:
            pays_noswap.append(pay)
            bets_noswap.append(bet)

    s_swap = roi_summary(pays_swap, bets_swap)
    s_noswap = roi_summary(pays_noswap, bets_noswap)
    return {
        "fav_swap": {"roi": s_swap, "n_races": len(pays_swap)},
        "fav_noswap": {"roi": s_noswap, "n_races": len(pays_noswap)},
        "roi_diff": s_swap["roi"] - s_noswap["roi"] if pays_swap and pays_noswap else None,
    }


# ── 最小標本数見積もり ────────────────────────────────────────────────

def estimate_min_n(hit_rate: float = 0.30, effect_roi: float = 0.20,
                   alpha: float = 0.05, power: float = 0.80) -> int:
    """ROI差effect_roi ppを検出するための必要レース数（per-race 正規近似）。

    per-race 払戻の正規近似:
      E[pay]    = hit_rate * avg_odds * 100  (円)
      E[bet]    = 3 * 100 = 300              (3点買い想定)
      ROI       = E[pay] / E[bet]
      Var[pay]  = hit_rate*(1-hit_rate) * (avg_odds*100)^2  (ベルヌーイ近似)
      Var[ROI]  = Var[pay] / E[bet]^2
      std[ROI]  = sqrt(Var[ROI])

    effect_roi (0.20 = 20pp ROI差) を1-sided 5% / 80% power で検出する n。
    典型値: ≤6車trio 的中率30% / 平均オッズ15倍 / 3点買い → std_roi ≈ 2.5 → n ≈ 300R
    """
    z_alpha = 1.645  # 1-sided 5%
    z_beta = 0.842   # 80% power
    avg_odds = 15.0   # 典型的な≤6車trio配当 (上限値)
    bet_per_race = 300.0  # 3点×100円
    var_pay = hit_rate * (1 - hit_rate) * (avg_odds * 100) ** 2
    std_roi = math.sqrt(var_pay) / bet_per_race
    # 2標本差の検出（比較セル用）
    n = 2 * ((z_alpha + z_beta) * std_roi / effect_roi) ** 2
    return max(int(math.ceil(n)), 100)


# ── レポート生成 ──────────────────────────────────────────────────────

def _fmt_roi(s: dict, n: int) -> str:
    if n == 0:
        return "0R  --"
    return (f"{n}R ROI{s['roi']:.0%} "
            f"[{s['ci_lo']:.0%},{s['ci_hi']:.0%}] 除最大{s['roi_ex_max']:.0%}")


def print_report(data: dict, date_from: str | None, date_to: str | None) -> str:
    """標準出力へ結果を表示し、Markdown レポート文字列を返す。"""
    desc = data["desc_all"]
    races = data["edge_le6"]
    n_le6 = len(races)
    period = f"{date_from or 'all'} 〜 {date_to or 'latest'}"
    today = date.today().isoformat()
    min_n = estimate_min_n()

    lines = []

    # ─ ヘッダー ───────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  money-flow 初期観察  対象期間: {period}  ≤6車レース: {n_le6}R")
    print(f"  ⚠ 現時点の数字はあくまで暫定。結論を出せる最小標本数 ≈ {min_n}R（≤6車）")
    print(f"{'='*100}")

    lines.append(f"# G04 money-flow 初期観察レポート（{today}更新）\n")
    lines.append(f"> 生成: `scripts/exp_moneyflow_wt.py --from {date_from} --to {date_to}`  \n")
    lines.append(f"> **データ期間**: {period}  ")
    lines.append(f"> **≤6車・結果確定レース（エッジ検定対象）**: {n_le6}R  ")
    lines.append(f"> **結論を出せる最小標本数**: ≈{min_n}R  ")
    lines.append(f">  \n> ⚠ 現時点は統計的結論を出せる標本数に達していない。")
    lines.append(f"> 本レポートは「データが溜まり次第すぐ回せるハーネス」と「初期記述統計」の確認目的。\n")

    # ─ 記述統計 ─────────────────────────────────────────────────────
    print("\n  ◆ 1. 記述統計: bet_type別 朝→確定ドリフト\n")
    lines.append("## 1. 記述統計: bet_type別 朝→確定ドリフト\n")
    lines.append("| bet_type | 目数 | 短縮率 | ドリフト中央 | Q25 | Q75 | ガミ帯反転率 |")
    lines.append("|---|---|---|---|---|---|---|")

    for bt in ["trio", "trifecta", "quinellaPlace", "exacta", "quinella"]:
        d = desc.get(bt)
        if d is None:
            continue
        row = (f"| {bt} | {d['n']:,} | {d['pct_shortened']:.1%} "
               f"| {d['drift_median']:.3f} | {d['drift_q25']:.3f} | {d['drift_q75']:.3f} "
               f"| {d['gami_flip_rate']:.1%} |")
        print(f"    {bt:<18} 目数{d['n']:,}  短縮{d['pct_shortened']:.1%}"
              f"  ドリフト中央{d['drift_median']:.3f} [{d['drift_q25']:.3f},{d['drift_q75']:.3f}]"
              f"  ガミ帯反転{d['gami_flip_rate']:.1%}")
        lines.append(row)

    lines.append("")

    # ── trio オッズ帯別 ───────────────────────────────────────────────
    print("\n  ◆ 1-1. trio オッズ帯別ドリフト\n")
    lines.append("### 1-1. trio 朝オッズ帯別ドリフト\n")
    trio_d = desc.get("trio", {})
    if trio_d.get("by_band"):
        lines.append("| 朝オッズ帯 | 目数 | 短縮率 | ドリフト中央 |")
        lines.append("|---|---|---|---|")
        for band, bd in sorted(trio_d["by_band"].items()):
            row = (f"| {band} | {bd['n']} | {bd['pct_shortened']:.1%} "
                   f"| {bd['median_drift']:.3f} |")
            print(f"    {band:<8} {bd['n']:>5}目  短縮{bd['pct_shortened']:.1%}"
                  f"  中央ドリフト{bd['median_drift']:.3f}")
            lines.append(row)
        lines.append("")

    # ── hit vs non-hit ────────────────────────────────────────────────
    print("\n  ◆ 1-2. trio 的中目 vs 非的中目 短縮率差（smart money 仮説）\n")
    lines.append("### 1-2. trio 的中目 vs 非的中目の朝→確定短縮率差（smart money 仮説）\n")
    hvnh = trio_d.get("hit_vs_nonhit", {})
    if hvnh:
        n_hit = hvnh["n_hit"]
        n_nh = hvnh["n_nonhit"]
        psh = hvnh.get("hit_pct_shortened")
        psn = hvnh.get("nonhit_pct_shortened")
        dff = hvnh.get("diff_pct_shortened")
        hdm = hvnh.get("hit_drift_median")
        ndm = hvnh.get("nonhit_drift_median")
        if psh is not None:
            print(f"    的中目  {n_hit:>4}目  短縮率{psh:.1%}  ドリフト中央{hdm:.3f}")
            print(f"    非的中目{n_nh:>4}目  短縮率{psn:.1%}  ドリフト中央{ndm:.3f}")
            print(f"    差（的中−非的中）: {dff:+.1%}  "
                  f"({'⚑スマートマネー方向' if dff > 0 else '逆方向（参考）'})")
            lines.append(f"| 区分 | 目数 | 短縮率 | ドリフト中央 |")
            lines.append(f"|---|---|---|---|")
            lines.append(f"| 的中目 | {n_hit} | {psh:.1%} | {hdm:.3f} |")
            lines.append(f"| 非的中目 | {n_nh} | {psn:.1%} | {ndm:.3f} |")
            lines.append(f"| **差(的中−非的中)** | - | **{dff:+.1%}** | - |")
            lines.append(f"\n> 差が正なら「当たる目は朝から短縮されていた」＝スマートマネー方向。")
            lines.append(f"> ただし現時点 {n_hit}的中目 は暫定値。")
        else:
            print(f"    的中目/非的中目データ不足（{n_hit}的中目）")
            lines.append(f"> 的中目データ不足（{n_hit}的中目）。今後データが溜まり次第再確認。")
    lines.append("")

    # ─ エッジ検定 ────────────────────────────────────────────────────
    print(f"\n  ◆ 2. 事前登録エッジ検定（≤6車・リーク無しモデル・doc18セマンティクス）\n")
    lines.append("## 2. 事前登録エッジ検定（≤6車・リーク無しモデル・doc18セマンティクス）\n")
    lines.append("> 以下の3セルは事前登録済み。これ以外の探索的数字は「**参考・追試しない**」。  \n")

    # セル a
    ca = cell_a(races)
    print(f"  セル a: モデル上位目×短縮/伸長 ROI差")
    print(f"    短縮目: {_fmt_roi(ca['shortened']['roi'], ca['shortened']['n_races'])}")
    print(f"    伸長目: {_fmt_roi(ca['elongated']['roi'], ca['elongated']['n_races'])}")
    diff_a = ca.get("roi_diff")
    if diff_a is not None:
        print(f"    ROI差（短縮−伸長）: {diff_a:+.0%}")
    lines.append("### セル a: モデル上位目 × 短縮/伸長 ROI 差\n")
    lines.append(f"| 区分 | 評価 |")
    lines.append(f"|---|---|")
    lines.append(f"| 短縮目 | {_fmt_roi(ca['shortened']['roi'], ca['shortened']['n_races'])} |")
    lines.append(f"| 伸長目 | {_fmt_roi(ca['elongated']['roi'], ca['elongated']['n_races'])} |")
    lines.append(f"| ROI差（短縮−伸長） | {diff_a:+.0%} |" if diff_a is not None
                 else "| ROI差 | データ不足 |")
    lines.append("")

    # セル b
    cb = cell_b(races)
    print(f"\n  セル b: C0戦略×短縮ゲート")
    print(f"    C0全体:   {_fmt_roi(cb['c0_all']['roi'], cb['c0_all']['n_races'])}")
    print(f"    C0+短縮ゲート: {_fmt_roi(cb['c0_gate']['roi'], cb['c0_gate']['n_races'])}  "
          f"(ゲート通過率{cb['gate_pct']:.0%})")
    lines.append("### セル b: C0戦略（3連複・最安≥5倍） × 推奨目短縮ゲート\n")
    lines.append(f"| 区分 | 評価 | 備考 |")
    lines.append(f"|---|---|---|")
    lines.append(f"| C0 全体 | {_fmt_roi(cb['c0_all']['roi'], cb['c0_all']['n_races'])} | baseline |")
    lines.append(f"| C0 + 短縮ゲート | {_fmt_roi(cb['c0_gate']['roi'], cb['c0_gate']['n_races'])} "
                 f"| 通過率{cb['gate_pct']:.0%} |")
    lines.append("")

    # セル c
    cc = cell_c(races)
    print(f"\n  セル c: fav_mismatch × 本命交代")
    print(f"    fav_mismatch×交代あり: {_fmt_roi(cc['fav_swap']['roi'], cc['fav_swap']['n_races'])}")
    print(f"    fav_mismatch×交代なし: {_fmt_roi(cc['fav_noswap']['roi'], cc['fav_noswap']['n_races'])}")
    diff_c = cc.get("roi_diff")
    if diff_c is not None:
        print(f"    ROI差（交代あり−なし）: {diff_c:+.0%}")
    lines.append("### セル c: fav_mismatch × 市場本命の朝→直前交代\n")
    lines.append(f"| 区分 | 評価 |")
    lines.append(f"|---|---|")
    lines.append(f"| fav_mismatch + 本命交代あり | "
                 f"{_fmt_roi(cc['fav_swap']['roi'], cc['fav_swap']['n_races'])} |")
    lines.append(f"| fav_mismatch + 本命交代なし | "
                 f"{_fmt_roi(cc['fav_noswap']['roi'], cc['fav_noswap']['n_races'])} |")
    lines.append(f"| ROI差（交代あり−なし） | {diff_c:+.0%} |" if diff_c is not None
                 else "| ROI差 | データ不足 |")
    lines.append("")

    # ─ 結論と再実行手順 ──────────────────────────────────────────────
    print(f"\n  ◆ 3. 現時点の判定と再実行手順\n")
    lines.append("## 3. 現時点の判定\n")
    lines.append(f"- **≤6車スナップショット対象**: {n_le6}R（うちエッジ検定可能: 結果確定分）")
    lines.append(f"- **結論を出せる最小標本数**: ≈{min_n}R（80%検出力・ROI差20%pp）")
    lines.append(f"- **現状**: 標本数が最小基準の約{n_le6/min_n:.0%}。")
    lines.append(f"  数字は暫定であり、いずれのセルも現時点では採否を判定できない。")
    lines.append(f"- **多重比較への防衛**: 上記3セル以外の探索的数字は「参考・追試しない」。")
    lines.append(f"  ガミ帯反転率・hit_vs_nonhit等は記述統計であり検定セルではない。\n")

    lines.append("## 4. 再実行手順\n")
    lines.append("```bash")
    lines.append("# スナップショット収集（G03が稼働後・毎朝cronで自動収集）")
    lines.append("# 再実行:")
    lines.append("python3 scripts/exp_moneyflow_wt.py --from 2026-06-08 --to YYYY-MM-DD")
    lines.append("# レポート生成:")
    lines.append("python3 scripts/exp_moneyflow_wt.py --from 2026-06-08 --to YYYY-MM-DD --report")
    lines.append("```\n")

    lines.append("## 5. 探索的参考数字（追試しない）\n")
    lines.append("> 以下はハーネス動作確認の副産物。多重比較リスクがあるため、")
    lines.append("> 上記3事前登録セルの結論に使用しない。\n")
    lines.append("| 指標 | 値 | 解釈 |")
    lines.append("|---|---|---|")
    trio_d2 = desc.get("trio", {})
    if trio_d2:
        lines.append(f"| trio全目短縮率 | {trio_d2['pct_shortened']:.1%} | 参考 |")
        lines.append(f"| trio全目ドリフト中央 | {trio_d2['drift_median']:.3f} | 参考 |")
        lines.append(f"| ガミ帯反転率(trio) | {trio_d2['gami_flip_rate']:.1%} | 参考 |")

    lines.append("")

    print(f"\n  最小標本数到達時期の目安:")
    print(f"    ≤6車レース発生率 ≈ 30/345 ≈ 8.7% → 1日あたり ~1R")
    print(f"    {min_n}R 到達 ≈ {min_n}日 ≈ {min_n/30:.0f}ヶ月")
    lines.append(f"## 6. 標本数到達見込み\n")
    lines.append(f"- ≤6車レース比率（現実績）: {n_le6}/{len(_races_with_morning_snapshot(date_from, date_to))}R "
                 f"= {n_le6/max(len(_races_with_morning_snapshot(date_from, date_to)), 1):.1%}")
    lines.append(f"- 1日あたり約{n_le6/5:.1f}R（{n_le6}R/5日から外挿・G03稼働後に更新）")
    lines.append(f"- {min_n}R到達まで約{min_n/(n_le6/5):.0f}日 ≈ {min_n/(n_le6/5)/30:.0f}ヶ月")
    lines.append(f"  （ただし G03 でスナップショット収集が拡充されれば短縮）\n")

    print(f"\n{'='*100}\n")

    return "\n".join(lines)


# ── メイン ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="money-flow 検証ハーネス (G04)",
    )
    ap.add_argument(
        "--from", dest="date_from", default="2026-06-08",
        help="データ取得開始日 (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--to", dest="date_to", default=None,
        help="データ取得終了日 (YYYY-MM-DD、省略=最新)",
    )
    ap.add_argument(
        "--report", action="store_true",
        help="docs/analysis/23-moneyflow-initial.md を生成",
    )
    args = ap.parse_args()

    print(f"\n[G04 money-flow] 期間: {args.date_from} 〜 {args.date_to or '最新'}", flush=True)
    data = collect(args.date_from, args.date_to)

    report_md = print_report(data, args.date_from, args.date_to)

    if args.report:
        out_path = Path(__file__).resolve().parent.parent / "docs/analysis/23-moneyflow-initial.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        print(f"  レポートを保存しました: {out_path}")


if __name__ == "__main__":
    main()
