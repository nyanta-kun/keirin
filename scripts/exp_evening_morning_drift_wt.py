"""朝→夕方 intraday オッズドリフト分析（doc36）

NOTE:
  wt_odds_snapshot の snapshot_type='evening' は「当日18〜20時」収集。
  「前夜→当日朝」の overnight drift ではなく、同日開門時→夕方の intraday drift。
  実用上の意味: 当日8時の段階でモデル予測を出し、夕方18時時点でのオッズ変化を観察。
  → 終盤レース（18時以降）に対しては pre-race フィルターとして利用可能。

データ:
  morning + evening 両方あるレース: 387R（2026-06-10〜2026-06-14 / 全HOLD期間内）
  ≤6車かつC0戦略対象: 見込み 20-30R（統計的結論は不可・方法論確立が目的）

事前登録セル（これ以外の探索的数字は「参考・追試しない」）:
  a) pred1/pred2 の朝→夕方「短縮」vs「伸長」別の確定ROI差
  b) C0戦略 × 全推奨目majority短縮ゲート の有無
  c) 朝→夕方ドリフト方向と朝→最終オッズ方向の一致率（実用性チェック）

使い方:
  python3 scripts/exp_evening_morning_drift_wt.py
  python3 scripts/exp_evening_morning_drift_wt.py --report
"""
import sys
import re
import argparse
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)

# ── 期間定義（doc18標準）────────────────────────────────────────────────
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-15")

LGB_PARAMS = dict(
    objective="binary", n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)

GAMI_THRESHOLD = 5.0
MIN_RACES_FOR_CONCLUSION = 300  # 結論に必要な最小レース数（20%pp 差・80%検出力）


# ── DB ユーティリティ ──────────────────────────────────────────────────

def _load_trio_boards(race_keys: list[str], snap_type: str) -> dict[str, dict]:
    """snapshot_type の trio 盤面を race_key → {frozenset(3): odds} で返す。"""
    boards: dict[str, dict] = defaultdict(dict)
    CH = 500
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
                    fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                except ValueError:
                    continue
                if len(fr) == 3:
                    boards[rk][fr] = float(ov)
    return dict(boards)


def _load_final_boards(race_keys: list[str]) -> dict[str, dict]:
    """wt_odds から確定 trio 盤面を返す。"""
    boards: dict[str, dict] = defaultdict(dict)
    CH = 500
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
                    fr = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                except ValueError:
                    continue
                if len(fr) == 3:
                    boards[rk][fr] = float(ov)
    return dict(boards)


def _load_n_entries(race_keys: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    CH = 500
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


def _dual_snapshot_race_keys() -> list[str]:
    """morning と evening 両方のスナップショットがある race_key を返す。"""
    with get_connection() as c:
        rows = c.execute("""
            SELECT e.race_key FROM
              (SELECT DISTINCT race_key FROM wt_odds_snapshot WHERE snapshot_type='evening') e
            JOIN
              (SELECT DISTINCT race_key FROM wt_odds_snapshot WHERE snapshot_type='morning') m
              ON e.race_key = m.race_key
        """).fetchall()
    return [r[0] for r in rows]


# ── player-level implied top3 probability ────────────────────────────

def _implied_top3_probs(trio_board: dict) -> dict[int, float]:
    """trio 盤面から player ごとの implied top3 確率（正規化）を返す。

    p_i ∝ Σ_{combo ∋ i} 1/odds(combo)
    """
    q: dict[int, float] = {}
    for combo, ov in trio_board.items():
        if ov <= 0:
            continue
        inv = 1.0 / ov
        for f in combo:
            q[f] = q.get(f, 0.0) + inv
    total = sum(q.values())
    if total <= 0:
        return {}
    return {f: v / total for f, v in q.items()}


def _drift(morning_probs: dict[int, float], evening_probs: dict[int, float],
           frame_no: int) -> float | None:
    """frame_no の implied prob ドリフト（夕方 - 朝）を返す。
    正 → 夕方に implied prob が増加 = オッズ短縮（market が支持）
    負 → 夕方に implied prob が減少 = オッズ伸長
    """
    mp = morning_probs.get(frame_no)
    ep = evening_probs.get(frame_no)
    if mp is None or ep is None or mp <= 0:
        return None
    return ep - mp


# ── レース構造体ビルド ─────────────────────────────────────────────────

def build_race_records(
    df: pd.DataFrame,
    morning_boards: dict[str, dict],
    evening_boards: dict[str, dict],
    final_boards: dict[str, dict],
    n_entries_map: dict[str, int],
) -> list[dict]:
    """C0 戦略評価用レース構造体リスト。"""
    records = []
    for rk, grp in df.groupby("race_key"):
        # 出走表基準 ≤6車
        if n_entries_map.get(rk, 99) > 6:
            continue
        mb = morning_boards.get(rk, {})
        eb = evening_boards.get(rk, {})
        fb = final_boards.get(rk, {})
        if not mb or not eb:
            continue

        # 結果確定チェック
        fin = grp[grp["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        dns = set(grp[grp["finish_order"] == 0]["frame_no"].astype(int).tolist())

        # 全エントリーでランキング（doc18）
        g = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        frames = g["frame_no"].astype(int).tolist()
        p1, p2 = frames[0], frames[1]
        thirds = frames[2:]

        # 欠車 void
        axis_void = (p1 in dns) or (p2 in dns)

        # implied top3 prob（朝・夕方）
        m_probs = _implied_top3_probs(mb)
        e_probs = _implied_top3_probs(eb)

        p1_drift = _drift(m_probs, e_probs, p1)
        p2_drift = _drift(m_probs, e_probs, p2)

        # C0: trio 3点（最安≥GAMI_THRESHOLD）
        legs: list[dict] = []
        if not axis_void:
            for t in thirds:
                if t in dns:
                    continue
                combo = frozenset((p1, p2, t))
                fo = fb.get(combo) if fb else None
                mo = mb.get(combo)
                eo = eb.get(combo)
                if fo is None or fo <= 0:
                    continue
                m_drift_leg = ((1.0 / eo - 1.0 / mo) if (mo and mo > 0 and eo and eo > 0)
                               else None)
                legs.append({
                    "final_odds": fo,
                    "morning_odds": mo,
                    "evening_odds": eo,
                    "hit": (combo == top3),
                    "m_drift": m_drift_leg,
                    # 夕方→最終 方向一致チェック用
                    "eve_shortened": (eo < mo) if (eo and mo) else None,
                    "final_shortened": (fo < mo) if (fo and mo) else None,
                })

        min_final = min((l["final_odds"] for l in legs if l["final_odds"] > 0),
                        default=None)
        gami_ok = (min_final is not None and min_final >= GAMI_THRESHOLD)

        records.append({
            "race_key": rk,
            "date": str(grp["race_date"].iloc[0]),
            "p1": p1, "p2": p2,
            "top3": top3, "dns": dns,
            "axis_void": axis_void,
            "p1_drift": p1_drift,
            "p2_drift": p2_drift,
            "legs": legs,
            "gami_ok": gami_ok,
            "min_final": min_final,
        })
    return records


# ── セル計算 ─────────────────────────────────────────────────────────

def roi_of(pays: list[float], bets: list[float]) -> float | None:
    total_bet = sum(bets)
    if total_bet <= 0:
        return None
    return sum(pays) / total_bet * 100.0


def cell_a(records: list[dict]) -> dict:
    """セル a: pred1/pred2 短縮 vs 伸長 別 ROI 差。"""
    pays_short, bets_short = [], []
    pays_long, bets_long = [], []

    for r in records:
        if r["axis_void"] or not r["gami_ok"] or not r["legs"]:
            continue
        p1d = r["p1_drift"]
        if p1d is None:
            continue
        pay = sum(l["final_odds"] * 100 for l in r["legs"] if l["hit"])
        bet = len(r["legs"]) * 100
        if p1d >= 0:  # pred1 短縮（implied prob 増加 = オッズ縮小）
            pays_short.append(pay)
            bets_short.append(bet)
        else:
            pays_long.append(pay)
            bets_long.append(bet)

    roi_s = roi_of(pays_short, bets_short)
    roi_l = roi_of(pays_long, bets_long)
    return {
        "shortened": {"roi": roi_s, "n": len(pays_short)},
        "elongated": {"roi": roi_l, "n": len(pays_long)},
        "roi_diff": (roi_s - roi_l) if (roi_s is not None and roi_l is not None) else None,
    }


def cell_b(records: list[dict]) -> dict:
    """セル b: C0 × majority短縮ゲート。"""
    pays_all, bets_all = [], []
    pays_gate, bets_gate = [], []

    for r in records:
        if r["axis_void"] or not r["gami_ok"] or not r["legs"]:
            continue
        pay = sum(l["final_odds"] * 100 for l in r["legs"] if l["hit"])
        bet = len(r["legs"]) * 100
        pays_all.append(pay)
        bets_all.append(bet)

        # majority 短縮: pred1 AND pred2 が両方短縮、かつ脚の過半数が短縮
        p1d = r["p1_drift"]
        p2d = r["p2_drift"]
        if p1d is None or p2d is None:
            continue
        n_legs = len(r["legs"])
        n_leg_short = sum(
            1 for l in r["legs"]
            if l["eve_shortened"] is True
        )
        if p1d >= 0 and p2d >= 0 and n_leg_short >= math.ceil(n_legs / 2):
            pays_gate.append(pay)
            bets_gate.append(bet)

    roi_all = roi_of(pays_all, bets_all)
    roi_gate = roi_of(pays_gate, bets_gate)
    return {
        "c0_all": {"roi": roi_all, "n": len(pays_all)},
        "c0_gate": {"roi": roi_gate, "n": len(pays_gate)},
        "gate_pct": len(pays_gate) / max(len(pays_all), 1),
    }


def cell_c(records: list[dict]) -> dict:
    """セル c: 朝→夕方ドリフト方向と朝→最終方向の一致率。

    一致率が高い → 夕方18時段階で「最終方向を予測できる」→ 終盤レースへの pre-race 実用性あり
    """
    n_agree, n_total = 0, 0
    agree_by_race: list[bool] = []

    for r in records:
        for l in r["legs"]:
            es = l["eve_shortened"]
            fs = l["final_shortened"]
            if es is None or fs is None:
                continue
            n_total += 1
            if es == fs:
                n_agree += 1

        # レース単位での一致（pred1 の方向）
        if r["p1_drift"] is not None:
            p1d_shortened = r["p1_drift"] >= 0
            # final での pred1 方向
            m_p1 = _implied_top3_probs(morning_boards_cache.get(r["race_key"], {})).get(r["p1"])
            e_p1 = _implied_top3_probs(evening_boards_cache.get(r["race_key"], {})).get(r["p1"])
            f_p1 = _implied_top3_probs(final_boards_cache.get(r["race_key"], {})).get(r["p1"])
            if m_p1 and f_p1:
                final_shortened = f_p1 >= m_p1
                agree_by_race.append(p1d_shortened == final_shortened)

    return {
        "leg_agreement_rate": n_agree / n_total if n_total > 0 else None,
        "n_legs": n_total,
        "race_p1_agreement_rate": (
            sum(agree_by_race) / len(agree_by_race) if agree_by_race else None
        ),
        "n_races_c": len(agree_by_race),
    }


# グローバルキャッシュ（cell_c 用）
morning_boards_cache: dict[str, dict] = {}
evening_boards_cache: dict[str, dict] = {}
final_boards_cache: dict[str, dict] = {}


# ── メイン ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="朝→夕方 intraday ドリフト分析（doc36）")
    ap.add_argument("--report", action="store_true",
                    help="docs/analysis/36-intraday-drift.md を生成")
    args = ap.parse_args()

    print("\n[doc36] 朝→夕方 intraday オッズドリフト分析", flush=True)
    print()

    # ── 1. dual-snapshot race_keys ─────────────────────────────────────
    print("dual-snapshot レース特定中...", flush=True)
    dual_keys = _dual_snapshot_race_keys()
    print(f"  morning + evening 両方あり: {len(dual_keys)}R")

    # ── 2. n_entries 取得・≤6車フィルタ ───────────────────────────────
    n_entries_map = _load_n_entries(dual_keys)
    le6_keys = [rk for rk in dual_keys if n_entries_map.get(rk, 99) <= 6]
    print(f"  うち ≤6車（出走表基準）: {len(le6_keys)}R")

    # ── 3. 特徴量ビルド + リーク無しモデル ─────────────────────────────
    print("\n特徴量ビルド中（TRAIN〜HOLD）...", flush=True)
    df_full = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
    fit = df_full[
        (df_full["race_date"] <= TRAIN[1]) & (df_full["finish_order"] >= 1)
    ]
    print(f"  TRAIN 学習: {len(fit):,} rows", flush=True)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"])

    df_snap = df_full[df_full["race_key"].isin(set(dual_keys))].copy()
    df_snap["pred_prob"] = m.predict_proba(prepare_X(df_snap))[:, 1]

    # ≤6車に絞る
    df_le6 = df_snap[df_snap["race_key"].isin(set(le6_keys))].copy()

    # 結果確定
    done = df_le6.groupby("race_key")["finish_order"].apply(
        lambda s: (s.between(1, 3)).sum() >= 3
    )
    df_le6 = df_le6[df_le6["race_key"].isin(done[done].index)]
    live_keys = df_le6["race_key"].unique().tolist()
    print(f"  うち結果確定: {len(live_keys)}R")

    # ── 4. 盤面ロード ──────────────────────────────────────────────────
    print("\n盤面ロード中...", flush=True)
    morning_boards_cache.update(_load_trio_boards(live_keys, "morning"))
    evening_boards_cache.update(_load_trio_boards(live_keys, "evening"))
    final_boards_cache.update(_load_final_boards(live_keys))

    # ── 5. レース構造体ビルド ──────────────────────────────────────────
    records = build_race_records(
        df_le6, morning_boards_cache, evening_boards_cache,
        final_boards_cache, n_entries_map,
    )
    c0_records = [r for r in records if r["gami_ok"] and not r["axis_void"]]
    print(f"\n  ≤6車 結果確定: {len(records)}R")
    print(f"  うち C0戦略対象（ガミ≥5倍・軸欠車なし）: {len(c0_records)}R")

    # ── 6. 記述統計 ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("記述統計: 朝→夕方 implied top3 prob ドリフト")
    print(f"{'='*80}")
    drifts_p1 = [r["p1_drift"] for r in c0_records if r["p1_drift"] is not None]
    drifts_p2 = [r["p2_drift"] for r in c0_records if r["p2_drift"] is not None]
    if drifts_p1:
        print(f"  pred1 drift (n={len(drifts_p1)}): "
              f"median={np.median(drifts_p1):+.4f}  "
              f"短縮率={np.mean([d >= 0 for d in drifts_p1]):.1%}")
    if drifts_p2:
        print(f"  pred2 drift (n={len(drifts_p2)}): "
              f"median={np.median(drifts_p2):+.4f}  "
              f"短縮率={np.mean([d >= 0 for d in drifts_p2]):.1%}")

    # leg-level drift
    all_leg_drifts = []
    for r in c0_records:
        for l in r["legs"]:
            if l["eve_shortened"] is not None:
                all_leg_drifts.append(l["eve_shortened"])
    if all_leg_drifts:
        print(f"  3点脚短縮率: {np.mean(all_leg_drifts):.1%}  (n={len(all_leg_drifts)})")

    # ── 7. 事前登録セル ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("事前登録セル")
    print(f"{'='*80}")

    # セル a
    ca = cell_a(c0_records)
    print("\n  [セル a] pred1 短縮/伸長別 C0 ROI")
    for tag, key in [("短縮(pred1 drift≥0)", "shortened"), ("伸長(pred1 drift<0)", "elongated")]:
        d = ca[key]
        roi_str = f"{d['roi']:.1f}%" if d["roi"] is not None else "--"
        print(f"    {tag}: {roi_str}  ({d['n']}R)")
    diff = ca["roi_diff"]
    print(f"    ROI差（短縮−伸長）: {diff:+.1f}pp" if diff is not None else "    ROI差: データ不足")

    # セル b
    cb = cell_b(c0_records)
    print("\n  [セル b] C0 × majority短縮ゲート")
    roi_all = cb["c0_all"]["roi"]
    roi_gate = cb["c0_gate"]["roi"]
    print(f"    C0 全体:  {roi_all:.1f}%  ({cb['c0_all']['n']}R)" if roi_all is not None
          else f"    C0 全体: --  ({cb['c0_all']['n']}R)")
    print(f"    C0+ゲート: {roi_gate:.1f}%  ({cb['c0_gate']['n']}R・"
          f"ゲート通過率{cb['gate_pct']:.0%})" if roi_gate is not None
          else f"    C0+ゲート: --  ({cb['c0_gate']['n']}R・ゲート通過率{cb['gate_pct']:.0%})")

    # セル c
    cc = cell_c(c0_records)
    print("\n  [セル c] 朝→夕方方向 vs 朝→最終方向 一致率")
    lag = cc["leg_agreement_rate"]
    rar = cc["race_p1_agreement_rate"]
    print(f"    脚レベル一致率: {lag:.1%}  ({cc['n_legs']}脚)" if lag is not None
          else "    脚レベル一致率: データ不足")
    print(f"    pred1 レース単位一致率: {rar:.1%}  ({cc['n_races_c']}R)" if rar is not None
          else "    pred1 レース単位一致率: データ不足")

    print(f"\n  ⚠ 現時点 {len(c0_records)}R（C0対象）：統計的結論に必要な最小標本数は"
          f"≈{MIN_RACES_FOR_CONCLUSION}R。方向確認・方法論確立のみ。")

    # ── 8. データ蓄積見込み ────────────────────────────────────────────
    n_days = 5  # 2026-06-10〜14
    rate_le6 = len(le6_keys) / max(len(dual_keys), 1)
    c0_rate = len(c0_records) / max(len(le6_keys), 1)
    daily_c0 = len(c0_records) / n_days
    days_to_conclude = (MIN_RACES_FOR_CONCLUSION - len(c0_records)) / max(daily_c0, 0.1)
    print(f"\n  蓄積見込み: ≤6車率={rate_le6:.0%}  C0率={c0_rate:.0%}  "
          f"1日あたりC0≈{daily_c0:.1f}R  "
          f"→ 結論まで残り約{days_to_conclude:.0f}日 ≈ {days_to_conclude/30:.0f}ヶ月")

    print()

    # ── 9. レポート生成 ────────────────────────────────────────────────
    if args.report:
        _write_report(records, c0_records, ca, cb, cc, dual_keys, le6_keys)


def _write_report(records, c0_records, ca, cb, cc, dual_keys, le6_keys):
    lines = []
    n_c0 = len(c0_records)

    lines.append("# doc36: 朝→夕方 intraday オッズドリフト分析（2026-06-15）\n")
    lines.append("> **結論**: 統計的判断には標本数不足（現在 {}R・必要 {}R）。"
                 "方向確認・方法論確立が目的。\n".format(n_c0, MIN_RACES_FOR_CONCLUSION))

    lines.append("## データ概要\n")
    lines.append(f"| 項目 | 値 |")
    lines.append(f"|---|---|")
    lines.append(f"| morning+evening 両snapshot対象 | {len(dual_keys)}R |")
    lines.append(f"| うち ≤6車（出走表基準） | {len(le6_keys)}R |")
    lines.append(f"| うち C0対象（ガミ≥5倍・軸欠車なし・結果確定） | {n_c0}R |")
    lines.append(f"| 期間 | 2026-06-10〜2026-06-14 (HOLD 内) |\n")

    lines.append("## NOTE: 「夕方」の意味\n")
    lines.append("- `snapshot_type='evening'` は当日18〜20時に収集。")
    lines.append("- 「前夜→翌朝」の overnight drift **ではなく**、同日朝（8時）→夕方（18時）の intraday drift。")
    lines.append("- 実用上、18時以降に発走するレース（最終3〜4走）に対しては pre-race フィルターとして利用可能。\n")

    lines.append("## 事前登録セル\n")

    # セル a
    lines.append("### セル a: pred1 短縮/伸長別 C0 ROI\n")
    lines.append("| 区分 | ROI | n |")
    lines.append("|---|---|---|")
    for tag, key in [("pred1 短縮（drift≥0）", "shortened"), ("pred1 伸長（drift<0）", "elongated")]:
        d = ca[key]
        roi_str = f"{d['roi']:.1f}%" if d["roi"] is not None else "--"
        lines.append(f"| {tag} | {roi_str} | {d['n']}R |")
    diff = ca.get("roi_diff")
    lines.append(f"\nROI差（短縮−伸長）: **{diff:+.1f}pp**" if diff is not None
                 else "\nROI差: データ不足（どちらかが0R）")
    lines.append("")

    # セル b
    lines.append("### セル b: C0 × majority短縮ゲート\n")
    lines.append("| 区分 | ROI | n | 備考 |")
    lines.append("|---|---|---|---|")
    ra = cb["c0_all"]["roi"]
    rg = cb["c0_gate"]["roi"]
    lines.append(f"| C0 全体 | {ra:.1f}% | {cb['c0_all']['n']}R | baseline |"
                 if ra is not None else f"| C0 全体 | -- | {cb['c0_all']['n']}R | baseline |")
    lines.append(f"| C0 + ゲート | {rg:.1f}% | {cb['c0_gate']['n']}R | "
                 f"通過率{cb['gate_pct']:.0%} |"
                 if rg is not None else f"| C0 + ゲート | -- | {cb['c0_gate']['n']}R | "
                 f"通過率{cb['gate_pct']:.0%} |")
    lines.append("")

    # セル c
    lines.append("### セル c: 朝→夕方 方向 vs 朝→最終 方向 一致率\n")
    lag = cc["leg_agreement_rate"]
    rar = cc["race_p1_agreement_rate"]
    lines.append(f"| 粒度 | 一致率 | n |")
    lines.append(f"|---|---|---|")
    lines.append(f"| 脚レベル | {lag:.1%} | {cc['n_legs']}脚 |"
                 if lag is not None else f"| 脚レベル | -- | {cc['n_legs']}脚 |")
    lines.append(f"| pred1 レース単位 | {rar:.1%} | {cc['n_races_c']}R |"
                 if rar is not None else f"| pred1 レース単位 | -- | {cc['n_races_c']}R |")
    lines.append("\n> 一致率 >80% → 夕方18時で最終方向を予測でき、終盤レースへの実用性あり。\n")

    lines.append("## 結論\n")
    lines.append(f"- 現時点 {n_c0}R は統計的結論の最小標本数（{MIN_RACES_FOR_CONCLUSION}R）の "
                 f"{n_c0/MIN_RACES_FOR_CONCLUSION:.0%}。")
    lines.append("- 方向シグナルの記述統計は参考値として活用。")
    lines.append("- **最重要確認**: セル c の一致率が >80% なら、終盤レースへの 18 時フィルター実用化を検討。\n")

    lines.append("## ハーネス\n")
    lines.append("```bash")
    lines.append("python3 scripts/exp_evening_morning_drift_wt.py")
    lines.append("python3 scripts/exp_evening_morning_drift_wt.py --report")
    lines.append("```\n")

    out_path = Path(__file__).resolve().parent.parent / "docs/analysis/36-intraday-drift.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  レポート保存: {out_path}")


if __name__ == "__main__":
    main()
