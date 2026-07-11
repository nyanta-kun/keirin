"""live実測レポートCLI — picks_history(route='wt')の集計・ドリフト割引率・必要標本数。

G02: 採否判断の唯一の裁定者（live実測）を見える化する。
  1. ランク別成績: 7PLUS_R(表示SS)/7PLUS_ST(表示S)/7PLUS_STP(表示S+) 別
     n・的中率・投資額・払戻・ROI・bootstrap CI・最大払戻除去ROI
  2. タグ別成績: detail.json の fav_mismatch / upset_tier / top3_sum帯 を race_key で picks_history に突合
  3. 朝→確定ドリフト割引率: wt_odds_snapshot(morning/evening) vs wt_odds(確定) のオッズ帯別ドリフト率
  4. 必要標本数の推定: 現在の的中率・払戻分布を所与として ROI CI下限 >100% に必要な残R数
  5. --from/--to 期間指定、--format md でマークダウン出力

ランク体系は notify_prerace_wt.py（2026-07-10〜）に準拠。内部 rank 列は 7PLUS_R/7PLUS_ST/7PLUS_STP
のみが購入対象（見送り miwokuri=True・候補 7PLUS_CAND は集計から除外・notify_results_wt.py の
集計条件 `rank IN (...) AND NOT COALESCE(miwokuri,FALSE) AND bet_amount>0` と統一）。

注: DB 書込みなし・Discord通知なし・標準入出力のみ。
"""
import sys
import json
import argparse
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.database import get_connection

# scripts/ にある roi_summary を再利用
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from roi_robustness_wt import roi_summary  # noqa: E402

# ── ランク体系（2026-07-10〜・notify_prerace_wt.py と同一） ──────────────
# 内部rank（DB格納値） → 表示ラベル。購入対象はこの3つのみ（見送り/候補は集計除外）。
RANKS = ["7PLUS_R", "7PLUS_ST", "7PLUS_STP"]
RANK_LABELS = {"7PLUS_R": "SS", "7PLUS_ST": "S", "7PLUS_STP": "S+"}

# ── picks_history 集計 ─────────────────────────────────────────────


def _load_picks(date_from: str | None, date_to: str | None) -> list[dict]:
    """picks_history から route='wt' の購入確定行（見送り・候補を除く）を取得。

    購入対象ランクは 7PLUS_R(表示SS・三連複)/7PLUS_ST(表示S・三連単F)/7PLUS_STP(表示S+・同増額) の3種。
    見送り(miwokuri=True)・候補(7PLUS_CAND)・bet_amount=0（プレースホルダー行）は集計対象外
    （notify_results_wt.py の _query_stats と同一条件）。
    """
    placeholders = ",".join("?" * len(RANKS))
    sql = f"""
        SELECT race_date, race_key, rank, n_combos, hit, payout, bet_amount
        FROM picks_history
        WHERE route = 'wt'
          AND rank IN ({placeholders})
          AND NOT COALESCE(miwokuri, FALSE)
          AND bet_amount > 0
    """
    params: list = list(RANKS)
    if date_from:
        sql += " AND race_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND race_date <= ?"
        params.append(date_to)
    sql += " ORDER BY race_date, race_key"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {"race_date": r[0], "race_key": r[1], "rank": r[2],
         "n_combos": r[3] or 1, "hit": bool(r[4]),
         "payout": r[5] or 0, "bet_amount": r[6] or 0}
        for r in rows
    ]


# ── タグ突合 ─────────────────────────────────────────────────────────


def _load_tags(date_from: str | None, date_to: str | None) -> dict[str, dict]:
    """detail JSON ファイルから race_key → タグ辞書 を返す。

    昼と夜の両ファイルを走査。タグ: fav_mismatch, upset_tier, top3_sum帯。
    """
    picks_dir = Path(__file__).resolve().parent.parent / "data" / "picks"
    result: dict[str, dict] = {}
    for p in sorted(picks_dir.glob("wave_picks_wt_*_detail.json")):
        # ファイル名から日付抽出: wave_picks_wt_2026-06-11_detail.json
        stem = p.stem  # wave_picks_wt_2026-06-11_detail
        parts = stem.split("_")
        # 日付は 4番目要素 (0-indexed: wave/picks/wt/DATE/detail)
        if len(parts) < 5:
            continue
        date_str = parts[3]  # e.g. "2026-06-11"
        if date_from and date_str < date_from:
            continue
        if date_to and date_str > date_to:
            continue
        try:
            entries = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for e in entries:
            rk = e.get("race_key", "")
            if not rk:
                continue
            t3s = e.get("top3_sum", None)
            result[rk] = {
                "fav_mismatch": e.get("fav_mismatch", None),
                "upset_tier": e.get("upset_tier", None),
                "top3_sum": t3s,
                "top3_sum_band": _top3_band(t3s),
            }
    return result


def _top3_band(v) -> str | None:
    """top3_sum 値をQ帯に変換（固定しきい値）。"""
    if v is None:
        return None
    if v < 1.70:
        return "Q1_loose(<1.70)"
    if v < 1.90:
        return "Q2(1.70-1.90)"
    if v < 2.08:
        return "Q3(1.90-2.08)"
    return "Q4_chalk(>=2.08)"


# ── ランク別集計 ──────────────────────────────────────────────────────


def _rank_section(picks: list[dict]) -> dict[str, dict]:
    """ランク別 (7PLUS_R/7PLUS_ST/7PLUS_STP・表示SS/S/S+) の roi_summary を返す。"""
    buckets: dict[str, tuple[list, list]] = {}
    for r in picks:
        rank = r["rank"]
        pays, bets = buckets.setdefault(rank, ([], []))
        pays.append(r["payout"])
        bets.append(r["bet_amount"])
    result = {}
    for rank, (pays, bets) in buckets.items():
        result[rank] = roi_summary(pays, bets)
    return result


# ── タグ別集計 ────────────────────────────────────────────────────────


def _tag_section(picks: list[dict], tags: dict[str, dict]) -> dict:
    """タグ有無別の live ROI を計算する。

    picks は _load_picks で購入確定ランク(7PLUS_R/7PLUS_ST/7PLUS_STP)のみに
    絞り込み済みのため、ここでは追加のランク除外は不要（全件が対象）。
    """
    main_picks = picks
    tag_results: dict[str, dict] = {}

    def _collect(label, predicate):
        pays, bets = [], []
        for p in main_picks:
            # picks_history.race_key は #7R/#7ST 等のサフィックス付き。
            # detail.json 由来のタグは素の race_key キーのため base で突合する。
            rk = p["race_key"].split("#", 1)[0]
            tag = tags.get(rk, {})
            if predicate(tag):
                pays.append(p["payout"])
                bets.append(p["bet_amount"])
        tag_results[label] = roi_summary(pays, bets)

    _collect("fav_mismatch=True", lambda t: t.get("fav_mismatch") is True)
    _collect("fav_mismatch=False", lambda t: t.get("fav_mismatch") is False)
    _collect("fav_mismatch=未記録", lambda t: t.get("fav_mismatch") is None)

    for band in ["Q1_loose(<1.70)", "Q2(1.70-1.90)", "Q3(1.90-2.08)", "Q4_chalk(>=2.08)"]:
        _collect(f"upset_tier={band}", lambda t, b=band: t.get("top3_sum_band") == b)

    return tag_results


# ── ドリフト割引率 ────────────────────────────────────────────────────


def _drift_section() -> dict:
    """wt_odds_snapshot(morning/evening) vs wt_odds(確定) のドリフト率分布。

    drift_ratio = final / snapshot。1.0 未満 = 実際のオッズは backtest 上限より低い（下振れ）。
    オッズ帯は snapshot 値で区切る。
    """
    sql = """
        SELECT s.odds_value AS snap, w.odds_value AS final, s.snapshot_type
        FROM wt_odds_snapshot s
        JOIN wt_odds w
          ON s.race_key = w.race_key
         AND s.bet_type = w.bet_type
         AND s.combination = w.combination
        WHERE s.odds_value > 0 AND w.odds_value > 0
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return {"morning": {}, "evening": {}}

    # オッズ帯の定義（snapshotオッズ基準）
    bands = [
        ("< 5",    0, 5),
        ("5–10",   5, 10),
        ("10–20",  10, 20),
        ("20–50",  20, 50),
        ("50–100", 50, 100),
        ("100+",   100, 1e9),
    ]

    by_type: dict[str, dict[str, list]] = {"morning": {b[0]: [] for b in bands},
                                            "evening": {b[0]: [] for b in bands}}
    for snap, final, stype in rows:
        if stype not in by_type:
            continue
        ratio = final / snap
        for label, lo, hi in bands:
            if lo <= snap < hi:
                by_type[stype][label].append(ratio)
                break

    result = {}
    for stype, band_data in by_type.items():
        result[stype] = {}
        for label, ratios in band_data.items():
            if not ratios:
                result[stype][label] = None
                continue
            arr = np.array(ratios)
            result[stype][label] = {
                "n": len(arr),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            }
    return result


# ── 必要標本数推定 ────────────────────────────────────────────────────


def _required_n_section(pays: list[float], bets: list[float],
                         target_roi: float = 1.0,
                         n_sim: int = 5000, seed: int = 42) -> dict:
    """現在の払戻・投資分布を所与として、ROI CI下限が target_roi を超えるのに必要な追加R数を推定。

    方法: 現在のサンプルから (払戻, 投資) ペアをブートストラップで再標本化し、
    追加 k R 分を今の分布から無作為に加えたときの CI 下限を計算する。
    k を二分探索で求める（最大 5000 R まで）。
    """
    if not pays or sum(bets) == 0:
        return {"current_n": 0, "needed_additional": None, "note": "データなし"}

    pay_arr = np.array(pays, dtype=float)
    bet_arr = np.array(bets, dtype=float)
    n_now = len(pay_arr)

    rng = np.random.default_rng(seed)

    def _ci_lo_at(k_extra: int) -> float:
        """現在 n_now + k_extra R のときの bootstrap CI 下限（中央値）。"""
        ci_los = []
        for _ in range(200):  # 200回のモンテカルロ
            # 既存サンプルを k_extra R分拡張（同分布から再標本化）
            extra_idx = rng.integers(0, n_now, size=k_extra) if k_extra > 0 else np.array([], dtype=int)
            ext_pay = np.concatenate([pay_arr, pay_arr[extra_idx]])
            ext_bet = np.concatenate([bet_arr, bet_arr[extra_idx]])
            n_total = len(ext_pay)
            # bootstrap CI
            idx = rng.integers(0, n_total, size=(500, n_total))
            boot = ext_pay[idx].sum(axis=1) / ext_bet[idx].sum(axis=1)
            ci_los.append(float(np.percentile(boot, 2.5)))
        return float(np.median(ci_los))

    # 現在のCI下限
    cur_summary = roi_summary(list(pays), list(bets), seed=seed)
    ci_lo_now = cur_summary["ci_lo"]

    if ci_lo_now >= target_roi:
        return {"current_n": n_now, "needed_additional": 0, "ci_lo_now": ci_lo_now,
                "note": "すでにCI下限>100%"}

    # 二分探索: 上限 5000 R
    lo_k, hi_k = 0, 5000
    if _ci_lo_at(hi_k) < target_roi:
        return {"current_n": n_now, "needed_additional": ">5000",
                "ci_lo_now": ci_lo_now,
                "note": "5000R追加後もCI下限<100%（現在の平均ROIが100%未満）"}

    while hi_k - lo_k > 20:
        mid = (lo_k + hi_k) // 2
        if _ci_lo_at(mid) >= target_roi:
            hi_k = mid
        else:
            lo_k = mid

    return {"current_n": n_now, "needed_additional": hi_k,
            "ci_lo_now": ci_lo_now,
            "note": f"現在{n_now}R→追加約{hi_k}R（計{n_now+hi_k}R）でCI下限>100%見込み"}


# ── フォーマット ──────────────────────────────────────────────────────


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_summary(s: dict) -> str:
    if s["n"] == 0:
        return "(データなし)"
    return (f"n={s['n']} 的中{s['hit_rate']:.1%} "
            f"ROI {s['roi']:.1%} [{s['ci_lo']:.1%},{s['ci_hi']:.1%}] "
            f"除max {s['roi_ex_max']:.1%}")


def _render_text(result: dict, date_from, date_to) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"  live実測レポート[wt]  期間: {date_from or '全期間'} 〜 {date_to or '最新'}")
    lines.append("  払戻=実績値（picks_history）。backtest上限値とは異なる。")
    lines.append("=" * 80)

    # 1. ランク別成績
    lines.append("\n▼ 1. ランク別成績")
    lines.append(f"  {'ランク':<7}{'n':>5}{'的中率':>9}{'ROI':>8}{'95%CI':>22}{'除max ROI':>11}{'投資(円)':>12}{'払戻(円)':>12}")
    lines.append("  " + "-" * 82)
    rank_data = result["rank"]
    for rank in RANKS:
        s = rank_data.get(rank)
        if s is None:
            continue
        label = RANK_LABELS[rank]
        inv = result["rank_raw"][rank]["total_bet"]
        ret = result["rank_raw"][rank]["total_pay"]
        lines.append(
            f"  {label:<7}{s['n']:>5}{s['hit_rate']:>8.1%}{s['roi']:>7.1%} "
            f"[{s['ci_lo']:>7.1%},{s['ci_hi']:>7.1%}]{s['roi_ex_max']:>10.1%}"
            f"{inv:>12,}{ret:>12,}"
        )
    # 合計 (SS/S/S+ 全ランク)
    main_s = result.get("main_total")
    if main_s:
        inv_m = result["rank_raw"].get("_main_inv", 0)
        ret_m = result["rank_raw"].get("_main_pay", 0)
        lines.append("  " + "-" * 82)
        lines.append(
            f"  {'SS+S+S+':<7}{main_s['n']:>5}{main_s['hit_rate']:>8.1%}{main_s['roi']:>7.1%} "
            f"[{main_s['ci_lo']:>7.1%},{main_s['ci_hi']:>7.1%}]{main_s['roi_ex_max']:>10.1%}"
            f"{inv_m:>12,}{ret_m:>12,}"
        )
    lines.append("  ※ N<50の層は標本不足で暫定。")

    # 2. タグ別成績
    lines.append("\n▼ 2. タグ別成績（mainのみ・fav_mismatch は 2026-06-11 朝から記録）")
    lines.append(f"  {'タグ':<32}{'n':>5}{'的中率':>9}{'ROI':>8}{'95%CI':>22}{'除max ROI':>11}")
    lines.append("  " + "-" * 90)
    for label, s in result.get("tag", {}).items():
        if s["n"] == 0:
            continue
        lines.append(
            f"  {label:<32}{s['n']:>5}{s['hit_rate']:>8.1%}{s['roi']:>7.1%} "
            f"[{s['ci_lo']:>7.1%},{s['ci_hi']:>7.1%}]{s['roi_ex_max']:>10.1%}"
        )

    # 3. ドリフト割引率
    lines.append("\n▼ 3. 朝→確定オッズ ドリフト率（final/morning）")
    lines.append("  1.0未満=確定は朝より低い（backtest上限値からの下振れ）")
    drift = result.get("drift", {})
    for stype in ["morning", "evening"]:
        d = drift.get(stype, {})
        if not d:
            continue
        lines.append(f"\n  [{stype}スナップショット]")
        lines.append(f"  {'オッズ帯':>12}{'n':>7}{'p10':>8}{'p25':>8}{'中央値':>9}{'p75':>8}{'p90':>8}")
        lines.append("  " + "-" * 62)
        for band, stats in d.items():
            if stats is None:
                continue
            lines.append(
                f"  {band:>12}{stats['n']:>7}"
                f"{stats['p10']:>8.3f}{stats['p25']:>8.3f}"
                f"{stats['median']:>8.3f}{stats['p75']:>8.3f}{stats['p90']:>8.3f}"
            )
        lines.append("  → backtest上限値×中央ドリフト率 = 期待live ROI換算表（morningスナップ帯別）:")
        lines.append(_drift_conversion_table(d, result.get("main_total")))

    # 4. 必要標本数
    lines.append("\n▼ 4. 必要標本数の推定（ROI CI下限 >100% まで）")
    for label, req in result.get("required_n", {}).items():
        lines.append(f"  [{label}]")
        lines.append(f"    現在: {req.get('current_n', 0)}R  CI下限(現在): {req.get('ci_lo_now', 0):.1%}")
        lines.append(f"    必要追加R: {req.get('needed_additional', '?')}  {req.get('note', '')}")

    lines.append("\n" + "=" * 80)
    lines.append("  ※ 採否判断は CI・最大払戻除去後・中央払戻で総合判断。点推定ROI絶対値で判断しない。")
    lines.append("=" * 80)
    return "\n".join(lines)


def _drift_conversion_table(morning_drift: dict, main_summary: dict | None) -> str:
    """backtest上限値×ドリフト中央値の換算行を作る。"""
    if main_summary is None or main_summary["n"] == 0:
        return "    (主戦略 main の成績データなし)"
    bt_roi = main_summary.get("roi", 1.0)
    lines = []
    for band, stats in morning_drift.items():
        if stats is None:
            continue
        expected = bt_roi * stats["median"]
        lines.append(f"    {band:>12}: BT{bt_roi:.0%} × {stats['median']:.3f} → {expected:.0%}")
    return "\n".join(lines) if lines else "    (データなし)"


def _render_md(result: dict, date_from, date_to) -> str:
    lines = []
    lines.append("# live実測レポート[wt]")
    lines.append(f"\n**期間**: {date_from or '全期間'} 〜 {date_to or '最新'}  ")
    lines.append("払戻=実績値（picks_history）。backtest上限値（最終オッズ）とは異なる。\n")

    # 1. ランク別成績
    lines.append("## 1. ランク別成績\n")
    lines.append("| ランク | n | 的中率 | ROI | 95%CI | 除max ROI | 投資(円) | 払戻(円) |")
    lines.append("|--------|--:|------:|----:|:------|----------:|---------:|---------:|")
    rank_data = result["rank"]
    for rank in RANKS:
        s = rank_data.get(rank)
        if s is None:
            continue
        label = RANK_LABELS[rank]
        inv = result["rank_raw"][rank]["total_bet"]
        ret = result["rank_raw"][rank]["total_pay"]
        ci = f"[{s['ci_lo']:.1%},{s['ci_hi']:.1%}]"
        lines.append(f"| {label} | {s['n']} | {s['hit_rate']:.1%} | {s['roi']:.1%} | {ci} | {s['roi_ex_max']:.1%} | {inv:,} | {ret:,} |")
    main_s = result.get("main_total")
    if main_s:
        inv_m = result["rank_raw"].get("_main_inv", 0)
        ret_m = result["rank_raw"].get("_main_pay", 0)
        ci = f"[{main_s['ci_lo']:.1%},{main_s['ci_hi']:.1%}]"
        lines.append(f"| **SS+S+S+** | **{main_s['n']}** | **{main_s['hit_rate']:.1%}** | **{main_s['roi']:.1%}** | {ci} | **{main_s['roi_ex_max']:.1%}** | **{inv_m:,}** | **{ret_m:,}** |")
    lines.append("\n※ N<50 の層は標本不足で暫定。")

    # 2. タグ別
    lines.append("\n## 2. タグ別成績\n")
    lines.append("fav_mismatch は 2026-06-11 朝から記録開始。N が少ない段階では参考値。\n")
    lines.append("| タグ | n | 的中率 | ROI | 95%CI | 除max ROI |")
    lines.append("|------|--:|------:|----:|:------|----------:|")
    for label, s in result.get("tag", {}).items():
        if s["n"] == 0:
            continue
        ci = f"[{s['ci_lo']:.1%},{s['ci_hi']:.1%}]"
        lines.append(f"| {label} | {s['n']} | {s['hit_rate']:.1%} | {s['roi']:.1%} | {ci} | {s['roi_ex_max']:.1%} |")

    # 3. ドリフト
    lines.append("\n## 3. 朝→確定オッズ ドリフト率\n")
    lines.append("`final/morning`。1.0未満=backtest上限値より確定オッズが低い（下振れ）。\n")
    drift = result.get("drift", {})
    for stype in ["morning", "evening"]:
        d = drift.get(stype, {})
        if not d:
            continue
        lines.append(f"### {stype}スナップショット\n")
        lines.append("| オッズ帯 | n | p10 | p25 | 中央値 | p75 | p90 |")
        lines.append("|----------|--:|----:|----:|------:|----:|----:|")
        for band, stats in d.items():
            if stats is None:
                continue
            lines.append(
                f"| {band} | {stats['n']} | {stats['p10']:.3f} | {stats['p25']:.3f} | "
                f"{stats['median']:.3f} | {stats['p75']:.3f} | {stats['p90']:.3f} |"
            )

    # 換算表
    morning_d = drift.get("morning", {})
    main_s = result.get("main_total")
    if morning_d and main_s and main_s["n"] > 0:
        lines.append("\n### BT上限値×ドリフト中央値 → 期待live ROI換算表\n")
        bt_roi = main_s.get("roi", 1.0)
        lines.append(f"backtest live-backtest ROI: {bt_roi:.1%}\n")
        lines.append("| オッズ帯 | BT ROI | ドリフト中央値 | 期待live ROI |")
        lines.append("|----------|-------:|---------------:|-------------:|")
        for band, stats in morning_d.items():
            if stats is None:
                continue
            expected = bt_roi * stats["median"]
            lines.append(f"| {band} | {bt_roi:.1%} | {stats['median']:.3f} | {expected:.1%} |")

    # 4. 必要標本数
    lines.append("\n## 4. 必要標本数の推定\n")
    lines.append("ROI の bootstrap CI 下限が 100% を超えるために必要な追加レース数（現在の分布を所与）。\n")
    for label, req in result.get("required_n", {}).items():
        lines.append(f"**{label}**: 現在 {req.get('current_n', 0)}R /"
                     f" CI下限 {req.get('ci_lo_now', 0):.1%} /"
                     f" 必要追加 {req.get('needed_additional', '?')}R  \n{req.get('note', '')}\n")

    lines.append("\n---")
    lines.append("※ 採否判断は CI・最大払戻除去後・中央払戻で総合判断。点推定ROI絶対値で判断しない。")
    return "\n".join(lines)


# ── メイン ────────────────────────────────────────────────────────────


def build_report(date_from: str | None = None, date_to: str | None = None) -> dict:
    """レポートデータ（辞書）を構築して返す。テストから直接呼べる純粋な集計関数。"""
    picks = _load_picks(date_from, date_to)
    tags = _load_tags(date_from, date_to)

    # ランク別（7PLUS_R/7PLUS_ST/7PLUS_STP の3種のみ・_load_picks で購入確定行に絞り込み済み）
    rank_summaries = _rank_section(picks)
    rank_raw: dict[str, dict] = {}
    for rank in RANKS:
        ps = [p for p in picks if p["rank"] == rank]
        rank_raw[rank] = {
            "total_bet": sum(p["bet_amount"] for p in ps),
            "total_pay": sum(p["payout"] for p in ps),
        }

    # SS+S+S+ 合計（全購入ランク合算）
    main_picks = picks
    main_pays = [p["payout"] for p in main_picks]
    main_bets = [p["bet_amount"] for p in main_picks]
    main_total = roi_summary(main_pays, main_bets) if main_picks else None
    rank_raw["_main_inv"] = sum(main_bets)
    rank_raw["_main_pay"] = sum(main_pays)

    # タグ別
    tag_summaries = _tag_section(picks, tags)

    # ドリフト
    drift = _drift_section()

    # 必要標本数
    required: dict[str, dict] = {}
    if main_picks:
        required["SS+S+S+合算"] = _required_n_section(main_pays, main_bets)
    # fav_mismatch タグ別
    fm_picks = [p for p in main_picks
                if tags.get(p["race_key"].split("#", 1)[0], {}).get("fav_mismatch") is True]
    if fm_picks:
        required["fav_mismatch=True"] = _required_n_section(
            [p["payout"] for p in fm_picks],
            [p["bet_amount"] for p in fm_picks],
        )

    return {
        "picks": picks,
        "rank": rank_summaries,
        "rank_raw": rank_raw,
        "main_total": main_total,
        "tag": tag_summaries,
        "drift": drift,
        "required_n": required,
    }


def main():
    ap = argparse.ArgumentParser(description="live実測レポートCLI (picks_history route='wt')")
    ap.add_argument("--from", dest="date_from", default=None, help="開始日 YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="終了日 YYYY-MM-DD")
    ap.add_argument("--format", dest="fmt", default="text", choices=["text", "md"],
                    help="出力フォーマット (text/md)")
    args = ap.parse_args()

    result = build_report(args.date_from, args.date_to)

    if args.fmt == "md":
        print(_render_md(result, args.date_from, args.date_to))
    else:
        print(_render_text(result, args.date_from, args.date_to))


if __name__ == "__main__":
    main()
