"""rebuild（月次vintage・本番判定関数）の出力を確認窓の既知数値と突合する（2026-08-06）。

## なぜ必要か

7A は 2026-08-05 の再定義（A群のみ・PR#10）以降、picks_history が再構築されて
おらず旧定義（A群+E群の混合）のまま残っている。7SS に至っては 0 行。
これを本番と同じ条件で作り直す前に、**rebuild 経路が確認窓の既知数値
（7A 6.28件/日・的中53.2%・ROI80.8% / 7SS 1.90件/日・的中41.2%・ROI85.9%）を
再現するか**を先に検算する。

⚠️ 完全一致は期待しない。既知数値は `exp_axis_rule_decomposition.py` 系の
   ハーネスで出したもので、rebuild とは次の3点が構造的に違う:
     1. 軸規則: exp は `a1_pw`（全7車の argmax pred_win）/ rebuild は本番の
        `rank_7s_select_axis`（pred_win上位3 ∩ pred_prob上位3 の重なりから選定）
     2. モデル: exp は窓ごとに自前で学習 / rebuild は月次凍結 vintage
     3. 特徴量の構築範囲: exp は 2022-12〜 の長期間 / rebuild は月単位
        （なお live は当日1日のみ＝ rebuild の方が live に近い。src/cli/main.py）
   本スクリプトが答えるのは「**同じ順序関係・同じ水準帯に収まるか**」であって
   小数点の一致ではない。

## 出すもの

確認窓 2024-07〜2025-06 の12ヶ月を月次vintageで回し、7S / A群(=7A) /
E群同ライン(=7SS) / E群別ライン(空白1) / 両方不合格(空白2) / overlap2 に
分解して件/日・的中・ROI・的中中央値を月別と合算で出す。

DB書き込みなし（読み取りのみ）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_7a_7ss_rebuild_reconcile.py
    PYTHONPATH=. .venv/bin/python scripts/exp_7a_7ss_rebuild_reconcile.py --two-head
        （bad モデルを渡さず旧2ヘッド軸で回す＝3ヘッド化の影響を切り分ける）
"""
from __future__ import annotations

import argparse
import statistics
import sys
from calendar import monthrange
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_wt_candidate_cache import month_candidates  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
)
from src.wt_vintage_config import bad_model_name, monthly_windows  # noqa: E402

CONFIRM_FROM, CONFIRM_TO = "2024-07-01", "2025-06-30"

# 既知数値（確認窓・memory keirin_rank_7ss_and_7b_2026_08_05 / keirin_7car_coverage_gaps）
KNOWN_BY_GROUP = {
    "7S":            (3.68, 41.0, 84.4),
    "A群(=7A)":      (6.28, 53.2, 80.8),
    "E群 同L(=7SS)": (1.90, 41.2, 85.9),
    "E群 別L(空白1)": (5.44, 26.7, 75.0),
}

GROUPS = ["7S", "A群(=7A)", "E群 同L(=7SS)", "E群 別L(空白1)",
          "両方不合格(空白2)", "overlap2"]


def classify(c: dict) -> str:
    """候補を被覆マップ上の群へ振り分ける（strategy_wt の定義から導出）。"""
    if c["wt_overlap_n"] not in (0, 1):
        return "overlap2"
    ok_a = c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
    ok_e = c["entropy"] <= RANK_7S_ENTROPY_MAX
    if ok_a and ok_e:
        return "7S"
    if (not ok_a) and ok_e:
        return "A群(=7A)"
    if ok_a and (not ok_e):
        return "E群 同L(=7SS)" if c["same_line"] else "E群 別L(空白1)"
    return "両方不合格(空白2)"


def settle(c: dict):
    """三連複 軸2車+総流し（盤面にある目だけ）を精算する。backfill と同一規則。"""
    a1, a2 = c["axis1"], c["axis2"]
    combos = [frozenset({a1, a2, x}) for x in c["others"]
              if frozenset({a1, a2, x}) in c["trio"]]
    if not combos:
        return None
    hit = c["actual_top3"] in combos
    # 払戻は backfill と同じく trio 最終オッズ×100（10円未満切り捨て）
    pay = (round(c["trio"][c["actual_top3"]] * 100) // 10 * 10) if hit else 0
    return len(combos) * 100, pay, hit


def agg(rows: list, days: int) -> dict | None:
    rows = [r for r in rows if r]
    if not rows:
        return None
    bet = sum(r[0] for r in rows)
    pay = sum(r[1] for r in rows)
    hits = [r for r in rows if r[2]]
    return dict(
        n=len(rows), per_day=len(rows) / days,
        hit=100 * len(hits) / len(rows),
        roi=100 * pay / bet if bet else 0.0,
        med=statistics.median([r[1] / r[0] for r in hits]) if hits else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-head", action="store_true",
                    help="bad モデルを渡さず旧2ヘッド軸で回す（3ヘッド化の影響切り分け）")
    ap.add_argument("--from", dest="date_from", default=CONFIRM_FROM)
    ap.add_argument("--to", dest="date_to", default=CONFIRM_TO)
    args = ap.parse_args()

    windows = [w for w in monthly_windows()
               if w[0] >= args.date_from and w[1] <= args.date_to]
    axis_label = "2ヘッド(旧)" if args.two_head else "3ヘッド(本番)"
    print(f"確認窓 {args.date_from}〜{args.date_to}  {len(windows)}ヶ月  軸={axis_label}",
          flush=True)

    by_group: dict[str, list] = {g: [] for g in GROUPS}
    total_days = 0
    for date_from, date_to, eval_model, win_model in windows:
        y, m = int(date_from[:4]), int(date_from[5:7])
        days = monthrange(y, m)[1]
        total_days += days
        bad_model = None if args.two_head else bad_model_name(eval_model)
        cands = month_candidates(date_from, date_to, eval_model, win_model, bad_model)

        month_rows: dict[str, list] = {g: [] for g in GROUPS}
        for c in cands:
            month_rows[classify(c)].append(settle(c))
        for g in GROUPS:
            by_group[g].extend(month_rows[g])

        a = agg(month_rows["A群(=7A)"], days)
        s = agg(month_rows["E群 同L(=7SS)"], days)
        print(f"  {date_from[:7]}  候補{len(cands):5}  "
              f"7A n={a['n']:4} 的中{a['hit']:5.1f}% ROI{a['roi']:6.1f}%   "
              f"7SS n={s['n']:3} 的中{s['hit']:5.1f}% ROI{s['roi']:6.1f}%"
              if a and s else f"  {date_from[:7]}  候補{len(cands):5}  (集計不能)",
              flush=True)

    print(f"\n===== 合算（{total_days}日） =====")
    print(f"{'群':<16}{'n':>6}{'件/日':>8}{'的中':>8}{'ROI':>8}{'的中中央値':>10}   既知(確認窓)")
    for g in GROUPS:
        a = agg(by_group[g], total_days)
        if not a:
            print(f"{g:<16}{'n=0':>6}")
            continue
        known = KNOWN_BY_GROUP.get(g)
        kn = (f"  {known[0]:.2f}/日 {known[1]:.1f}% {known[2]:.1f}%" if known else "  —")
        print(f"{g:<16}{a['n']:>6}{a['per_day']:>8.2f}{a['hit']:>7.1f}%"
              f"{a['roi']:>7.1f}%{a['med']:>9.2f}倍{kn}")


if __name__ == "__main__":
    main()
