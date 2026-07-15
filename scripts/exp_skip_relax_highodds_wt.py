"""見送り条件緩和×高オッズ目のみ購入の検証（実精算方式・2026-07-15）。

現行SS: レース単位 min(全目)≥7 ∧ gap12≥0.10 ∧ gap23≥1pt ∧ doc53ポリシー → 全目購入。
仮説: 見送りレースに中配当的中がある → レース見送り条件を外し、
      買い目単位で「三連複オッズ ≥ N倍」の目だけ購入したら ROI はどうか。

集計は実精算方式（盤面ランキング・落車失格=外れ計上・欠車=盤面から除外済み）。
eval_clean_split_wt.collect（gap12≥0.07 の候補プール）を共用。

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_skip_relax_highodds_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30 2026-07-01:2026-07-09
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.strategy_wt import ss_policy


def eval_legs(rows, min_leg_odds, gap12_min=None, gap23_min=None,
              use_policy=False, gami_min=None, stake=100):
    """買い目単位オッズフィルタ戦略の集計。

    returns (n_races, n_hits, n_legs, bet, payout)
      - レース条件: gap12_min / gap23_min / gami_min(min全目) / use_policy(選抜・4分戦見送り)
        None は条件なし
      - 買い目条件: 三連複オッズ >= min_leg_odds の目のみ購入
    """
    n = h = legs_total = b = pp = 0
    for r in rows:
        if gap12_min is not None and r["gap12"] < gap12_min:
            continue
        if gap23_min is not None and r["gap23_pt"] < gap23_min:
            continue
        if use_policy:
            skip, _ = ss_policy(r["race_type"], r["avg_gap"], r["n_lines"], r["all_solo"])
            if skip:
                continue
        all_legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                    for t in r["frames"][2:]}
        all_legs = {t: o for t, o in all_legs.items() if o}
        if not all_legs:
            continue
        if gami_min is not None and min(all_legs.values()) < gami_min:
            continue
        legs = {t: o for t, o in all_legs.items() if o >= min_leg_odds}
        if not legs:
            continue
        pay = 0
        for t, o in legs.items():
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = (int(o * 100) // 10 * 10) * (stake // 100)
                break
        n += 1
        legs_total += len(legs)
        h += 1 if pay > 0 else 0
        b += len(legs) * stake
        pp += pay
    return n, h, legs_total, b, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（実精算方式・買い目単位オッズフィルタ検証）", flush=True)
    model = E.load_model(args.model)

    variants = [
        # (label, kwargs)
        ("現行SS(基準)",              dict(min_leg_odds=0, gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN, use_policy=True, gami_min=E.SS_GAMI)),
        ("指数条件のみ・全目",         dict(min_leg_odds=0, gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN)),
        ("見送りなし・目≥7",          dict(min_leg_odds=7)),
        ("見送りなし・目≥10",         dict(min_leg_odds=10)),
        ("見送りなし・目≥15",         dict(min_leg_odds=15)),
        ("見送りなし・目≥20",         dict(min_leg_odds=20)),
        ("gap12≥0.10のみ・目≥10",     dict(min_leg_odds=10, gap12_min=E.SS_GAP12)),
        ("指数条件のみ・目≥10",        dict(min_leg_odds=10, gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN)),
        ("指数+ポリシー・目≥10",       dict(min_leg_odds=10, gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN, use_policy=True)),
        ("指数条件のみ・目≥15",        dict(min_leg_odds=15, gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN)),
    ]

    for w in args.windows:
        f, t = w.split(":")
        rows = E.collect(model, f, t)
        days = len({r["rk"][:8] for r in rows}) or 1
        print(f"\n===== {f} 〜 {t}（候補{len(rows)}R / 開催{days}日） =====")
        print(f"{'区分':<20} {'R数':>5} {'R/日':>5} {'点数':>6} {'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>7}")
        for label, kw in variants:
            n, h, nl, b, pp = eval_legs(rows, **kw)
            if n == 0:
                print(f"{label:<20} {'0':>5}")
                continue
            print(f"{label:<20} {n:>5} {n/days:>5.1f} {nl:>6} {h/n:>6.1%} {b:>9,} {pp:>9,} {pp/b:>6.1%}")


if __name__ == "__main__":
    main()
