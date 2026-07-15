"""doc53 統合ポリシー（選抜カット/4分戦カット/格差増額）の実精算方式での再検証（2026-07-16）。

doc53 採用時（2026-07-12）の検証は旧・完走者ランキング方式（生存者バイアス込み）だったため、
実精算方式（盤面ランキング・落車失格=外れ計上）で各成分の有効性を再検証する。

SS数値条件（min全目≥7 ∧ gap12≥0.10 ∧ gap23≥1pt・全目購入）を通過したレースを母集団に:
  1. セグメント別成績（選抜 / 4分戦 / 格差≥1.5 / その他）
  2. カット・増額の組み合わせ別の合計成績

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_ss_policy_realistic_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.strategy_wt import SS_LINE_GAP_BOOST, SS_N_LINES_SKIP, is_senbatsu


def base_races(rows):
    """SS数値条件を通過したレースの (r, legs, pay100) リスト。pay100=100円/点時の払戻。"""
    out = []
    for r in rows:
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < E.SS_GAMI:
            continue
        if r["gap12"] < E.SS_GAP12 or r["gap23_pt"] < E.GAP23_MIN:
            continue
        pay = 0
        for t, o in legs.items():
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = int(o * 100) // 10 * 10
                break
        out.append((r, legs, pay))
    return out


def _seg(r):
    """doc53 の判定順にセグメント分類（選抜 → 4分戦 → 格差boost → その他）。"""
    if is_senbatsu(r["race_type"]):
        return "選抜"
    if r["n_lines"] is not None and r["n_lines"] >= SS_N_LINES_SKIP and not r["all_solo"]:
        return "4分戦"
    if r["avg_gap"] is not None and r["avg_gap"] >= SS_LINE_GAP_BOOST:
        return "格差≥1.5"
    return "その他"


def _agg(items, stake_fn=lambda seg: 100):
    n = h = b = pp = 0
    for r, legs, pay in items:
        stake = stake_fn(_seg(r))
        n += 1
        h += 1 if pay > 0 else 0
        b += len(legs) * stake
        pp += pay * (stake // 100)
    return n, h, b, pp


def _roi_ci(items, stake_fn, n_boot=2000, seed=7):
    """レース単位ブートストラップの ROI 95%CI。"""
    if not items:
        return None
    rng = np.random.default_rng(seed)
    bets, pays = [], []
    for r, legs, pay in items:
        stake = stake_fn(_seg(r))
        bets.append(len(legs) * stake)
        pays.append(pay * (stake // 100))
    bets, pays = np.array(bets), np.array(pays)
    idx = rng.integers(0, len(bets), size=(n_boot, len(bets)))
    rois = pays[idx].sum(axis=1) / bets[idx].sum(axis=1)
    return np.percentile(rois, [2.5, 97.5])


def _print_line(label, n, h, b, pp, days, ci=None):
    if n == 0:
        print(f"  {label:<24} {'0':>5}")
        return
    ci_s = f"  CI[{ci[0]:.0%},{ci[1]:.0%}]" if ci is not None else ""
    print(f"  {label:<24} {n:>5} {n/days:>5.2f} {h/n:>6.1%} {b:>9,} {pp:>9,} {pp/b:>7.1%}{ci_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（実精算方式・doc53ポリシー成分検証）", flush=True)
    model = E.load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = E.collect(model, f, t)
        days = len({r["rk"][:8] for r in rows}) or 1
        base = base_races(rows)
        print(f"\n===== {f} 〜 {t}（SS数値条件通過 {len(base)}R / 開催{days}日） =====")
        print(f"  {'区分':<24} {'R数':>5} {'R/日':>5} {'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>8}")

        # 1) セグメント別（100円/点固定）
        print("  --- セグメント別（100円/点） ---")
        for seg in ("選抜", "4分戦", "格差≥1.5", "その他"):
            items = [x for x in base if _seg(x[0]) == seg]
            _print_line(f"[{seg}]", *_agg(items), days)

        # 2) ポリシー組み合わせ
        print("  --- ポリシー組み合わせ ---")
        flat = lambda seg: 100
        boost = lambda seg: 200 if seg == "格差≥1.5" else 100
        variants = [
            ("ポリシーなし(数値のみ)", base, flat),
            ("選抜カットのみ",   [x for x in base if _seg(x[0]) != "選抜"], flat),
            ("4分戦カットのみ",  [x for x in base if _seg(x[0]) != "4分戦"], flat),
            ("両カット(増額なし)", [x for x in base if _seg(x[0]) not in ("選抜", "4分戦")], flat),
            ("現行フル(両カット+増額)", [x for x in base if _seg(x[0]) not in ("選抜", "4分戦")], boost),
        ]
        for label, items, sfn in variants:
            ci = _roi_ci(items, sfn)
            _print_line(label, *_agg(items, sfn), days, ci)


if __name__ == "__main__":
    main()
