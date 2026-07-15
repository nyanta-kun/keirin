"""欠車判明レースの残存目ミスプライス仮説の検証（実精算方式・2026-07-16）。

旧方式で「SSの利益は欠車レース集中（ROI 4-11倍）」と見えていたのは
完走者ランキング（未来情報）による人工物と判明済み（keirin-survivor-bias-inflation）。
ただし裏の経済仮説は発走前情報のみで構成できるため実精算方式で独立検証する:
  「欠車は締切前に判明し盤面から該当組合せが消える。市場の再価格が不完全なら
    残存目に過大オッズが残り、欠車判明レースは +EV になる」

セグメント定義（発走前に判定可能）:
  出走表 n_entries=7 ∧ 最終オッズ盤面の掲載車 <7 → 欠車判明レース
  （wt_odds=最終盤面。締切直前の欠車はごく僅かに混入し得る点は限界として注記）

各買いルールを 欠車判明レース vs 完走盤面レース（対照群）で比較する。
使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_scratch_races_realistic_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.strategy_wt import ss_policy


def eval_rule(rows, min_leg_odds=0, gap12_min=None, gap23_min=None,
              gami_min=None, senbatsu_cut=False, stake=100):
    n = h = b = pp = 0
    per_race = []
    for r in rows:
        if gap12_min is not None and r["gap12"] < gap12_min:
            continue
        if gap23_min is not None and r["gap23_pt"] < gap23_min:
            continue
        if senbatsu_cut and ss_policy(r["race_type"])[0]:
            continue
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs:
            continue
        if gami_min is not None and min(legs.values()) < gami_min:
            continue
        legs = {t: o for t, o in legs.items() if o >= min_leg_odds}
        if not legs:
            continue
        pay = 0
        for t, o in legs.items():
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = (int(o * 100) // 10 * 10) * (stake // 100)
                break
        bet = len(legs) * stake
        n += 1
        h += 1 if pay > 0 else 0
        b += bet
        pp += pay
        per_race.append((bet, pay))
    return n, h, b, pp, per_race


def roi_ci(per_race, n_boot=2000, seed=7):
    if not per_race:
        return None
    rng = np.random.default_rng(seed)
    bets = np.array([x[0] for x in per_race])
    pays = np.array([x[1] for x in per_race])
    idx = rng.integers(0, len(bets), size=(n_boot, len(bets)))
    rois = pays[idx].sum(axis=1) / bets[idx].sum(axis=1)
    return np.percentile(rois, [2.5, 97.5])


RULES = [
    ("SS条件そのまま",   dict(gap12_min=E.SS_GAP12, gap23_min=E.GAP23_MIN, gami_min=E.SS_GAMI, senbatsu_cut=True)),
    ("ゲートなし全目",    dict()),
    ("gap12≥0.10全目",  dict(gap12_min=E.SS_GAP12)),
    ("ゲートなし目≥10",  dict(min_leg_odds=10)),
    ("gap12≥0.10目≥10", dict(min_leg_odds=10, gap12_min=E.SS_GAP12)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（欠車判明レース vs 完走盤面レース・実精算）", flush=True)
    model = E.load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = E.collect(model, f, t)
        # collect は n_entries==7 限定 → 盤面車数<7 = 欠車判明レース
        scratch = [r for r in rows if len(r["frames"]) < 7]
        full = [r for r in rows if len(r["frames"]) == 7]
        days = len({r["rk"][:8] for r in rows}) or 1
        print(f"\n===== {f} 〜 {t}（候補{len(rows)}R: 欠車判明{len(scratch)}R / 完走盤面{len(full)}R / {days}日） =====")
        for seg_label, seg in (("欠車判明", scratch), ("完走盤面(対照)", full)):
            print(f"  --- {seg_label} ---")
            print(f"  {'ルール':<18} {'R数':>5} {'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>8}  CI95%")
            for label, kw in RULES:
                n, h, b, pp, pr = eval_rule(seg, **kw)
                if n == 0:
                    print(f"  {label:<18} {'0':>5}")
                    continue
                ci = roi_ci(pr)
                ci_s = f"[{ci[0]:.0%},{ci[1]:.0%}]" if ci is not None else ""
                print(f"  {label:<18} {n:>5} {h/n:>6.1%} {b:>9,} {pp:>9,} {pp/b:>7.1%}  {ci_s}")


if __name__ == "__main__":
    main()
