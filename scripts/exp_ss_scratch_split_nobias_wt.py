"""無バイアス版 SS の欠車有無レース分割（2026-07-15）。

keirin-ss-budget-structure の「SS利益は欠車レース集中（フル7車はROI0.71）」は
バイアスあり評価（完走者ランキング）下の知見。無バイアス版で同じ分解が
成立するかを TEST 2026-04〜06 / FWD 7月 で確認する。

分類:
  full   : 7車全員完走
  dnf    : 出走したが非完走(0/NULL)あり（落車/失格/直前欠車）
  absent : 出走表から行ごと消えている（事前欠車・rows<7）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.trainer import load_model
from eval_clean_split_nobias_wt import collect_nobias
from eval_clean_split_wt import SS_GAP12, SS_GAMI, GAP23_MIN
from src.strategy_wt import ss_policy


def eval_ss_split(rows, mode="refund"):
    cats = {}
    for r in rows:
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < SS_GAMI:
            continue
        if r["gap12"] < SS_GAP12 or r["gap23_pt"] < GAP23_MIN:
            continue
        skip, stake = ss_policy(r["race_type"], r["avg_gap"], r["n_lines"], r["all_solo"])
        if skip:
            continue
        if len(r["frames"]) < 7:
            cat = "absent(事前欠車)"
        elif r["nonfin"]:
            cat = "dnf(非完走あり)"
        else:
            cat = "full(7車完走)"
        pay = bet = 0
        hit = False
        for t, o in legs.items():
            combo = frozenset({r["p1"], r["p2"], t})
            if combo & r["nonfin"]:
                if mode == "refund":
                    continue
            bet += stake
            if combo == r["top3"]:
                pay += int(o * stake)
                hit = True
        if bet == 0:
            continue
        c = cats.setdefault(cat, [0, 0, 0, 0])
        c[0] += 1
        c[1] += 1 if hit else 0
        c[2] += bet
        c[3] += pay
    return cats


def main():
    model = load_model("lgbm_wt_2026h1_eval")
    for f, t in (("2026-04-01", "2026-06-30"), ("2026-07-01", "2026-07-10")):
        rows = collect_nobias(model, f, t)
        print(f"\n===== {f}〜{t} SS 欠車有無分割（返還モデル）=====")
        cats = eval_ss_split(rows)
        tot = [0, 0, 0, 0]
        for cat in ("full(7車完走)", "dnf(非完走あり)", "absent(事前欠車)"):
            c = cats.get(cat)
            if not c:
                print(f"{cat:<16} 0R")
                continue
            n, h, b, pp = c
            for i, v in enumerate(c):
                tot[i] += v
            print(f"{cat:<16} {n:>4}R 的中{h/n:>6.1%} 投資{b:>8,} 払戻{pp:>8,} ROI{pp/b:>7.1%}")
        n, h, b, pp = tot
        if n:
            print(f"{'合計':<16} {n:>4}R 的中{h/n:>6.1%} 投資{b:>8,} 払戻{pp:>8,} ROI{pp/b:>7.1%}")


if __name__ == "__main__":
    main()
