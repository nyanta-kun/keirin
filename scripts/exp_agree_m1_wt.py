"""◎一致レース（WT◎=システム◎）の本命ランク検討（実精算・2026-07-16）。

目標: 的中率重視・人気決着許容で ROI100% 前後。
母集団: 7車・盤面7車 ∧ WT◎存在 ∧ システム◎（モデル1位）= WT◎。
軸1 = 一致◎。U/M と同じ2段:
  A) 2車目軸の網羅選定（WT○/モデル2位/市場2位/得点上位/同ライン相方/同ライン番手・先頭）
  B) 相手（流し目）のオッズ帯フィルタ:
     全目 / 大穴カット(目<=50) / 超人気目カット(目>=5) / 両方(5-50)
     ※人気決着許容のため低オッズ目は残す方向が基本

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_agree_m1_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_dark_pair_features_wt import collect
from exp_mismatch_m1_wt import load_marks, mate_of
from src.models.trainer import load_model

STAKE = 100


def prep_agree(races, marks):
    out = []
    for r in races:
        mk = marks.get(r["rk"], {})
        wt_top = next((fno for fno, pm in mk.items() if pm == 1), None)
        if wt_top is None:
            continue
        m1 = min(r["riders"], key=lambda x: r["riders"][x]["model_rank"])
        if m1 != wt_top:
            continue
        qi = {}
        for combo, ov in r["trio"].items():
            if 0 < ov < 9000:
                for fno in combo:
                    qi[fno] = qi.get(fno, 0.0) + 1.0 / ov
        mrank = {fno: k + 1 for k, (fno, _) in
                 enumerate(sorted(qi.items(), key=lambda x: -x[1]))}
        r2 = dict(r)
        r2["m1f"] = m1
        r2["mrank"] = mrank
        r2["marks"] = mk
        out.append(r2)
    return out


def trio_eval(r, a1, a2, lo=0.0, hi=1e9):
    bet = pay = 0
    for t in r["board"]:
        if t in (a1, a2):
            continue
        ov = r["trio"].get(frozenset({a1, a2, t}))
        if not ov or ov < lo or ov > hi:
            continue
        bet += STAKE
        if frozenset({a1, a2, t}) == r["top3"]:
            pay += int(ov * 100) // 10 * 10
    return bet, pay


def a2_wt_maru(r):
    return next((f for f, pm in r["marks"].items() if pm == 2), None)


def a2_model2(r):
    return min((f for f in r["riders"] if f != r["m1f"]),
               key=lambda x: r["riders"][x]["model_rank"])


def a2_mkt2(r):
    cands = [f for f in r["mrank"] if f != r["m1f"]]
    return min(cands, key=lambda f: r["mrank"][f]) if cands else None


def a2_rp_best(r):
    cands = [f for f in r["riders"] if f != r["m1f"]]
    return min(cands, key=lambda f: r["riders"][f]["rp_rank"]) if cands else None


AXES = [
    ("WT○(相手印)",   a2_wt_maru),
    ("モデル2位",      a2_model2),
    ("市場2位",       a2_mkt2),
    ("得点最上位(非◎)", a2_rp_best),
    ("同ライン相方",    lambda r: mate_of(r, r["m1f"])),
    ("同ライン逃相方",   lambda r: mate_of(r, r["m1f"], "逃")),
]

LEG_BANDS = [("全目", 0.0, 1e9), ("目<=50", 0.0, 50.0), ("目>=5", 5.0, 1e9), ("5-50", 5.0, 50.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（◎一致×本命軸・実精算）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect(model, f, t)
        marks = load_marks([r["rk"] for r in races])
        pop = prep_agree(races, marks)
        days = len({r["rk"][:8] for r in races}) or 1
        m1hit = sum(1 for r in pop if r["m1f"] in r["top3"])
        print(f"\n===== {f} 〜 {t}（7車 {len(races)}R・◎一致 {len(pop)}R = {len(pop)/days:.1f}R/日 / "
              f"一致◎の3着内率 {m1hit/len(pop):.1%}） =====")
        print(f"  {'2車目軸':<14} {'R数':>5} {'軸成立':>6}" +
              "".join(f" {b[0]:>10}" for b in LEG_BANDS))
        for name, fn in AXES:
            n = ph = 0
            band_bp = [[0, 0] for _ in LEG_BANDS]
            for r in pop:
                a2 = fn(r)
                if a2 is None or a2 == r["m1f"]:
                    continue
                n += 1
                if {r["m1f"], a2} <= r["top3"]:
                    ph += 1
                for i, (_, lo, hi) in enumerate(LEG_BANDS):
                    bb, pp = trio_eval(r, r["m1f"], a2, lo, hi)
                    band_bp[i][0] += bb
                    band_bp[i][1] += pp
            if n == 0:
                print(f"  {name:<14} {'—':>5}")
                continue
            cells = "".join(
                f" {bp[1]/bp[0]:>9.1%}" if bp[0] else f" {'—':>9}"
                for bp in band_bp)
            print(f"  {name:<14} {n:>5} {ph/n:>6.1%}{cells}")


if __name__ == "__main__":
    main()
