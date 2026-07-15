"""◎不一致レースにおけるシステム◎軸の三連複検証（実精算・2026-07-16）。

母集団: 7車・盤面7車 ∧ WT◎(prediction_mark==1)が存在 ∧ システム1位(モデル指数1位) ≠ WT◎。
軸1 = システム1位。U戦略と同じ2段アプローチ:
  A) 2車目軸の網羅選定（WT◎/モデル2位/市場1位/得点1位/同ライン相方/同ライン逃相方 + 関係性テーブル）
  B) 有望軸について相手（流し目）のオッズ帯フィルタ（>=15倍カット等）

集計は「不一致のみ」と「不一致∧波乱見込み（ent>=1.84∧盤面min>=4.3・凍結値）」の両母集団。

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_mismatch_m1_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_dark_pair_features_wt import collect, pos_label
from src.models.trainer import load_model
from src.database import get_connection
from src.strategy_wt import U_ENTROPY_MIN, U_MTO_MIN

STAKE = 100
LEG_MIN = 15.0


def load_marks(race_keys):
    marks = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, pm in c.execute(q, chunk):
                marks.setdefault(rk, {})[int(fno)] = pm
    return marks


def prep(races, marks):
    """不一致レースに m1 / wt_top / 市場順位 / 得点1位を付与して返す。"""
    out = []
    for r in races:
        mk = marks.get(r["rk"], {})
        wt_top = next((fno for fno, pm in mk.items() if pm == 1), None)
        if wt_top is None:
            continue
        m1 = min(r["riders"], key=lambda x: r["riders"][x]["model_rank"])
        if m1 == wt_top:
            continue
        qi = {}
        for combo, ov in r["trio"].items():
            if 0 < ov < 9000:
                for fno in combo:
                    qi[fno] = qi.get(fno, 0.0) + 1.0 / ov
        mrank = {fno: k + 1 for k, (fno, _) in
                 enumerate(sorted(qi.items(), key=lambda x: -x[1]))}
        rp1 = min(r["riders"], key=lambda x: r["riders"][x]["rp_rank"])
        r2 = dict(r)
        r2["m1f"] = m1
        r2["wt_top"] = wt_top
        r2["mrank"] = mrank
        r2["rp1f"] = rp1
        out.append(r2)
    return out


def mate_of(r, fno, style_filter=None):
    """fno と同ラインの相方（先頭⇔番手）。style_filter 指定時は脚質も要求。"""
    me = r["riders"][fno]
    if me["lg"] is None or me["lsize"] < 2:
        return None
    want = 1 if me["lpos"] == 2 else 2
    for o, orr in r["riders"].items():
        if o == fno or orr["lg"] != me["lg"]:
            continue
        if orr["lpos"] == want and (style_filter is None or orr["style"] == style_filter):
            return o
    if style_filter is not None:  # 位置指定で見つからなければ同ラインの該当脚質
        for o, orr in r["riders"].items():
            if o != fno and orr["lg"] == me["lg"] and orr["style"] == style_filter:
                return o
    return None


def trio_eval(r, a1, a2, leg_min=0.0):
    bet = pay = 0
    for t in r["board"]:
        if t in (a1, a2):
            continue
        ov = r["trio"].get(frozenset({a1, a2, t}))
        if not ov or ov < leg_min:
            continue
        bet += STAKE
        if frozenset({a1, a2, t}) == r["top3"]:
            pay += int(ov * 100) // 10 * 10
    return bet, pay


AXES = [
    ("WT◎",         lambda r: r["wt_top"]),
    ("モデル2位",     lambda r: min((f for f in r["riders"] if f != r["m1f"]),
                                 key=lambda x: r["riders"][x]["model_rank"])),
    ("市場1位",      lambda r: min(r["mrank"], key=lambda k: r["mrank"][k])),
    ("得点1位",      lambda r: r["rp1f"]),
    ("同ライン相方",   lambda r: mate_of(r, r["m1f"])),
    ("同ライン逃相方",  lambda r: mate_of(r, r["m1f"], "逃")),
]


def run_pop(pop, label, days):
    m1hit = sum(1 for r in pop if r["m1f"] in r["top3"])
    print(f"  ◇ {label}: {len(pop)}R ({len(pop)/days:.1f}R/日) / システム◎の3着内率 "
          f"{m1hit/len(pop):.1%}" if pop else f"  ◇ {label}: 0R")
    if not pop:
        return
    print(f"    {'2車目軸':<12} {'R数':>4} {'軸成立':>6} {'全目ROI':>8} {'目>=15ROI':>9}")
    for name, fn in AXES:
        n = ph = b0 = p0 = b15 = p15 = 0
        for r in pop:
            a2 = fn(r)
            if a2 is None or a2 == r["m1f"]:
                continue
            n += 1
            if {r["m1f"], a2} <= r["top3"]:
                ph += 1
            bb, pp = trio_eval(r, r["m1f"], a2)
            b0 += bb; p0 += pp
            bb, pp = trio_eval(r, r["m1f"], a2, LEG_MIN)
            b15 += bb; p15 += pp
        if not b0:
            print(f"    {name:<12} {'—':>4}")
            continue
        r15 = f"{p15/b15:>8.1%}" if b15 else "       —"
        print(f"    {name:<12} {n:>4} {ph/n:>6.1%} {p0/b0:>8.1%} {r15}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（◎不一致×システム◎軸・実精算）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect(model, f, t)
        marks = load_marks([r["rk"] for r in races])
        pop_all = prep(races, marks)
        days = len({r["rk"][:8] for r in races}) or 1
        pop_upset = [r for r in pop_all
                     if r["entropy"] >= U_ENTROPY_MIN and r["mto"] >= U_MTO_MIN]
        print(f"\n===== {f} 〜 {t}（7車 {len(races)}R・◎不一致 {len(pop_all)}R） =====")
        run_pop(pop_all, "不一致のみ", days)
        run_pop(pop_upset, "不一致∧波乱見込み(凍結ゲート)", days)


if __name__ == "__main__":
    main()
