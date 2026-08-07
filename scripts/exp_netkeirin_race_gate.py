"""配分後の「レース単位の下限」は要るか（2026-08-07・ユーザー依頼）。

ユーザー: 「金額配分した上で、そのレースの**期待値として下限の設定は不要か**。
            期待値が低いレースでも買うべきか見送るべきかの判断が要る。
            低くても**的中率が高ければ100%以上が確保できる**なら
            的中率を上げるという目的にはできる」

ユーザーの整理が構造的に正しい:

    実質的中率 = 的中率 × (1 − ガミ率)

配分を dutch 系にすると全点の払戻がそろうので **ガミ率は「保証下限」だけで決まる**。
よって最大化すべきは「**保証下限 >= 100% を満たすレースの中で的中率が高いもの**」であり、
期待値（ROI の期待）は目的関数ではない。両方をゲートとして測って確かめる。

朝8:00 に計算できる2つの量:

  保証下限 floor = min_i(賭け金_i × 想定着地オッズ_i) / 予算
      dutch 系なら全点ほぼ同じ ＝ 1 / Σ(1/想定オッズ)。**1.0 以上ならガミ無し**
  期待値   ev    = Σ_i モデル確率_i × 賭け金_i × 想定着地オッズ_i / 予算

想定着地オッズは `exp_netkeirin_landing_odds.py` の blend（朝オッズ×モデル）で作る。
⚠️ 読み取り専用。
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.stake_allocation import LANDING_LAMBDA  # noqa: E402
from scripts.exp_netkeirin_landing_odds import (  # noqa: E402
    BUDGET, DETAIL, MORNING_FROM, allocate, build, load_odds, load_top3,
    weights_for,
)

LAMBDA = LANDING_LAMBDA   # 本番と同じ値を使う（別に持つと必ず食い違う）
DRIFT = 0.85           # 朝→最終の下振れ中央値。**水準**の補正に使う（比率には効かない）
TAKEOUT = 0.25         # 三連複の控除率（book の水準推定に使う）


def naive_probs(p3):
    """35組の素朴確率（p3 の積を正規化）。"""
    out = {frozenset(c): p3[c[0]] * p3[c[1]] * p3[c[2]]
           for c in itertools.combinations(sorted(p3), 3)}
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else out


def estimate_landing(legs, pts, morning, p3, nprob):
    """想定着地オッズ ô_i を作る。

    比率は blend の重み、**水準（買う点の合成ブック）** は
      ・朝オッズがあれば Σ(1/朝オッズ) / DRIFT
      ・無ければ モデルの的中確率 / (1 − 控除率)
    から与える。ô_i = (Σw / w_i) / book。
    """
    has_m = all(morning.get(t) is not None for t in legs)
    w = weights_for("blend" if has_m else "model", legs, morning, p3, LAMBDA)
    tot_w = sum(w.values())
    if has_m:
        book = sum(1.0 / morning[t] for t in legs) / DRIFT
    else:
        book = sum(nprob.get(pts[t], 0.0) for t in legs) / (1 - TAKEOUT)
    book = max(book, 1e-6)
    return w, {t: (tot_w / w[t]) / book for t in legs}


def main():
    p3map = {r["rk"]: r["p3"] for r in pickle.load(open(DETAIL, "rb"))}
    with get_connection() as conn:
        recs = build(conn, p3map)
        bases = sorted({r["base"] for r in recs})
        board = load_odds(conn, bases, None)
        morning = load_odds(conn, bases, "morning")
        top3 = load_top3(conn, bases)
    snap_days = {r["date"] for r in recs if r["base"] in morning}
    recs = [r for r in recs if r["date"] in snap_days]

    rows = []
    book_err = []
    for r in recs:
        fo, mo = board.get(r["base"], {}), morning.get(r["base"], {})
        win = top3.get(r["base"])
        if win is None:
            continue
        pts = {t: frozenset({*r["axes"], t}) for t in r["legs"]}
        f = {t: fo.get(p) for t, p in pts.items()}
        if any(v is None for v in f.values()):
            continue
        m = {t: mo.get(p) for t, p in pts.items()}
        nprob = naive_probs(r["p3"])
        w, est = estimate_landing(r["legs"], pts, m, r["p3"], nprob)
        stakes = allocate(w, est)
        bet = sum(stakes.values())
        wt = next((t for t, p in pts.items() if p == win), None)
        ret = int(stakes[wt] * f[wt]) // 10 * 10 if wt is not None else 0
        floor = min(stakes[t] * est[t] for t in r["legs"]) / bet
        ev = sum(nprob.get(pts[t], 0.0) * stakes[t] * est[t] for t in r["legs"]) / bet
        rows.append(dict(date=r["date"], rank=r["rank"], bet=bet, ret=ret,
                         hit=int(ret > 0), rhit=int(ret >= bet and ret > 0),
                         floor=floor, ev=ev,
                         phit=sum(nprob.get(pts[t], 0.0) for t in r["legs"])))
        book_err.append((sum(1.0 / est[t] for t in r["legs"]),
                         sum(1.0 / f[t] for t in r["legs"])))

    days = len({r["date"] for r in rows})
    print(f"7車 {len(rows):,} レース / {days}日（{MORNING_FROM}〜2026-08-06）"
          f"  配分= blend λ={LAMBDA}")

    be = sorted(a - b for a, b in book_err)
    print(f"\n想定ブック − 実ブック: 中央値 {be[len(be)//2]:+.3f} / "
          f"25%点 {be[len(be)//4]:+.3f} / 75%点 {be[3*len(be)//4]:+.3f}")
    print("  （0 に近いほど『保証下限』の見積りが当たっている。"
          "正なら実際はもっと稼げる／負なら見込みより下振れ）")

    def report(title, key, thresholds, ge=True):
        print(f"\n-- {title} --")
        print(f"{'閾値':>8s} {'n':>6s} {'件/日':>7s} {'残存%':>7s} "
              f"{'的中%':>7s} {'実質的中%':>10s} {'ガミ率%':>8s} {'ROI%':>7s}")
        for thr in thresholds:
            g = [x for x in rows if (x[key] >= thr if ge else x[key] <= thr)]
            if not g:
                continue
            h = sum(x["hit"] for x in g)
            rh = sum(x["rhit"] for x in g)
            print(f"{thr:>8.2f} {len(g):>6,d} {len(g)/days:>7.2f} "
                  f"{100*len(g)/len(rows):>7.1f} {100*h/len(g):>7.2f} "
                  f"{100*rh/len(g):>10.2f} {100*(1-rh/max(h,1)):>8.1f} "
                  f"{100*sum(x['ret'] for x in g)/sum(x['bet'] for x in g):>7.1f}")

    report("保証下限 floor >= 閾値（ユーザー案の『100%以上が確保できるか』）",
           "floor", [0.0, 0.90, 1.00, 1.10, 1.20, 1.40, 1.60])
    report("期待値 ev >= 閾値", "ev", [0.0, 0.65, 0.70, 0.75, 0.80, 0.85])
    report("モデル的中確率 phit >= 閾値（比較用）",
           "phit", [0.0, 0.40, 0.50, 0.55, 0.60, 0.65])

    print("\n-- floor >= 1.0 を満たすレースの内訳 --")
    g = [x for x in rows if x["floor"] >= 1.0]
    for rank in sorted({x["rank"] for x in rows}):
        a = [x for x in rows if x["rank"] == rank]
        b = [x for x in g if x["rank"] == rank]
        if not a:
            continue
        print(f"  {rank:<5s} 全{len(a):>4,d}件 → floor>=1.0 は {len(b):>4,d}件"
              f"（{100*len(b)/len(a):>5.1f}%）"
              + (f" 的中 {100*sum(x['hit'] for x in b)/len(b):5.2f}% / "
                 f"実質 {100*sum(x['rhit'] for x in b)/len(b):5.2f}%" if b else ""))


if __name__ == "__main__":
    main()
