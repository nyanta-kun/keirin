"""7C の3列目を「実力差」で可変にする（2026-08-07）。

ユーザー要望:
  「相手の実力差の考慮ができればと思います。固定ではなく、3着内に入れなそうな
    ところを除外として**点数可変**とします。またその場合のROI」

3列目候補は軸2車を除く5車。ここから「3着内に入れなそうな車」を落として点数を
可変にする。落とし方は**オッズ非依存**（朝8:00に確定する必要があるため）で、
モデル3着内率 p3 だけを使う。

  A 絶対閾値   : p3 >= T の車だけ買う
  B 相対閾値   : p3 >= α ×（相手5車の最大 p3）の車だけ買う
  C ギャップ切り: 相手5車を p3 降順に並べ、**最大の落差**の上側だけ買う
  D シェア累積 : 相手5車の p3 を正規化し、上位から累積 X% に達するまで買う

比較対象は固定点数（総流し5点 / 上位4車 / 上位3車）。
指標は 的中率・**実質的中率（払戻>=投資）**・ガミ率・ROI・平均点数・収支。

⚠️ オッズは精算にのみ使う。選別・相手選びには一切使わない。読み取りのみ。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100
P3_SUM_MIN = 1.5734          # 網羅25%の選別閾値（確定済み）
MIN_LEGS = 1                 # 0点（＝実質見送り）を許すかどうかの下限

# 🔴 1レースの予算枠（ユーザー指定 2026-08-07）。点数で均等割りし 100円単位へ切り捨てる
# （7H1 の RANK_7H1_BUDGET_* / RANK_7H1_UNIT と同じ方式）。
# 点数が**レースごとに変わる**ルールでは、固定額/点とROIが一致しない:
#   固定額/点 … 点数の多いレースほど分母に強く効く（重み付き平均）
#   予算枠    … どのレースも投資10,000円で等しく効く（レース単位の平均）
# ユーザーの実運用は予算枠方式なので、こちらを主指標として読むこと。
BUDGET = 10000
UNIT = 100


def unit_stake(k: int) -> int:
    """点数 k のときの1点あたり賭け金（100円単位・切り捨て）。"""
    return max(UNIT, (BUDGET // k) // UNIT * UNIT)


def legs_abs(others: list[int], p3: dict, T: float) -> list[int]:
    return [c for c in others if p3[c] >= T]


def legs_rel(others: list[int], p3: dict, a: float) -> list[int]:
    top = max(p3[c] for c in others)
    return [c for c in others if p3[c] >= a * top]


def legs_gap(others: list[int], p3: dict, _=None) -> list[int]:
    """p3 降順に並べ、隣接する最大の落差の**上側**を買う（最低1車）。"""
    s = sorted(others, key=lambda c: -p3[c])
    if len(s) < 2:
        return s
    gaps = [(p3[s[i]] - p3[s[i + 1]], i) for i in range(len(s) - 1)]
    _, i = max(gaps)
    return s[:i + 1]


def legs_share(others: list[int], p3: dict, x: float) -> list[int]:
    s = sorted(others, key=lambda c: -p3[c])
    tot = sum(p3[c] for c in s)
    if tot <= 0:
        return s
    out, acc = [], 0.0
    for c in s:
        out.append(c)
        acc += p3[c] / tot
        if acc >= x:
            break
    return out


RULES: list[tuple[str, object, object]] = [
    ("固定 総流し5点", lambda o, p, _: o, None),
    ("固定 上位4車", lambda o, p, _: sorted(o, key=lambda c: -p[c])[:4], None),
    ("固定 上位3車", lambda o, p, _: sorted(o, key=lambda c: -p[c])[:3], None),
    *[(f"A 絶対 p3>={t:.0%}", legs_abs, t) for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)],
    *[(f"B 相対 >={a:.0%}×最強", legs_rel, a) for a in (0.4, 0.5, 0.6, 0.7, 0.8)],
    ("C 最大ギャップ切り", legs_gap, None),
    *[(f"D 累積{x:.0%}まで", legs_share, x) for x in (0.6, 0.7, 0.8, 0.9)],
]


def build(races) -> pd.DataFrame:
    rows = []
    for r in races:
        p3, board = r["p3"], r["board"]
        if len(p3) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2 = ranked[0], ranked[1]
        if p3[a1] + p3[a2] < P3_SUM_MIN:
            continue
        others = ranked[2:]
        rec = dict(rk=r["rk"], date=r["date"], sum2=p3[a1] + p3[a2],
                   win="確認" if r["date"] <= CONFIRM_END else "掃引",
                   third_in=next(iter(top3 - {a1, a2})) if len(top3 - {a1, a2}) == 1 else None,
                   axes_ok=int(a1 in top3 and a2 in top3))
        for name, fn, arg in RULES:
            legs = fn(others, p3, arg)
            if len(legs) < MIN_LEGS:
                legs = sorted(others, key=lambda c: -p3[c])[:MIN_LEGS]
            combos = [frozenset({a1, a2, t}) for t in legs
                      if frozenset({a1, a2, t}) in board]
            if not combos:
                continue
            hit = top3 in combos
            k = len(combos)
            od = board[top3]
            # ① 予算枠方式（本番・1レース10,000円を点数で均等割り）
            s = unit_stake(k)
            bet_b = k * s
            ret_b = (round(od * s) // 10 * 10) if hit else 0
            # ② 固定100円/点（従来の比較用）
            bet_f = k * STAKE
            ret_f = (round(od * 100) // 10 * 10) if hit else 0
            rec[f"h__{name}"] = int(hit)
            rec[f"k__{name}"] = k
            rec[f"b__{name}"] = bet_b
            rec[f"r__{name}"] = ret_b
            rec[f"n__{name}"] = int(hit and ret_b >= bet_b)
            rec[f"bf__{name}"] = bet_f
            rec[f"rf__{name}"] = ret_f
            rec[f"o__{name}"] = od if hit else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def report(d: pd.DataFrame, wname: str) -> pd.DataFrame:
    days = d.date.nunique()
    out = []
    for name, _, _a in RULES:
        h, k_, b, r_, n_, o = (f"h__{name}", f"k__{name}", f"b__{name}",
                               f"r__{name}", f"n__{name}", f"o__{name}")
        if h not in d.columns:
            continue
        g = d.dropna(subset=[h])
        hits = g[g[h] == 1]
        out.append(dict(
            rule=name, pts=g[k_].mean(),
            hit=100 * g[h].mean(), net=100 * g[n_].mean(),
            gami=100 * (g[h].sum() - g[n_].sum()) / max(g[h].sum(), 1),
            roi=100 * g[r_].sum() / g[b].sum(),
            roi_flat=100 * g[f"rf__{name}"].sum() / g[f"bf__{name}"].sum(),
            avg=float(hits[o].mean()) if len(hits) else 0.0,
            bet_day=g[b].sum() / days, pl_day=(g[r_].sum() - g[b].sum()) / days,
        ))
    df = pd.DataFrame(out)
    print(f"\n=== {wname}  n={len(d):,}R / {days}日 "
          f"({len(d)/days:.2f}件/日)  ※ROIは1レース{BUDGET:,}円の予算枠方式 ===")
    print(f"{'絞り方':22s} {'平均点':>6s} {'的中%':>6s} {'実質的中%':>8s} "
          f"{'ガミ%':>6s} {'ROI%':>6s} {'(参考)100円/点':>12s} {'的中時平均':>8s} "
          f"{'投資/日':>9s} {'収支/日':>10s}")
    for _, x in df.iterrows():
        print(f"{x.rule:22s} {x.pts:6.2f} {x.hit:6.2f} {x.net:8.2f} "
              f"{x.gami:6.1f} {x.roi:6.1f} {x.roi_flat:12.1f} {x.avg:8.2f} "
              f"{x.bet_day:9,.0f} {x.pl_day:+10,.0f}")
    return df


def main() -> None:
    d = build(pickle.load(open(DETAIL, "rb")))
    ev, cf = d[d.win == "掃引"], d[d.win == "確認"]
    a = report(ev, "評価窓 2025-07〜2026-08")
    b = report(cf, "確認窓 2024-07〜2025-06")

    m = a.merge(b, on="rule", suffixes=("_e", "_c"))
    m["roi_min"] = m[["roi_e", "roi_c"]].min(axis=1)
    print("\n=== 両窓そろって良いもの（ROI の低い方でソート）===")
    print(f"{'絞り方':22s} {'点':>5s} | {'評価 的中%':>9s} {'ROI%':>6s} | "
          f"{'確認 的中%':>9s} {'ROI%':>6s} | {'両窓最小ROI':>10s}")
    for _, x in m.sort_values("roi_min", ascending=False).iterrows():
        print(f"{x.rule:22s} {x.pts_e:5.2f} | {x.hit_e:9.2f} {x.roi_e:6.1f} | "
              f"{x.hit_c:9.2f} {x.roi_c:6.1f} | {x.roi_min:10.1f}")

    # 点数分布（可変ルールが実際どうばらけるか）
    print("\n=== 可変ルールの点数分布（評価窓）===")
    for name in ("A 絶対 p3>=25%", "B 相対 >=50%×最強", "C 最大ギャップ切り",
                 "D 累積80%まで"):
        col = f"k__{name}"
        if col not in ev.columns:
            continue
        vc = ev[f"k__{name}"].value_counts().sort_index()
        dist = "  ".join(f"{int(k)}点:{100*v/len(ev):4.1f}%" for k, v in vc.items())
        print(f"  {name:22s} {dist}")

    # 3着に来た車が「除外された側」だった割合＝取りこぼしの内訳
    print("\n=== 取りこぼし: 軸2車は当たったのに3列目で外した率（評価窓）===")
    ok = ev[ev.axes_ok == 1]
    print(f"  軸2車がともに3着内: {len(ok):,}R ({100*len(ok)/len(ev):.2f}%)")
    for name, _, _a in RULES:
        h = f"h__{name}"
        if h not in ok.columns:
            continue
        miss = 100 * (1 - ok[h].mean())
        print(f"    {name:22s} 3列目で取りこぼし {miss:5.2f}%")


if __name__ == "__main__":
    main()
