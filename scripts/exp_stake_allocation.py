"""レース内の傾斜配分でROIが上がるかを検証する（2026-08-05・ユーザー依頼）。

ユーザー依頼「現在の各モデルは二軸総流しの均等としているが、**1レース10,000円**の
傾斜配分をオッズ・相手の3着内確率配分などで行えばROI向上が可能ではないか」。

## 構造（これを踏まえないと設計を誤る）

三連複・軸2車固定なら **3着に入る相手は1車＝的中する点は必ず1点だけ**。よって

    ROI = Σ_races (当たった点への賭け金 × その点のオッズ) / Σ_races 10,000

**等分が最適でなくなるのは、点ごとの期待値 p_i×o_i にばらつきがあるときだけ**で、
極論すれば最適配分は「期待値最大の1点に全額」という degenerate 解になる。つまり
本検証の実体は「**点ごとの期待値差を事前に見抜けるか**」である。

⚠️ ROI はスケール不変なので **10,000円という金額自体は ROI に影響しない**
（100円単位の端数も3〜5点なら効かない）。効くのは配分の形だけ。金額は
実運用の見え方を揃えるために使う。

## 配分3系統

| 系統 | 重み | 必要なもの | 検証していること |
|---|---|---|---|
| A オッズのみ | `w ∝ o^k` | オッズ | **相手3〜5車の中に人気-穴バイアスがあるか**。モデル不要＝逆選択が原理的に起きない。k=−1 は均等払戻(dutching)・k=0 は等分・k>0 は穴寄せ |
| B モデル確率のみ | `w ∝ p3^k` | モデル | **オッズ不要＝入稿時にも使える**。相手の3着内確率の序列に配分価値があるか |
| C モデルEV | `w ∝ (p3·o)^k` / 上位1点集中 | 両方 | 上値は最大だが**最も危険** |

**⚠️ C への事前警告**: 確定済みの事実として「**モデルはオッズに予測精度で負ける**」
（[[keirin_clean_baseline_market_efficiency_2026_07_30]]）。モデルEVで傾斜をかけると
**モデルが確率を過大評価している点にだけ金が寄る**逆選択が構造的に起きる。
期待は持たずに測る。

⚠️ 検証は**最終オッズ**。実購入は発走15分前なので、**オッズ依存度が高い配分ほど
実運用との乖離が大きい**（A・C は影響を受け、B は受けない）。

⚠️ 掃引窓（2025-07〜2026-07）のみ。**確認窓は使わない**（候補を絞ってから一度きり）。

DB書き込みなし。予測はキャッシュ利用。

使い方:
    python scripts/exp_stake_allocation.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7b_order_disagree, rank_7b_select_legs,
    rank_7s_field_entropy, rank_7s_wt_overlap_n, rank_7ss_same_line,
)

BUDGET = 10_000        # 円/レース（ユーザー指定）
UNIT = 100             # 100円単位に丸める
SWEEP = {"w1": ("2026-04-13", "2026-07-15", 94), "w2": ("2026-01-01", "2026-04-12", 102),
         "w3": ("2025-10-01", "2025-12-31", 92), "w4": ("2025-07-01", "2025-09-30", 92)}
SWEEP_TRAIN_FROM = "2024-04-01"
CACHE_DIR = REPO / "data" / "exp_cache"
FINALS = ("準決勝",)


def cached_preds(tf, tt, train_from):
    p = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{train_from}.pkl"
    if not p.exists():
        raise SystemExit(f"[FATAL] 予測キャッシュがありません: {p}")
    return pd.read_pickle(p)


def load_trio(keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            ch = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(p) == 3:
                        out[rk][p] = v
    return out


def build(df):
    races = []
    for w, (tf, tt, _d) in SWEEP.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, SWEEP_TRAIN_FROM),
            on=["race_key", "frame_no"], how="inner")
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            rows = list(g.itertuples(index=False))
            fo = {int(x.frame_no): (int(x.finish_order)
                                    if x.finish_order == x.finish_order
                                    and x.finish_order is not None else 0) for x in rows}
            top3 = {f for f, v in fo.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            mk = {int(x.frame_no): x.prediction_mark for x in rows}
            lg = {int(x.frame_no): x.line_group for x in rows}
            r = {"rk": rk, "w": w, "top3": top3,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in rows},
                 "pw": {int(x.frame_no): float(x.ppw) for x in rows},
                 "bad": {int(x.frame_no): float(x.pbad) for x in rows},
                 "race_type": str(rows[0].race_type)}
            a1 = max(r["pw"], key=lambda f: r["pw"][f])
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            cand = [f for f in r["p3"] if f != a1]
            if not cand:
                continue
            a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
            r["a1"], r["a2"] = a1, a2
            r["ov"] = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["axis_ok"] = r["asum"] <= RANK_7S_AXIS_SUM_MAX
            r["ent_ok"] = r["ent"] <= RANK_7S_ENTROPY_MAX
            r["same_line"] = rank_7ss_same_line(a1, a2, lg)
            r["order_dis"] = rank_7b_order_disagree(r["pw"], r["hon"])
            r["others"] = sorted(set(r["p3"]) - {a1, a2})
            r["legs3"] = rank_7b_select_legs(r["others"], r["p3"], r["ana"])
            races.append(r)
    trio = load_trio(sorted({r["rk"] for r in races}))
    return [r for r in races if trio.get(r["rk"])], trio


# ---- ランク定義（本番と同一） ----
RANKS = {
    "7SS": (lambda r: r["ov"] in (0, 1) and r["axis_ok"] and not r["ent_ok"]
            and r["same_line"], "full"),
    "7S":  (lambda r: r["ov"] in (0, 1) and r["axis_ok"] and r["ent_ok"], "full"),
    "7A":  (lambda r: r["ov"] in (0, 1) and not r["axis_ok"] and r["ent_ok"], "full"),
    "7B(参考・畳む)": (lambda r: r["ov"] == 2 and r["order_dis"] is True, "three"),
    "空白3×準決勝(新候補)": (lambda r: r["ov"] == 2 and r["order_dis"] is not True
                       and r["race_type"] in FINALS, "three"),
}


def allocate(weights: list[float]) -> list[int]:
    """重みを BUDGET へ正規化し UNIT 単位に丸める（端数は最大重みの点へ寄せる）。"""
    tot = sum(weights)
    if tot <= 0:
        n = len(weights)
        base = (BUDGET // n // UNIT) * UNIT
        out = [base] * n
    else:
        out = [int((BUDGET * w / tot) // UNIT) * UNIT for w in weights]
    rest = BUDGET - sum(out)
    if rest > 0:
        out[int(np.argmax(weights)) if sum(weights) > 0 else 0] += rest
    return out


def evaluate(races, trio, sel_fn, mode, wfn):
    """wfn(o_list, p_list) -> 重みリスト。窓別ROIと全体ROIを返す。"""
    per_w = defaultdict(lambda: [0.0, 0.0])   # [bet, ret]
    tot = [0.0, 0.0]
    n = 0
    for r in races:
        if not sel_fn(r):
            continue
        board = trio[r["rk"]]
        a1, a2 = r["a1"], r["a2"]
        legs = [x for x in (r["others"] if mode == "full" else r["legs3"])
                if frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        o = [board[frozenset({a1, a2, x})] for x in legs]
        p = [r["p3"][x] for x in legs]
        stakes = allocate(wfn(o, p))
        rest = r["top3"] - {a1, a2}
        ret = 0.0
        if len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1:
            win = next(iter(rest))
            if win in legs:
                i = legs.index(win)
                ret = (round(o[i] * stakes[i]) // 10) * 10
        per_w[r["w"]][0] += BUDGET; per_w[r["w"]][1] += ret
        tot[0] += BUDGET; tot[1] += ret
        n += 1
    if n == 0:
        return None
    rois = [100 * v[1] / v[0] for _, v in sorted(per_w.items())]
    return dict(n=n, roi=100 * tot[1] / tot[0], rois=rois)


def _pow(vals, k):
    return [max(v, 1e-9) ** k for v in vals]


def _onehot(vals, hi=True):
    i = int(np.argmax(vals)) if hi else int(np.argmin(vals))
    return [1.0 if j == i else 0.0 for j in range(len(vals))]


SCHEMES = [
    ("等分（現行）",            lambda o, p: [1.0] * len(o)),
    ("── A オッズのみ", None),
    ("A k=-1.0 均等払戻(dutch)", lambda o, p: _pow(o, -1.0)),
    ("A k=-0.5 やや人気寄せ",    lambda o, p: _pow(o, -0.5)),
    ("A k=+0.5 やや穴寄せ",      lambda o, p: _pow(o, +0.5)),
    ("A k=+1.0 穴寄せ",         lambda o, p: _pow(o, +1.0)),
    ("A 最人気1点に全額",        lambda o, p: _onehot(o, hi=False)),
    ("A 最高オッズ1点に全額",     lambda o, p: _onehot(o, hi=True)),
    ("── B モデル確率のみ（オッズ不要＝入稿時可）", None),
    ("B k=+1 p3比例",          lambda o, p: _pow(p, +1.0)),
    ("B k=+2 p3二乗",          lambda o, p: _pow(p, +2.0)),
    ("B k=-1 p3逆比例",        lambda o, p: _pow(p, -1.0)),
    ("B p3最上位1点に全額",      lambda o, p: _onehot(p, hi=True)),
    ("── C モデルEV（逆選択の危険あり）", None),
    ("C k=+1 EV比例",          lambda o, p: _pow([a * b for a, b in zip(o, p)], 1.0)),
    ("C k=+2 EV二乗",          lambda o, p: _pow([a * b for a, b in zip(o, p)], 2.0)),
    ("C EV最上位1点に全額",      lambda o, p: _onehot([a * b for a, b in zip(o, p)], hi=True)),
]


def main():
    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t, _ in SWEEP.values())
    df = build_features_wt(load_raw_data_wt(min_date=SWEEP_TRAIN_FROM, max_date=max_to))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    print("掃引窓を構築 ...", flush=True)
    races, trio = build(df)
    print(f"  7車・オッズ有り {len(races)} レース\n")

    for rank, (sel, mode) in RANKS.items():
        base = evaluate(races, trio, sel, mode, lambda o, p: [1.0] * len(o))
        if not base:
            print(f"■ {rank}: 該当なし\n")
            continue
        pts = 5 if mode == "full" else 3
        print("=" * 96)
        print(f"■ {rank}   n={base['n']}  買い目={pts}点  予算={BUDGET:,}円/レース")
        print(f"  {'配分':<28}{'ROI':>9}{'等分比':>9}     窓別ROI(w1 w2 w3 w4)")
        for lbl, fn in SCHEMES:
            if fn is None:
                print(f"  {lbl}")
                continue
            m = evaluate(races, trio, sel, mode, fn)
            if not m:
                continue
            d = m["roi"] - base["roi"]
            flag = "✓" if len(m["rois"]) == 4 and all(x >= 75 for x in m["rois"]) else " "
            print(f"  {lbl:<28}{m['roi']:>8.1f}%{d:>+8.1f}pt  {flag} "
                  + " ".join(f"{x:6.1f}" for x in m["rois"]))
        print()

    print("=" * 96)
    print("  ✓ = 4窓すべてで ROI>=75%")
    print("  ※ ROIはスケール不変なので予算額そのものは結果に影響しない（配分の形だけが効く）。")
    print("  ※ A・C は最終オッズ依存。実購入は発走15分前なので乖離する。B はオッズ不要。")
    print("  ⚠️ 掃引窓。採否は確認窓＋ブートストラップで決めること。")


if __name__ == "__main__":
    main()
