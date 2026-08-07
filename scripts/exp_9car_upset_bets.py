"""選別した9車の母集団の上に、**実装可能な点数の買い目**を載せて測る。

[[keirin_pooled_upset_screen_2026_08_08]]（PR#43）で、6/7/9車の統合学習で作った
波乱スコアの上位20%を取ると、9車の高配当帯の ratio が 0.887 → 1.027 になった
（帯ROI 67.3% → 77.0%）。ただしあれは**帯を丸ごと買った理論値**で、
9車の300倍+帯は約100点あり 1レース1万円では1点100円にしかならない。

本スクリプトは同じ母集団の上で **N点の実際の買い目**に落とし、
30万+率・ROI・裾依存まで測る。

## 買い目の作り方（既に確定している制約から一意に決まる）

- **帯内の目選びはモデルがランダムに負ける**（`exp_highpay_trifecta_design.py`）。
  よって選び方は**オッズ昇順**（＝要求ラインぎりぎりの目から拾う）が算術上の最適
- 1レース1万円をN点に等分するなら、30万円に届く条件は **各点のオッズ >= 30N**
  （[[keirin_highpay_payout_ceiling_2026_08_06]] の恒等式）
- したがって買い目は `オッズ >= T の中で最も安い N点` の1族に絞られる。
  `--t-mode payout` は T = 30N（30万円ちょうどを狙う）、
  `--t-mode fixed` は T を固定（30万円は諦めて的中率を取る）

対照として **オッズ非依存**の構成（モデル3着内率 k位1着固定のフォーメーション。
PR#40 の最良形）も同じ母集団で測り、「オッズを使う価値」を分離する。

## 測るもの

- **30万+率**（1レース1万円なので「高額払戻」の定義そのもの）
- ROI と **上位k本を除いたROI**（裾依存）／月次100%超の月数
- **選別あり / なしの対比**（選別が買い目レベルまで効いているか）

⚠️ 最終オッズで測っている。朝の板は下振れするので実運用の値ではない
（[[keirin_highpay_tail_mispricing_2026_08_08]] の朝→最終ドリフト参照）。
本スクリプトの目的は**構造の比較**であって実運用値の推定ではない。

## 🔴 結論（2026-08-08・9車 3,435R・選別687R）

### ① オッズを使えるなら「30倍以上で最も安い1点に1万円」が圧倒的に強い

| 構成 | n | 的中 | ROI | **30万+** | 中央払戻 | 除上10 |
|---|---|---|---|---|---|---|
| **1点 T=30倍（非選別・全レース）** | 2,748 | 2.44% | **79.7%** | **2.44%** | 319,000 | 66.2% |
| 1点 T=30倍（選別） | 687 | 2.47% | 79.9% | 2.47% | 316,000 | 30.9% |
| 2点 T=60倍 | 2,748 | 2.18% | 70.9% | 2.18% | 318,500 | 57.7% |
| 10点 T=300倍 | 2,748 | 2.00% | 64.5% | 2.00% | 319,900 | 51.9% |

**30万+ 2.44% は 7H1 の 0.54% の約4.5倍**で、本プロジェクトで見つかった中で最良。
点数を増やすと要求オッズが同率で上がって相殺されるので **1点が最適**
（[[keirin_highpay_payout_ceiling_2026_08_06]] の恒等式どおり）。

### ② ❌ 選別は買い目レベルでは効かない（オッズを使う構成では）

    1点 T=30倍   選別 2.47% / 非選別 2.44%   （差 +0.03pt）
    2点 T=60倍   選別 2.33% / 非選別 2.18%
    10点 T=300倍 選別 2.18% / 非選別 2.00%

マージンを掛けても同じで、むしろ選別側が下回ることが多い
（margin 1.3 の 1点: 選別 1.31% / 非選別 1.60%）。選別後は n=687・的中10件前後
なので、光って見えるセル（margin 1.3 の5点 3.78%/ROI 154%）はすべてノイズ。

**機序は既知**: 「レース選別はオッズ情報の代替として機能する」。
オッズで帯を固定した時点で選別の役目は既に果たされており、上乗せの余地が無い。
PR#43 の ratio 改善は **≥300倍帯を丸ごと**買ったときの話で、
その帯の**最も安い数点だけ**を買う構成には伝わらなかった。

### ③ ✅ 選別が効くのはオッズ非依存の構成のほう

| 構成 | 選別 30万+ | 非選別 30万+ | 選別 ROI | 非選別 ROI |
|---|---|---|---|---|
| k=7 3x4（9点） | **1.02%** | 0.76% | 78.1% | 76.8% |
| k=5 2x4（6点） | **0.87%** | 0.58% | **106.8%** | 83.6% |

こちらは 30万+ が 1.3〜1.5倍になる。ただし **裾依存が極端**（除上10 で 4.7〜8.3%）で
ROI 106.8% を期待値として扱ってはいけない。

### 到達点の整理

| 方式 | 30万+ | ROI | オッズ | 備考 |
|---|---|---|---|---|
| **9車 全レース・30倍以上の最安1点** | **2.44%** | 79.7% | **発走前の板が必須** | 選別不要 |
| 同・安全マージン1.3 | 1.60% | 67.3% | 朝の板でも可 | 実運用に近い |
| 9車 選別 × k=5 2x4（6点） | 0.87% | 106.8%（裾依存） | 不要 | 朝で完結 |
| 参考: 7H1（7車・本番実績） | 0.54% | 81.1% | 不要 | 三連複が的中を下支え |

## 使い方

    .venv/bin/python scripts/exp_9car_upset_bets.py
    .venv/bin/python scripts/exp_9car_upset_bets.py --margin 1.3
    .venv/bin/python scripts/exp_9car_upset_bets.py --sel-q 0.1 --t-mode fixed --t-fixed 300

DB へは書き込まない。
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_pooled_upset_screen import (  # noqa: E402
    MODEL_COLS, _load, _make_target, _walk_forward,
)
from src.database import get_connection  # noqa: E402

ODDS_MAX = 9000.0
RACE_BUDGET = 10_000
UNIT = 100
BIG = 300_000


def _boards(n_entries: int) -> dict[str, dict[str, float]]:
    """9車レースの三連単の板（全目のオッズ）と決着目を返す。

    ⚠️ 絞り込みは `wt_races` への JOIN で書く（`race_key IN (SELECT ...)` は
    `wt_odds` のプランを壊す）。9車だけなら約240万行で扱える。
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT o.race_key, o.combination, o.odds_value
            FROM wt_odds o JOIN wt_races r USING(race_key)
            WHERE o.bet_type='trifecta' AND r.cancel=0 AND r.n_entries={n_entries}
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
        """)
        board: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for o in cur:
            board[o["race_key"]][o["combination"]] = float(o["odds_value"])

        cur.execute(f"""
            SELECT e.race_key,
                   max(CASE WHEN e.finish_order=1 THEN e.frame_no END) f1,
                   max(CASE WHEN e.finish_order=2 THEN e.frame_no END) f2,
                   max(CASE WHEN e.finish_order=3 THEN e.frame_no END) f3,
                   count(*) n_rows
            FROM wt_entries e JOIN wt_races r USING(race_key)
            WHERE r.cancel=0 AND r.n_entries={n_entries} GROUP BY e.race_key
        """)
        win = {}
        for e in cur:
            if e["f1"] and e["f2"] and e["f3"] and e["n_rows"] == n_entries:
                win[e["race_key"]] = f"{e['f1']}-{e['f2']}-{e['f3']}"
    return {rk: {"board": b, "win": win[rk]} for rk, b in board.items() if rk in win}


def _band_legs(board: dict[str, float], thr: float, n: int) -> list[str]:
    """オッズ `thr` 以上の目のうち**最も安い n 点**。足りなければ空。

    「最も安い」を選ぶのは、帯内の順位付けでモデルがランダムに負けると確定して
    いるため。要求ラインぎりぎりを拾うのが算術上の最適になる。
    """
    cand = sorted(((o, c) for c, o in board.items() if o >= thr))
    if len(cand) < n:
        return []
    return [c for _, c in cand[:n]]


def _eval(races: list[dict], sel: np.ndarray, boards: dict, legs_fn) -> dict:
    n = 0
    cost = 0.0
    pays: list[float] = []
    by_month: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    big = over500 = 0
    for i, r in enumerate(races):
        if not sel[i]:
            continue
        b = boards.get(r["race_key"])
        if not b:
            continue
        legs = legs_fn(b["board"])
        if not legs:
            continue
        stake = RACE_BUDGET // len(legs) // UNIT * UNIT
        if stake < UNIT:
            continue
        n += 1
        c = stake * len(legs)
        cost += c
        pay = stake * b["board"][b["win"]] if b["win"] in legs else 0.0
        if pay:
            pays.append(pay)
            if pay >= BIG:
                big += 1
            if b["board"][b["win"]] >= 500:
                over500 += 1
        m = by_month[r["date"][:7]]
        m[0] += c
        m[1] += pay
    if not n:
        return {}
    tail = sorted(pays, reverse=True)
    months = [v[1] / v[0] for v in by_month.values() if v[0] >= 200_000]
    return {
        "n": n, "hit": len(pays), "roi": sum(pays) / cost * 100,
        "big": big, "big_pct": big / n * 100, "over500": over500,
        "med": float(np.median(pays)) if pays else 0.0,
        "roi_ex3": sum(tail[3:]) / cost * 100 if len(tail) > 3 else 0.0,
        "roi_ex10": sum(tail[10:]) / cost * 100 if len(tail) > 10 else 0.0,
        "m100": sum(1 for x in months if x >= 1.0), "months": len(months),
    }


def _fmt(tag: str, r: dict) -> str:
    if not r:
        return f"{tag:<26} （対象なし）"
    return (f"{tag:<26} n={r['n']:>5} 的中={r['hit']/r['n']*100:>5.2f}% "
            f"ROI={r['roi']:>6.1f}% (除上3 {r['roi_ex3']:>5.1f}/除上10 {r['roi_ex10']:>5.1f}) "
            f"500倍+={r['over500']:>3} 30万+={r['big']:>3}({r['big_pct']:>4.2f}%) "
            f"中央={int(r['med']):>7} 月次100%超={r['m100']}/{r['months']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="6,7,9", help="学習に使う車数")
    ap.add_argument("--eval-ne", type=int, default=9)
    ap.add_argument("--sel-q", type=float, default=0.2, help="波乱スコア上位いくつを採るか")
    ap.add_argument("--target-q", type=float, default=0.25)
    ap.add_argument("--a-thr", type=float, default=300.0, help="学習・診断用の波乱下限")
    ap.add_argument("--t-mode", choices=("payout", "fixed"), default="payout")
    ap.add_argument("--t-fixed", type=float, default=300.0)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="要求ラインの安全マージン。朝の板は発走までに下振れするので "
                         "T = 30N x margin で買う（1.0 は最終オッズ前提の理論値）")
    ap.add_argument("--features", choices=("card", "all"), default="card")
    args = ap.parse_args()

    pool = [int(x) for x in args.pool.split(",")]
    rows = _load(pool, args.a_thr, 50.0)
    drop = ("race_key", "date", "win_odds", "impA", "impB")
    cols = [c for c in rows[0] if c not in drop]
    if args.features == "card":
        cols = [c for c in cols if c not in MODEL_COLS]
    y_train = _make_target(rows, "quantile", args.a_thr, args.target_q)
    s = _walk_forward(rows, cols, y_train)

    ne = np.array([r["n_entries"] for r in rows])
    ok = (~np.isnan(s)) & (ne == args.eval_ne)
    thr = np.nanquantile(s[ok], 1 - args.sel_q)
    sel = ok & (s >= thr)
    non = ok & (s < thr)
    days = len({r["date"] for i, r in enumerate(rows) if ok[i]})
    print(f"{args.eval_ne}車 評価対象 {ok.sum()}R / 選別（上位{args.sel_q:.0%}）{sel.sum()}R"
          f"（{sel.sum()/days:.2f}件/日）\n")

    boards = _boards(args.eval_ne)

    print("=== オッズ使用: 「オッズ >= T の中で最も安い N点」 ===")
    if args.t_mode == "payout":
        print("    T = 30N（1点あたりが 30万円ちょうどに届く最小ライン）")
    for n_legs in (1, 2, 3, 5, 8, 10):
        t = (30 * n_legs * args.margin if args.t_mode == "payout"
             else args.t_fixed * args.margin)
        fn = (lambda b, t=t, k=n_legs: _band_legs(b, t, k))
        print(_fmt(f"  {n_legs}点 (T={t:.0f}倍) 選別", _eval(rows, sel, boards, fn)))
        print(_fmt(f"  {n_legs}点 (T={t:.0f}倍) 非選別", _eval(rows, non, boards, fn)))

    print("\n=== 対照: オッズ非依存（モデル3着内率 k位1着固定フォーメーション）===")
    order_of = _rank_orders(args.eval_ne)
    for k, m2, m3 in ((7, 3, 4), (7, 2, 3), (5, 2, 4)):
        sel_r = _eval_formation(rows, sel, boards, order_of, k, m2, m3)
        non_r = _eval_formation(rows, non, boards, order_of, k, m2, m3)
        print(_fmt(f"  k={k} {m2}x{m3} 選別", sel_r))
        print(_fmt(f"  k={k} {m2}x{m3} 非選別", non_r))


def _rank_orders(n_entries: int) -> dict[str, list[int]]:
    """レースごとの「モデル3着内率の降順の車番」。オッズ非依存の対照で使う。"""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, e.frame_no, e.pred_top3_pct
            FROM wt_entries e JOIN wt_races r USING(race_key)
            WHERE r.cancel=0 AND r.n_entries={n_entries} AND e.pred_top3_pct IS NOT NULL
        """)
        by: dict[str, list] = collections.defaultdict(list)
        for e in cur:
            by[e["race_key"]].append((float(e["pred_top3_pct"]), e["frame_no"]))
    return {rk: [f for _, f in sorted(v, reverse=True)]
            for rk, v in by.items() if len(v) == n_entries}


def _eval_formation(races, sel, boards, orders, k, m2, m3) -> dict:
    def make(rk):
        order = orders.get(rk)
        if not order or k > len(order):
            return []
        lead = order[k - 1]
        rest = [f for f in order if f != lead]
        return [f"{lead}-{a}-{b}" for a in rest[:m2] for b in rest[:m3] if b != a]

    n = 0
    cost = 0.0
    pays: list[float] = []
    by_month: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    big = over500 = 0
    for i, r in enumerate(races):
        if not sel[i]:
            continue
        b = boards.get(r["race_key"])
        if not b:
            continue
        legs = [x for x in make(r["race_key"]) if x in b["board"]]
        if not legs:
            continue
        stake = RACE_BUDGET // len(legs) // UNIT * UNIT
        if stake < UNIT:
            continue
        n += 1
        cost += stake * len(legs)
        pay = stake * b["board"][b["win"]] if b["win"] in legs else 0.0
        if pay:
            pays.append(pay)
            if pay >= BIG:
                big += 1
            if b["board"][b["win"]] >= 500:
                over500 += 1
        m = by_month[r["date"][:7]]
        m[0] += stake * len(legs)
        m[1] += pay
    if not n:
        return {}
    tail = sorted(pays, reverse=True)
    months = [v[1] / v[0] for v in by_month.values() if v[0] >= 200_000]
    return {
        "n": n, "hit": len(pays), "roi": sum(pays) / cost * 100,
        "big": big, "big_pct": big / n * 100, "over500": over500,
        "med": float(np.median(pays)) if pays else 0.0,
        "roi_ex3": sum(tail[3:]) / cost * 100 if len(tail) > 3 else 0.0,
        "roi_ex10": sum(tail[10:]) / cost * 100 if len(tail) > 10 else 0.0,
        "m100": sum(1 for x in months if x >= 1.0), "months": len(months),
    }


if __name__ == "__main__":
    main()
