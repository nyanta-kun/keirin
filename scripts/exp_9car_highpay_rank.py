"""9車立ての高配当ランク（7H3 候補）— 買い目構造の掃引と確認。

## なぜ9車なのか（2026-08-08 の実測）

決着した三連単オッズの分布（2024-01〜・板が揃ったレース）:

| 車数 | レース数 | 中央 | ≥300倍 | ≥500倍 | ≥1000倍 |
|---|---|---|---|---|---|
| 7車 | 60,036 | 35.7倍 | 9.31% | **5.15%** | 1.91% |
| **9車** | 5,678 | 77.4倍 | 20.50% | **13.40%** | 6.38% |

**9車は500倍超の発生率が7車の2.6倍**。さらに帯を丸ごと買ったときの素のROIも
7車より一貫して高い（500-1000倍 67.5% vs 60.9% / 1000-2000倍 64.7% vs 52.7%）。
＝ 高配当を狙う母集団としては9車のほうが構造的に有利なのに、
現行の穴推奨 7H1 は7車専用（9車はバスト予測 AUC 0.5967 で不成立）。

## 何を掃引するか

オッズ非依存（＝朝の入稿で完結する）買い目だけを対象にする:

    1着 = ランキング k 位の1車で固定
    2着 = ランキング上位 m2 車（k を除く）
    3着 = ランキング上位 m3 車（k と2着を除く）   → m2 × (m3-1) 点

`--rank-by model` は `wt_entries.pred_top3_pct`（walk-forward バックフィル済み・
**2024-01 以降しか無い**）、`--rank-by rp` は `race_point`（全期間ある）。

## 窓（宣言）

| 窓 | 既定 | 使い方 |
|---|---|---|
| 掃引窓 | 2025-07-01 〜 | 構成（k, m2, m3）を決める |
| 確認窓 | 2024-01-01 〜 2025-06-30 | 決めた構成を一度きり評価 |
| 独立窓 | 〜2023-12-31 | `--rank-by rp` のときだけ使える完全未使用期間 |

## 見るもの（平均だけで決めない）

- **30万+率**（1レース1万円なので「高額払戻」の定義そのもの）
- ROI と、**上位k本を除いたROI**（裾依存）
- **月次で 100% を超えた月数**（平均が窓別の反転を隠す）

## 🔴 結論（2026-08-08）: 9車の高配当ランクは**新設しない**

母集団は有利なのに、それを活かす選別が無いため全指標で 7H1 以下になった。
最良構成 `k=7・2着上位3・3着上位4`（9点）+ 三連複 3〜7位BOX（10点）:

| 指標 | 掃引窓 | **確認窓** | **独立窓** | 参考: 7H1 本番 |
|---|---|---|---|---|
| 的中率 | 9.83% | **9.13%** | **9.13%** | **18.47%** |
| ROI | 94.6% | **62.6%** | **70.2%** | **81.1%** |
| 30万+ | 0.74% | **0.56%** | **0.54%** | 0.54% |
| 除・上10 の ROI | 41.5% | 24.7% | 18.7% | — |
| 月次100%超 | 5/14 | 3/18 | 2/13 | — |

- **30万+ は 7H1 と同等**だが、**的中率は半分・ROI は 11〜18pt 下で控除率の壁を割る**
- 掃引窓の 30万+ 0.74〜1.68% は確認窓で必ず半減する（0.21〜0.72%）
- 裾依存が 7H1 より深刻（上位10本を除くと ROI 19〜25%）
- 唯一の利点は在庫が約3.6件/日 増えること。**質の悪い推奨を増やすだけ**で割に合わない

**効かなかった理由は買い目でも選別でもなく、9車には「市場が持っていない予測」が
1つも無いこと**（7車のバスト確率 AUC 0.6848 に相当するものが9車では 0.5967）。
[[keirin_7h2_third_upset_rejected_2026_08_06]] の一般則がそのまま当てはまる。

### 副次的に確認できたこと（今後に使える）

- **モデルランキングは9車では競走得点ランキングを上回らない**（確認窓 30万+
  model 0.56% vs rp 0.62%）。7車では model が rp を明確に上回った（0.76 vs 0.60）ので、
  **また1つ「7車の知見が9車へ移らない」事例が増えた**
- **三連複の相手から上位2車を外すと中央払戻が 2,400円 → 11,000円（4.6倍）**になる。
  的中率は 35%→9% に落ちるが、**上位5車BOXは的中しても元本の1/4しか戻らない**
  （＝ほぼ全部ガミ）ので実質的中はむしろ増えている。
  「市場と一致した相手を買うと配当が消える」（7B の中核仮説）が9車の三連複でも成立する
- フラット選別（`rp_std低 ∧ max_line<=3`）を重ねると掃引窓 30万+ は 1.08→1.68% に
  上がるが確認窓は 0.56→0.72% どまりで、n も 979R へ落ちる

## 使い方

    .venv/bin/python scripts/exp_9car_highpay_rank.py --rank-by model
    .venv/bin/python scripts/exp_9car_highpay_rank.py --rank-by rp --flat-gate
    .venv/bin/python scripts/exp_9car_highpay_rank.py --rank-by rp \
        --trio-box 5 --trio-skip 2 --tf-budget 7500

DB へは書き込まない。
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import get_connection  # noqa: E402

N_ENTRIES = 9
FULL_BOARD = N_ENTRIES * (N_ENTRIES - 1) * (N_ENTRIES - 2)   # 504
ODDS_MAX = 9000.0
RACE_BUDGET = 10_000
UNIT = 100
BIG = 300_000          # 「高額払戻」の定義（ユーザー基準）

SWEEP_FROM = "2025-07-01"
CONFIRM_FROM = "2024-01-01"


def _load(rank_by: str) -> dict[str, dict]:
    """レース単位の {ランキング, 決着目, 全オッズ, 構造特徴} を返す。

    ⚠️ 絞り込みは必ず `wt_races` への JOIN で書く。`race_key IN (SELECT ...)` に
    すると `wt_odds`（2,200万行）のプランが崩れて15秒→15分以上になる。
    """
    col = "pred_top3_pct" if rank_by == "model" else "race_point"
    races: dict[str, dict] = {}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, r.race_date, e.frame_no, e.{col} AS score,
                   e.finish_order, e.race_point, e.line_size
            FROM wt_entries e JOIN wt_races r USING(race_key)
            WHERE r.cancel=0 AND r.n_entries={N_ENTRIES} AND e.{col} IS NOT NULL
        """)
        rows = cur.fetchall()
        by_race: dict[str, list] = collections.defaultdict(list)
        for e in rows:
            by_race[e["race_key"]].append(e)
        del rows

        cur.execute(f"""
            SELECT o.race_key, o.combination, o.odds_value
            FROM wt_odds o JOIN wt_races r USING(race_key)
            WHERE o.bet_type='trifecta' AND r.cancel=0 AND r.n_entries={N_ENTRIES}
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
        """)
        odds: dict[str, dict] = collections.defaultdict(dict)
        for o in cur:
            odds[o["race_key"]][o["combination"]] = float(o["odds_value"])

    for rk, ents in by_race.items():
        # 事前欠車は行自体が消える（CLAUDE.md「finish_order=0 の意味」）ので、
        # 9行そろっていないレースは「9車立てとして発走していない」＝対象外。
        if len(ents) != N_ENTRIES:
            continue
        board = odds.get(rk)
        if not board or len(board) < FULL_BOARD * 0.9:
            continue
        fin = {e["finish_order"]: e["frame_no"] for e in ents
               if e["finish_order"] in (1, 2, 3)}
        if len(fin) < 3:
            continue
        win = f"{fin[1]}-{fin[2]}-{fin[3]}"
        if win not in board:
            continue
        rps = [float(e["race_point"]) for e in ents if e["race_point"] is not None]
        races[rk] = {
            "date": ents[0]["race_date"],
            "order": [e["frame_no"] for e in sorted(ents, key=lambda x: -float(x["score"]))],
            "win": win, "board": board, "win_odds": board[win],
            "rp_std": float(np.std(rps)) if rps else np.nan,
            "max_line": max(int(e["line_size"] or 1) for e in ents),
        }
    return races


def _attach_trio(races: dict[str, dict], box_n: int, skip: int = 0) -> int:
    """ランキング上位 `box_n` 車の三連複BOXの目とオッズを各レースへ付ける。

    7H1 が控除率の壁の上で「商品」として成立している実装上の理由は、
    三連単だけだと的中率が数%しかないところへ**三連複を併買して的中を下支え**
    している点にある（[[keirin_highpay_payout_ceiling_2026_08_06]] Phase 9・
    三複2,500 : 三単7,500 が実用最適）。9車でも同じことができるかを見る。
    """
    from itertools import combinations
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT o.race_key, o.combination, o.odds_value
            FROM wt_odds o JOIN wt_races r USING(race_key)
            WHERE o.bet_type='trio' AND r.cancel=0 AND r.n_entries={N_ENTRIES}
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
        """)
        board: dict[str, dict] = collections.defaultdict(dict)
        for o in cur:
            key = frozenset(int(x) for x in o["combination"].replace("-", "=").split("="))
            board[o["race_key"]][key] = float(o["odds_value"])
    ok = 0
    for rk, race in races.items():
        b = board.get(rk)
        if not b:
            continue
        pool = race["order"][skip:skip + box_n]
        if len(pool) < 3:
            continue
        legs = [frozenset(c) for c in combinations(pool, 3)]
        if any(x not in b for x in legs):
            continue
        race["trio_legs"] = legs
        race["trio_odds"] = b
        race["trio_win"] = frozenset(int(x) for x in race["win"].split("-"))
        ok += 1
    return ok


def _bet(race: dict, k: int, m2: int, m3: int) -> tuple[list[str], int]:
    """(買い目, 1点あたりの賭け金) を返す。組めなければ ([], 0)。"""
    order = race["order"]
    if k > len(order):
        return [], 0
    lead = order[k - 1]
    rest = [f for f in order if f != lead]
    legs = [f"{lead}-{a}-{b}" for a in rest[:m2] for b in rest[:m3] if b != a]
    legs = [x for x in legs if x in race["board"]]
    if not legs:
        return [], 0
    stake = RACE_BUDGET // len(legs) // UNIT * UNIT
    return (legs, stake) if stake >= UNIT else ([], 0)


def _evaluate(races: list[dict], k: int, m2: int, m3: int,
              tf_budget: int = RACE_BUDGET) -> dict:
    """`tf_budget` 未満なら残りを三連複BOXへ回す（races に trio_legs がある場合のみ）。"""
    trio_budget = RACE_BUDGET - tf_budget
    n = cost = 0
    payouts: list[float] = []          # 的中したレースの払戻（外れは0を入れない）
    by_month: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    hits = big = over500 = 0
    for race in races:
        legs, _ = _bet(race, k, m2, m3)
        if not legs:
            continue
        trio = race.get("trio_legs") if trio_budget else None
        if trio_budget and not trio:
            continue                   # 三連複の板が無いレースは併買版の対象外
        stake = tf_budget // len(legs) // UNIT * UNIT
        if stake < UNIT:
            continue
        n += 1
        c = stake * len(legs)
        pay = stake * race["win_odds"] if race["win"] in legs else 0.0
        if trio:
            u = trio_budget // len(trio) // UNIT * UNIT
            c += u * len(trio)
            if race["trio_win"] in trio:
                pay += u * race["trio_odds"][race["trio_win"]]
        cost += c
        if pay:
            hits += 1
            payouts.append(pay)
            if pay >= BIG:
                big += 1
            if race["win_odds"] >= 500:
                over500 += 1
        m = by_month[race["date"][:7]]
        m[0] += c
        m[1] += pay
    if not n:
        return {}
    tail = sorted(payouts, reverse=True)
    months = [v[1] / v[0] for v in by_month.values() if v[0] >= 200_000]
    return {
        "n": n, "hit": hits, "roi": sum(payouts) / cost * 100,
        "over500": over500, "big": big, "big_pct": big / n * 100,
        "med_pay": float(np.median(payouts)) if payouts else 0.0,
        "roi_ex3": (sum(tail[3:]) / cost * 100) if len(tail) > 3 else 0.0,
        "roi_ex10": (sum(tail[10:]) / cost * 100) if len(tail) > 10 else 0.0,
        "months_100": sum(1 for r in months if r >= 1.0), "months": len(months),
    }


def _fmt(tag: str, r: dict) -> str:
    if not r:
        return f"{tag:>18} （対象なし）"
    return (f"{tag:>18} n={r['n']:>5} 的中={r['hit']/r['n']*100:>5.2f}% "
            f"ROI={r['roi']:>6.1f}% (除上3 {r['roi_ex3']:>5.1f}% / 除上10 {r['roi_ex10']:>5.1f}%) "
            f"500倍+={r['over500']:>3} 30万+={r['big']:>3}({r['big_pct']:>4.2f}%) "
            f"中央払戻={int(r['med_pay']):>7} 月次100%超={r['months_100']}/{r['months']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank-by", choices=("model", "rp"), default="model",
                    help="model=pred_top3_pct(2024-01〜のみ) / rp=race_point(全期間)")
    ap.add_argument("--flat-gate", action="store_true",
                    help="rp_std 低半分 ∧ max_line<=3 のレースだけに絞る")
    ap.add_argument("--top", type=int, default=8, help="掃引窓の上位いくつを確認窓へ持ち込むか")
    ap.add_argument("--trio-box", type=int, default=0,
                    help="三連複BOXに使うランキング上位車数（0=三連単のみ）")
    ap.add_argument("--trio-skip", type=int, default=0,
                    help="三連複BOXで上位から何車を外すか（市場の本命どころを避ける）")
    ap.add_argument("--tf-budget", type=int, default=RACE_BUDGET,
                    help="三連単へ回す予算。残りが三連複へ行く（--trio-box 指定時）")
    args = ap.parse_args()

    races = _load(args.rank_by)
    print(f"読み込み {len(races)}R（9車ちょうど・板9割以上・ランキング={args.rank_by}）")
    if args.trio_box:
        n_ok = _attach_trio(races, args.trio_box, args.trio_skip)
        lo, hi = args.trio_skip + 1, args.trio_skip + args.trio_box
        print(f"三連複BOX（ランキング{lo}〜{hi}位の{args.trio_box}車＝"
              f"{args.trio_box*(args.trio_box-1)*(args.trio_box-2)//6}点）を併買 → 板あり {n_ok}R")

    if args.flat_gate:
        cut = float(np.nanmedian([r["rp_std"] for r in races.values()]))
        races = {k: v for k, v in races.items()
                 if v["rp_std"] <= cut and v["max_line"] <= 3}
        print(f"フラット選別（rp_std<={cut:.3f} ∧ max_line<=3）→ {len(races)}R")

    sweep = [r for r in races.values() if r["date"] >= SWEEP_FROM]
    confirm = [r for r in races.values()
               if CONFIRM_FROM <= r["date"] < SWEEP_FROM]
    indep = [r for r in races.values() if r["date"] < CONFIRM_FROM]
    print(f"掃引窓 {SWEEP_FROM}〜 {len(sweep)}R / 確認窓 {CONFIRM_FROM}〜 {len(confirm)}R"
          f" / 独立窓 〜{CONFIRM_FROM} {len(indep)}R\n")

    grid = [(k, m2, m3) for k in range(1, 9)
            for (m2, m3) in ((2, 3), (2, 4), (3, 4), (3, 5), (4, 5))]

    # --- 掃引窓: 30万+率で並べる（この窓の数字で採否を決めてはいけない） ---
    scored = []
    for (k, m2, m3) in grid:
        r = _evaluate(sweep, k, m2, m3, args.tf_budget)
        if r and r["n"] >= 200:
            scored.append(((k, m2, m3), r))
    scored.sort(key=lambda x: -x[1]["big_pct"])
    print("=== 掃引窓（構成を決めるためだけに使う）30万+率 上位 ===")
    for cfg, r in scored[:args.top]:
        print(_fmt(f"k={cfg[0]} {cfg[1]}x{cfg[2]}", r))

    # --- 確認窓: 掃引窓で決めた上位構成をそのまま当てる（再最適化しない） ---
    print(f"\n=== 確認窓（一度きり・{CONFIRM_FROM}〜{SWEEP_FROM}）===")
    for cfg, _ in scored[:args.top]:
        print(_fmt(f"k={cfg[0]} {cfg[1]}x{cfg[2]}", _evaluate(confirm, *cfg, args.tf_budget)))

    if indep:
        print(f"\n=== 独立窓（〜{CONFIRM_FROM}・探索で一度も触っていない）===")
        for cfg, _ in scored[:args.top]:
            print(_fmt(f"k={cfg[0]} {cfg[1]}x{cfg[2]}", _evaluate(indep, *cfg, args.tf_budget)))


if __name__ == "__main__":
    main()
