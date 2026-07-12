"""SS買い目の券種・配分検証（1レース1万円固定・返還モデル・2026-07-12改訂）

現行SS（三連複 軸2車-全・min全目>=7・gap12>=0.10・gap23>=1pt）のレース選定は
そのままに、同一レース集合へ「1レース1万円」を以下の構成で投じた場合の
ROI を比較する:

  - 三連複 全目（均等 / オッズ反比例=均等払戻）
  - 三連単 2車軸マルチ 3着全（6*(n-2)点、均等 / オッズ反比例）
  - 三連単 1・2着固定 3着全（2*(n-2)点、均等 / オッズ反比例）
  - 三連単 p1->p2->全（(n-2)点、均等）
  - ミックス（三連複+三連単の資金配分 50/50, 70/30 等）

## 返還モデル（2026-07-12 改訂の要点）
初版は「完走者の目だけ買った」扱いで、欠車レース（目数3-4）に定額資金が
集中する非現実的な計算になっていた（欠車は事前に知り得ない）。
本版は買い目を発走前の全 n 車から構築し、欠車が絡む目は返還
（賭け金に数えず払戻もなし）として扱う。ROI = 払戻 / リスク投資額。

母集団は eval_clean_split_wt.collect の 7車ちょうど限定（本番と同一）。
金額は100円単位に丸め（最大剰余法）て合計1万円（返還分を含む名目額）。

使い方:
  .venv/bin/python scripts/exp_ss_trifecta_budget_wt.py
"""
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_clean_split_wt import (  # noqa: E402
    GAP23_MIN, SS_GAMI, SS_GAP12, collect,
)
from src.database import get_connection  # noqa: E402
from src.models.trainer import load_model  # noqa: E402

BUDGET = 10_000
UNIT = 100


def alloc(weights: dict, budget: int) -> dict:
    """weights に比例して budget を 100円単位で配分（最大剰余法）。"""
    total_w = sum(weights.values())
    if total_w <= 0:
        return {}
    units = budget // UNIT
    raw = {k: w / total_w * units for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    rem = units - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    return {k: v * UNIT for k, v in base.items() if v > 0}


def ss_races(rows):
    """現行SS条件を満たすレースのみ抽出（eval_ss と同一判定）。"""
    out = []
    for r in rows:
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < SS_GAMI:
            continue
        if r["gap12"] < SS_GAP12 or r["gap23_pt"] < GAP23_MIN:
            continue
        r = dict(r)
        r["ss_legs"] = legs  # third_frame -> trio odds（完走・オッズ有の目のみ）
        out.append(r)
    return out


def attach_prerace(races):
    """発走前の全出走車から thirds_pre（相手候補）と started（出走車）を付与する。

    買い目は発走前情報のみで構築し、欠車が絡む目は返還として評価する。
    """
    with get_connection() as c:
        fa = defaultdict(set)
        for rk, fn in c.execute("SELECT race_key, frame_no FROM wt_entries"):
            fa[rk].add(int(fn))
    for r in races:
        pre = fa.get(r["rk"], set(r["frames"]))
        r["thirds_pre"] = sorted(pre - {r["p1"], r["p2"]})
        r["started"] = set(r["frames"])
    return races


# ─── チケット集合の生成（値は最終オッズ / None=欠車等で返還） ─────────

def tickets_trio(r):
    """三連複 軸2車-全（発走前 n-2 目）"""
    return {("trio", frozenset({r["p1"], r["p2"], t})): r["ss_legs"].get(t)
            for t in r["thirds_pre"]}


def tickets_tri_multi(r):
    """三連単 2車軸マルチ 3着全: {p1,p2,t} の全順列"""
    out = {}
    for t in r["thirds_pre"]:
        for perm in permutations((r["p1"], r["p2"], t)):
            out[("tri", perm)] = r["tri"].get(perm) if t in r["started"] else None
    return out


def tickets_tri_12fix(r):
    """三連単 1・2着=軸2車(両順)固定 3着全"""
    out = {}
    for t in r["thirds_pre"]:
        for a, b in ((r["p1"], r["p2"]), (r["p2"], r["p1"])):
            out[("tri", (a, b, t))] = (r["tri"].get((a, b, t))
                                       if t in r["started"] else None)
    return out


def tickets_tri_p1p2(r):
    """三連単 p1->p2->全"""
    return {("tri", (r["p1"], r["p2"], t)):
            (r["tri"].get((r["p1"], r["p2"], t)) if t in r["started"] else None)
            for t in r["thirds_pre"]}


def is_hit(key, r):
    kind, comb = key
    if kind == "trio":
        return comb == r["top3"]
    return comb == r["order"]


# ─── 戦略定義 ───────────────────────────────────────────────────────

def make_weights(tickets, mode):
    """配分ウェイト。オッズ不明（返還見込み）の目は均等側は1.0、
    反比例側は quoted の平均ウェイトを与える（発走前は欠車を知り得ないため）。"""
    if mode == "eq":
        return {k: 1.0 for k in tickets}
    if mode == "inv":  # 払戻均等（オッズ反比例）
        quoted = [1.0 / o for o in tickets.values() if o]
        mean_w = sum(quoted) / len(quoted) if quoted else 1.0
        return {k: (1.0 / o if o else mean_w) for k, o in tickets.items()}
    raise ValueError(mode)


STRATEGIES = [
    # (label, [(ticket_fn, mode, budget_share)])
    ("三連複 均等(現行構成)", [(tickets_trio, "eq", 1.0)]),
    ("三連複 払戻均等", [(tickets_trio, "inv", 1.0)]),
    ("単マルチ 均等", [(tickets_tri_multi, "eq", 1.0)]),
    ("単マルチ 払戻均等", [(tickets_tri_multi, "inv", 1.0)]),
    ("単12固定 均等", [(tickets_tri_12fix, "eq", 1.0)]),
    ("単12固定 払戻均等", [(tickets_tri_12fix, "inv", 1.0)]),
    ("単p1p2固定 均等", [(tickets_tri_p1p2, "eq", 1.0)]),
    ("複50+単マルチ50", [(tickets_trio, "eq", 0.5), (tickets_tri_multi, "eq", 0.5)]),
    ("複70+単12固定30", [(tickets_trio, "eq", 0.7), (tickets_tri_12fix, "eq", 0.3)]),
    ("複50+単12固定50", [(tickets_trio, "eq", 0.5), (tickets_tri_12fix, "eq", 0.5)]),
    ("複30+単12固定70", [(tickets_trio, "eq", 0.3), (tickets_tri_12fix, "eq", 0.7)]),
    ("複70+単p1p2固定30", [(tickets_trio, "eq", 0.7), (tickets_tri_p1p2, "eq", 0.3)]),
]


def eval_strategy(races, parts):
    """返還モデルで戦略を評価する。bet=リスク投資額（返還分を除く）。"""
    n = hits = bet = pay = 0
    monthly = defaultdict(lambda: [0, 0])  # month -> [bet, pay]
    per_race_returns = []
    for r in races:
        stakes = {}   # key -> (amount, odds_or_None)
        ok = True
        for fn, mode, share in parts:
            tk = fn(r)
            if not any(o for o in tk.values()):
                ok = False
                break
            for k, amt in alloc(make_weights(tk, mode), int(BUDGET * share)).items():
                prev = stakes.get(k, (0, tk[k]))[0]
                stakes[k] = (prev + amt, tk[k])
        if not ok or not stakes:
            continue
        b = sum(amt for amt, o in stakes.values() if o)  # 返還分は除外
        p = sum(int(o * amt) for k, (amt, o) in stakes.items()
                if o and is_hit(k, r))
        n += 1
        hits += 1 if p > 0 else 0
        bet += b
        pay += p
        m = r["rk"][:6]
        monthly[m][0] += b
        monthly[m][1] += p
        per_race_returns.append(p - b)
    return n, hits, bet, pay, monthly, per_race_returns


def run_window(label, model_name, date_from, date_to):
    model = load_model(model_name)
    rows = collect(model, date_from, date_to)
    races = attach_prerace(ss_races(rows))
    n_dns = sum(1 for r in races if len(r["started"]) < len(r["thirds_pre"]) + 2)
    days = len({r["rk"][:8] for r in races}) or 1
    print(f"\n===== {label}: {date_from}〜{date_to} "
          f"(SS {len(races)}R / {days}日 / 欠車あり{n_dns}R=返還扱い) =====")
    print(f"{'戦略':<20} {'R数':>4} {'的中率':>6} {'リスク投資':>11} {'払戻':>11} "
          f"{'ROI':>7} {'月別ROI':<40}")
    for name, parts in STRATEGIES:
        n, h, b, p, monthly, _ = eval_strategy(races, parts)
        if n == 0 or b == 0:
            print(f"{name:<20} {'0':>4}")
            continue
        mm = " ".join(f"{m[4:]}:{v[1] / v[0]:.2f}" for m, v in sorted(monthly.items()))
        print(f"{name:<20} {n:>4} {h / n:>6.1%} {b:>11,} {p:>11,} "
              f"{p / b:>6.1%} {mm:<40}")


def main():
    # クリーン分割: eval モデル(学習〜2026-03)で 2026-04〜06 は真のOOS
    run_window("TEST(OOS)", "lgbm_wt_2026h1_eval", "2026-04-01", "2026-06-30")
    # 本番モデル(〜2026-06-30 refit)で 7月は真のフォワード
    run_window("FWD(7月)", "lgbm_wt_2026h1", "2026-07-01", "2026-07-10")
    # 参考: eval モデルの学習期間内（レース選定はin-sample・構成間比較の参考）
    run_window("参考(in-sample)", "lgbm_wt_2026h1_eval", "2025-07-01", "2026-03-31")


if __name__ == "__main__":
    main()
