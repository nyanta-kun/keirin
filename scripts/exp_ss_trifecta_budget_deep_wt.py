"""SS券種比較の頑健性検証（paired bootstrap・DD・上位依存度・返還モデル）

exp_ss_trifecta_budget_wt.py（2026-07-12 返還モデル改訂版）の続き。
OOSプール（TEST 2026-04〜06 + FWD 7月・7車ちょうど限定）で per-race の
リスク投資・払戻を揃え、現行構成（三連複均等）との差のブートストラップCI、
最大連敗・最大ドローダウン・上位5レース依存度を出す。
1・2着固定の順序重み変種（p1先行 60/40, 70/30）も追加比較。

使い方:
  .venv/bin/python scripts/exp_ss_trifecta_budget_deep_wt.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_ss_trifecta_budget_wt import (  # noqa: E402
    BUDGET, alloc, attach_prerace, is_hit, make_weights, ss_races,
    tickets_tri_12fix, tickets_tri_multi, tickets_tri_p1p2, tickets_trio,
)
from eval_clean_split_wt import collect  # noqa: E402
from src.models.trainer import load_model  # noqa: E402


def tickets_tri_12fix_w(r, w_first):
    """1・2着固定で p1先行 に w_first、p2先行に 1-w_first の重み。
    返り値: key -> (odds_or_None, weight)"""
    out = {}
    for t in r["thirds_pre"]:
        for (a, b), w in (((r["p1"], r["p2"]), w_first),
                          ((r["p2"], r["p1"]), 1.0 - w_first)):
            o = r["tri"].get((a, b, t)) if t in r["started"] else None
            out[("tri", (a, b, t))] = (o, w)
    return out


STRATEGIES = [
    ("三連複 均等(基準)", [(tickets_trio, "eq", 1.0)]),
    ("単マルチ 均等", [(tickets_tri_multi, "eq", 1.0)]),
    ("単12固定 均等", [(tickets_tri_12fix, "eq", 1.0)]),
    ("単12固定 払戻均等", [(tickets_tri_12fix, "inv", 1.0)]),
    ("単12固定 p1厚め60", [(tickets_tri_12fix_w, 0.6, 1.0)]),
    ("単12固定 p1厚め70", [(tickets_tri_12fix_w, 0.7, 1.0)]),
    ("単p1p2固定 均等", [(tickets_tri_p1p2, "eq", 1.0)]),
    ("複50+単12固定50", [(tickets_trio, "eq", 0.5), (tickets_tri_12fix, "eq", 0.5)]),
    ("複30+単12固定70", [(tickets_trio, "eq", 0.3), (tickets_tri_12fix, "eq", 0.7)]),
    ("複50+単マルチ50", [(tickets_trio, "eq", 0.5), (tickets_tri_multi, "eq", 0.5)]),
]


def per_race_bet_pay(races, parts):
    """返還モデルで per-race の (リスク投資, 払戻) を返す。"""
    bets, pays = [], []
    for r in races:
        stakes = {}  # key -> (amount, odds_or_None)
        for fn, mode, share in parts:
            if isinstance(mode, float):  # 重み付き 12固定
                tk_w = fn(r, mode)
                tk = {k: o for k, (o, _) in tk_w.items()}
                weights = {k: w for k, (_, w) in tk_w.items()}
            else:
                tk = fn(r)
                weights = make_weights(tk, mode)
            for k, amt in alloc(weights, int(BUDGET * share)).items():
                prev = stakes.get(k, (0, tk[k]))[0]
                stakes[k] = (prev + amt, tk[k])
        b = sum(amt for amt, o in stakes.values() if o)
        p = sum(int(o * amt) for k, (amt, o) in stakes.items()
                if o and is_hit(k, r))
        bets.append(b)
        pays.append(p)
    return np.array(bets, dtype=float), np.array(pays, dtype=float)


def stats(bets, pays):
    ret = pays - bets
    cum = np.cumsum(ret)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    max_dd = float(np.max(peak - cum))
    streak = best = 0
    for p in pays:
        streak = streak + 1 if p == 0 else 0
        best = max(best, streak)
    top5 = float(np.sort(pays)[-5:].sum() / pays.sum()) if pays.sum() > 0 else 0.0
    return max_dd, best, top5


def main():
    rng = np.random.default_rng(42)
    races = []
    for model_name, f, t in (("lgbm_wt_2026h1_eval", "2026-04-01", "2026-06-30"),
                             ("lgbm_wt_2026h1", "2026-07-01", "2026-07-10")):
        races += ss_races(collect(load_model(model_name), f, t))
    races = attach_prerace(races)
    n = len(races)
    print(f"OOSプール: SS {n}R（2026-04-01〜07-10・7車限定）・1R名目{BUDGET:,}円・返還モデル")

    results = {name: per_race_bet_pay(races, parts) for name, parts in STRATEGIES}
    base_bets, base_pays = results["三連複 均等(基準)"]

    idx = rng.integers(0, n, size=(10_000, n))
    print(f"\n{'戦略':<18} {'ROI':>7} {'ROI 95%CI':>15} {'Δvs基準CI':>17} "
          f"{'P(Δ>0)':>7} {'maxDD':>9} {'連敗':>4} {'top5寄与':>7}")
    for name, (bets, pays) in results.items():
        roi = pays.sum() / bets.sum()
        boot_roi = pays[idx].sum(axis=1) / bets[idx].sum(axis=1)
        lo, hi = np.percentile(boot_roi, [2.5, 97.5])
        boot_base = base_pays[idx].sum(axis=1) / base_bets[idx].sum(axis=1)
        diff = boot_roi - boot_base
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        p_pos = float((diff > 0).mean())
        max_dd, streak, top5 = stats(bets, pays)
        print(f"{name:<18} {roi:>6.1%} [{lo:>5.1%},{hi:>6.1%}] "
              f"[{dlo:>+6.1%},{dhi:>+6.1%}] {p_pos:>6.1%} "
              f"{max_dd:>8,.0f} {streak:>4d} {top5:>6.1%}")


if __name__ == "__main__":
    main()
