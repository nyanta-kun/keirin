"""買い目構造 × レース条件 × オッズ上下カットの ROI スイープ（2026-07-15）。

ユーザー要求: 以下の構造で「オッズ上下カットにより ROI>100% になる条件」を探索する。
  S1: 1車の穴を軸にした三連複（軸全流し / 軸+指数上位相手）
  S2: 1車の穴の頭固定 三連単（2-3着=指数上位）
  S3: 1車の堅軸1着固定 + 2-3着穴狙い 三連単
  S4: 2車連軸 三連複（現行SS形）
  S5: 2車連軸 三連単（現行ST形: 1着=1位, 2着=2-3位, 3着=残り）

過学習対策（keirin-survivor-bias-inflation の教訓に準拠）:
  - EXPLORE 2025-07-01〜2026-03-31: ≤2025-06-30 学習モデル(M1)でスコア＝真のOOS
  - VALIDATE 2026-04-01〜2026-07-10: ≤2026-03-31 学習モデル(M2)でスコア＝真のOOS
  - EXPLORE で条件を全数スイープ → 生存者(ROI≥1.05, n≥200, hits≥20)のみ VALIDATE で判定
  - 欠車/失格(finish_order 0/NULL)絡みの買い目は **返還**（リポジトリ確立のfloorモデル）
  - 払戻 = 最終オッズ×100（picks_history実払戻と17/18一致の実証済み近似）
  - オッズ未収録の目は買わない（発走前に最終オッズは既知＝現実的）

本番モデル・戦略は変更しない。
"""
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import numpy as np
import lightgbm as lgb

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)

M1_TO = "2025-06-30"
EX_FROM, EX_TO = "2025-07-01", "2026-03-31"
M2_TO = "2026-03-31"
VA_FROM, VA_TO = "2026-04-01", "2026-07-10"
PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              random_state=42, verbose=-1)
STAKE = 100

TRIO_CUTS = [(lo, hi) for lo in (0, 5, 10, 15) for hi in (1e9, 50, 30, 20) if hi > lo]
TRI_CUTS = [(lo, hi) for lo in (0, 10, 20, 40) for hi in (1e9, 300, 150, 80) if hi > lo]

SURV_ROI, SURV_N, SURV_HITS = 1.05, 200, 20


# ---------------------------------------------------------------- structures
def gen_bets(ctx):
    """レース文脈 → {cond_name: (bet_type, [combo,...])} を返す。
    combo は trio=frozenset(3車番), trifecta=tuple(順序3車番)。"""
    mr = ctx["mrank"]      # model rank → frame_no（1-indexed rank）
    pop = ctx["poprank"]   # frame_no → market popularity rank (1=人気)
    p = ctx["probs"]       # rank順 pred_prob list
    frames = ctx["frames"]
    n = len(frames)
    if n < 6:
        return {}
    gap12, gap23 = p[0] - p[1], p[1] - p[2]
    ratio1 = p[0] / (3.0 / n)
    out = {}

    def others(ax, pool):
        return [f for f in pool if f != ax]

    # --- S1: 穴1車軸 三連複 ---
    for adef, ax in (("mr3", mr[2]), ("mr4", mr[3]),
                     ("value", next((mr[i] for i in range(4)
                                     if pop.get(mr[i], 1) >= 5), None))):
        if ax is None:
            continue
        rest = others(ax, frames)
        out[f"S1trio_ax{adef}_all"] = ("trio", [frozenset((ax, a, b))
                                                for a, b in combinations(rest, 2)])
        top4o = others(ax, mr[:5])[:4]
        out[f"S1trio_ax{adef}_top4"] = ("trio", [frozenset((ax, a, b))
                                                 for a, b in combinations(top4o, 2)])

    # --- S2: 穴頭固定 三連単（2-3着=指数上位4） ---
    for adef, ax in (("mr3", mr[2]), ("mr4", mr[3]),
                     ("value", next((mr[i] for i in range(4)
                                     if pop.get(mr[i], 1) >= 5), None))):
        if ax is None:
            continue
        top4o = others(ax, mr[:5])[:4]
        out[f"S2tri_head{adef}"] = ("trifecta", [(ax, a, b)
                                                 for a, b in permutations(top4o, 2)])

    # --- S3: 堅軸1着固定 + 2-3着穴 三連単 ---
    for cname, cond in (("g12_15", gap12 >= 0.15), ("g12_20", gap12 >= 0.20),
                        ("ratio16", ratio1 >= 1.6)):
        if not cond:
            continue
        pool_wide = mr[2:7]        # 指数3-7位
        pool_mid = mr[2:5]         # 指数3-5位
        out[f"S3tri_{cname}_p37"] = ("trifecta", [(mr[0], a, b)
                                                  for a, b in permutations(pool_wide, 2)])
        out[f"S3tri_{cname}_p35"] = ("trifecta", [(mr[0], a, b)
                                                  for a, b in permutations(pool_mid, 2)])

    # --- S4: 2車軸 三連複（現行SS形） ---
    thirds = mr[2:7]
    s4 = [frozenset((mr[0], mr[1], t)) for t in thirds]
    out["S4trio_all"] = ("trio", s4)
    if gap12 >= 0.10 and gap23 >= 0.01:
        out["S4trio_ssgate"] = ("trio", s4)
    if gap12 >= 0.15:
        out["S4trio_g12_15"] = ("trio", s4)

    # --- S5: 2車軸 三連単（現行ST形: 1着=1位, 2着=2-3位, 3着=残り） ---
    s5 = [(mr[0], sec, t) for sec in (mr[1], mr[2])
          for t in frames if t not in (mr[0], sec)]
    out["S5tri_all"] = ("trifecta", s5)
    if gap12 >= 0.15:
        out["S5tri_g12_15"] = ("trifecta", s5)
    if gap12 >= 0.25:
        out["S5tri_g12_25"] = ("trifecta", s5)
    return out


# ---------------------------------------------------------------- settlement
def settle_window(df_win, label):
    """window の全レースについて全構造×オッズカットの (pay,bet) を集計する。"""
    cells = defaultdict(lambda: [[], []])   # key → [pays_per_race, bets_per_race]
    race_keys = df_win["race_key"].unique().tolist()
    ctxs = {}
    for rk, g in df_win.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        frames = g["frame_no"].astype(int).tolist()
        fins = dict(zip(g["frame_no"].astype(int),
                        [None if (f != f or f is None) else int(f)
                         for f in g["finish_order"].astype(float)]))
        scratch = {f for f, fo in fins.items() if fo is None or fo == 0}
        fin3 = sorted((fo, f) for f, fo in fins.items() if fo and fo >= 1)[:3]
        order = tuple(f for _, f in fin3) if len(fin3) == 3 else None
        ctxs[rk] = dict(mrank=frames, frames=sorted(frames),
                        probs=g["pred_prob"].tolist(), scratch=scratch,
                        order=order, top3=frozenset(order) if order else None)

    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            boards = defaultdict(dict)
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio":
                    boards[rk][("trio", frozenset(parts))] = float(od)
                elif len(parts) == 3:
                    boards[rk][("trifecta", tuple(parts))] = float(od)

            for rk in chunk:
                ctx = ctxs.get(rk)
                bd = boards.get(rk)
                if ctx is None or not bd:
                    continue
                # 市場人気proxy: 三連単で頭のときの最低オッズ
                pop_odds = {}
                for (bt, cb), od in bd.items():
                    if bt == "trifecta":
                        h = cb[0]
                        if od < pop_odds.get(h, 1e9):
                            pop_odds[h] = od
                ctx["poprank"] = {f: r + 1 for r, (f, _) in enumerate(
                    sorted(pop_odds.items(), key=lambda kv: kv[1]))}

                for cond, (bt, combos) in gen_bets(ctx).items():
                    cuts = TRIO_CUTS if bt == "trio" else TRI_CUTS
                    priced = []
                    for cb in combos:
                        od = bd.get((bt, cb))
                        if od is None:
                            continue
                        refund = bool(ctx["scratch"] & set(cb)) or ctx["order"] is None
                        hit = (not refund) and (
                            cb == ctx["top3"] if bt == "trio" else cb == ctx["order"])
                        priced.append((od, refund, hit))
                    for lo, hi in cuts:
                        pay = bet = 0
                        for od, refund, hit in priced:
                            if not (od >= lo and od <= hi):
                                continue
                            bet += STAKE
                            if refund:
                                pay += STAKE
                            elif hit:
                                pay += od * STAKE
                        if bet > 0:
                            key = (cond, lo, hi)
                            cells[key][0].append(pay)
                            cells[key][1].append(bet)
    print(f"  {label}: {len(race_keys)}R 集計完了 / cells={len(cells)}")
    return cells


def summarize(cells, min_n=1):
    rows = []
    for (cond, lo, hi), (pays, bets) in cells.items():
        n = len(pays)
        if n < min_n:
            continue
        tp, tb = sum(pays), sum(bets)
        hits = sum(1 for p, b in zip(pays, bets) if p > b)  # 返還のみはhit扱いしない近似
        roi = tp / tb if tb else 0.0
        rows.append(dict(cond=cond, lo=lo, hi=hi, n=n, hits=hits,
                         roi=roi, bet=tb, pay=tp))
    return rows


def boot_ci(pays, bets, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    pays, bets = np.array(pays), np.array(bets)
    idx = rng.integers(0, len(pays), size=(iters, len(pays)))
    rois = pays[idx].sum(axis=1) / np.maximum(bets[idx].sum(axis=1), 1)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=VA_TO)
    df = build_features_wt(raw)
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)

    tr_mask = df["finish_order"] >= 1
    m1 = lgb.LGBMClassifier(**PARAMS)
    sub = df[tr_mask & (df["race_date"] <= M1_TO)]
    m1.fit(sub[FEATURE_COLS_WT].fillna(0).values, sub[TARGET_COL_WT].values)
    m2 = lgb.LGBMClassifier(**PARAMS)
    sub = df[tr_mask & (df["race_date"] <= M2_TO)]
    m2.fit(sub[FEATURE_COLS_WT].fillna(0).values, sub[TARGET_COL_WT].values)
    print("  M1(≤2025-06) / M2(≤2026-03) 学習完了")

    ex = df[(df["race_date"] >= EX_FROM) & (df["race_date"] <= EX_TO) & (df["_ne"] == 7)].copy()
    va = df[(df["race_date"] >= VA_FROM) & (df["race_date"] <= VA_TO) & (df["_ne"] == 7)].copy()
    ex["pred_prob"] = m1.predict_proba(ex[FEATURE_COLS_WT].fillna(0).values)[:, 1]
    va["pred_prob"] = m2.predict_proba(va[FEATURE_COLS_WT].fillna(0).values)[:, 1]
    print(f"EXPLORE {ex['race_key'].nunique()}R / VALIDATE {va['race_key'].nunique()}R")

    ex_cells = settle_window(ex, "EXPLORE")
    rows = summarize(ex_cells, min_n=SURV_N)
    rows.sort(key=lambda r: -r["roi"])

    print(f"\n===== EXPLORE 上位30（n≥{SURV_N}）=====")
    print(f"{'cond':<26}{'odds帯':>14}{'nR':>6}{'hits':>6}{'ROI':>8}")
    for r in rows[:30]:
        hi = "∞" if r["hi"] >= 1e9 else f"{r['hi']:.0f}"
        print(f"{r['cond']:<26}{r['lo']:>6.0f}-{hi:>7}{r['n']:>6}{r['hits']:>6}{r['roi']:>8.1%}")

    surv = [r for r in rows if r["roi"] >= SURV_ROI and r["hits"] >= SURV_HITS]
    print(f"\n生存者（ROI≥{SURV_ROI:.0%}, n≥{SURV_N}, hits≥{SURV_HITS}）: {len(surv)}件")
    if not surv:
        print("→ EXPLORE 段階で生存者なし。VALIDATE 省略。")
        return

    va_cells = settle_window(va, "VALIDATE")
    print(f"\n===== VALIDATE（生存者のみ・untouched 窓）=====")
    print(f"{'cond':<26}{'odds帯':>14}{'EX ROI':>8} | {'VA nR':>6}{'VA hits':>8}{'VA ROI':>8}{'  95%CI':>16}")
    for r in surv:
        key = (r["cond"], r["lo"], r["hi"])
        pays, bets = va_cells.get(key, [[], []])
        hi = "∞" if r["hi"] >= 1e9 else f"{r['hi']:.0f}"
        if not pays:
            print(f"{r['cond']:<26}{r['lo']:>6.0f}-{hi:>7}{r['roi']:>8.1%} |   (VA該当なし)")
            continue
        tb = sum(bets)
        roi = sum(pays) / tb if tb else 0
        hits = sum(1 for p, b in zip(pays, bets) if p > b)
        lo_ci, hi_ci = boot_ci(pays, bets)
        flag = "  ◎" if lo_ci > 1.0 else ("  ○" if roi >= 1.0 else "")
        print(f"{r['cond']:<26}{r['lo']:>6.0f}-{hi:>7}{r['roi']:>8.1%} |"
              f"{len(pays):>6}{hits:>8}{roi:>8.1%}  [{lo_ci:.0%},{hi_ci:.0%}]{flag}")
    print("\n判定: ◎=CI下限>100%（強い） ○=点推定>100%（要フォワード追試） 無印=不成立。")


if __name__ == "__main__":
    main()
