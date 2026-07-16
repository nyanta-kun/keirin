"""三連複 買い目削減によるROI改善検証 (doc50)

現行: pred1+pred2 → pred3/pred4/pred5  固定3点
課題: 全目一律購入 → 不要目（低オッズ・低着率）の引き算による効率改善余地

検証軸:
  Part1: 目別ポジション分析  — pred3/pred4/pred5 の各「3着目」着率・個別ROI
  Part2: 点数削減策           — 2点(pred3+pred4) / 1点(pred3のみ)
  Part3: 個別コンボオッズ足切り — 各目のオッズ < Xなら購入しない
  Part4: gap34/gap45 分岐      — pred3とpred4の確率差で点数を動的に変更
  Part5: 複合策               — 点数削減 + オッズ足切り

戦略略称:
  S0    : 現行 3点（ベースライン）
  P1    : pred3 のみ 1点
  P2    : pred3+pred4 2点
  O5    : 各目オッズ≥5倍のみ購入（個別足切り・従来のレース単位とは別）
  O8    : 各目オッズ≥8倍のみ
  O10   : 各目オッズ≥10倍のみ
  O_hi50: 各目オッズ≤50倍のみ（超高オッズ目を除去）
  G_p2  : gap34≥0.03→pred3のみ / <0.03→pred3+pred4
  G_p3  : gap34≥0.05→pred3のみ / <0.05→pred3+pred4
  H1    : P2 + O8 (pred3+pred4 かつ 各目≥8倍)
  H2    : P2 + O10
  H3    : G_p2 + O8
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)
from src.database import get_connection
from src.evaluation.backtest_wt import _assign_tier
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_RACE_THRESHOLD = 5.0   # 現行: 3点最安目でのレース足切り（ベースライン維持）
N_THIRDS = 3                 # 購入する"流し先"の最大数


# ─── データ準備 ──────────────────────────────────────────────────────────────

def _parse_combo(s, ordered=False):
    parts = re.split(r"[-=]", str(s))
    try:
        nums = [int(p) for p in parts]
        return tuple(nums) if ordered else frozenset(nums)
    except Exception:
        return None


def load_all():
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    with get_connection() as conn:
        ri = pd.read_sql("SELECT race_key, n_entries, grade FROM wt_races", conn)
        trio_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'", conn
        )

    df = df.merge(ri, on="race_key", how="left")

    print("  TRAIN 期間のみでリーク無しモデル学習中...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(prepare_X(fit).reset_index(drop=True), fit["top3_flag"].reset_index(drop=True).values)
    print(f"  学習完了 ({len(fit):,} 行)", flush=True)

    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # trio オッズマップ: {race_key: {frozenset: payout_yen}}
    trio_df["k"] = trio_df["combination"].apply(_parse_combo)
    trio_df = trio_df.dropna(subset=["k"])
    trio_map: dict[str, dict] = {}
    for r in trio_df.itertuples(index=False):
        trio_map.setdefault(r.race_key, {})[r.k] = int(round(r.odds_value * 100))

    # 実際の結果: {race_key: frozenset(top3 frames)}
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    return df, trio_map, actual_trio


# ─── レース指標 ──────────────────────────────────────────────────────────────

def race_metrics(grp):
    grp = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    if len(grp) < 4:
        return None
    probs = grp["pred_prob"].tolist()
    p1, p2, p3 = probs[0], probs[1], probs[2]
    p4 = probs[3] if len(probs) > 3 else 0.0
    p5 = probs[4] if len(probs) > 4 else 0.0
    gap12 = p1 - p2
    gap34 = p3 - p4
    gap45 = p4 - p5
    n = int(grp["n_entries"].iloc[0])
    ratio = p1 / (3 / n) if n > 0 else 0
    tier = _assign_tier(gap12, ratio)
    frames = grp["frame_no"].astype(int).tolist()
    return {
        "gap12": gap12, "gap34": gap34, "gap45": gap45,
        "tier": tier, "ratio": ratio,
        "pred1": frames[0], "pred2": frames[1],
        "pred3": frames[2],
        "pred4": frames[3] if len(frames) > 3 else None,
        "pred5": frames[4] if len(frames) > 4 else None,
        "thirds": frames[2:2 + N_THIRDS],
    }


def get_period(d):
    if TRAIN[0] <= d <= TRAIN[1]: return "TRAIN"
    if VAL[0]   <= d <= VAL[1]:   return "VAL"
    if HOLD[0]  <= d <= HOLD[1]:  return "HOLD"
    return None


# ─── Part1: 目別ポジション分析 ─────────────────────────────────────────────

def positional_analysis(df, trio_map, actual_trio):
    """pred3/pred4/pred5 それぞれが「3着目」として的中した時の統計。"""
    pos_recs = []

    for race_key, grp in df.groupby("race_key"):
        period = get_period(grp["race_date"].iloc[0])
        if period is None: continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6: continue

        m = race_metrics(grp)
        if m is None: continue

        p1, p2, thirds = m["pred1"], m["pred2"], m["thirds"]
        race_trio = trio_map.get(race_key, {})
        actual_t = actual_trio.get(race_key, frozenset())

        # レース単位ガミ足切り（現行と同じ）
        combo_keys = [frozenset({p1, p2, t}) for t in thirds]
        valid_odds = [race_trio.get(k, 0) for k in combo_keys if race_trio.get(k, 0) > 0]
        if not valid_odds or min(valid_odds) < GAMI_RACE_THRESHOLD * 100:
            continue

        for pos, t in enumerate(thirds):
            k = frozenset({p1, p2, t})
            odds_yen = race_trio.get(k, 0)
            hit = int(actual_t == k)
            pos_recs.append({
                "period": period,
                "pos": pos + 3,        # 3=pred3, 4=pred4, 5=pred5
                "hit": hit,
                "odds_yen": odds_yen,
                "pay": odds_yen if hit else 0,
            })

    recs = pd.DataFrame(pos_recs)
    val_hold = recs[recs["period"].isin(["VAL", "HOLD"])]

    print(f"\n{'='*70}")
    print("Part1: 目別ポジション分析（≤6車・ガミ≥5倍・VAL+HOLD）")
    print(f"{'='*70}")
    print(f"  {'位置':>4}  {'n':>6}  {'的中%':>8}  {'avg_odds':>10}  {'ROI':>9}")
    print("  " + "-" * 48)
    for pos in [3, 4, 5]:
        s = val_hold[val_hold["pos"] == pos]
        if len(s) == 0: continue
        hit_pct = s["hit"].mean() * 100
        avg_odds = s["odds_yen"].mean() / 100
        roi = s["pay"].sum() / (len(s) * 100) * 100
        mark = "★" if roi >= 100 else ""
        print(f"  pred{pos}:  {len(s):>6}  {hit_pct:>7.1f}%  {avg_odds:>9.1f}倍  {roi:>8.1f}%{mark}")

    # オッズ帯別
    print(f"\n  【個別コンボオッズ帯別 hit率・ROI（VAL+HOLD 全3ポジション合計）】")
    bins = [(0, 500), (500, 800), (800, 1000), (1000, 1500), (1500, 3000), (3000, 10000)]
    print(f"  {'オッズ帯':>14}  {'n':>6}  {'的中%':>8}  {'ROI':>9}")
    print("  " + "-" * 46)
    for lo, hi in bins:
        s = val_hold[(val_hold["odds_yen"] >= lo) & (val_hold["odds_yen"] < hi)]
        if len(s) == 0: continue
        label = f"{lo//100:.0f}〜{hi//100:.0f}倍"
        hit_pct = s["hit"].mean() * 100
        roi = s["pay"].sum() / (len(s) * 100) * 100
        mark = "★" if roi >= 100 else ""
        print(f"  {label:>14}  {len(s):>6}  {hit_pct:>7.1f}%  {roi:>8.1f}%{mark}")

    return recs


# ─── Part2-5: 戦略ROI ────────────────────────────────────────────────────────

class Strat:
    def __init__(self, name):
        self.name = name
        self.rec = []

    def add(self, period, pay, n_bets, unit=100):
        if n_bets <= 0: return
        self.rec.append({"period": period, "pay": pay,
                          "cost": n_bets * unit, "hit": int(pay > 0),
                          "n_bets": n_bets})

    def summary(self):
        df = pd.DataFrame(self.rec)
        rows = []
        for p in ["TRAIN", "VAL", "HOLD"]:
            s = df[df["period"] == p]
            if len(s) == 0: continue
            cost = s["cost"].sum()
            pay = s["pay"].sum()
            roi = pay / cost * 100 if cost > 0 else 0
            avg_bets = s["n_bets"].mean()
            rows.append({
                "period": p, "n": len(s),
                "avg_bets": avg_bets,
                "hit%": s["hit"].sum() / len(s) * 100,
                "ROI%": roi,
            })
        return pd.DataFrame(rows)


def evaluate_strategies(df, trio_map, actual_trio):
    strategy_defs = {
        "S0":    ("現行 3点",                lambda odds_yen, m: [i for i in range(3)]),
        "P1":    ("pred3 のみ 1点",           lambda o, m: [0]),
        "P2":    ("pred3+pred4 2点",          lambda o, m: [0, 1]),
        "O5":    ("各目≥5倍のみ(個別)",        lambda o, m: [i for i, ov in enumerate(o) if ov >= 500]),
        "O8":    ("各目≥8倍のみ",              lambda o, m: [i for i, ov in enumerate(o) if ov >= 800]),
        "O10":   ("各目≥10倍のみ",             lambda o, m: [i for i, ov in enumerate(o) if ov >= 1000]),
        "O15":   ("各目≥15倍のみ",             lambda o, m: [i for i, ov in enumerate(o) if ov >= 1500]),
        "O_hi50":("各目≤50倍のみ",             lambda o, m: [i for i, ov in enumerate(o) if 0 < ov <= 5000]),
        "G_p2":  ("gap34≥0.03→1点/他→2点",   None),   # 特別処理
        "G_p3":  ("gap34≥0.05→1点/他→2点",   None),
        "H1":    ("P2+O8 (2点&各目≥8倍)",     lambda o, m: [i for i in [0, 1] if o[i] >= 800]),
        "H2":    ("P2+O10 (2点&各目≥10倍)",   lambda o, m: [i for i in [0, 1] if o[i] >= 1000]),
        "H3":    ("G_p2+O8 (gap動的&≥8倍)",   None),
        "H4":    ("P2+O5 (2点&各目≥5倍)",     lambda o, m: [i for i in [0, 1] if o[i] >= 500]),
    }

    strats = {k: Strat(f"{k}: {v[0]}") for k, v in strategy_defs.items()}

    for race_key, grp in df.groupby("race_key"):
        period = get_period(grp["race_date"].iloc[0])
        if period is None: continue
        n_entries = grp["n_entries"].iloc[0]
        if n_entries > 6: continue

        m = race_metrics(grp)
        if m is None: continue

        p1, p2, thirds = m["pred1"], m["pred2"], m["thirds"]
        race_trio = trio_map.get(race_key, {})
        actual_t = actual_trio.get(race_key, frozenset())

        # レース単位ガミ足切り（共通）
        combo_keys = [frozenset({p1, p2, t}) for t in thirds]
        odds_yen = [race_trio.get(k, 0) for k in combo_keys]  # 0=取得不可
        valid_min = min((o for o in odds_yen if o > 0), default=0)
        if valid_min < GAMI_RACE_THRESHOLD * 100:
            continue

        gap34 = m["gap34"]

        for key, (name, fn) in strategy_defs.items():
            if fn is not None:
                indices = fn(odds_yen, m)
            elif key == "G_p2":
                indices = [0] if gap34 >= 0.03 else [0, 1]
            elif key == "G_p3":
                indices = [0] if gap34 >= 0.05 else [0, 1]
            elif key == "H3":
                base = [0] if gap34 >= 0.03 else [0, 1]
                indices = [i for i in base if odds_yen[i] >= 800]
            else:
                indices = []

            # 有効なインデックスのみ（オッズ 0 = 取得不可は除外）
            indices = [i for i in indices if i < len(thirds) and odds_yen[i] > 0]
            if not indices:
                continue

            # 的中チェック
            pay = 0
            for i in indices:
                k = combo_keys[i]
                if actual_t == k:
                    pay = odds_yen[i]
                    break

            strats[key].add(period, pay, len(indices))

    return strats


# ─── Part3 補足: gap12帯 × 戦略 ─────────────────────────────────────────────

def gap12_band_roi(df, trio_map, actual_trio):
    """VAL+HOLD で gap12 帯 × S0/P2/H1 ROI 比較"""
    bands = [
        ("<0.06",     0.00, 0.06),
        ("0.06-0.10", 0.06, 0.10),
        ("0.10-0.15", 0.10, 0.15),
        ("0.15+",     0.15, 1.00),
    ]
    key_strats = ["S0", "P2", "H1", "H2"]

    rec_rows = []
    for band_label, lo, hi in bands:
        totals = {k: {"pay": 0, "cost": 0, "hit": 0, "n": 0} for k in key_strats}

        for race_key, grp in df.groupby("race_key"):
            period = get_period(grp["race_date"].iloc[0])
            if period not in ("VAL", "HOLD"): continue
            n_entries = grp["n_entries"].iloc[0]
            if n_entries > 6: continue
            m = race_metrics(grp)
            if m is None: continue
            if not (lo <= m["gap12"] < hi): continue

            p1, p2, thirds = m["pred1"], m["pred2"], m["thirds"]
            race_trio = trio_map.get(race_key, {})
            actual_t = actual_trio.get(race_key, frozenset())
            combo_keys = [frozenset({p1, p2, t}) for t in thirds]
            odds_yen = [race_trio.get(k, 0) for k in combo_keys]
            valid_min = min((o for o in odds_yen if o > 0), default=0)
            if valid_min < GAMI_RACE_THRESHOLD * 100: continue

            def eval_indices(indices):
                indices = [i for i in indices if i < len(thirds) and odds_yen[i] > 0]
                if not indices: return 0, 0
                pay = 0
                for i in indices:
                    if actual_t == combo_keys[i]:
                        pay = odds_yen[i]
                        break
                return pay, len(indices)

            idx_map = {
                "S0": list(range(len(thirds))),
                "P2": [0, 1],
                "H1": [i for i in [0, 1] if odds_yen[i] >= 800],
                "H2": [i for i in [0, 1] if odds_yen[i] >= 1000],
            }
            for k, idxs in idx_map.items():
                pay, nb = eval_indices(idxs)
                if nb > 0:
                    totals[k]["pay"]  += pay
                    totals[k]["cost"] += nb * 100
                    totals[k]["hit"]  += int(pay > 0)
                    totals[k]["n"]    += 1

        row = {"band": band_label}
        for k in key_strats:
            t = totals[k]
            row[f"{k}_n"]    = t["n"]
            row[f"{k}_hit"]  = t["hit"] / t["n"] * 100 if t["n"] > 0 else 0
            row[f"{k}_roi"]  = t["pay"] / t["cost"] * 100 if t["cost"] > 0 else 0
        rec_rows.append(row)

    print(f"\n{'='*88}")
    print("Part補足: gap12帯別 ROI（VAL+HOLD）  S0=3点 / P2=2点 / H1=2点&≥8倍 / H2=2点&≥10倍")
    print(f"{'='*88}")
    print(f"{'帯':<12} {'n':>5}  {'S0 hit%':>9}{'S0 ROI':>8}  {'P2 hit%':>9}{'P2 ROI':>8}  "
          f"{'H1 hit%':>9}{'H1 ROI':>8}  {'H2 hit%':>9}{'H2 ROI':>8}")
    print("-" * 88)
    for row in rec_rows:
        def roi_s(roi): return f"{roi:>7.1f}%" + ("★" if roi >= 100 else " ")
        def hit_s(h):   return f"{h:>8.1f}%"
        n = row["S0_n"]
        print(f"{row['band']:<12} {n:>5}  "
              f"{hit_s(row['S0_hit'])}{roi_s(row['S0_roi'])}  "
              f"{hit_s(row['P2_hit'])}{roi_s(row['P2_roi'])}  "
              f"{hit_s(row['H1_hit'])}{roi_s(row['H1_roi'])}  "
              f"{hit_s(row['H2_hit'])}{roi_s(row['H2_roi'])}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def print_table(strats):
    print(f"\n{'='*78}")
    print("Part2-5: 戦略別 ROI サマリ")
    print(f"{'='*78}")
    print(f"  {'戦略':<44} {'期間':<6} {'対象R':>7} {'avg点':>6} {'的中%':>7} {'ROI':>8}")
    print("  " + "-" * 72)
    for key, strat in strats.items():
        summ = strat.summary()
        if summ.empty: continue
        for _, row in summ.iterrows():
            mark = "★" if row["ROI%"] >= 100 else ""
            print(f"  {strat.name:<44} {row['period']:<6} {row['n']:>7} "
                  f"{row['avg_bets']:>5.1f}点 {row['hit%']:>6.1f}% {row['ROI%']:>7.1f}%{mark}")
        print()


def main():
    print("三連複 買い目削減によるROI改善検証 (doc50)")
    print("データ準備中...")
    df, trio_map, actual_trio = load_all()

    positional_analysis(df, trio_map, actual_trio)
    strats = evaluate_strategies(df, trio_map, actual_trio)
    print_table(strats)
    gap12_band_roi(df, trio_map, actual_trio)

    print(f"\n{'='*78}")
    print("解釈ポイント")
    print(f"{'='*78}")
    print("  ・Part1: pred5 が pred3 より低着率・低ROI なら P2(2点)への削減でROI改善余地あり")
    print("  ・O8/O10: 個別コンボの低オッズ目をカット → 高オッズ目に集中")
    print("  ・G_p2/p3: gap34 大なら pred3 に集中・小なら pred3+4 の分散が有効")
    print("  ・H1/H2: 2点削減 + 個別足切りの複合策")
    print("  ・全て最終オッズ上限値。採否判断は live 実測のみ。")


if __name__ == "__main__":
    main()
