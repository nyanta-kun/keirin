"""【レース選択×車単位信号スタックの掛け合わせ】（2026-07-30）。

これまでに独立に発見した2つの改善軸を掛け合わせ、複合効果を測る。

  A. [[keirin_signal_stack_weight_optimization_2026_07_30]]
     車単位信号スタック（LightGBM残差学習・M2）: TEST上位1%で比1.169（n=941・実用性は低い）
  B. [[keirin_netkeirin_race_selection_verification_2026_07_30]]
     レース選択（top2_prob_sum上位20%＝Mr.T的「二軸の堅さ」）: TEST比1.123（n=2,388・実用的規模）

問い: **「堅いレース」に絞った上で、その中でさらにM2スコアが高い軸ペアを選べば、
比はどこまで伸びるか。** 測定単位はメモの指摘通り quinella（二車複・1-2着占有、
順序不問）に統一する（trio/3着内ではなくMr.Tの実際の買い目構造に対応する単位）。

## 方法

1. [[keirin_c_candidates_market_test_2026_07_30]] / [[keirin_interaction_features_market_test_2026_07_30]]
   / [[keirin_signal_stack_weight_optimization_2026_07_30]] と同じ特徴量セットで
   M2（LightGBM・init_score=logit(market_top3_marginal)）を学習
2. レース単位の「堅さ」= その レースの pred_top3_pct 上位2車の合計
   （`top2_prob_sum`）をTRAINで五分位/十分位カットしTESTに固定適用
3. 各レースで M2 予測比の高い順に2車を軸ペアとして選び、quinella市場の
   含意確率と比較（実測=そのペアが1-2着を占めたか）
4. 「堅さ」で絞った場合と絞らない場合、さらに「堅さ×M2上位絞り込み」の
   二重フィルタでの比を比較する

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

from exp_signal_stack_weight_optimization import (  # noqa: E402
    EPS, FEATURES, MEAS_FROM, MEAS_TO, MIN_BOARD, ROI_BREAKEVEN, TAKEOUT_RETURN,
    TRAIN_TO, build_history, load_all, logit,
)

MIN_SEG = 150


def load_quinella_odds(race_keys):
    out = {}
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'quinella' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(sorted(int(x) for x in str(comb).split("=")))
                except ValueError:
                    continue
                if len(parts) != 2:
                    continue
                out.setdefault(rk, {})[parts] = fv
    return out


def normalize(odds_map):
    raw = {k: TAKEOUT_RETURN / o for k, o in odds_map.items() if o > 0}
    tot = sum(raw.values())
    if tot <= 0:
        return None
    return {k: v / tot for k, v in raw.items()}


class Acc:
    __slots__ = ("n", "hit", "mkt", "d", "d2")

    def __init__(self):
        self.n = 0
        self.hit = self.mkt = self.d = self.d2 = 0.0

    def add(self, y, mp):
        self.n += 1
        self.hit += y
        self.mkt += mp
        d = y - mp
        self.d += d
        self.d2 += d * d

    def report(self):
        n = self.n
        act, mkt = self.hit / n, self.mkt / n
        md = self.d / n
        var = max(self.d2 / n - md * md, 0.0)
        t = md / math.sqrt(var / n) if var > 0 else 0.0
        ratio = act / mkt if mkt > 0 else 0.0
        return {"n": n, "act": act * 100, "mkt": mkt * 100, "ratio": ratio, "t": t,
                "roi": TAKEOUT_RETURN * ratio * 100}


def zstats(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    import statistics
    m = statistics.mean(v)
    sd = statistics.pstdev(v)
    return (m, sd) if sd > 0 else None


def main():
    races, entries = load_all()
    hist = build_history(races, entries)

    targets = [rk for rk, m in races.items()
               if m["n_entries"] == 7 and MEAS_FROM <= m["date"] <= MEAS_TO]
    print(f"[meas] 対象: {len(targets)}R", flush=True)

    rows = []           # per-car feature rows（学習用）
    race_info = {}      # race_key -> {"w":..., "top2_prob_sum":..., "frames":[...]}
    by_month = defaultdict(list)
    for rk in targets:
        by_month[races[rk]["date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        trio_boards_needed = None  # 使わない（quinellaのみで判定するため軽量化）
        qn_boards = load_quinella_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            qn = qn_boards.get(rk)
            if not ents or len(ents) != 7 or not qn:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            qn_mkt = normalize(qn)
            if qn_mkt is None or len(qn_mkt) < 15:
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 2:
                continue
            fin.sort()
            actual_pair = tuple(sorted([fin[0][1], fin[1][1]]))

            w = "TRAIN" if meta["date"] <= TRAIN_TO else "TEST"
            top3p = sorted((float(e["pred_top3_pct"]) for e in ents), reverse=True)
            top2_prob_sum = top3p[0] + top3p[1]

            elos = [hist.get((rk, e["player_id"]), {}).get("elo") for e in ents]
            rps = [float(e["race_point"]) if e["race_point"] is not None else None
                   for e in ents]
            ze, zr = zstats(elos), zstats(rps)
            bcs = [float(e["b_count"]) if e["b_count"] is not None else 0.0 for e in ents]
            bsort = sorted(bcs, reverse=True)
            b_gap = bsort[0] - bsort[1] if len(bsort) >= 2 else 0.0
            import statistics as _st

            def gini(vals):
                v = sorted(x for x in vals if x is not None)
                n = len(v)
                if n < 2:
                    return 0.0
                s = sum(v)
                if s <= 0:
                    return 0.0
                cum = sum((i + 1) * x for i, x in enumerate(v))
                return (2 * cum) / (n * s) - (n + 1) / n
            b_gi = gini(bcs)
            n_senko = sum(1 for e in ents if str(e["style"] or "") == "逃")
            classes = {str(e["player_class"] or "?") for e in ents}
            mixed = 1 if len(classes) > 1 else 0

            def rank_of(key):
                v = [(float(e[key]) if e[key] is not None else -1.0, int(e["frame_no"]))
                     for e in ents]
                v.sort(reverse=True)
                return {fr: i + 1 for i, (_x, fr) in enumerate(v)}
            s_rk, b_rk = rank_of("s_count"), rank_of("b_count")

            lines = defaultdict(dict)
            for e in ents:
                if e["line_group"] is not None and e["line_pos"] is not None:
                    lines[e["line_group"]][int(e["line_pos"])] = e
            bms = {}
            for _lg, pos in lines.items():
                if 2 in pos and 3 in pos and pos[2]["race_point"] and pos[3]["race_point"]:
                    g = float(pos[2]["race_point"]) - float(pos[3]["race_point"])
                    bms[int(pos[2]["frame_no"])] = g
                    bms[int(pos[3]["frame_no"])] = g

            try:
                import datetime as _dt
                hh = _dt.datetime.fromtimestamp(
                    int(str(meta["start_at"])),
                    tz=_dt.timezone(_dt.timedelta(hours=9))).hour
            except (TypeError, ValueError, KeyError):
                hh = 12
            is_mid = 1 if hh >= 20 else 0

            frames_here = []
            for idx, e in enumerate(ents):
                fr = int(e["frame_no"])
                mp_top3 = float(e["pred_top3_pct"]) / 100.0    # 車単位モデル学習用のダミーmp
                h = hist.get((rk, e["player_id"]), {})
                pa = h.get("prev_agari")
                sty = str(e["style"] or "?")
                is_nige = 1 if sty == "逃" else 0
                bm = bms.get(fr)

                r = {
                    "race_key": rk, "frame": fr, "w": w,
                    "logit_mp": logit(mp_top3),   # M0/M1/M2はこのbase(市場top3周辺)で残差学習
                    "score_z": ((rps[idx] - zr[0]) / zr[1]) if (zr and rps[idx] is not None) else 0.0,
                    "elo_resid": (((elos[idx] - ze[0]) / ze[1]) - ((rps[idx] - zr[0]) / zr[1]))
                                 if (ze and zr and elos[idx] is not None and rps[idx] is not None) else 0.0,
                    "elo_trend_30d": h.get("elo_trend_30d") or 0.0,
                    "n30": float(h.get("n30") or 0), "n90": float(h.get("n90") or 0),
                    "days_since": float(h.get("days_since") or 0),
                    "travel_burden": float(h.get("travel_burden") or 0),
                    "prev_dnf": float(h.get("prev_dnf") or 0),
                    "is_nige": float(is_nige), "is_oi": 1.0 if sty == "追" else 0.0,
                    "n_senko": float(n_senko),
                    "is_lone_senko": 1.0 if (is_nige and n_senko == 1) else 0.0,
                    "b_top2_gap": b_gap, "b_gini": b_gi,
                    "b_rank": float(b_rk.get(fr, 0)), "s_rank": float(s_rk.get(fr, 0)),
                    "prev_best_agari_out": 1.0 if (pa and pa[1] == 1 and pa[0] > 3) else 0.0,
                    "prev_finish_minus_agari": float(pa[0] - pa[1]) if pa else 0.0,
                    "prev_agari_rank": float(pa[1]) if pa else 0.0,
                    "line_pos": float(e["line_pos"] or 0), "line_size": float(e["line_size"] or 1),
                    "is_isolated": 1.0 if (e["line_size"] or 1) == 1 else 0.0,
                    "bante_minus_sanbante": float(bm) if bm is not None else 0.0,
                    "is_ordered_line": 1.0 if (bm is not None and bm > 1.0) else 0.0,
                    "car_no": float(fr), "is_midnight": float(is_mid),
                    "is_mixed_class": float(mixed),
                }
                rows.append(r)
                frames_here.append(fr)

            race_info[rk] = {"w": w, "top2_prob_sum": top2_prob_sum,
                             "frames": frames_here, "qn_mkt": qn_mkt, "actual_pair": actual_pair}
        print(f"  {ym}: {len(rks)}R (累計{len(rows)}車)", flush=True)

    import lightgbm as lgb

    tr = [r for r in rows if r["w"] == "TRAIN"]
    te = [r for r in rows if r["w"] == "TEST"]
    print(f"\n[fit] TRAIN {len(tr)}車 / TEST {len(te)}車")

    def X(rs, cols):
        return np.array([[r[c] for c in cols] for r in rs], dtype=np.float64)

    y_tr_proxy = None  # 実際のyラベル(車が実際に上位2着に入ったか)を作る
    frame_top2 = {}    # (race_key, frame) -> 1/0 車が実際に1-2着だったか
    for rk, info in race_info.items():
        for fr in info["frames"]:
            frame_top2[(rk, fr)] = 1.0 if fr in info["actual_pair"] else 0.0

    y_tr = np.array([frame_top2.get((r["race_key"], r["frame"]), 0.0) for r in tr])
    y_te = np.array([frame_top2.get((r["race_key"], r["frame"]), 0.0) for r in te])

    feat_no_mp = [c for c in FEATURES if c != "logit_mp"]
    X2_tr, X2_te = X(tr, feat_no_mp), X(te, feat_no_mp)
    base_tr = np.array([r["logit_mp"] for r in tr])
    base_te = np.array([r["logit_mp"] for r in te])
    ds = lgb.Dataset(X2_tr, label=y_tr, init_score=base_tr,
                     feature_name=feat_no_mp, free_raw_data=False)
    m2 = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 31,
                    "min_data_in_leaf": 500, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 1,
                    "lambda_l2": 5.0, "verbose": -1, "seed": 42,
                    "deterministic": True}, ds, num_boost_round=400)
    p_te = 1.0 / (1.0 + np.exp(-(base_te + m2.predict(X2_te, raw_score=True))))

    score_by = {}
    for r, s in zip(te, p_te):
        score_by[(r["race_key"], r["frame"])] = s

    # TRAINで top2_prob_sum の五分位・十分位カットを決定
    tr_sums = sorted(info["top2_prob_sum"] for rk, info in race_info.items() if info["w"] == "TRAIN")
    q5_cuts = [tr_sums[len(tr_sums) * i // 5] for i in range(1, 5)]
    d10_cuts = [tr_sums[len(tr_sums) * i // 10] for i in range(1, 10)]

    def quantile(val, cuts):
        return sum(1 for x in cuts if val >= x) + 1

    # ---- レース単位で軸ペアを M2 スコアで選び、quinella比を測る ----
    acc = defaultdict(lambda: defaultdict(Acc))    # dim -> (w, seg) -> Acc

    for rk, info in race_info.items():
        if info["w"] != "TEST":
            continue
        frames = info["frames"]
        cand = sorted(frames, key=lambda fr: -score_by.get((rk, fr), -9))
        if len(cand) < 2:
            continue
        pair = tuple(sorted(cand[:2]))
        mp = info["qn_mkt"].get(pair)
        if mp is None or mp <= 0:
            continue
        y = 1.0 if pair == info["actual_pair"] else 0.0

        q5 = quantile(info["top2_prob_sum"], q5_cuts)
        d10 = quantile(info["top2_prob_sum"], d10_cuts)
        acc["ALL"][("TEST", "全体")].add(y, mp)
        acc["Q5"][("TEST", f"Q{q5}/5")].add(y, mp)
        if d10 == 10:
            acc["D10"][("TEST", "上位10%以内")].add(y, mp)
        if d10 >= 9:
            acc["D10"][("TEST", "上位20%以内")].add(y, mp)
        if d10 < 9:
            acc["D10"][("TEST", "下位80%")].add(y, mp)

    print("\n" + "=" * 112)
    print("M2軸選定(quinella比) × レース選択(top2_prob_sum) の掛け合わせ")
    print("=" * 112)
    print(f"{'区分':<20}{'セグメント':<16}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for dim, title in (("ALL", "ベースライン"), ("Q5", "五分位"), ("D10", "累積上位%")):
        for seg in sorted({k[1] for k in acc[dim]}):
            a = acc[dim].get(("TEST", seg))
            if not a or a.n < MIN_SEG:
                continue
            p = a.report()
            flag = "  ★" if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3 else ""
            print(f"{title:<20}{seg:<16}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                  f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{flag}")

    print("\n" + "=" * 112)
    print("【結論】比 ≥ {:.3f} かつ t>3 のセグメント".format(ROI_BREAKEVEN))
    print("=" * 112)
    found = False
    for dim in acc:
        for seg in {k[1] for k in acc[dim]}:
            a = acc[dim].get(("TEST", seg))
            if not a or a.n < MIN_SEG:
                continue
            p = a.report()
            if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3:
                found = True
                print(f"  ★[{dim}] {seg}: 比={p['ratio']:.3f} n={p['n']} ROI={p['roi']:.1f}%")
    if not found:
        print("  該当なし。掛け合わせても1.333には届かない。")


if __name__ == "__main__":
    main()
