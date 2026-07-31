"""【発見済み全信号の積み上げ・ウェイト最適化・交互作用の自動探索】（2026-07-30）。

## 目的：到達可能な上界を確定させる

本セッションで rider-level の本物の信号を6つ発見したが、いずれも比 1.02〜1.06 だった
（[[keirin_c_candidates_market_test_2026_07_30]] /
  [[keirin_interaction_features_market_test_2026_07_30]]）。
単純な「シグナル個数」の積み上げでは 1.056（ROI 79%）で止まった。

本スクリプトは「**最適に重み付けし、交互作用も自由に使ったら 1.333 に届くか**」という
上界の問いに答える。届かなければ、この特徴量セットでの探索は数学的に閉じる。

## 手法：市場含意確率をベースラインに置いた残差学習

`inputs/外部データ源と交互作用特徴量_深掘り調査.md` II-5 の推奨:
「implied_prob を目的変数から差し引いた残差を学習させる（residual learning）など、
  明示的に『市場が説明できない部分』だけを学習する設計にすること」

これを実装する。3モデルを TRAIN のみで学習し TEST で評価:

- **M0（市場のみ）**: logit(p) = a + b·logit(mp)
  → 市場の再較正だけ。ここからの改善が我々の付加価値
- **M1（線形ウェイト）**: logit(p) = a + b·logit(mp) + Σ wᵢ·xᵢ
  → **ウェイト見直しに相当**。係数が解釈可能で、どの信号にどれだけ重みが乗るか分かる
- **M2（LightGBM・交互作用自動探索）**: init_score = logit(mp) で残差のみを学習
  → **掛け合わせに相当**。木モデルなので任意の交互作用・非線形を自動で拾う。
    人手で設計した交互作用より広い空間を探索するため、これが実質的な**上界**になる

## 評価

    比 = 実測3着内率 ÷ 市場の3着内含意確率
    ROI(軸) = 0.75 × 比 → ROI100%超には 比 ≥ 1.333

- 予測比 `p_hat / mp` で TEST を十分位に分割し、各分位の実測比を出す
  （較正が正しければ単調増加し、最上位分位が到達可能な最大の比になる）
- 上位1%・5%・per-race top1 も出す（絞り込みの極限）
- **ペア単位のS7型ROI**も測る（実際の買い目は軸2車＋5点流しなので）

## 重要な注意

M2 が TEST で高い比を出しても、それは「TRAINで見つけた重みがTESTでも通用した」
という意味であり、**多重比較の心配は小さい**（単一のモデルを1回だけ評価する構成）。
逆に M2 でも届かなければ、人手の組み合わせ探索を続けても無駄と確定できる。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
Elo は 2023-01 から warm-up。DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

from exp_c_candidates_market_test import (  # noqa: E402
    ELO_INIT, ELO_K, ISLAND, MEAS_FROM, MEAS_TO, MIN_BOARD, PREF_LATLON,
    ROI_BREAKEVEN, TAKEOUT_RETURN, TRAIN_TO, haversine, load_trio_odds,
)

HIST_FROM = "2023-01-01"
JST = timezone(timedelta(hours=9))
FH_MIN, FH_MAX = 8.0, 20.0
EPS = 1e-6

FEATURES = [
    "logit_mp",            # 市場（ベースライン）
    "score_z",             # レース内の得点z（基本）
    "elo_resid",           # ★C-5: elo z − 得点 z
    "elo_trend_30d",
    "n30", "n90",          # ★C-1: 連戦負荷
    "days_since", "travel_burden", "prev_dnf",
    "is_nige", "is_oi", "n_senko", "is_lone_senko",   # ★II-2: 単独先行
    "b_top2_gap", "b_gini", "b_rank", "s_rank",
    "prev_best_agari_out", "prev_finish_minus_agari", "prev_agari_rank",  # ★C-7
    "line_pos", "line_size", "is_isolated",
    "bante_minus_sanbante", "is_ordered_line",        # ★II-3
    "car_no", "is_midnight", "is_mixed_class",
]


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def valid_fh(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if FH_MIN <= f <= FH_MAX else None


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


def zstats(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    sd = statistics.pstdev(v)
    return (m, sd) if sd > 0 else None


def load_all():
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, race_no, cup_id, venue_id, n_entries, start_at "
            "FROM wt_races WHERE cancel = 0 AND race_date BETWEEN ? AND ?",
            (HIST_FROM, MEAS_TO)).fetchall()
        venues = {str(r["venue_code"]): r["prefecture"]
                  for r in c.execute("SELECT venue_code, prefecture FROM venue_info").fetchall()}
    races = {r["race_key"]: {
        "date": str(r["race_date"]), "race_no": r["race_no"] or 0,
        "cup_id": r["cup_id"], "n_entries": r["n_entries"],
        "start_at": r["start_at"], "pref": venues.get(str(r["venue_id"])),
    } for r in rrows}
    print(f"[load] races: {len(races)}", flush=True)

    keys = list(races)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, player_id, pred_top3_pct, player_class, "
                 "       style, race_point, s_count, b_count, line_group, line_pos, "
                 "       line_size, finish_order, final_half "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())}", flush=True)
    return races, by_race


def build_history(races, entries_by_race):
    """chronological に走査して選手単位の point-in-time 履歴特徴を作る。"""
    order = sorted(races, key=lambda rk: (races[rk]["date"], races[rk]["race_no"]))
    elo = defaultdict(lambda: ELO_INIT)
    elo_hist = defaultdict(list)
    last_race, last_cup, cup_travel = {}, {}, {}
    race_dates = defaultdict(list)
    prev_agari = {}          # pid -> (finish_order, agari_rank, n_ranked)
    out = {}

    for rk in order:
        meta = races[rk]
        ents = entries_by_race.get(rk)
        if not ents:
            continue
        d = meta["date"]
        dt = date.fromisoformat(d)
        cur_pref = meta["pref"]
        cur_ll = PREF_LATLON.get(cur_pref) if cur_pref else None

        for e in ents:
            pid = e["player_id"]
            prev = last_race.get(pid)
            days_since = (dt - date.fromisoformat(prev[0])).days if prev else None
            prev_dnf = prev[1] if prev else 0

            ck = (meta["cup_id"], pid)
            if ck not in cup_travel:
                lc = last_cup.get(pid)
                if lc and lc[0] != meta["cup_id"]:
                    _, ld, lp = lc
                    dk = haversine(PREF_LATLON.get(lp) if lp else None, cur_ll)
                    gap = (dt - date.fromisoformat(ld)).days
                    cup_travel[ck] = (dk, gap)
                else:
                    cup_travel[ck] = (None, None)
            dk, gap = cup_travel[ck]
            burden = (dk / max(gap, 1)) if (dk is not None and gap is not None) else None

            dl = race_dates.get(pid, [])
            n30 = sum(1 for x in dl if 0 < (dt - date.fromisoformat(x)).days <= 30)
            n90 = sum(1 for x in dl if 0 < (dt - date.fromisoformat(x)).days <= 90)

            et = None
            h = elo_hist.get(pid)
            if h:
                et = elo[pid] - h[0][1]
                for d0, e0 in reversed(h):
                    if (dt - d0).days >= 30:
                        et = elo[pid] - e0
                        break

            out[(rk, pid)] = {
                "days_since": days_since, "prev_dnf": prev_dnf,
                "travel_burden": burden, "n30": n30, "n90": n90,
                "elo": elo[pid], "elo_trend_30d": et,
                "prev_agari": prev_agari.get(pid),
            }

        # 更新: Elo
        fin = [(int(e["finish_order"]), e["player_id"]) for e in ents
               if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
        if len(fin) >= 2:
            fin.sort()
            pids = [p for _, p in fin]
            n = len(pids)
            delta = defaultdict(float)
            for a in range(n):
                for b in range(a + 1, n):
                    pa, pb = pids[a], pids[b]
                    ea = 1.0 / (1.0 + 10 ** ((elo[pb] - elo[pa]) / 400.0))
                    g = ELO_K * (1.0 - ea) / (n - 1)
                    delta[pa] += g
                    delta[pb] -= g
            for p, g in delta.items():
                elo_hist[p].append((dt, elo[p]))
                elo[p] += g

        # 更新: 前走の上がり順位
        ranked = []
        for e in ents:
            fo = e["finish_order"]
            fh = valid_fh(e["final_half"])
            if fo is not None and int(fo) >= 1 and fh is not None:
                ranked.append((fh, int(fo), e["player_id"]))
        if len(ranked) >= 3:
            ranked.sort()
            for i, (_fh, fo, pid) in enumerate(ranked):
                prev_agari[pid] = (fo, i + 1, len(ranked))

        for e in ents:
            pid = e["player_id"]
            fo = e["finish_order"]
            dnf = 1 if (fo is not None and int(fo) == 0 and e["final_half"] is not None) else 0
            last_race[pid] = (d, dnf)
            last_cup[pid] = (meta["cup_id"], d, cur_pref)
            race_dates[pid].append(d)

    print(f"[hist] {len(out)} 件", flush=True)
    return out


def main():
    races, entries = load_all()
    hist = build_history(races, entries)

    targets = [rk for rk, m in races.items()
               if m["n_entries"] == 7 and MEAS_FROM <= m["date"] <= MEAS_TO]
    print(f"[meas] 対象: {len(targets)}R", flush=True)

    rows = []          # dict per rider
    pairs = []         # per race のペア情報（S7型ROI用）
    by_month = defaultdict(list)
    for rk in targets:
        by_month[races[rk]["date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        boards = load_trio_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            board = boards.get(rk)
            if not ents or len(ents) != 7 or not board or len(board) < MIN_BOARD:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 3:
                continue
            fin.sort()
            tm = 0
            for _, fr in fin[:3]:
                tm |= 1 << fr

            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            frames = sorted(int(e["frame_no"]) for e in ents)
            marg = {fr: 0.0 for fr in frames}
            pair_mkt = defaultdict(float)
            for m, v in mk_raw.items():
                p = v / tot
                fs = [fr for fr in frames if (m >> fr) & 1]
                for fr in fs:
                    marg[fr] += p
                for a, b in combinations(fs, 2):
                    pair_mkt[(a, b)] += p

            w = "TRAIN" if meta["date"] <= TRAIN_TO else "TEST"
            try:
                hh = datetime.fromtimestamp(int(str(meta["start_at"])), tz=JST).hour
            except (TypeError, ValueError):
                hh = 12
            is_mid = 1 if hh >= 20 else 0

            elos = [hist.get((rk, e["player_id"]), {}).get("elo") for e in ents]
            rps = [float(e["race_point"]) if e["race_point"] is not None else None
                   for e in ents]
            ze, zr = zstats(elos), zstats(rps)
            bcs = [float(e["b_count"]) if e["b_count"] is not None else 0.0 for e in ents]
            bsort = sorted(bcs, reverse=True)
            b_gap = bsort[0] - bsort[1] if len(bsort) >= 2 else 0.0
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

            race_rows = {}
            for idx, e in enumerate(ents):
                fr = int(e["frame_no"])
                mp = marg[fr]
                if mp <= 0:
                    continue
                h = hist.get((rk, e["player_id"]), {})
                pa = h.get("prev_agari")
                sty = str(e["style"] or "?")
                is_nige = 1 if sty == "逃" else 0
                bm = bms.get(fr)

                r = {
                    "race_key": rk, "frame": fr, "w": w, "mp": mp,
                    "y": 1.0 if (tm >> fr) & 1 else 0.0,
                    "logit_mp": logit(mp),
                    "score_z": ((rps[idx] - zr[0]) / zr[1]) if (zr and rps[idx] is not None) else 0.0,
                    "elo_resid": (((elos[idx] - ze[0]) / ze[1]) - ((rps[idx] - zr[0]) / zr[1]))
                                 if (ze and zr and elos[idx] is not None and rps[idx] is not None) else 0.0,
                    "elo_trend_30d": h.get("elo_trend_30d") or 0.0,
                    "n30": float(h.get("n30") or 0), "n90": float(h.get("n90") or 0),
                    "days_since": float(h.get("days_since") or 0),
                    "travel_burden": float(h.get("travel_burden") or 0),
                    "prev_dnf": float(h.get("prev_dnf") or 0),
                    "is_nige": float(is_nige),
                    "is_oi": 1.0 if sty == "追" else 0.0,
                    "n_senko": float(n_senko),
                    "is_lone_senko": 1.0 if (is_nige and n_senko == 1) else 0.0,
                    "b_top2_gap": b_gap, "b_gini": b_gi,
                    "b_rank": float(b_rk.get(fr, 0)), "s_rank": float(s_rk.get(fr, 0)),
                    "prev_best_agari_out": 1.0 if (pa and pa[1] == 1 and pa[0] > 3) else 0.0,
                    "prev_finish_minus_agari": float(pa[0] - pa[1]) if pa else 0.0,
                    "prev_agari_rank": float(pa[1]) if pa else 0.0,
                    "line_pos": float(e["line_pos"] or 0),
                    "line_size": float(e["line_size"] or 1),
                    "is_isolated": 1.0 if (e["line_size"] or 1) == 1 else 0.0,
                    "bante_minus_sanbante": float(bm) if bm is not None else 0.0,
                    "is_ordered_line": 1.0 if (bm is not None and bm > 1.0) else 0.0,
                    "car_no": float(fr), "is_midnight": float(is_mid),
                    "is_mixed_class": float(mixed),
                }
                rows.append(r)
                race_rows[fr] = r

            if race_rows:
                pairs.append({"race_key": rk, "w": w, "top3_mask": tm,
                              "pair_mkt": dict(pair_mkt), "frames": sorted(race_rows)})
        print(f"  {ym}: {len(rks)}R (累計 {len(rows)} 車)", flush=True)

    # ---------------- 学習 ----------------
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    tr = [r for r in rows if r["w"] == "TRAIN"]
    te = [r for r in rows if r["w"] == "TEST"]
    print(f"\n[fit] TRAIN {len(tr)} 車 / TEST {len(te)} 車")

    def X(rs, cols):
        return np.array([[r[c] for c in cols] for r in rs], dtype=np.float64)

    y_tr = np.array([r["y"] for r in tr])
    y_te = np.array([r["y"] for r in te])
    mp_tr = np.array([r["mp"] for r in tr])
    mp_te = np.array([r["mp"] for r in te])

    # M0: 市場のみ
    X0_tr, X0_te = X(tr, ["logit_mp"]), X(te, ["logit_mp"])
    m0 = LogisticRegression(max_iter=2000).fit(X0_tr, y_tr)
    p0 = m0.predict_proba(X0_te)[:, 1]

    # M1: 線形ウェイト（市場 + 全信号）
    X1_tr, X1_te = X(tr, FEATURES), X(te, FEATURES)
    mu, sd = X1_tr.mean(axis=0), X1_tr.std(axis=0)
    sd[sd == 0] = 1.0
    m1 = LogisticRegression(max_iter=5000, C=1.0).fit((X1_tr - mu) / sd, y_tr)
    p1 = m1.predict_proba((X1_te - mu) / sd)[:, 1]

    # M2: LightGBM（init_score = logit(mp) で残差のみ学習＝交互作用の自動探索）
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
    p2 = 1.0 / (1.0 + np.exp(-(base_te + m2.predict(X2_te, raw_score=True))))

    print("\n" + "=" * 112)
    print("M1 の学習済みウェイト（標準化後の係数・|値|降順）＝ウェイト見直しの結果")
    print("=" * 112)
    coefs = sorted(zip(FEATURES, m1.coef_[0]), key=lambda x: -abs(x[1]))
    for name, c in coefs:
        print(f"    {name:<26}{c:+.4f}")

    print("\n" + "=" * 112)
    print("M2 の特徴量重要度（gain・上位12）＝掛け合わせで実際に使われた変数")
    print("=" * 112)
    imp = sorted(zip(feat_no_mp, m2.feature_importance("gain")), key=lambda x: -x[1])
    tot_g = sum(v for _n, v in imp) or 1.0
    for name, g in imp[:12]:
        print(f"    {name:<26}{g/tot_g*100:>6.2f}%")

    # ---------------- 評価 ----------------
    def eval_bins(pred, label):
        ratio = pred / np.maximum(mp_te, EPS)
        order = np.argsort(-ratio)
        print("\n" + "-" * 112)
        print(f"{label}: 予測比 p_hat/mp の十分位別 実測比")
        print("-" * 112)
        print(f"{'分位':<12}{'車数':>9}{'実測%':>9}{'市場%':>9}{'実測比':>9}"
              f"{'t値':>8}{'→ROI%':>9}{'予測比平均':>11}")
        n = len(order)
        for d in range(10):
            idx = order[int(n * d / 10):int(n * (d + 1) / 10)]
            yy, mm = y_te[idx], mp_te[idx]
            act, mk = yy.mean(), mm.mean()
            dd = yy - mm
            t = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd))) if len(dd) > 1 else 0.0
            rr = act / mk if mk > 0 else 0.0
            flag = "  ★" if rr >= ROI_BREAKEVEN and t > 3 else ""
            print(f"{'D'+str(d+1)+(' (最高)' if d==0 else ''):<12}{len(idx):>9}"
                  f"{act*100:>9.2f}{mk*100:>9.2f}{rr:>9.3f}{t:>+8.2f}"
                  f"{TAKEOUT_RETURN*rr*100:>9.1f}{ratio[idx].mean():>11.3f}{flag}")
        for pct in (0.05, 0.01):
            idx = order[:max(int(n * pct), 50)]
            yy, mm = y_te[idx], mp_te[idx]
            act, mk = yy.mean(), mm.mean()
            dd = yy - mm
            t = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd))) if len(dd) > 1 else 0.0
            rr = act / mk if mk > 0 else 0.0
            print(f"{'上位'+str(int(pct*100))+'%':<12}{len(idx):>9}{act*100:>9.2f}"
                  f"{mk*100:>9.2f}{rr:>9.3f}{t:>+8.2f}{TAKEOUT_RETURN*rr*100:>9.1f}")
        return ratio

    print("\n" + "=" * 112)
    print("TEST 評価: 比 ≥ 1.333 なら ROI 100%超")
    print("=" * 112)
    eval_bins(p0, "M0（市場のみ・再較正）")
    eval_bins(p1, "M1（線形ウェイト最適化）")
    r2 = eval_bins(p2, "M2（LightGBM・交互作用自動探索）★実質的な上界")

    # ---------------- per-race 軸選定 & ペア単位 S7型ROI ----------------
    pred_by = {}
    for r, pr in zip(te, r2):
        pred_by[(r["race_key"], r["frame"])] = (pr, r["mp"], r["y"])

    print("\n" + "-" * 112)
    print("M2 による per-race 軸選定と S7型（軸2車＋5点流し）ROI")
    print("-" * 112)
    n1 = h1 = 0
    s_act = s_mkt = 0.0
    p_n = p_hit = 0
    p_mkt_sum = 0.0
    for pinfo in pairs:
        if pinfo["w"] != "TEST":
            continue
        cand = [(pred_by[(pinfo["race_key"], fr)][0], fr) for fr in pinfo["frames"]
                if (pinfo["race_key"], fr) in pred_by]
        if len(cand) < 2:
            continue
        cand.sort(reverse=True)
        # top1 = 軸1車
        pr, fr = cand[0]
        _p, mp, y = pred_by[(pinfo["race_key"], fr)]
        n1 += 1
        h1 += int(y)
        s_act += y
        s_mkt += mp
        # top2 = 軸ペア → S7型5点流し
        a, b = sorted([cand[0][1], cand[1][1]])
        mk = pinfo["pair_mkt"].get((a, b))
        if mk and mk > 0:
            tm = pinfo["top3_mask"]
            hit = 1 if ((tm >> a) & 1 and (tm >> b) & 1) else 0
            p_n += 1
            p_hit += hit
            p_mkt_sum += mk
    if n1:
        rr = (s_act / n1) / (s_mkt / n1)
        print(f"  軸1車（per-race top1）: n={n1}  実測3着内 {s_act/n1*100:.2f}% / "
              f"市場 {s_mkt/n1*100:.2f}%  比 {rr:.3f}  → ROI {TAKEOUT_RETURN*rr*100:.1f}%")
    if p_n:
        pr_ = (p_hit / p_n) / (p_mkt_sum / p_n)
        print(f"  軸2車ペア（S7型5点流し）: n={p_n}  実測的中 {p_hit/p_n*100:.2f}% / "
              f"市場 {p_mkt_sum/p_n*100:.2f}%  比 {pr_:.3f}  → **ROI {TAKEOUT_RETURN*pr_*100:.1f}%**")

    print("\n" + "=" * 112)
    print("【結論】")
    print("=" * 112)
    best = max((y_te[np.argsort(-(p / np.maximum(mp_te, EPS)))[:max(int(len(te)*0.01), 50)]].mean()
                / mp_te[np.argsort(-(p / np.maximum(mp_te, EPS)))[:max(int(len(te)*0.01), 50)]].mean())
               for p in (p0, p1, p2))
    print(f"  3モデル×上位1%での最大の実測比 = {best:.3f}（必要値 {ROI_BREAKEVEN:.3f}）")
    if best >= ROI_BREAKEVEN:
        print("  → ROI100%超に到達する組み合わせが存在する。")
    else:
        print("  → **最適ウェイト・自動交互作用探索をもってしても 1.333 に届かない。**")
        print("     人手での信号の組み替え・重み調整による探索は数学的に閉じたと判断できる。")


if __name__ == "__main__":
    main()
