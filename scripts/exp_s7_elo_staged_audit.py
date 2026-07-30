"""却下済み案「Elo」のS7 honest全期間再検証（2026-07-29）。

exp_s7_gate_staged_audit.py と同じ手法（9四半期vintage+tail、一貫した
walk-forwardモデル）で、Eloレーティング特徴を追加したモデルを新たに学習し、
現行S7(D構成: axis_sum<=1.3 ∧ entropy<=1.8329 ∧ mark3<=1)のhonest全期間ROIを
baseline（Eloなし・既に検証済みのD=86.3%）と比較する。

Elo自体はexp_elo_linecoop_wt.py::compute_elo_and_linecoop()のElo部分のみ再利用
（ライン連携は別候補として今回は対象外）。

学習は各区間ごとに「その時点までのデータのみ」で行う真のwalk-forward
（quarterly vintageモデルと同じ境界: --from 2022-12-01 --test-from/--test-to）。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.preprocessing.feature_wt import (
    build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT, TARGET_COL_WT,
)
from src.strategy_wt import (
    S7_STAKE, s7_field_entropy, s7_select_axis, s7_wt_mark3_overlap_n, s7_wt_overlap_n,
)

ELO_K = 24.0
ELO_SCALE = 400.0
ELO_INIT = 1500.0
ELO_COLS = ["elo", "elo_rank", "elo_z"]

PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1, random_state=42)

S7_AXIS_SUM_MAX = 1.3
S7_ENTROPY_MAX = 1.8329
S7_MARK3_OVERLAP_MAX = 1

# (train_to, test_from, test_to) 四半期境界。lgbm_wt_eval_qXXXXと同じ境界。
WINDOWS = [
    ("2024-01-01", "2024-03-31"),
    ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"),
    ("2026-01-01", "2026-04-12"),
    ("2026-04-13", "2026-07-28"),
]


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """point-in-time Elo（exp_elo_linecoop_wt.pyのElo部分のみ抽出）。"""
    d = df[["race_key", "race_date", "start_at", "player_id", "finish_order"]].copy()
    d["fin"] = pd.to_numeric(d["finish_order"], errors="coerce")
    race_order = (d.groupby("race_key")
                  .agg(race_date=("race_date", "first"), start_at=("start_at", "first"))
                  .sort_values(["race_date", "start_at"])
                  .index.tolist())
    rating = defaultdict(lambda: ELO_INIT)
    groups = {rk: g for rk, g in d.groupby("race_key", sort=False)}
    out_elo = {}
    for rk in race_order:
        g = groups[rk]
        pids = g["player_id"].tolist()
        fins = g["fin"].tolist()
        pre = {p: rating[p] for p in pids}
        for p in pids:
            out_elo[(rk, p)] = pre[p]
        finished = [(p, f) for p, f in zip(pids, fins) if f is not None and f >= 1]
        delta = defaultdict(float)
        for i in range(len(finished)):
            for j in range(i + 1, len(finished)):
                pa, fa = finished[i]
                pb, fb = finished[j]
                if fa == fb:
                    continue
                ea = 1.0 / (1.0 + 10 ** ((pre[pb] - pre[pa]) / ELO_SCALE))
                sa = 1.0 if fa < fb else 0.0
                delta[pa] += ELO_K * (sa - ea)
                delta[pb] += ELO_K * ((1.0 - sa) - (1.0 - ea))
        for p, dl in delta.items():
            rating[p] += dl
    df = df.copy()
    key = list(zip(df["race_key"], df["player_id"]))
    df["elo"] = [out_elo.get(k, ELO_INIT) for k in key]
    grp = df.groupby("race_key")["elo"]
    df["elo_rank"] = grp.rank(ascending=False)
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(1.0).replace(0.0, 1.0)
    df["elo_z"] = ((df["elo"] - mean) / std).clip(-5, 5)
    return df


def _load_trio_boards(race_keys):
    import re
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def score_window(df_all_feat: pd.DataFrame, test_from: str, test_to: str, use_elo: bool):
    """指定ウィンドウをwalk-forward学習し、S7(D構成)候補をhonestにスコアする。"""
    cols = list(FEATURE_COLS_WT) + (ELO_COLS if use_elo else [])
    tr = df_all_feat[df_all_feat["race_date"] < test_from]
    tr = tr[tr["finish_order"].notna()]
    m_top3 = lgb.LGBMClassifier(**PARAMS)
    m_top3.fit(tr[cols].fillna(0).values, tr[TARGET_COL_WT].values)
    m_win = lgb.LGBMClassifier(**PARAMS)
    m_win.fit(tr[cols].fillna(0).values, tr["win_flag"].values)

    ev = df_all_feat[(df_all_feat["race_date"] >= test_from)
                      & (df_all_feat["race_date"] <= test_to)].copy()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (test_from, test_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    ev = ev[ev["race_key"].isin(set(rks7))].copy()
    if ev.empty:
        return []
    X = ev.reindex(columns=cols).fillna(0)
    ev["pred_prob"] = m_top3.predict_proba(X)[:, 1]
    ev["pred_win"] = m_win.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(ev["race_key"].unique().tolist())
    pm = _load_payouts_wt(ev["race_key"].unique().tolist())

    out = []
    for rk, g in ev.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue
        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s7_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = s7_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue
        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3 = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)
        combos = []
        for x in others:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combos.append(key)
        if not combos:
            continue
        hit = actual_top3 in combos
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
        pay = trio_pay * S7_STAKE // 100 if hit else 0
        bet = len(combos) * S7_STAKE
        out.append({
            "axis_sum": axis_sum, "entropy": entropy,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3,
            "hit": int(hit), "payout": pay, "bet_amount": bet,
        })
    return out


def d_filter(c):
    return (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
            and c["entropy"] <= S7_ENTROPY_MAX
            and (c["wt_mark3_overlap_n"] is not None
                 and c["wt_mark3_overlap_n"] <= S7_MARK3_OVERLAP_MAX))


def summarize(cands):
    sel = [c for c in cands if d_filter(c)]
    n = len(sel)
    hits = sum(c["hit"] for c in sel)
    bet = sum(c["bet_amount"] for c in sel)
    pay = sum(c["payout"] for c in sel)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "exp_cache" / "s7_elo_audit_candidates.pkl"


def main():
    import pickle
    if CACHE_PATH.exists():
        print(f"[elo-audit] キャッシュから読み込み: {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as f:
            all_base, all_elo = pickle.load(f)
    else:
        print("[elo-audit] データロード + Elo計算中...", flush=True)
        raw = load_raw_data_wt(min_date="2022-12-01", max_date="2026-07-28")
        print(f"  raw rows={len(raw)}", flush=True)
        raw = compute_elo(raw)
        print(f"  Elo計算完了 (range {raw['elo'].min():.0f}-{raw['elo'].max():.0f})", flush=True)
        df = build_features_wt(raw)
        print(f"  features built: {len(df)} rows", flush=True)

        all_base, all_elo = [], []
        for test_from, test_to in WINDOWS:
            print(f"\n[elo-audit] window {test_from}〜{test_to}", flush=True)
            base_cands = score_window(df, test_from, test_to, use_elo=False)
            elo_cands = score_window(df, test_from, test_to, use_elo=True)
            nb, hb, hrb, betb, payb, roib = summarize(base_cands)
            ne_, he, hre, bete, paye, roie = summarize(elo_cands)
            print(f"  baseline D: n={nb} hit={hrb:.1f}% ROI={roib:.1f}%", flush=True)
            print(f"  +elo     D: n={ne_} hit={hre:.1f}% ROI={roie:.1f}%", flush=True)
            all_base.extend(base_cands)
            all_elo.extend(elo_cands)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((all_base, all_elo), f)
        print(f"[elo-audit] キャッシュ保存: {CACHE_PATH}", flush=True)

    print("\n================ 全期間合計（walk-forward・S7 D構成・同一閾値） ================")
    for name, cands in (("baseline(Eloなし)", all_base), ("+elo(同一閾値)", all_elo)):
        n, hits, hitrate, bet, pay, roi = summarize(cands)
        mark = " ★100%超" if roi > 100 else ""
        print(f"{name:<20} n={n:>6} hit={hitrate:>6.1f}% bet={bet:>10,} pay={pay:>10,} ROI={roi:>7.1f}%{mark}")

    # ── +eloモデル自身の分布で axis_sum/entropy 閾値を再較正できないか sweep ──
    print("\n[elo-audit] --- +eloモデルのwt_overlap∈{0,1}母集団でaxis_sum/entropyをsweep ---")
    base_pool = [c for c in all_elo if c["wt_overlap_n"] in (0, 1)]
    print(f"  母集団(wt_overlap∈{{0,1}}のみ): n={len(base_pool)}")

    def _filt(axis_max, ent_max, mark3_req):
        def f(c):
            if c["axis_sum"] > axis_max or c["entropy"] > ent_max:
                return False
            if mark3_req and (c["wt_mark3_overlap_n"] is None or c["wt_mark3_overlap_n"] > S7_MARK3_OVERLAP_MAX):
                return False
            return True
        return f

    print(f"{'axis_sum<=':<12}{'entropy<=':<12}{'mark3':<8}{'n':>8}{'hit%':>8}{'ROI':>10}")
    best = None
    for axis_max in (0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8):
        for ent_max in (1.5, 1.7, 1.8329, 2.0, 2.2, 999):
            for mark3_req in (False, True):
                sel = [c for c in base_pool if _filt(axis_max, ent_max, mark3_req)(c)]
                n = len(sel)
                if n < 15:
                    continue
                hits = sum(c["hit"] for c in sel)
                bet = sum(c["bet_amount"] for c in sel)
                pay = sum(c["payout"] for c in sel)
                roi = pay / bet * 100 if bet else 0
                hitrate = hits / n * 100 if n else 0
                if best is None or roi > best[0]:
                    best = (roi, axis_max, ent_max, mark3_req, n, hitrate)
                if roi > 95:
                    mark = " ★100%超" if roi > 100 else ""
                    print(f"{axis_max:<12}{ent_max:<12}{str(mark3_req):<8}{n:>8}{hitrate:>7.1f}%{roi:>9.1f}%{mark}")
    if best:
        print(f"\n  最良: axis_sum<={best[1]} entropy<={best[2]} mark3={best[3]} n={best[4]} hit={best[5]:.1f}% ROI={best[0]:.1f}%")


if __name__ == "__main__":
    main()
