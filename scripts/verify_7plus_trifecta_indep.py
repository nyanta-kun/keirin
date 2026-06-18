"""
Independent verification: can a 7+ car TRIFECTA strategy reach reproducible ROI>100%?

Designed from scratch (no reuse of prior 7+ experiment code). Strategy follows the
user's 4-phase decomposition, with all phase logic implemented here:

  Phase 1 (波乱選定):  Select races where the pred-favorite likely does NOT win 1st
                       but stays in 2nd-3rd ("favorite-survives upset" => high trifecta).
  Phase 2 (軸選定):    Given that upset, pick the 1st-place axis (best alternative line head
                       / pred2 / etc) using observed conditional accuracy.
  Phase 3 (相手/着順):  Build the trifecta formation (1st axis x 2nd box x 3rd box) anchoring
                       the favorite into the 2-3 slots.
  Phase 4 (合成オッズ足切り): drop bets/races the market already prices in (market-implied
                       probability filter via odds; combo-EV filter).

Judging rule: "reproducible profit" ONLY if BOTH large-sample TRAIN (2023-07..2026-02)
and OOS TEST (2026-03..2026-06-08) show ROI>100%, with bootstrap CI lower bound and
no single-payout domination. Otherwise => noise / non-reproducible.

Run: .venv/bin/python3 scripts/verify_7plus_trifecta_indep.py
"""
import re
import sys
import numpy as np
import pandas as pd

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection

RNG = np.random.default_rng(42)
SPLIT = "2023-07-01", "2026-02-28", "2026-03-01", "2026-06-08"
SEP = re.compile(r"[-=→]")


def load_block(min_date, max_date):
    df = build_features_wt(load_raw_data_wt(min_date=min_date, max_date=max_date))
    model = load_model("lgbm_wt")
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    sz = df.groupby("race_key")["frame_no"].transform("count")
    df = df[sz >= 7].copy()
    # per-race pred rank (1 = best)
    df["prank"] = df.groupby("race_key")["pred_prob"].rank(ascending=False, method="first")
    df["frame_no"] = df["frame_no"].astype(int)
    return df


def load_trifecta_odds(keys):
    """Return dict race_key -> {(a,b,c): odds_value}."""
    keys = list(keys)
    out = {}
    with get_connection() as con:
        CH = 800
        for i in range(0, len(keys), CH):
            chunk = keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            rows = con.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE bet_type='trifecta' AND race_key IN ({ph})", chunk
            ).fetchall()
            for r in rows:
                parts = SEP.split(r["combination"])
                if len(parts) != 3:
                    continue
                try:
                    t = (int(parts[0]), int(parts[1]), int(parts[2]))
                except ValueError:
                    continue
                ov = r["odds_value"]
                if ov is None or ov <= 0:
                    continue
                out.setdefault(r["race_key"], {})[t] = float(ov)
    return out


def race_table(df):
    """One row per (race_key, frame_no) with the columns we need; plus result tuple."""
    cols = ["race_key", "frame_no", "pred_prob", "prank", "finish_order",
            "line_group", "is_line_leader", "score_rank", "race_point"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def winning_combo(g):
    """Return (a,b,c) of the actual 1st,2nd,3rd by frame_no, or None if incomplete."""
    fin = g[g["finish_order"].between(1, 3)]
    if fin["finish_order"].nunique() < 3 or len(fin) < 3:
        return None
    a = fin[fin["finish_order"] == 1]["frame_no"]
    b = fin[fin["finish_order"] == 2]["frame_no"]
    c = fin[fin["finish_order"] == 3]["frame_no"]
    if len(a) != 1 or len(b) != 1 or len(c) != 1:
        return None
    return (int(a.iloc[0]), int(b.iloc[0]), int(c.iloc[0]))


# ----- Phase 1: upset signal -----
def upset_signal(g):
    """
    Heuristic upset score: low favorite dominance + competitive field.
    Returns (fav_frame, upset_score, fav_prob, gap, second_line_flag).
    Higher score => more likely the favorite does NOT win 1st.
    """
    g = g.sort_values("prank")
    fav = g.iloc[0]
    p1 = float(fav["pred_prob"])
    p2 = float(g.iloc[1]["pred_prob"]) if len(g) > 1 else 0.0
    gap = p1 - p2
    # is favorite isolated from a line? (line_size proxy via line_group counts)
    favline = fav.get("line_group", None)
    if favline is not None and "line_group" in g:
        line_sz = (g["line_group"] == favline).sum()
    else:
        line_sz = 1
    # upset score: small gap & lowish p1 & favorite in small/isolated line => upset prone
    score = (0.25 - p1) + (0.06 - gap) * 2.0 + (2 - line_sz) * 0.05
    return int(fav["frame_no"]), score, p1, gap, line_sz


# ----- Phase 2: pick 1st-place axis given upset -----
def first_axis_candidates(g):
    """
    Under "favorite won't win", candidate 1st-axis = best pred among non-favorite,
    preferring a different line head. Return ordered list of frame_no candidates.
    """
    g = g.sort_values("prank")
    fav = g.iloc[0]
    favline = fav.get("line_group", None)
    rest = g.iloc[1:]
    # rank rest by pred_prob; tie-break: different line head first
    rest = rest.copy()
    rest["diff_line"] = (rest.get("line_group", -1) != favline).astype(int) if "line_group" in rest else 0
    rest = rest.sort_values(["pred_prob"], ascending=False)
    return [int(x) for x in rest["frame_no"].tolist()], int(fav["frame_no"])


def main():
    results = {}
    blocks = {"TRAIN": (SPLIT[0], SPLIT[1]), "TEST": (SPLIT[2], SPLIT[3])}
    loaded = {}
    for name, (a, b) in blocks.items():
        print(f"[load] {name} {a}..{b}", file=sys.stderr)
        df = load_block(a, b)
        loaded[name] = df

    for name, df in loaded.items():
        rt = race_table(df)
        keys = rt["race_key"].unique().tolist()
        print(f"[odds] {name} loading odds for {len(keys)} races", file=sys.stderr)
        odds = load_trifecta_odds(keys)
        results[name] = run_strategies(rt, odds)
    report(results)


def boot_ci(stake_pay, n_boot=2000):
    """stake_pay: list of (stake, payout) per race. Returns (roi, lo, hi, hit, n)."""
    if not stake_pay:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    arr = np.array(stake_pay, dtype=float)  # (n,2) stake, payout
    n = len(arr)
    roi = arr[:, 1].sum() / arr[:, 0].sum() * 100
    hit = (arr[:, 1] > 0).mean() * 100
    idx = RNG.integers(0, n, size=(n_boot, n))
    s = arr[idx, 0].sum(axis=1)
    p = arr[idx, 1].sum(axis=1)
    rois = p / s * 100
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return (roi, lo, hi, hit, n)


def run_strategies(rt, odds):
    """
    Evaluate several formation strategies. Each strategy returns per-race (stake, payout).
    Stake = 100 yen per combination bet. Payout = odds*100 if winning combo bet else 0.
    """
    grp = dict(tuple(rt.groupby("race_key")))
    strat_out = {}

    # config of strategies (phase combos)
    # name -> dict of params
    strategies = {
        # baseline: no upset filter, favorite-1st formation (sanity / market baseline)
        "S0_fav1st_box": dict(upset_q=None, axis="fav", second=3, third=3, ev_min=None, mkt_max=None),
        # Phase1+2+3: upset races, alt-line axis 1st, favorite forced into 2nd, box 3rd
        "S1_upset_alt1st": dict(upset_q=0.5, axis="alt", second=2, third=3, ev_min=None, mkt_max=None),
        # tighter upset selection
        "S2_upset_q25": dict(upset_q=0.25, axis="alt", second=2, third=3, ev_min=None, mkt_max=None),
        # +Phase4 EV floor
        "S3_upset_ev": dict(upset_q=0.25, axis="alt", second=2, third=3, ev_min=1.0, mkt_max=None),
        # +Phase4 market-implied prob cap (avoid priced-in combos)
        "S4_upset_mktcap": dict(upset_q=0.25, axis="alt", second=2, third=3, ev_min=None, mkt_max=0.04),
        # combine EV floor + market cap, very tight upset
        "S5_tight_ev_mkt": dict(upset_q=0.15, axis="alt", second=2, third=3, ev_min=1.2, mkt_max=0.05),
        # alt-axis but allow alt to also take 2nd (favorite -> 3rd), wider relevant combos
        "S6_alt_fav3rd": dict(upset_q=0.25, axis="alt", second=2, third=3, ev_min=1.0, mkt_max=None, fav_slot="any"),
    }

    # precompute upset scores per race for quantile threshold
    upset_scores = {}
    for k, g in grp.items():
        if len(g) < 7:
            continue
        fav, sc, p1, gap, lsz = upset_signal(g)
        upset_scores[k] = sc

    for sname, cfg in strategies.items():
        sp = []  # (stake, payout)
        sp_combos = []  # diagnostic: per winning bet payout
        thr = None
        if cfg["upset_q"] is not None:
            thr = np.quantile(list(upset_scores.values()), 1 - cfg["upset_q"])
        for k, g in grp.items():
            if len(g) < 7:
                continue
            if k not in odds:
                continue
            if thr is not None and upset_scores.get(k, -1e9) < thr:
                continue
            res = winning_combo(g)
            # build bet set
            cand, fav = first_axis_candidates(g)
            if not cand:
                continue
            if cfg["axis"] == "fav":
                axis_first = [fav]
            else:
                axis_first = cand[:1]  # best non-favorite
            # 2nd-slot pool, 3rd-slot pool: take top pred frames overall
            gg = g.sort_values("prank")
            top_frames = [int(x) for x in gg["frame_no"].tolist()]
            # formation: 1st in axis_first; 2nd in next-best (incl favorite if upset survives);
            # 3rd box of next frames.
            second_pool = []
            for f in top_frames:
                if f not in axis_first and len(second_pool) < cfg["second"]:
                    second_pool.append(f)
            # ensure favorite is in second pool (Phase1 thesis: fav survives to 2-3)
            fav_slot = cfg.get("fav_slot", "second")
            if fav_slot == "second" and fav not in second_pool and fav not in axis_first:
                second_pool = [fav] + second_pool[:-1] if second_pool else [fav]
            third_pool = []
            for f in top_frames:
                if f not in axis_first and len(third_pool) < (cfg["third"] + 1):
                    third_pool.append(f)
            combos = set()
            for a in axis_first:
                for b in second_pool:
                    if b == a:
                        continue
                    for c in third_pool:
                        if c == a or c == b:
                            continue
                        combos.add((a, b, c))
            if not combos:
                continue
            ov = odds[k]
            # Phase 4 filters per combo
            bets = []
            for cm in combos:
                o = ov.get(cm)
                if o is None:
                    continue
                mkt_p = 1.0 / o  # market-implied prob (gross of takeout)
                if cfg["mkt_max"] is not None and mkt_p > cfg["mkt_max"]:
                    continue
                if cfg["ev_min"] is not None:
                    # use model: approximate combo prob = product of place-ish? use pred ranks
                    # crude model prob: normalized pred_prob of the 3 frames product
                    pmap = dict(zip(g["frame_no"].astype(int), g["pred_prob"]))
                    pp = pmap.get(cm[0], 0) * pmap.get(cm[1], 0) * pmap.get(cm[2], 0)
                    # scale: there are ~ n*(n-1)*(n-2) combos; rough EV
                    ev = pp * o * 50  # heuristic scale factor
                    if ev < cfg["ev_min"]:
                        continue
                bets.append((cm, o))
            if not bets:
                continue
            stake = 100 * len(bets)
            payout = 0.0
            if res is not None:
                for cm, o in bets:
                    if cm == res:
                        payout = o * 100
                        sp_combos.append(o)
            sp.append((stake, payout))
        roi, lo, hi, hit, n = boot_ci(sp)
        maxpay = max(sp_combos) if sp_combos else 0
        totalpay = sum(p for _, p in sp)
        maxshare = (maxpay / totalpay * 100) if totalpay > 0 else 0
        strat_out[sname] = dict(roi=roi, lo=lo, hi=hi, hit=hit, n=n,
                                maxpay=maxpay, maxshare=maxshare, nwin=len(sp_combos))
    return strat_out


def report(results):
    print("\n" + "=" * 100)
    print("7+ TRIFECTA INDEPENDENT VERIFICATION")
    print("=" * 100)
    snames = list(next(iter(results.values())).keys())
    hdr = f"{'strategy':22} | {'block':5} | {'n':>5} | {'ROI%':>7} | {'CI95':>17} | {'hit%':>5} | {'maxshare%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in snames:
        for blk in results:
            r = results[blk][s]
            ci = f"[{r['lo']:.0f},{r['hi']:.0f}]"
            print(f"{s:22} | {blk:5} | {r['n']:>5} | {r['roi']:>7.1f} | {ci:>17} | {r['hit']:>5.1f} | {r['maxshare']:>9.1f}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
