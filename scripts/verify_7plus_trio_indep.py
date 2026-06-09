"""
独立検証: 7車以上(7+)レースの三連複(trio)で ROI>100% を再現的に達成できるか。

設計思想（独立・自前）:
  Phase1 波乱選定 : pred1位(指数最上位)の top3確率(p1_prob) が低いレースほど
                    本命が飛ぶ(top3外)率が高い。これを波乱シグナルとする。
                    さらに「本命が4着以下=top3完全脱落」サブセットも別途評価する。
  Phase2 軸選定   : 本命が飛ぶ前提での軸候補(pred2 / 別ライン最上位 / pred1温存)を比較。
  Phase3 相手選出 : 軸からの相手をpred順 box/formation で構成。
  Phase4 足切り   : モデルのtop3確率から trio確率(独立近似)→EV=確率*オッズ。
                    市場の織り込み度(EV)で参加レース/買い目を足切りできるか。

判定: 大標本TRAIN(2023-07〜2026-02)とOOS TEST(2026-03〜2026-06-08)の双方で
      ROI>100% かつ ブートストラップ95%CI下限>100% のときのみ「再現的黒字」。

実行: .venv/bin/python3 scripts/verify_7plus_trio_indep.py
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection

TRAIN = ("2023-07-01", "2026-02-28")
TEST = ("2026-03-01", "2026-06-08")
RNG = np.random.default_rng(42)


def load_block(min_date, max_date, model):
    df = build_features_wt(load_raw_data_wt(min_date=min_date, max_date=max_date))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    sz = df.groupby("race_key")["frame_no"].count()
    keys7 = sz[sz >= 7].index
    d = df[df.race_key.isin(keys7)].copy()
    d["prank"] = d.groupby("race_key")["pred_prob"].rank(ascending=False, method="first")
    return d


def load_trio_odds(keys):
    keys = list(keys)
    chunks = []
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            ks = "','".join(keys[i:i + 900])
            chunks.append(pd.read_sql_query(
                f"SELECT race_key,combination,odds_value FROM wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ('{ks}')", c))
    od = pd.concat(chunks, ignore_index=True)
    od["cs"] = od.combination.apply(lambda s: frozenset(int(x) for x in s.split("=")))
    return {(r.race_key, r.cs): r.odds_value for r in od.itertuples()}


def build_race_table(d):
    """race単位の盤面(軸候補・波乱シグナル・実績top3)を作る。"""
    rows = []
    for rk, sub in d.groupby("race_key"):
        sub = sub.sort_values("prank")
        fr = sub.frame_no.astype(int).tolist()
        probs = sub.pred_prob.values
        act = frozenset(sub[sub.top3_flag == 1].frame_no.astype(int))
        if len(act) != 3:
            continue
        p1 = sub.iloc[0]
        pmap = dict(zip(sub.frame_no.astype(int), sub.pred_prob))
        # 別ライン最上位: pred1と別ラインのうちpred最上位
        p1_line = p1.line_group if "line_group" in sub.columns else None
        alt = None
        for _, r in sub.iloc[1:].iterrows():
            if p1_line is None or r.line_group != p1_line:
                alt = int(r.frame_no)
                break
        if alt is None:
            alt = fr[1]
        rows.append(dict(
            race_key=rk, frames=fr, pmap=pmap, act=act,
            p1_prob=float(probs[0]),
            p1_fell=int(p1.top3_flag == 0),
            axis_p2=fr[1],
            axis_alt=alt,
            axis_p1=fr[0],
        ))
    return pd.DataFrame(rows)


def trio_indep_prob(pmap, combo):
    return float(np.prod([pmap[f] for f in combo]))


# ---- formation pickers: race_row -> list[frozenset] of trios to bet ----
def pick_box(n):
    def f(rr):
        return [frozenset(c) for c in combinations(rr.frames[:n], 3)]
    return f


def pick_axis_box(axis_field, n_partners):
    """axis固定 + pred上位n_partners人(axis除く)から相手2人を総当り。"""
    def f(rr):
        axis = getattr(rr, axis_field)
        partners = [x for x in rr.frames if x != axis][:n_partners]
        return [frozenset([axis, a, b]) for a, b in combinations(partners, 2)]
    return f


def evaluate(RT, odd_map, picker, race_filter=None, ev_min=None):
    """ROIをレース単位(payoff,invest)で集計し、bootstrap CIも返す。"""
    per_race = []  # (invest, payoff)
    hits = 0
    nbets = 0
    for rr in RT.itertuples():
        if race_filter is not None and not race_filter(rr):
            continue
        combos = picker(rr)
        if ev_min is not None:
            kept = []
            for cs in combos:
                o = odd_map.get((rr.race_key, cs))
                if o is None:
                    continue
                ev = trio_indep_prob(rr.pmap, cs) * o
                if ev >= ev_min:
                    kept.append(cs)
            combos = kept
        if not combos:
            continue
        inv = 100.0 * len(combos)
        pay = 0.0
        if rr.act in combos:
            o = odd_map.get((rr.race_key, rr.act), 0.0)
            pay = o * 100.0
            hits += 1
        per_race.append((inv, pay))
        nbets += len(combos)
    if not per_race:
        return None
    arr = np.array(per_race)
    inv_tot = arr[:, 0].sum()
    pay_tot = arr[:, 1].sum()
    roi = pay_tot / inv_tot * 100
    n = len(arr)
    # bootstrap CI on ROI (resample races)
    bs = []
    for _ in range(2000):
        idx = RNG.integers(0, n, n)
        s = arr[idx]
        si = s[:, 0].sum()
        bs.append(s[:, 1].sum() / si * 100 if si else 0)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=n, nbets=nbets, hit=hits / n * 100, roi=roi,
                ci_lo=lo, ci_hi=hi, bets_per_race=nbets / n,
                max_payoff=arr[:, 1].max())


def fmt(name, r):
    if r is None:
        return f"{name:42s} -- no bets --"
    return (f"{name:42s} n={r['n']:5d} bpr={r['bets_per_race']:4.1f} "
            f"hit={r['hit']:5.1f}% ROI={r['roi']:6.1f}% "
            f"CI[{r['ci_lo']:5.1f},{r['ci_hi']:6.1f}] maxpay={r['max_payoff']:.0f}")


def run_split(label, d):
    print(f"\n{'='*100}\n{label}  (7+ races, trio)\n{'='*100}")
    RT = build_race_table(d)
    odd_map = load_trio_odds(RT.race_key.unique())
    print(f"races(valid top3)={len(RT)}  本命飛ぶ率={RT.p1_fell.mean()*100:.1f}%")

    # 波乱フィルタ: p1_prob 下位40% (TRAIN/TESTそれぞれの分位で揃える)
    thr40 = RT.p1_prob.quantile(0.40)
    thr20 = RT.p1_prob.quantile(0.20)
    haran40 = lambda rr: rr.p1_prob <= thr40
    haran20 = lambda rr: rr.p1_prob <= thr20
    fell = lambda rr: rr.p1_fell == 1  # 後ろ向き(実績)。上限性能の参考用

    print("\n-- Phase1+3 全レース box系 (波乱フィルタなし) --")
    for nm, n in [("box4", 4), ("box5", 5), ("box6", 6)]:
        print(fmt(f"全:{nm}", evaluate(RT, odd_map, pick_box(n))))

    print("\n-- Phase1 波乱(p1_prob下位40%)で box系 --")
    for nm, n in [("box4", 4), ("box5", 5), ("box6", 6)]:
        print(fmt(f"波乱40:{nm}", evaluate(RT, odd_map, pick_box(n), race_filter=haran40)))

    print("\n-- Phase1 強波乱(p1_prob下位20%)で box系 --")
    for nm, n in [("box5", 5), ("box6", 6)]:
        print(fmt(f"波乱20:{nm}", evaluate(RT, odd_map, pick_box(n), race_filter=haran20)))

    print("\n-- Phase2 軸選定比較 (波乱40%, 軸固定+相手pred上位5) --")
    for ax in ["axis_p2", "axis_alt", "axis_p1"]:
        print(fmt(f"波乱40:{ax}+5",
                  evaluate(RT, odd_map, pick_axis_box(ax, 5), race_filter=haran40)))

    print("\n-- Phase4 EV足切り (波乱40%, box6, EV閾値) --")
    for ev in [0.8, 1.0, 1.2, 1.5]:
        print(fmt(f"波乱40:box6 EV>={ev}",
                  evaluate(RT, odd_map, pick_box(6), race_filter=haran40, ev_min=ev)))

    print("\n-- Phase4 EV足切り (全レース, top6 box universe, EV閾値) --")
    for ev in [1.0, 1.5, 2.0]:
        print(fmt(f"全:box6 EV>={ev}",
                  evaluate(RT, odd_map, pick_box(6), ev_min=ev)))

    print("\n-- 参考(上限): 本命が実際に飛んだレースのみ(後知恵) --")
    for nm, n in [("box5", 5), ("box6", 6)]:
        print(fmt(f"[oracle]飛:{nm}",
                  evaluate(RT, odd_map, pick_box(n), race_filter=fell)))
        print(fmt(f"[oracle]飛:alt+5",
                  evaluate(RT, odd_map, pick_axis_box("axis_alt", 5), race_filter=fell)))
    return RT, odd_map


def main():
    model = load_model("lgbm_wt")
    print("loading TRAIN block...", flush=True)
    dtr = load_block(*TRAIN, model)
    print("loading TEST block...", flush=True)
    dte = load_block(*TEST, model)
    run_split("TRAIN 2023-07..2026-02", dtr)
    run_split("TEST(OOS) 2026-03..2026-06-08", dte)


if __name__ == "__main__":
    main()
