"""S/S+（三連単1着固定F）S通常帯の赤字対策 検証スイープ（2026-07-12）

背景: 7車ちょうど限定ハーネス（97f3b7b）でクリーンOOS(2026-04〜06)を再評価した結果、
S+帯 ROI 277% に対し S通常帯 89.5% の赤字と判明。本スクリプトで以下を検証する。

  1. ST_GAP12 × ST_GAMI の閾値スイープ（S+帯は現行条件のまま固定し、S通常帯のみ変更）
  2. S通常帯への gap34>=0.04 追加
  3. 種別「選抜」（race_type LIKE %選抜%）カット
  4. 構成比較 (a)現行 / (b)S+のみ / (c)最良スイープ / (d)最良+選抜カット
  5. OOS レース単位ブートストラップ（1万回・seed固定）で現行比 ROI 差の 95%CI

方針: 探索（スイープ）は in-sample(2025-07〜2026-03) で選び、
クリーンOOS(2026-04〜06) は少数構成の確認にのみ使う。
7月フォワード(07-01〜07-10) は小 n の参考として分離報告。

使い方:
  .venv/bin/python scripts/exp_st_threshold_sweep_wt.py [--cache-dir DIR]

決定論的（seed=20260712 固定）・既存ファイル無変更。
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_clean_split_wt import collect  # 2026-07-12修正済: 7車ちょうど限定
from src.database import get_connection
from src.models.trainer import load_model

SEED = 20260712
N_BOOT = 10000

# 現行本番条件（notify_prerace_wt.py と同一）
ST_GAP12 = 0.15
ST_GAMI = 10.0
STP_GAP12 = 0.25
STP_GAP34 = 0.04
ST_STAKE = 100
STP_STAKE = 200

WINDOWS = {
    "IS":  ("lgbm_wt_2026h1_eval", "2025-07-01", "2026-03-31"),
    "OOS": ("lgbm_wt_2026h1_eval", "2026-04-01", "2026-06-30"),
    "FWD": ("lgbm_wt_2026h1", "2026-07-01", "2026-07-10"),
}


def load_race_types(race_keys):
    """race_key -> race_type（None あり）"""
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, race_type FROM wt_races WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, rt in c.execute(q, chunk):
                out[rk] = rt or ""
    return out


def prepare_rows(rows, race_types):
    """各候補レースに買い目盤・S+判定など評価用フィールドを事前計算して付与する。

    返り値: list[dict]  keys:
      rk, month, gap12, gap34, min_odds, n_pts, hit_odds(的中時オッズ or None),
      is_plus(現行S+帯: gap12>=0.25 ∧ gap34>=0.04 ∧ min>=10), sensen(選抜か)
    combos が引けないレースは除外（本番同様、盤欠損は購入対象外）。
    """
    out = []
    for r in rows:
        combos = {}
        for s in (r["p2"], r["r3"]):
            for t in r["frames"]:
                if t in (r["p1"], s):
                    continue
                ov = r["tri"].get((r["p1"], s, t))
                if ov:
                    combos[(r["p1"], s, t)] = ov
        if not combos:
            continue
        m = min(combos.values())
        out.append({
            "rk": r["rk"],
            "month": r["rk"][:6],
            "gap12": r["gap12"],
            "gap34": r["gap34"],
            "min_odds": m,
            "n_pts": len(combos),
            "hit_odds": combos.get(r["order"]),
            "is_plus": (r["gap12"] >= STP_GAP12 and r["gap34"] >= STP_GAP34
                        and m >= ST_GAMI),
            "sensen": "選抜" in (race_types.get(r["rk"], "") or ""),
        })
    out.sort(key=lambda x: x["rk"])
    return out


def eval_config(prows, base_gap12=ST_GAP12, base_gami=ST_GAMI,
                base_gap34=None, cut_sensen_base=False,
                plus_on=True, base_on=True, cut_sensen_plus=False):
    """構成を評価してレース単位の bet レコード列を返す。

    S+帯は現行条件（is_plus）固定・200円/点。
    S通常帯は (not is_plus) ∧ gap12>=base_gap12 ∧ min>=base_gami
              ∧ (base_gap34 指定時 gap34>=base_gap34)
              ∧ (cut_sensen_base 時 選抜除外)
    返り値: list[(rk, month, band, stake, payout)]  band: 'plus'|'base'
    """
    bets = []
    for p in prows:
        if plus_on and p["is_plus"] and not (cut_sensen_plus and p["sensen"]):
            stake = p["n_pts"] * STP_STAKE
            pay = int(p["hit_odds"] * STP_STAKE) if p["hit_odds"] else 0
            bets.append((p["rk"], p["month"], "plus", stake, pay))
            continue
        if not base_on or p["is_plus"]:
            continue
        if p["gap12"] < base_gap12 or p["min_odds"] < base_gami:
            continue
        if base_gap34 is not None and p["gap34"] < base_gap34:
            continue
        if cut_sensen_base and p["sensen"]:
            continue
        stake = p["n_pts"] * ST_STAKE
        pay = int(p["hit_odds"] * ST_STAKE) if p["hit_odds"] else 0
        bets.append((p["rk"], p["month"], "base", stake, pay))
    return bets


def summarize(bets, n_days):
    n = len(bets)
    if n == 0:
        return dict(n=0, per_day=0.0, hit=0.0, stake=0, pay=0, roi=float("nan"),
                    profit=0)
    stake = sum(b[3] for b in bets)
    pay = sum(b[4] for b in bets)
    hits = sum(1 for b in bets if b[4] > 0)
    return dict(n=n, per_day=n / max(n_days, 1), hit=hits / n, stake=stake,
                pay=pay, roi=pay / stake, profit=pay - stake)


def band(bets, name):
    return [b for b in bets if b[2] == name]


def monthly_roi(bets):
    agg = defaultdict(lambda: [0, 0])
    for _, mo, _, st, pa in bets:
        agg[mo][0] += st
        agg[mo][1] += pa
    return {mo: (v[1] / v[0] if v[0] else float("nan"), v[0]) for mo, v in sorted(agg.items())}


def max_drawdown(bets):
    """rk 順の累積損益の最大ドローダウン（円）"""
    cum = peak = dd = 0
    for _, _, _, st, pa in sorted(bets, key=lambda b: b[0]):
        cum += pa - st
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def bootstrap_roi_diff(prows, cfg_kwargs, base_kwargs, n_boot=N_BOOT, seed=SEED):
    """レース単位リサンプリングで ROI(cfg) - ROI(現行) の分布を返す。

    prows 全体（候補レース）を復元抽出。両構成とも同一リサンプル上で ROI を計算。
    """
    def per_race(kwargs):
        st = np.zeros(len(prows))
        pa = np.zeros(len(prows))
        idx = {p["rk"]: i for i, p in enumerate(prows)}
        for rk, _, _, s, p in eval_config(prows, **kwargs):
            i = idx[rk]
            st[i] += s
            pa[i] += p
        return st, pa

    st_a, pa_a = per_race(cfg_kwargs)
    st_b, pa_b = per_race(base_kwargs)
    rng = np.random.default_rng(seed)
    n = len(prows)
    diffs = np.empty(n_boot)
    pdiffs = np.empty(n_boot)  # 利益差（円）
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        sa, ta = st_a[idx].sum(), pa_a[idx].sum()
        sb, tb = st_b[idx].sum(), pa_b[idx].sum()
        ra = ta / sa if sa > 0 else np.nan
        rb = tb / sb if sb > 0 else np.nan
        diffs[k] = ra - rb
        pdiffs[k] = (ta - sa) - (tb - sb)
    diffs = diffs[~np.isnan(diffs)]
    return ((float(np.percentile(diffs, 2.5)), float(np.median(diffs)),
             float(np.percentile(diffs, 97.5))),
            (float(np.percentile(pdiffs, 2.5)), float(np.median(pdiffs)),
             float(np.percentile(pdiffs, 97.5))))


def fmt(s):
    if s["n"] == 0:
        return f"{'0':>5} {'-':>5} {'-':>6} {'-':>10} {'-':>10} {'-':>7}"
    return (f"{s['n']:>5} {s['per_day']:>5.1f} {s['hit']:>6.1%} "
            f"{s['stake']:>10,} {s['pay']:>10,} {s['roi']:>6.1%}")


def get_rows(win, cache_dir):
    model_name, f, t = WINDOWS[win]
    if cache_dir:
        cp = Path(cache_dir) / f"rows_{win}_{f}_{t}.pkl"
        if cp.exists():
            with open(cp, "rb") as fp:
                return pickle.load(fp)
    rows = collect(load_model(model_name), f, t)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cache_dir) / f"rows_{win}_{f}_{t}.pkl", "wb") as fp:
            pickle.dump(rows, fp)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None,
                    help="collect 結果の pickle キャッシュ置き場（再実行高速化用）")
    args = ap.parse_args()

    data = {}
    for win in WINDOWS:
        print(f"collect {win} ...", flush=True)
        rows = get_rows(win, args.cache_dir)
        rts = load_race_types([r["rk"] for r in rows])
        prows = prepare_rows(rows, rts)
        n_days = len({p["rk"][:8] for p in prows}) or 1
        data[win] = (prows, n_days)
        print(f"  {win}: 候補{len(prows)}R / {n_days}日", flush=True)

    CUR = dict()  # 現行
    hdr = f"{'R数':>5} {'R/日':>5} {'的中率':>6} {'投資':>10} {'払戻':>10} {'ROI':>7}"

    # ---------- 0. 現行ベースライン ----------
    print("\n===== 0. 現行構成 (S+帯200円 + S通常帯100円) =====")
    for win, (prows, nd) in data.items():
        bets = eval_config(prows, **CUR)
        print(f"[{win}] {'区分':<10} {hdr}")
        print(f"      {'S計':<10} {fmt(summarize(bets, nd))}")
        print(f"      {'  S+帯':<10} {fmt(summarize(band(bets, 'plus'), nd))}")
        print(f"      {'  S通常帯':<9} {fmt(summarize(band(bets, 'base'), nd))}")

    # ---------- 1. 閾値スイープ（S通常帯のみ・S+帯固定） ----------
    print("\n===== 1. S通常帯 閾値スイープ（S+帯は現行のまま固定） =====")
    print("      ※探索は IS で行い OOS は参考掲示（選択には使わない）")
    sweep_res = []
    print(f"{'gap12':>6} {'gami':>5} | {'IS n':>5} {'IS R/日':>6} {'IS 的中':>6} "
          f"{'IS ROI':>7} | {'OOS n':>5} {'OOS 的中':>6} {'OOS ROI':>7}")
    for g12 in (0.15, 0.18, 0.20, 0.22, 0.25):
        for gami in (10, 12, 15):
            kw = dict(base_gap12=g12, base_gami=float(gami))
            res = {}
            for win in ("IS", "OOS"):
                prows, nd = data[win]
                res[win] = summarize(band(eval_config(prows, **kw), "base"), nd)
            sweep_res.append((g12, gami, res))
            i, o = res["IS"], res["OOS"]
            print(f"{g12:>6.2f} {gami:>5} | {i['n']:>5} {i['per_day']:>6.1f} "
                  f"{i['hit']:>6.1%} {i['roi']:>6.1%} | {o['n']:>5} "
                  f"{o['hit']:>6.1%} {o['roi']:>6.1%}")

    # IS 基準の最良（n>=100 を条件に ROI 最大）
    cand = [(g, ga, r) for g, ga, r in sweep_res if r["IS"]["n"] >= 100]
    best_g12, best_gami, best_res = max(cand, key=lambda x: x[2]["IS"]["roi"])
    print(f"\nIS最良（n>=100）: gap12>={best_g12:.2f} ∧ min>={best_gami} "
          f"(IS ROI {best_res['IS']['roi']:.1%} / OOS ROI {best_res['OOS']['roi']:.1%})")
    # ボリューム維持型の IS 最良（IS で R/日>=3.5 を維持しつつ ROI 最大）
    cand_v = [(g, ga, r) for g, ga, r in sweep_res if r["IS"]["per_day"] >= 3.5]
    vol_g12, vol_gami, vol_res = max(cand_v, key=lambda x: x[2]["IS"]["roi"])
    print(f"IS最良（R/日>=3.5維持）: gap12>={vol_g12:.2f} ∧ min>={vol_gami} "
          f"(IS ROI {vol_res['IS']['roi']:.1%} / OOS ROI {vol_res['OOS']['roi']:.1%})")

    # ---------- 2. gap34 を S通常帯に追加 ----------
    print("\n===== 2. S通常帯に gap34>=0.04 を追加（gap12/gami は現行のまま） =====")
    kw34 = dict(base_gap34=STP_GAP34)
    for win, (prows, nd) in data.items():
        s = summarize(band(eval_config(prows, **kw34), "base"), nd)
        print(f"[{win}] S通常帯(gap34付) {fmt(s)}")

    # ---------- 3. 選抜カット ----------
    print("\n===== 3. 種別『選抜』カット（race_type LIKE %選抜%） =====")
    for label, kw in (
        ("S通常帯のみカット", dict(cut_sensen_base=True)),
        ("S+帯もカット", dict(cut_sensen_base=True, cut_sensen_plus=True)),
    ):
        print(f"-- {label} --")
        for win, (prows, nd) in data.items():
            bets = eval_config(prows, **kw)
            print(f"[{win}] S計 {fmt(summarize(bets, nd))} | "
                  f"S通常帯 {fmt(summarize(band(bets, 'base'), nd))}")
    # 参考: 選抜レース自体の現行成績
    print("-- 参考: 現行構成のうち選抜レース分 --")
    for win, (prows, nd) in data.items():
        sens = [p for p in prows if p["sensen"]]
        bets = eval_config(sens, **CUR)
        print(f"[{win}] 選抜のみ S計 {fmt(summarize(bets, nd))}")

    # ---------- 4. 構成比較 ----------
    best_kw = dict(base_gap12=best_g12, base_gami=float(best_gami))
    best_cut_kw = dict(base_gap12=best_g12, base_gami=float(best_gami),
                       cut_sensen_base=True)
    vol_kw = dict(base_gap12=vol_g12, base_gami=float(vol_gami))
    vol_cut_kw = dict(base_gap12=vol_g12, base_gami=float(vol_gami),
                      cut_sensen_base=True)
    configs = [
        ("(a) 現行 S+S+", CUR),
        ("(b) S+のみ（S通常廃止）", dict(base_on=False)),
        (f"(c) 最良スイープ g12>={best_g12:.2f}∧min>={best_gami}", best_kw),
        ("(d) (c)+S通常帯選抜カット", best_cut_kw),
        (f"(e) 量維持 g12>={vol_g12:.2f}∧min>={vol_gami}", vol_kw),
        ("(f) (e)+S通常帯選抜カット", vol_cut_kw),
        ("(g) 現行+S通常帯選抜カットのみ", dict(cut_sensen_base=True)),
    ]
    print("\n===== 4. 構成比較 =====")
    for win, (prows, nd) in data.items():
        print(f"\n[{win}] {'構成':<34} {hdr} {'利益':>10} {'最大DD':>9}")
        for label, kw in configs:
            bets = eval_config(prows, **kw)
            s = summarize(bets, nd)
            dd = max_drawdown(bets)
            print(f"{label:<38} {fmt(s)} {s['profit']:>+10,} {dd:>9,}")
        # 月別 ROI
        print(f"  -- 月別ROI ({win}) --")
        mos = sorted({p['month'] for p in prows})
        head = "  " + f"{'構成':<34}" + "".join(f"{mo[4:6]+'月':>9}" for mo in mos)
        print(head)
        for label, kw in configs:
            mr = monthly_roi(eval_config(prows, **kw))
            cells = "".join(
                f"{mr[mo][0]:>8.0%} " if mo in mr else f"{'-':>8} " for mo in mos)
            print(f"  {label:<36} {cells}")

    # ---------- 5. OOS ブートストラップ（現行比 ROI差 95%CI） ----------
    print(f"\n===== 5. OOS ブートストラップ（レース単位 {N_BOOT}回 seed={SEED}） =====")
    prows_oos, _ = data["OOS"]
    prows_is, _ = data["IS"]
    for label, kw in configs[1:]:
        (lo, med, hi), (plo, pmed, phi) = bootstrap_roi_diff(prows_oos, kw, CUR)
        (lo_i, med_i, hi_i), (plo_i, pmed_i, phi_i) = bootstrap_roi_diff(
            prows_is, kw, CUR)
        same_dir = (med > 0) == (med_i > 0)
        print(f"{label:<38} OOS ΔROI 中央値{med:+.1%} CI[{lo:+.1%},{hi:+.1%}] | "
              f"IS 中央値{med_i:+.1%} CI[{lo_i:+.1%},{hi_i:+.1%}] | "
              f"方向一致={'○' if same_dir else '×'}")
        print(f"{'':<38} OOS Δ利益 中央値{pmed:+,.0f}円 CI[{plo:+,.0f},{phi:+,.0f}] | "
              f"IS 中央値{pmed_i:+,.0f}円 CI[{plo_i:+,.0f},{phi_i:+,.0f}]")


if __name__ == "__main__":
    main()
