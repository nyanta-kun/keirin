"""7S/7A のレース選別を「配当構造」から見直す（2026-08-06・探索）。

## 背景（ユーザー依頼）

> 7S,7A のレース選択、軸選択の精度を見直す。現在はレース選択は WT◎◯ との印の
> 重なりなどを中心としている。指数のばらつきなどにより力差、ラインの強さが
> 適切に反映できていないと思う。二軸総流しのため、三連複が10倍以上つくレースの
> 選別、その中で買い方の検討をする。

## 本スクリプトがやること

1. **honest walk-forward 予測キャッシュ**（`data/exp_cache/wf_preds_*.pkl`・
   2024-07-01〜2026-08-04 を連続被覆）から本番と同一手順で軸を再構成し、
   7S / 7A / 7SS の母集団を復元する
2. レース単位の**オッズ非依存**な構造特徴（指数のばらつき・競走得点の力差・
   ライン強度）を作る
3. 「結果の三連複配当」を目的変数として、① 配当がどれだけ予測できるか
   ② 配当帯ごとに ROI がどう動くか（＝市場エッジが偏在するか）を測る

⚠️ オッズは `wt_odds`（最終オッズ）。DB 書き込みなし。
⚠️ 掃引窓 2025-07-01〜2026-07-15 / 確認窓 2024-07-01〜2025-06-30 に分けて出す。
   **確認窓は閾値を固定して一度きり**（既存プロトコル）。

使い方:
    python scripts/exp_7s7a_payout_structure.py [--rebuild]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

CACHE_DIR = REPO / "data" / "exp_cache"
DATASET = CACHE_DIR / "payout_structure_7car.pkl"

# 予測キャッシュ（TRAIN_FROM 混在は元プロトコル通り。確認窓=2022-12-01 / 掃引窓=2024-04-01）
PRED_FILES = [
    ("2024-07-01", "2024-09-30", "wf_preds_2024-07-01_2024-09-30_f60_2022-12-01.pkl"),
    ("2024-10-01", "2024-12-31", "wf_preds_2024-10-01_2024-12-31_f60_2022-12-01.pkl"),
    ("2025-01-01", "2025-03-31", "wf_preds_2025-01-01_2025-03-31_f60_2022-12-01.pkl"),
    ("2025-04-01", "2025-06-30", "wf_preds_2025-04-01_2025-06-30_f60_2022-12-01.pkl"),
    ("2025-07-01", "2025-09-30", "wf_preds_2025-07-01_2025-09-30_f60_2024-04-01.pkl"),
    ("2025-10-01", "2025-12-31", "wf_preds_2025-10-01_2025-12-31_f60_2024-04-01.pkl"),
    ("2026-01-01", "2026-04-12", "wf_preds_2026-01-01_2026-04-12_f60_2024-04-01.pkl"),
    ("2026-04-13", "2026-07-15", "wf_preds_2026-04-13_2026-07-15_f60_2024-04-01.pkl"),
    ("2026-07-16", "2026-08-04", "wf_preds_2026-07-16_2026-08-04_f60_2024-04-01.pkl"),
]

STAKE = 100
CONFIRM_END = "2025-06-30"      # 確認窓: 2024-07-01〜2025-06-30
SWEEP_FROM = "2025-07-01"       # 掃引窓: 2025-07-01〜2026-08-04


# ---------------------------------------------------------------------------
# データ組み立て
# ---------------------------------------------------------------------------
def _pg():
    import psycopg2
    return psycopg2.connect(os.environ["KEIRIN_DB_URL"])


def load_entries(d0: str, d1: str) -> pd.DataFrame:
    q = """
    SELECT r.race_key, r.race_date, r.venue_id, r.grade, r.race_type, r.day_index,
           e.frame_no, e.race_point, e.line_group, e.line_size, e.line_pos,
           e.is_line_leader, e.n_lines, e.prediction_mark, e.player_class,
           e.style, e.front_runner, e.finish_order, e.first_rate, e.third_rate
    FROM keirin.wt_entries e
    JOIN keirin.wt_races r ON e.race_key = r.race_key
    WHERE r.n_entries = 7 AND r.race_date BETWEEN %s AND %s
    ORDER BY r.race_date, e.race_key, e.frame_no
    """
    with _pg() as c:
        return pd.read_sql(q, c, params=(d0, d1))


def load_trio(keys: list[str]) -> dict[str, dict[frozenset, float]]:
    out: dict[str, dict[frozenset, float]] = defaultdict(dict)
    with _pg() as c:
        cur = c.cursor()
        for i in range(0, len(keys), 2000):
            ch = keys[i:i + 2000]
            cur.execute(
                "SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                "WHERE bet_type='trio' AND race_key = ANY(%s)", (ch,))
            for rk, comb, od in cur.fetchall():
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(p) == 3:
                        out[rk][p] = v
    return out


def load_preds() -> pd.DataFrame:
    parts = []
    for _, _, fn in PRED_FILES:
        p = CACHE_DIR / fn
        if not p.exists():
            raise SystemExit(f"予測キャッシュが無い: {p}")
        parts.append(pd.read_pickle(p))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=["race_key", "frame_no"], keep="first")


def _entropy(vals: list[float]) -> float:
    s = sum(vals)
    if s <= 0:
        return 0.0
    return -sum((v / s) * math.log(v / s) for v in vals if v > 0)


def build_dataset() -> pd.DataFrame:
    print("予測キャッシュ読み込み ...", flush=True)
    pred = load_preds()
    d0, d1 = "2024-07-01", "2026-08-04"
    print(f"出走表読み込み {d0}〜{d1} ...", flush=True)
    ent = load_entries(d0, d1)
    ent["frame_no"] = ent["frame_no"].astype(int)
    pred["frame_no"] = pred["frame_no"].astype(int)
    df = ent.merge(pred, on=["race_key", "frame_no"], how="inner")
    print(f"  結合後 {len(df):,} 行 / {df['race_key'].nunique():,} レース", flush=True)

    keys = sorted(df["race_key"].unique().tolist())
    print("三連複オッズ読み込み ...", flush=True)
    trio = load_trio(keys)
    print(f"  盤面あり {len(trio):,} レース", flush=True)

    rows = []
    for rk, g in df.groupby("race_key", sort=False):
        if len(g) != 7:
            continue
        board = trio.get(rk)
        if not board:
            continue
        fo = {}
        for t in g.itertuples(index=False):
            v = t.finish_order
            try:
                fo[int(t.frame_no)] = int(v) if v is not None and v == v else 0
            except (TypeError, ValueError):
                fo[int(t.frame_no)] = 0
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue
        res_odds = board.get(frozenset(top3))
        if res_odds is None:
            continue

        f = [int(x) for x in g["frame_no"]]
        p3 = {int(t.frame_no): float(t.pp3) for t in g.itertuples(index=False)}
        pw = {int(t.frame_no): float(t.ppw) for t in g.itertuples(index=False)}
        pb = {int(t.frame_no): float(t.pbad) for t in g.itertuples(index=False)}
        mk = {int(t.frame_no): t.prediction_mark for t in g.itertuples(index=False)}
        rp = {int(t.frame_no): (float(t.race_point) if t.race_point is not None
                                and t.race_point == t.race_point else np.nan)
              for t in g.itertuples(index=False)}

        # ---- 軸選定（本番と同一：3ヘッド） ----
        a1 = max(pw, key=lambda k: pw[k])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        cand = [k for k in p3 if k != a1]
        a2 = max(cand, key=lambda k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k])
        hon = next((k for k, m in mk.items() if m == 1), None)
        tai = next((k for k, m in mk.items() if m == 2), None)
        ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)

        legs = [x for x in f if x not in (a1, a2)
                and frozenset({a1, a2, x}) in board]
        rest = top3 - {a1, a2}
        hit = len(top3 & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
        ret = (round(res_odds * 100) // 10 * 10) if hit else 0

        # ---- ライン構造 ----
        lg = {}
        for t in g.itertuples(index=False):
            v = t.line_group
            lg[int(t.frame_no)] = f"L{v}" if v is not None and v == v else f"S{int(t.frame_no)}"
        lines = defaultdict(list)
        for k, v in lg.items():
            lines[v].append(k)
        line_p = sorted((sum(p3[x] for x in mem) for mem in lines.values()), reverse=True)
        line_rp = sorted((np.nansum([rp[x] for x in mem]) for mem in lines.values()),
                         reverse=True)
        sizes = sorted((len(m) for m in lines.values()), reverse=True)
        n_solo = sum(1 for m in lines.values() if len(m) == 1)

        # ---- 指数のばらつき ----
        v3 = sorted(p3.values(), reverse=True)
        vw = sorted(pw.values(), reverse=True)
        rpv = [x for x in rp.values() if x == x]
        rps = sorted(rpv, reverse=True)

        rows.append(dict(
            race_key=rk, race_date=str(g["race_date"].iloc[0]),
            grade=g["grade"].iloc[0], race_type=g["race_type"].iloc[0],
            venue_id=g["venue_id"].iloc[0],
            a1=a1, a2=a2, overlap=ov,
            asum=p3[a1] + p3[a2], ent=rank_7s_field_entropy(p3),
            p3a1=p3[a1], p3a2=p3[a2],
            hit=int(hit), ret=ret, bet=len(legs) * STAKE, n_legs=len(legs),
            res_odds=res_odds,
            # --- 指数のばらつき ---
            p3_std=float(np.std(v3)), p3_top2=v3[0] + v3[1], p3_top3=sum(v3[:3]),
            p3_g12=v3[0] - v3[1], p3_g23=v3[1] - v3[2], p3_g34=v3[2] - v3[3],
            pw_max=vw[0], pw_g12=vw[0] - vw[1], pw_ent=_entropy(list(pw.values())),
            # --- 競走得点の力差 ---
            rp_std=float(np.std(rps)) if len(rps) >= 2 else np.nan,
            rp_g12=(rps[0] - rps[1]) if len(rps) >= 2 else np.nan,
            rp_top_minus_mean=(rps[0] - float(np.mean(rps))) if rps else np.nan,
            # --- ライン強度 ---
            n_lines=len(lines), n_solo=n_solo, max_line=sizes[0],
            line_p_max=line_p[0],
            line_p_g12=(line_p[0] - line_p[1]) if len(line_p) >= 2 else np.nan,
            line_p_hhi=float(sum((x / sum(line_p)) ** 2 for x in line_p)),
            line_rp_g12=(line_rp[0] - line_rp[1]) if len(line_rp) >= 2 else np.nan,
            axes_same_line=int(lg[a1] == lg[a2]),
            axes_in_top_line=int(lg[a1] == lg[a2] and
                                 sum(p3[x] for x in lines[lg[a1]]) == line_p[0]),
            n_senko=int(sum(1 for t in g.itertuples(index=False)
                            if bool(t.front_runner))),
        ))
    out = pd.DataFrame(rows)
    print(f"データセット {len(out):,} レース", flush=True)
    return out


# ---------------------------------------------------------------------------
def rank_of(r) -> str | None:
    """本番の 7S / 7A / 7SS 判定を再現する（overlap∈{0,1} が前提ゲート）。"""
    if r.overlap not in (0, 1):
        return None
    a_ok = r.asum <= RANK_7S_AXIS_SUM_MAX
    e_ok = r.ent <= RANK_7S_ENTROPY_MAX
    if a_ok and e_ok:
        return "7S"
    if (not a_ok) and e_ok:
        return "7A"
    if a_ok and (not e_ok):
        # 7SS は「軸2車が同一ライン」が追加条件。外れる側は現状どのランクにも
        # 入らない空白（keirin_7car_coverage_gaps_2026_08_05）。
        return "7SS" if r.axes_same_line else "空白E"
    return None  # 両方不合格＝対象外


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if DATASET.exists() and not args.rebuild:
        df = pd.read_pickle(DATASET)
        print(f"[cache] {DATASET.name} {len(df):,} レース")
    else:
        df = build_dataset()
        df.to_pickle(DATASET)
        print(f"[save] {DATASET.name}")

    df["rank"] = [rank_of(r) for r in df.itertuples(index=False)]
    df["win"] = df["race_date"].apply(lambda d: "確認" if d <= CONFIRM_END else "掃引")

    print("\n" + "=" * 100)
    print("【1】母集団の復元（本番と同じ 3ヘッド軸 + 2ゲート + 7SS同一ライン）")
    print(f"  {'rank':<7}{'窓':<5}{'n':>7}{'的中':>8}{'ROI':>8}{'的中中央値':>11}{'≥10倍率':>9}")
    for rk in ("7S", "7A", "7SS", "空白E"):
        for w in ("確認", "掃引"):
            s = df[(df["rank"] == rk) & (df["win"] == w)]
            if s.empty:
                continue
            hits = s[s.hit == 1]
            print(f"  {rk:<7}{w:<5}{len(s):>7}{100*s.hit.mean():>7.1f}%"
                  f"{100*s.ret.sum()/s.bet.sum():>7.1f}%{hits.res_odds.median():>10.1f}倍"
                  f"{100*(hits.res_odds>=10).mean():>8.1f}%")

    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("【2】ROI の配当帯分解 — 「当たっても元本割れ」がどれだけ食っているか")
    print("     ※ 5点×100円=500円。三連複5倍未満の的中は賭け金割れ")
    bands = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
    for rk in ("7S", "7A", "7SS"):
        s = df[df["rank"] == rk]
        bet = s.bet.sum()
        print(f"\n  ── {rk}  n={len(s):,}  ROI={100*s.ret.sum()/bet:.1f}%")
        print(f"     {'配当帯':<14}{'的中数':>7}{'全体比':>8}{'回収寄与':>10}")
        for lo, hi in bands:
            h = s[(s.hit == 1) & (s.res_odds >= lo) & (s.res_odds < hi)]
            if h.empty:
                continue
            lbl = f"{lo:>3}〜{'∞' if hi > 1e8 else int(hi):>3}倍"
            print(f"     {lbl:<14}{len(h):>7}{100*len(h)/len(s):>7.1f}%"
                  f"{100*h.ret.sum()/bet:>9.1f}pt")
        # オラクル: 結果が5倍未満になるレースを事前に全部避けられたら
        cheap = s[s.res_odds < 5]
        rest = s[s.res_odds >= 5]
        print(f"     [オラクル] 結果<5倍のレース({len(cheap):,}件={100*len(cheap)/len(s):.1f}%)を"
              f"全部避けた場合 → n={len(rest):,} 的中{100*rest.hit.mean():.1f}% "
              f"ROI={100*rest.ret.sum()/rest.bet.sum():.1f}%")
        cheap10 = s[s.res_odds < 10]
        rest10 = s[s.res_odds >= 10]
        print(f"     [オラクル] 結果<10倍を全部避けた場合           → n={len(rest10):,} "
              f"的中{100*rest10.hit.mean():.1f}% ROI={100*rest10.ret.sum()/rest10.bet.sum():.1f}%")

    days = {w: df[df["win"] == w]["race_date"].nunique() for w in ("掃引", "確認")}
    print(f"\n  開催日数: 掃引={days['掃引']}日 / 確認={days['確認']}日")
    analyze_predictability(df)
    analyze_quartiles(df, days)


FEATS = [
    # 指数のばらつき（力差）
    "p3_std", "p3_top2", "p3_top3", "p3_g12", "p3_g23", "p3_g34",
    "pw_max", "pw_g12", "pw_ent", "ent", "asum", "p3a1", "p3a2",
    # 競走得点の力差
    "rp_std", "rp_g12", "rp_top_minus_mean",
    # ライン強度
    "n_lines", "n_solo", "max_line", "line_p_max", "line_p_g12",
    "line_p_hhi", "line_rp_g12", "axes_same_line", "n_senko",
]


def auc(y, x):
    """ROC-AUC（NaN 行は除外）。"""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    if len(np.unique(y)) < 2:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def analyze_predictability(df):
    print("\n" + "=" * 100)
    print("【3】「結果の三連複配当≥10倍」はオッズ非依存の構造から読めるか（掃引窓のみ）")
    print("     AUC 0.5=情報なし。参考: 本命バスト予測(7H1)は AUC 0.685 で ROI の壁を破った")
    sw = df[(df["win"] == "掃引") & df["rank"].isin(["7S", "7A", "7SS"])]
    y = (sw.res_odds >= 10).astype(int)
    print(f"     母集団 n={len(sw):,}  ≥10倍の基準率 {100*y.mean():.1f}%")
    scores = sorted(((abs(auc(y, sw[f]) - 0.5), f, auc(y, sw[f])) for f in FEATS),
                    reverse=True)
    print(f"     {'特徴':<20}{'AUC':>8}   {'特徴':<20}{'AUC':>8}")
    half = (len(scores) + 1) // 2
    for i in range(half):
        a = scores[i]
        line = f"     {a[1]:<20}{a[2]:>8.3f}"
        if i + half < len(scores):
            b = scores[i + half]
            line += f"   {b[1]:<20}{b[2]:>8.3f}"
        print(line)


def report(label, s, days_by_win, width=34):
    """1 行サマリ（掃引/確認を分けて出す）。"""
    out = []
    for w in ("掃引", "確認"):
        t = s[s["win"] == w]
        if len(t) < 20:
            out.append(None)
            continue
        h = t[t.hit == 1]
        out.append(dict(n=len(t), per_day=len(t) / days_by_win[w],
                        hit=100 * t.hit.mean(),
                        roi=100 * t.ret.sum() / t.bet.sum(),
                        med=float(h.res_odds.median()) if len(h) else 0.0,
                        ge10=100 * float((h.res_odds >= 10).mean()) if len(h) else 0.0))
    txt = f"  {label:<{width}}"
    for o in out:
        txt += ("      —  " * 5) if o is None else (
            f"{o['n']:>6}{o['per_day']:>6.2f}{o['hit']:>7.1f}%{o['roi']:>7.1f}%"
            f"{o['med']:>7.1f}{o['ge10']:>7.1f}%")
    print(txt)
    return out


def analyze_quartiles(df, days_by_win):
    print("\n" + "=" * 100)
    print("【4】構造特徴の四分位別 ROI（掃引窓で候補を探す → 確認窓で符号を見る）")
    hdr = f"  {'':<34}" + "".join(
        [f"{'n':>6}{'件/日':>6}{'的中':>8}{'ROI':>8}{'中央':>7}{'≥10':>8}" for _ in range(2)])
    for rk in ("7A", "7S"):
        base = df[df["rank"] == rk]
        print(f"\n  ══ {rk} ══ （左=掃引窓 2025-07〜2026-08 / 右=確認窓 2024-07〜2025-06）")
        print(hdr)
        report("基準（絞りなし）", base, days_by_win)
        for f in ("p3_std", "rp_std", "line_p_g12", "line_p_hhi", "n_solo",
                  "p3_g34", "pw_g12", "rp_g12", "n_lines"):
            sw = base[base["win"] == "掃引"]
            qs = sw[f].quantile([0.25, 0.5, 0.75]).tolist()
            if len(set(qs)) < 3:
                lo, hi = sw[f].min(), sw[f].max()
                vals = sorted(sw[f].dropna().unique())
                if len(vals) <= 4:
                    for v in vals:
                        report(f"{f} == {v:g}", base[base[f] == v], days_by_win)
                    continue
            edges = [-np.inf] + qs + [np.inf]
            for i in range(4):
                s = base[(base[f] > edges[i]) & (base[f] <= edges[i + 1])]
                report(f"{f} Q{i+1} ({edges[i]:.3g},{edges[i+1]:.3g}]", s, days_by_win)


if __name__ == "__main__":
    main()
