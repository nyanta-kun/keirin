"""安定2車抽出のレース条件探索・車数別（2026-07-15）

目的: 欠車前（全カード出走馬）の指数分布から「上位2車がともに3着内」
      = SS的中条件（オッズ非依存）の的中率とROIを、出走車数(6/7/9)ごとに
      別モデルで検証する。目標 hit2>=50%（最低30%）・ROIの確保。

方針（ユーザー指示 2026-07-15）:
  - 落車・失格・欠車が発生したレースはノイズ → レースごと除外。
    欠車(absent)はスクレイパーが選手行を除外するので、「行数==出走車数 かつ
    全員 finish_order>=1」のレースだけが純粋な力関係の結果。
  - 7車限定をやめ、出走車数ごとに分けて学習・検証（6/7/9車）。

汚染防止の窓:
  TRAIN    2022-12-01 〜 2026-02-28（モデル学習のみ）
  DISCOVER 2026-03-01 〜 2026-05-31（OOS・条件探索/検証）
  CONFIRM  2026-06-01 〜 2026-07-10（完全未使用ホールドアウト）

使い方:
  .venv/bin/python scripts/exp_stable_top2_wt.py            # フル（特徴量+車数別学習）
  .venv/bin/python scripts/exp_stable_top2_wt.py --from-cache  # 条件/ROIのみ再実行
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.preprocessing.feature_wt import (  # noqa: E402
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT)
from src.models.trainer import train_lgbm  # noqa: E402
from src.database import get_connection  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
TRAIN_END = "2026-02-28"
DISC = ("2026-03-01", "2026-05-31")
CONF = ("2026-06-01", "2026-07-10")
SIZES = [6, 7, 9]


def build_features_once() -> pd.DataFrame:
    print("① 特徴量構築 2022-12-01〜2026-07-10（全車）...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date="2022-12-01", max_date="2026-07-10"))
    with get_connection() as c:
        ne = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races "
            "WHERE race_date BETWEEN '2022-12-01' AND '2026-07-10'"))
    df["n_entries"] = df["race_key"].map(lambda k: ne.get(k))
    print(f"   rows={len(df):,}", flush=True)
    return df


def clean_races_of_size(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """出走車数==size かつ 全員完走のクリーンレースのみ返す（落車/失格/欠車除外）。"""
    dfx = df[df["n_entries"] == size].copy()
    dfx["_fin"] = (dfx["finish_order"] >= 1).astype(int)
    agg = dfx.groupby("race_key").agg(rows=("_fin", "size"), fin=("_fin", "sum"))
    clean = set(agg[(agg["rows"] == size) & (agg["fin"] == size)].index)
    n_all = dfx["race_key"].nunique()
    dfx = dfx[dfx["race_key"].isin(clean)].drop(columns="_fin")
    print(f"   [{size}車] 除外前{n_all:,} → クリーン{len(clean):,}レース", flush=True)
    return dfx


def build_race_table(df: pd.DataFrame, size: int) -> pd.DataFrame:
    dfx = clean_races_of_size(df, size)
    if dfx.empty:
        return pd.DataFrame()
    fit = dfx[(dfx["race_date"] <= TRAIN_END)].copy()
    print(f"   [{size}車] モデル学習 rows={len(fit):,}...", flush=True)
    model = train_lgbm(fit, feature_cols=FEATURE_COLS_WT, target_col="top3_flag")
    dfx["pred"] = model.predict_proba(prepare_X(dfx))[:, 1]

    rows = []
    for rk, g in dfx.groupby("race_key"):
        g = g.sort_values("pred", ascending=False).reset_index(drop=True)
        p = g["pred"].to_numpy()
        frames = g["frame_no"].astype(int).to_numpy()
        fo = g["finish_order"].to_numpy()
        top3 = {int(fr) for fr, o in zip(frames, fo)
                if pd.notna(o) and int(o) in (1, 2, 3)}
        if len(top3) < 3:
            continue
        a1, a2 = int(frames[0]), int(frames[1])
        pn = p / p.sum()
        rows.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0], "size": size,
            "hit2": int(a1 in top3 and a2 in top3),
            "hit1": int(a1 in top3),
            "a1": a1, "a2": a2,
            "thirds": sorted(int(x) for x in frames[2:]),
            "top3": sorted(top3),
            "gap12": p[0] - p[1], "gap23": p[1] - p[2], "gap13": p[0] - p[2],
            "gap2_rest": p[1] - p[2:].mean(),
            "top2_share": pn[0] + pn[1],
            "std": p.std(),
        })
    out = pd.DataFrame(rows)
    cache = CACHE_DIR / f"stable_top2_n{size}.pkl"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    return out


def seg(df, w):
    return df[(df["race_date"] >= w[0]) & (df["race_date"] <= w[1])]


def _parse_combo(comb):
    try:
        return [int(x) for x in re.split(r"[-=→]", str(comb))]
    except ValueError:
        return None


def load_odds_maps(race_keys):
    wide, trio = {}, {}
    rks = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(rks), 900):
            chunk = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('quinellaPlace','trio') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                parts = _parse_combo(comb)
                if not parts:
                    continue
                if bt == "quinellaPlace" and len(parts) == 2:
                    wide.setdefault(rk, {})[frozenset(parts)] = float(od)
                elif bt == "trio" and len(parts) == 3:
                    trio.setdefault(rk, {})[frozenset(parts)] = float(od)
    return wide, trio


def hit_sweep(df, col, ths, size):
    disc, conf = seg(df, DISC), seg(df, CONF)
    dd, cd = disc["race_date"].nunique() or 1, conf["race_date"].nunique() or 1
    print(f"  [{size}車] {col}>= : DISCOVER hit2(R/日)  →  CONFIRM hit2(R/日)")
    for th in ths:
        d, c = disc[disc[col] >= th], conf[conf[col] >= th]
        print(f"    >={th:<5} D:{d['hit2'].mean() if len(d) else 0:5.1%}"
              f"({len(d)/dd:4.1f}/日 n={len(d):>4})  "
              f"C:{c['hit2'].mean() if len(c) else 0:5.1%}({len(c)/cd:4.1f}/日 n={len(c):>4})")


def roi_eval(df, label, mask, size):
    sub = df[mask]
    wide, trio = load_odds_maps(sub["race_key"].unique().tolist())
    print(f"\n【ROI】[{size}車] {label}  (最終オッズ・100円/点)")
    for wlabel, w in (("DISC", DISC), ("CONF", CONF)):
        s = seg(sub, w)
        days = s["race_date"].nunique() or 1
        wn = wb = wp = wh = tn = tb = tp = th = 0
        for _, r in s.iterrows():
            od = wide.get(r["race_key"], {}).get(frozenset({r["a1"], r["a2"]}))
            if od is not None:
                wn += 1; wb += 100
                if r["hit2"]:
                    wh += 1; wp += int(od * 100)
            legs = {t: trio.get(r["race_key"], {}).get(frozenset({r["a1"], r["a2"], t}))
                    for t in r["thirds"]}
            legs = {t: o for t, o in legs.items() if o}
            if legs:
                tn += 1; tb += len(legs) * 100
                won = frozenset(r["top3"])
                for t, o in legs.items():
                    if frozenset({r["a1"], r["a2"], t}) == won:
                        tp += int(o * 100); th += 1; break
        wr = f"ワイド R={wn:>4}({wn/days:4.1f}/日) 的中{wh/wn:5.1%} ROI={wp/wb:6.1%}" if wn else "ワイド n/a"
        tr = f"三連複2-全 R={tn:>4} 的中{th/tn:5.1%} ROI={tp/tb:6.1%}" if tn else "三連複 n/a"
        print(f"  {wlabel}: {wr} | {tr}")


def report(tables):
    for size, df in tables.items():
        if df is None or df.empty:
            print(f"\n===== {size}車: データなし =====")
            continue
        disc, conf = seg(df, DISC), seg(df, CONF)
        dd, cd = disc["race_date"].nunique() or 1, conf["race_date"].nunique() or 1
        print("\n" + "=" * 78)
        print(f"===== {size}車 ベースライン（条件なし・クリーンレース）=====")
        print(f"  DISCOVER: R={len(disc):>5}({len(disc)/dd:4.1f}/日) hit2={disc['hit2'].mean():5.1%} 単={disc['hit1'].mean():5.1%}")
        print(f"  CONFIRM : R={len(conf):>5}({len(conf)/cd:4.1f}/日) hit2={conf['hit2'].mean():5.1%} 単={conf['hit1'].mean():5.1%}")
        hit_sweep(df, "top2_share", [0.35, 0.40, 0.45, 0.50, 0.55], size)
        hit_sweep(df, "gap23", [0.03, 0.05, 0.08, 0.10], size)
        roi_eval(df, "条件なし", pd.Series(True, index=df.index), size)
        roi_eval(df, "top2_share>=0.45", df["top2_share"] >= 0.45, size)
        roi_eval(df, "top2_share>=0.50", df["top2_share"] >= 0.50, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()
    tables = {}
    if args.from_cache:
        for s in SIZES:
            f = CACHE_DIR / f"stable_top2_n{s}.pkl"
            tables[s] = pd.read_pickle(f) if f.exists() else None
    else:
        df = build_features_once()
        for s in SIZES:
            print(f"\n② 車数{s} の学習・テーブル生成 ...", flush=True)
            tables[s] = build_race_table(df, s)
    report(tables)


if __name__ == "__main__":
    main()
