"""検証C2: 固め2車前提で3列目を構造特徴で絞る（2026-07-15）

固め2車（top2_share>=0.5・hit2~67%）が決まる前提で、残り5車の
  ①指数のばらつき（モデル3位と4位以下のgap）
  ②ライン連携（3列目が軸2車と同じライン＝番手/3番手か）
  ③ライン合計得点
から3列目を絞り、三連複ROIを確保できるか検証する。

per-horse キャッシュを生成（モデル train<=2026-02-28・クリーン7車）。
窓: DISCOVER 03-01〜05-31 / CONFIRM 06-01〜07-10

使い方:
  .venv/bin/python scripts/exp_third_structure_wt.py            # per-horse生成+分析
  .venv/bin/python scripts/exp_third_structure_wt.py --from-cache
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.preprocessing.feature_wt import (  # noqa: E402
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT)
from src.models.trainer import train_lgbm  # noqa: E402
from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import load_odds_maps, seg, DISC, CONF, CACHE_DIR  # noqa: E402

PH_CACHE = CACHE_DIR / "perhorse_n7.pkl"
TRAIN_END = "2026-02-28"


def build_perhorse():
    print("特徴量構築 + 7車クリーン抽出 ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date="2022-12-01", max_date="2026-07-10"))
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races "
                            "WHERE race_date BETWEEN '2022-12-01' AND '2026-07-10'"))
    df = df[df["race_key"].map(lambda k: ne.get(k)) == 7].copy()
    df["_fin"] = (df["finish_order"] >= 1).astype(int)
    agg = df.groupby("race_key").agg(rows=("_fin", "size"), fin=("_fin", "sum"))
    clean = set(agg[(agg["rows"] == 7) & (agg["fin"] == 7)].index)
    df = df[df["race_key"].isin(clean)].copy()
    print(f"  クリーン7車: {len(clean):,}レース", flush=True)

    fit = df[df["race_date"] <= TRAIN_END]
    print(f"  学習 rows={len(fit):,} ...", flush=True)
    model = train_lgbm(fit, feature_cols=FEATURE_COLS_WT, target_col="top3_flag")
    df["pred"] = model.predict_proba(prepare_X(df))[:, 1]
    df["model_rank"] = df.groupby("race_key")["pred"].rank(ascending=False, method="first")
    keep = ["race_key", "race_date", "frame_no", "pred", "model_rank", "finish_order",
            "race_point", "style", "line_group", "line_size", "is_line_leader"]
    out = df[keep].copy()
    out.to_pickle(PH_CACHE)
    print(f"  保存: {PH_CACHE} ({len(out):,} rows)", flush=True)
    return out


def analyze(ph):
    # レース単位に整形
    _, trio = load_odds_maps(ph["race_key"].unique().tolist())
    recs = []
    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        fr = g["frame_no"].astype(int).tolist()
        lg = dict(zip(g["frame_no"].astype(int), g["line_group"].astype(int)))
        rp = dict(zip(g["frame_no"].astype(int), g["race_point"].astype(float)))
        fo = dict(zip(g["frame_no"].astype(int), g["finish_order"]))
        stl = dict(zip(g["frame_no"].astype(int), g["style"].astype(str)))
        pred = g["pred"].tolist()
        a1, a2 = fr[0], fr[1]
        thirds = fr[2:]              # モデル3〜7位
        top3 = {f for f, o in fo.items() if pd.notna(o) and int(o) in (1, 2, 3)}
        if len(top3) < 3:
            continue
        pn = np.array(pred) / sum(pred)
        top2_share = pn[0] + pn[1]
        hit2 = int(a1 in top3 and a2 in top3)
        # 軸2車のライン集合
        axis_lines = {lg[a1], lg[a2]}
        # 3列目(軸以外のtop3馬)
        third = [f for f in top3 if f not in (a1, a2)]
        third = third[0] if third else None
        # ライン合計得点（グループごと）
        line_pts = {}
        for f in fr:
            line_pts[lg[f]] = line_pts.get(lg[f], 0) + rp[f]
        # 残り5車の指数ばらつき: モデル3位pred - 4位pred
        gap34 = pred[2] - pred[3]
        recs.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0],
            "top2_share": top2_share, "hit2": hit2,
            "a1": a1, "a2": a2, "thirds": thirds, "top3": sorted(top3),
            "gap34": gap34,
            "axis_lines": axis_lines,
            "lg": lg, "line_pts": line_pts, "stl": stl,
            "best_line": max(line_pts, key=line_pts.get),
            "third_actual": third,
            "third_style": (stl.get(third) if third else None),
            "third_in_axis_line": int(third is not None and lg.get(third) in axis_lines),
            "trio": trio.get(rk, {}),
        })
    R = pd.DataFrame(recs)
    print(f"\n7車レース: {len(R):,}")

    lock = R[R["top2_share"] >= 0.5]
    print(f"固め2車母集団(top2_share>=0.5): {len(lock):,}")

    # === 記述: 固め2車が的中(hit2=1)したレースで、3列目の性質 ===
    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(lock, w)
        hit = s[s["hit2"] == 1]
        if not len(hit):
            continue
        # 3列目のモデル順位分布
        rank_of_third = []
        for _, r in hit.iterrows():
            if r["third_actual"] is None:
                continue
            # third のモデル順位 = thirds内index+3
            try:
                rk3 = r["thirds"].index(r["third_actual"]) + 3
            except ValueError:
                rk3 = None
            rank_of_third.append(rk3)
        rank_of_third = [x for x in rank_of_third if x]
        in_line = hit["third_in_axis_line"].mean()
        print(f"\n== {wl} 固め2車的中レース n={len(hit)} ==")
        print(f"  3列目が軸ラインから出る率: {in_line:.1%}")
        vc = pd.Series(rank_of_third).value_counts(normalize=True).sort_index()
        print(f"  3列目のモデル順位分布: " +
              " ".join(f"{int(k)}位{v:.0%}" for k, v in vc.items()))

    # === ROI: 固め2車前提の絞り買い戦略 ===
    print("\n" + "=" * 70)
    print("三連複 絞り買い戦略（軸2車 固定・top2_share>=0.5・最終オッズ100円/点）")
    strategies = {
        "S0 全5点(2-全)": lambda r: r["thirds"],
        "S1 モデル3-4位のみ(2点)": lambda r: r["thirds"][:2],
        "S2 モデル3-5位(3点)": lambda r: r["thirds"][:3],
        "S3 軸ライン馬のみ": lambda r: [f for f in r["thirds"] if r["lg"].get(f) in r["axis_lines"]],
        "S4 モデル3-4位∪軸ライン馬": lambda r: sorted(set(r["thirds"][:2]) |
                                     {f for f in r["thirds"] if r["lg"].get(f) in r["axis_lines"]}),
        "S5 gap34>=0.03 & モデル3位1点": lambda r: (r["thirds"][:1] if r["gap34"] >= 0.03 else []),
        "S6 追・両脚質のみ": lambda r: [f for f in r["thirds"] if r["stl"].get(f) in ("追", "両")],
        "S7 最高得点ライン馬のみ": lambda r: [f for f in r["thirds"]
                                  if r["lg"].get(f) == r["best_line"]],
        "S8 モデル3-4位∩追両": lambda r: [f for f in r["thirds"][:2] if r["stl"].get(f) in ("追", "両")],
    }
    # 3列目の脚質分布（記述）
    lock2 = lock[lock["hit2"] == 1]
    if len(lock2):
        vc = lock2["third_style"].value_counts(normalize=True)
        print("固め2車的中時の3列目 脚質分布: " +
              " ".join(f"{k}{v:.0%}" for k, v in vc.items()))
    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(lock, w)
        days = s["race_date"].nunique() or 1
        print(f"\n--- {wl} ({days}日) ---")
        for name, fn in strategies.items():
            n = b = p = h = 0
            for _, r in s.iterrows():
                legs = {}
                for t in fn(r):
                    od = r["trio"].get(frozenset({r["a1"], r["a2"], t}))
                    if od:
                        legs[t] = od
                if not legs:
                    continue
                n += 1
                b += len(legs) * 100
                won = frozenset(r["top3"])
                for t, od in legs.items():
                    if frozenset({r["a1"], r["a2"], t}) == won:
                        p += int(od * 100); h += 1; break
            if n and b:
                print(f"  {name:<26} R={n:>4}({n/days:4.1f}/日) 的中={h/n:5.1%} "
                      f"投{b:>7,} ROI={p/b:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()
    if args.from_cache and PH_CACHE.exists():
        ph = pd.read_pickle(PH_CACHE)
        print(f"per-horse キャッシュ読込: {len(ph):,} rows")
    else:
        ph = build_perhorse()
    analyze(ph)


if __name__ == "__main__":
    main()
