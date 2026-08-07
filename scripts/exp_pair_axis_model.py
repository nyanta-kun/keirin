"""二軸（＝三連複2軸総流しの的中）の精度を上げるペア単位モデル（2026-08-07・探索）。

ユーザー要件（2026-08-07）:
  - 全レース（終日）を対象にできるベースモデルを1本作る
  - **オッズは使わない**（直前まで監視できず販売時間が確保できないため）
  - ガミ回避は購入者の金額配分に委ねる。**ROI ではなく的中率を最大化する**
  - 「二軸探偵」ブランドゆえ**軸2車の精度向上が必須**

目的関数はこれ一つ:
    P(軸1・軸2 がともに3着内)  ＝ 三連複2軸総流しの的中率（数学的に同値）

現行は「a1 = pw 最大 / a2 = z(p3) − 0.3·z(pb) 最大」という**車単位スコアの上位2車**。
車単位の周辺確率だけで選ぶと、7車中3車が3着内という**構造的な負の相関**と、
ライン関係による**同時発生の偏り**（memory: 同ライン 1.122x / 別ライン 0.672x）を
まったく使えていない。ここではペア（21通り）を直接スコアリングして比較する。

学習: 確認窓 2024-07-01〜2025-06-30 / 評価: 掃引窓 2025-07-01〜2026-08-04
（p3/pw/pb 自体が honest walk-forward vintage 予測なので二重に安全側）

⚠️ DB は読み取りのみ。
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import RANK_AXIS2_BAD_WEIGHT, _race_zscore  # noqa: E402

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CACHE = REPO / "data" / "exp_cache" / "pair_axis_dataset.pkl"
TRAIN_END = "2025-06-30"   # ここまでで学習
TEST_START = "2025-07-01"  # ここから評価


# --------------------------------------------------------------------------
# 車単位の属性を DB から取る（オッズ非依存の列のみ）
# --------------------------------------------------------------------------
ENTRY_COLS = ("frame_no", "prefecture", "player_class", "style", "race_point",
              "line_group", "line_size", "line_pos", "is_line_leader", "n_lines",
              "front_runner", "stalker", "deep_closer", "marker", "term")


def load_entries(keys: list[str]) -> dict[str, dict[int, dict]]:
    sql = f"""SELECT race_key, {", ".join(ENTRY_COLS)}
              FROM keirin.wt_entries WHERE race_key = ANY(%s)"""
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute(sql, (keys,))
        out: dict[str, dict[int, dict]] = {}
        for row in cur.fetchall():
            rk, rest = row[0], row[1:]
            d = dict(zip(ENTRY_COLS, rest))
            out.setdefault(rk, {})[d["frame_no"]] = d
        return out


# --------------------------------------------------------------------------
# ペア特徴量
# --------------------------------------------------------------------------
def _f(v, default=0.0):
    return default if v is None else float(v)


def build_dataset(races, entries) -> pd.DataFrame:
    rows = []
    for r in races:
        cars = sorted(r["p3"])
        if len(cars) != 7:
            continue
        ent = entries.get(r["rk"])
        if not ent or len(ent) < 7:
            continue
        top3 = set(r["top3"])
        p3, pw, pb = r["p3"], r["pw"], r["pb"]
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        # レース内の順位（1 が最上位）
        rk_p3 = {k: i + 1 for i, k in enumerate(sorted(p3, key=lambda x: -p3[x]))}
        rk_pw = {k: i + 1 for i, k in enumerate(sorted(pw, key=lambda x: -pw[x]))}
        p3v = np.array([p3[k] for k in cars])
        rp = np.array([_f(ent[k].get("race_point")) for k in cars if k in ent])
        race_feat = dict(
            p3_sd=float(p3v.std()), p3_max=float(p3v.max()),
            p3_gap12=float(np.sort(p3v)[-1] - np.sort(p3v)[-2]),
            n_lines=_f(ent[cars[0]].get("n_lines"), np.nan),
            rp_sd=float(rp.std()) if len(rp) else 0.0,
        )
        for i, j in itertools.combinations(cars, 2):
            ei, ej = ent.get(i), ent.get(j)
            if ei is None or ej is None:
                continue
            hi, lo = (i, j) if p3[i] >= p3[j] else (j, i)
            same_line = int(ei.get("line_group") is not None
                            and ei.get("line_group") == ej.get("line_group"))
            pos_i, pos_j = _f(ei.get("line_pos"), 9), _f(ej.get("line_pos"), 9)
            rows.append(dict(
                rk=r["rk"], date=r["date"], i=hi, j=lo,
                label=int(i in top3 and j in top3),
                # --- 車単位（強い方 hi / 弱い方 lo に正規化して並べる）---
                p3_hi=p3[hi], p3_lo=p3[lo],
                pw_hi=pw[hi], pw_lo=pw[lo],
                pb_hi=pb[hi], pb_lo=pb[lo],
                zp_hi=zp[hi], zp_lo=zp[lo], zb_hi=zb[hi], zb_lo=zb[lo],
                rank_p3_hi=rk_p3[hi], rank_p3_lo=rk_p3[lo],
                rank_pw_hi=rk_pw[hi], rank_pw_lo=rk_pw[lo],
                # --- ペアの合成 ---
                p3_prod=p3[hi] * p3[lo], p3_sum=p3[hi] + p3[lo],
                p3_diff=p3[hi] - p3[lo],
                rank_sum=rk_p3[hi] + rk_p3[lo],
                # --- ライン関係（ここが現行の未使用領域）---
                same_line=same_line,
                line_pos_hi=pos_i if p3[i] >= p3[j] else pos_j,
                line_pos_lo=pos_j if p3[i] >= p3[j] else pos_i,
                line_pos_sum=pos_i + pos_j,
                line_pos_gap=abs(pos_i - pos_j),
                line_size_hi=_f(ei.get("line_size") if p3[i] >= p3[j] else ej.get("line_size"), 1),
                line_size_lo=_f(ej.get("line_size") if p3[i] >= p3[j] else ei.get("line_size"), 1),
                both_leader=int(_f(ei.get("is_line_leader")) == 1
                                and _f(ej.get("is_line_leader")) == 1),
                any_single=int(_f(ei.get("line_size"), 1) == 1
                               or _f(ej.get("line_size"), 1) == 1),
                same_pref=int(ei.get("prefecture") is not None
                              and ei.get("prefecture") == ej.get("prefecture")),
                # --- 脚質 ---
                fr_sum=_f(ei.get("front_runner")) + _f(ej.get("front_runner")),
                st_sum=_f(ei.get("stalker")) + _f(ej.get("stalker")),
                dc_sum=_f(ei.get("deep_closer")) + _f(ej.get("deep_closer")),
                rp_hi=_f(ei.get("race_point") if p3[i] >= p3[j] else ej.get("race_point")),
                rp_lo=_f(ej.get("race_point") if p3[i] >= p3[j] else ei.get("race_point")),
                term_gap=abs(_f(ei.get("term")) - _f(ej.get("term"))),
                **race_feat,
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 現行方式のベースライン
# --------------------------------------------------------------------------
def baseline_axes(races) -> pd.DataFrame:
    rows = []
    for r in races:
        if len(r["p3"]) != 7:
            continue
        p3, pw, pb = r["p3"], r["pw"], r["pb"]
        top3 = set(r["top3"])
        a1 = max(pw, key=lambda k: pw[k])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in p3}
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        top2 = sorted(p3, key=lambda k: -p3[k])[:2]
        rows.append(dict(
            rk=r["rk"], date=r["date"],
            hit_3head=int(a1 in top3 and a2 in top3),
            hit_p3top2=int(top2[0] in top3 and top2[1] in top3),
        ))
    return pd.DataFrame(rows)


FEATS = None  # main で決める


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    print(f"読み込み: {len(races):,}R")

    if CACHE.exists():
        df = pd.read_pickle(CACHE)
        print(f"ペアデータ（キャッシュ）: {len(df):,}行")
    else:
        ent = load_entries([r["rk"] for r in races])
        print(f"wt_entries 取得: {len(ent):,}R")
        df = build_dataset(races, ent)
        df.to_pickle(CACHE)
        print(f"ペアデータ生成: {len(df):,}行 → {CACHE}")

    base = baseline_axes(races)
    base = base[base.rk.isin(set(df.rk))]
    te_b = base[base.date >= TEST_START]
    print("\n=== ベースライン（評価窓 2025-07-01〜）===")
    print(f"n={len(te_b):,}R  現行3ヘッド軸 二軸的中 {100*te_b.hit_3head.mean():.2f}%  "
          f"/ p3上位2車 {100*te_b.hit_p3top2.mean():.2f}%")

    # --- 記述: ライン関係が p3積を超える情報を持つか ---
    tr = df[df.date <= TRAIN_END]
    tr = tr.assign(dec=pd.qcut(tr.p3_prod, 10, labels=False, duplicates="drop"))
    print("\n=== 学習窓: p3積の十分位 × ライン関係 の実同時3着内率 ===")
    print(f"{'十分位':>5s} {'別ライン n':>10s} {'率%':>6s} {'同ライン n':>10s} {'率%':>6s} {'差pt':>6s}")
    for d, g in tr.groupby("dec"):
        a = g[g.same_line == 0]
        b = g[g.same_line == 1]
        if len(a) < 50 or len(b) < 50:
            continue
        print(f"{d:5.0f} {len(a):10,d} {100*a.label.mean():6.2f} "
              f"{len(b):10,d} {100*b.label.mean():6.2f} "
              f"{100*(b.label.mean()-a.label.mean()):6.2f}")

    # --- LightGBM ペアモデル ---
    import lightgbm as lgb
    drop = {"rk", "date", "i", "j", "label"}
    feats = [c for c in df.columns if c not in drop]
    tr = df[df.date <= TRAIN_END]
    te = df[df.date >= TEST_START]
    print(f"\n学習 {len(tr):,}行 / 評価 {len(te):,}行 / 特徴量 {len(feats)}")

    m = lgb.train(
        dict(objective="binary", learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=42),
        lgb.Dataset(tr[feats], tr.label), num_boost_round=600)

    te = te.assign(score=m.predict(te[feats]))
    pick = te.loc[te.groupby("rk").score.idxmax()]
    print("\n=== ペアモデル（評価窓）===")
    print(f"n={len(pick):,}R  二軸的中 {100*pick.label.mean():.2f}%")

    # p3積だけで選んだ場合（ライン情報なしの上限確認）
    pick_prod = te.loc[te.groupby("rk").p3_prod.idxmax()]
    print(f"        p3積 最大ペア   二軸的中 {100*pick_prod.label.mean():.2f}%")

    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n=== 重要度上位15 ===")
    for k, v in imp.head(15).items():
        print(f"  {k:16s} {v:12,.0f}")

    # 月別の安定性
    te2 = pick.assign(month=pick.date.str[:7])
    mb = base[base.date >= TEST_START].assign(month=lambda d: d.date.str[:7])
    print("\n=== 月別 二軸的中率（現行3ヘッド → ペアモデル）===")
    for mth in sorted(te2.month.unique()):
        a = mb[mb.month == mth].hit_3head
        b = te2[te2.month == mth].label
        print(f"  {mth}  n={len(b):5,d}  {100*a.mean():5.2f}% → {100*b.mean():5.2f}%  "
              f"({100*(b.mean()-a.mean()):+.2f}pt)")

    te.to_pickle(REPO / "data" / "exp_cache" / "pair_axis_scored.pkl")


if __name__ == "__main__":
    main()
