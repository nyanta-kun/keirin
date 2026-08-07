"""二軸精度モデル v2: 特徴量とペア構造を強化（2026-08-07・探索）。

v1（exp_pair_axis_model.py）で 53.48% → 54.50%（+1.02pt）。ライン関係は同一
p3積帯で最大 +15.6pt の差を持つのに、選択精度への寄与が小さい。ここでは
「ペアの同時発生」をもっと素直に表せる特徴と目的関数を試す:

  ① 選手属性を全投入（脚質率 ex_*・S/H/B・着別度数・級班・ギア）
  ② ペアの構造（同ライン隣接／前後関係／単騎同士／脚質の組合せ）
  ③ レース属性（グレード・種別・バンク・分戦数）
  ④ 目的関数: binary vs LambdaRank（レース内で1ペアを選ぶ問題そのもの）
  ⑤ 「相手が誰か」ではなく「その2車の**上に何車いるか**」という相対量

学習 ≤2025-06-30（うち末尾2ヶ月を early stopping 用）/ 評価 2025-07-01〜。
⚠️ オッズ不使用。DB は読み取りのみ。
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
CACHE = REPO / "data" / "exp_cache" / "pair_axis_dataset_v2.pkl"
VALID_START = "2025-05-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"

ENTRY_COLS = ("frame_no", "prefecture", "player_class", "style", "race_point",
              "line_group", "line_size", "line_pos", "is_line_leader", "n_lines",
              "front_runner", "stalker", "deep_closer", "marker", "term",
              "gear_ratio", "s_count", "h_count", "b_count",
              "first_rate", "second_rate", "third_rate",
              "ex_spurt_pct", "ex_thrust_pct", "ex_left_behind_pct",
              "ex_split_line_pct", "ex_snatch_pct")

_STYLE = {"逃": 0, "捲": 1, "追": 2, "両": 3}
_CLASS = {"S1": 0, "S2": 1, "SS": 2, "A1": 3, "A2": 4, "A3": 5, "L1": 6}


def _f(v, d=0.0):
    return d if v is None else float(v)


def load_meta(keys):
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute(f"""SELECT race_key, {", ".join(ENTRY_COLS)}
                        FROM keirin.wt_entries WHERE race_key = ANY(%s)""", (keys,))
        ent: dict[str, dict[int, dict]] = {}
        for row in cur.fetchall():
            d = dict(zip(ENTRY_COLS, row[1:]))
            ent.setdefault(row[0], {})[d["frame_no"]] = d
        cur.execute("""SELECT r.race_key, r.grade, r.race_type, r.day_index,
                              v.bank_length
                       FROM keirin.wt_races r
                       LEFT JOIN keirin.venue_info v ON v.venue_code = r.venue_id
                       WHERE r.race_key = ANY(%s)""", (keys,))
        rc = {r[0]: dict(grade=r[1], race_type=r[2], day_index=r[3], bank=r[4])
              for r in cur.fetchall()}
        return ent, rc


def build(races, ent, rc) -> pd.DataFrame:
    grades, rtypes = {}, {}
    rows = []
    for r in races:
        cars = sorted(r["p3"])
        if len(cars) != 7:
            continue
        e = ent.get(r["rk"])
        m = rc.get(r["rk"])
        if not e or len(e) < 7 or m is None:
            continue
        top3 = set(r["top3"])
        p3, pw, pb = r["p3"], r["pw"], r["pb"]
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        rk3 = {k: i + 1 for i, k in enumerate(sorted(p3, key=lambda x: -p3[x]))}
        rkw = {k: i + 1 for i, k in enumerate(sorted(pw, key=lambda x: -pw[x]))}
        p3v = np.array([p3[k] for k in cars])
        srt = np.sort(p3v)[::-1]
        rp = np.array([_f(e[k].get("race_point")) for k in cars])
        g = grades.setdefault(m["grade"], len(grades))
        t = rtypes.setdefault(m["race_type"], len(rtypes))
        rf = dict(p3_sd=float(p3v.std()), p3_max=float(srt[0]),
                  p3_gap12=float(srt[0] - srt[1]), p3_gap23=float(srt[1] - srt[2]),
                  p3_gap34=float(srt[2] - srt[3]),
                  n_lines=_f(e[cars[0]].get("n_lines"), 4),
                  rp_sd=float(rp.std()), rp_max=float(rp.max()),
                  grade_c=g, rtype_c=t, day_index=_f(m["day_index"], 1),
                  bank=_f(m["bank"], 400))
        for a, b in itertools.combinations(cars, 2):
            hi, lo = (a, b) if p3[a] >= p3[b] else (b, a)
            eh, el = e[hi], e[lo]
            gh, gl = eh.get("line_group"), el.get("line_group")
            same = int(gh is not None and gh == gl)
            ph, pl = _f(eh.get("line_pos"), 9), _f(el.get("line_pos"), 9)
            sh = _STYLE.get(eh.get("style"), 4)
            sl = _STYLE.get(el.get("style"), 4)
            # 「この2車より上に何車いるか」（相対量）
            above = sum(1 for k in cars if p3[k] > p3[lo] and k != hi)
            rows.append(dict(
                rk=r["rk"], date=r["date"], hi=hi, lo=lo,
                label=int(a in top3 and b in top3),
                p3_hi=p3[hi], p3_lo=p3[lo], pw_hi=pw[hi], pw_lo=pw[lo],
                pb_hi=pb[hi], pb_lo=pb[lo],
                zp_hi=zp[hi], zp_lo=zp[lo], zb_hi=zb[hi], zb_lo=zb[lo],
                rank3_hi=rk3[hi], rank3_lo=rk3[lo],
                rankw_hi=rkw[hi], rankw_lo=rkw[lo],
                p3_prod=p3[hi] * p3[lo], p3_sum=p3[hi] + p3[lo],
                p3_diff=p3[hi] - p3[lo], rank_sum=rk3[hi] + rk3[lo],
                n_above=above,
                # ライン構造
                same_line=same, line_pos_hi=ph, line_pos_lo=pl,
                line_adjacent=int(same and abs(ph - pl) == 1),
                hi_ahead=int(same and ph < pl),
                line_size_hi=_f(eh.get("line_size"), 1),
                line_size_lo=_f(el.get("line_size"), 1),
                both_leader=int(_f(eh.get("is_line_leader")) == 1
                                and _f(el.get("is_line_leader")) == 1),
                both_single=int(_f(eh.get("line_size"), 1) == 1
                                and _f(el.get("line_size"), 1) == 1),
                any_single=int(_f(eh.get("line_size"), 1) == 1
                               or _f(el.get("line_size"), 1) == 1),
                same_pref=int(eh.get("prefecture") is not None
                              and eh.get("prefecture") == el.get("prefecture")),
                # 脚質（組合せをコード化 + 同ラインとの交互作用）
                style_hi=sh, style_lo=sl, style_pair=sh * 5 + sl,
                style_pair_same=(sh * 5 + sl) if same else -1,
                # 選手属性
                cls_hi=_CLASS.get(eh.get("player_class"), 7),
                cls_lo=_CLASS.get(el.get("player_class"), 7),
                rp_hi=_f(eh.get("race_point")), rp_lo=_f(el.get("race_point")),
                rp_diff=_f(eh.get("race_point")) - _f(el.get("race_point")),
                gear_hi=_f(eh.get("gear_ratio"), 3.9), gear_lo=_f(el.get("gear_ratio"), 3.9),
                term_hi=_f(eh.get("term")), term_lo=_f(el.get("term")),
                s_hi=_f(eh.get("s_count")), s_lo=_f(el.get("s_count")),
                b_hi=_f(eh.get("b_count")), b_lo=_f(el.get("b_count")),
                fr_hi=_f(eh.get("front_runner")), fr_lo=_f(el.get("front_runner")),
                st_hi=_f(eh.get("stalker")), st_lo=_f(el.get("stalker")),
                dc_hi=_f(eh.get("deep_closer")), dc_lo=_f(el.get("deep_closer")),
                r1_hi=_f(eh.get("first_rate")), r1_lo=_f(el.get("first_rate")),
                r3_hi=_f(eh.get("third_rate")), r3_lo=_f(el.get("third_rate")),
                exs_hi=_f(eh.get("ex_spurt_pct")), exs_lo=_f(el.get("ex_spurt_pct")),
                ext_hi=_f(eh.get("ex_thrust_pct")), ext_lo=_f(el.get("ex_thrust_pct")),
                exl_hi=_f(eh.get("ex_left_behind_pct")),
                exl_lo=_f(el.get("ex_left_behind_pct")),
                exp_hi=_f(eh.get("ex_split_line_pct")),
                exp_lo=_f(el.get("ex_split_line_pct")),
                **rf))
    return pd.DataFrame(rows)


def baseline(races) -> pd.DataFrame:
    rows = []
    for r in races:
        if len(r["p3"]) != 7:
            continue
        p3, pw, pb = r["p3"], r["pw"], r["pb"]
        t3 = set(r["top3"])
        a1 = max(pw, key=lambda k: pw[k])
        zp, zb = _race_zscore(p3), _race_zscore(pb)
        sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in p3}
        a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
        rows.append(dict(rk=r["rk"], date=r["date"],
                         hit=int(a1 in t3 and a2 in t3)))
    return pd.DataFrame(rows)


def evaluate(te, score_col, label="") -> float:
    pick = te.loc[te.groupby("rk")[score_col].idxmax()]
    return 100 * pick.label.mean()


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    if CACHE.exists():
        df = pd.read_pickle(CACHE)
        print(f"ペアデータ（キャッシュ）: {len(df):,}行")
    else:
        ent, rc = load_meta([r["rk"] for r in races])
        df = build(races, ent, rc)
        df.to_pickle(CACHE)
        print(f"ペアデータ生成: {len(df):,}行")

    base = baseline(races)
    base = base[base.rk.isin(set(df.rk))]
    b_te = base[base.date >= TEST_START]
    print(f"\n=== 評価窓 {TEST_START}〜  n={len(b_te):,}R ===")
    print(f"現行3ヘッド軸           : {100*b_te.hit.mean():.2f}%")

    import lightgbm as lgb
    drop = {"rk", "date", "hi", "lo", "label"}
    feats = [c for c in df.columns if c not in drop]
    cats = ["style_hi", "style_lo", "style_pair", "style_pair_same",
            "cls_hi", "cls_lo", "grade_c", "rtype_c"]
    for c in cats:
        df[c] = df[c].astype("category")

    tr = df[df.date < VALID_START]
    va = df[(df.date >= VALID_START) & (df.date <= TRAIN_END)]
    te = df[df.date >= TEST_START].copy()
    print(f"学習 {len(tr):,} / 検証 {len(va):,} / 評価 {len(te):,} / 特徴量 {len(feats)}")

    print(f"p3積 最大ペア           : {evaluate(te, 'p3_prod'):.2f}%")

    res = {}
    # --- ① binary ---
    m1 = lgb.train(
        dict(objective="binary", learning_rate=0.05, num_leaves=127,
             min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=42),
        lgb.Dataset(tr[feats], tr.label, categorical_feature=cats),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(va[feats], va.label, categorical_feature=cats)],
        callbacks=[lgb.early_stopping(100, verbose=False)])
    te["s_bin"] = m1.predict(te[feats])
    res["binary"] = (evaluate(te, "s_bin"), m1.best_iteration)

    # --- ② LambdaRank（レース内で1ペアを選ぶ問題そのもの）---
    tr_s = tr.sort_values("rk")
    va_s = va.sort_values("rk")
    m2 = lgb.train(
        dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
             learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             verbose=-1, seed=42, label_gain=[0, 1]),
        lgb.Dataset(tr_s[feats], tr_s.label, categorical_feature=cats,
                    group=tr_s.groupby("rk", sort=False).size().values),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(va_s[feats], va_s.label, categorical_feature=cats,
                                group=va_s.groupby("rk", sort=False).size().values)],
        callbacks=[lgb.early_stopping(100, verbose=False)])
    te["s_rank"] = m2.predict(te[feats])
    res["lambdarank"] = (evaluate(te, "s_rank"), m2.best_iteration)

    # --- ③ 2つの平均（順位の合成）---
    te["s_avg"] = (te.groupby("rk").s_bin.rank(pct=True)
                   + te.groupby("rk").s_rank.rank(pct=True))
    res["bin+rank 合成"] = (evaluate(te, "s_avg"), 0)

    print()
    for k, (v, it) in res.items():
        print(f"ペアモデル {k:16s}: {v:.2f}%  "
              f"({v - 100*b_te.hit.mean():+.2f}pt, best_iter={it})")

    imp = pd.Series(m1.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n=== binary 重要度上位20 ===")
    for k, v in imp.head(20).items():
        print(f"  {k:16s} {v:12,.0f}")

    best = "s_rank" if res["lambdarank"][0] >= res["binary"][0] else "s_bin"
    pick = te.loc[te.groupby("rk")[best].idxmax()].assign(month=lambda d: d.date.str[:7])
    mb = b_te.assign(month=lambda d: d.date.str[:7])
    print(f"\n=== 月別（現行 → {best}）===")
    for mth in sorted(pick.month.unique()):
        a, c = mb[mb.month == mth].hit, pick[pick.month == mth].label
        print(f"  {mth}  n={len(c):5,d}  {100*a.mean():5.2f}% → {100*c.mean():5.2f}%  "
              f"({100*(c.mean()-a.mean()):+.2f}pt)")

    te.to_pickle(REPO / "data" / "exp_cache" / "pair_axis_scored_v2.pkl")


if __name__ == "__main__":
    main()
