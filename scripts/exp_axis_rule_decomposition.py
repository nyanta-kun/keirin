"""軸1・軸2のどちらの選定規則が不足しているかを切り分ける（2026-08-05）。

ユーザー指摘:
  「検証としては二軸同時の3着内率の向上が必須。ただし軸1・軸2それぞれ別の基準で
    選んでいるため、**どちらで選定した軸の精度が不足しているか**を原因整理する必要がある」

## 設計上の要点：母集団を固定する

軸を変えると `wt_overlap_n` / `axis_sum` / `entropy` の判定が変わり、7S/7A の
ゲートを通る**レース自体が入れ替わる**。ゲート後で比べると「精度が上がった」のか
「簡単なレースに移った」のか区別できない（9車で平均が符号反転を隠していたのと同型・
[[keirin_three_head_axis_2026_08_04]]）。

そこで本スクリプトは **7車立て・結果確定の全レースを固定母集団**として、
すべての規則を同じレース集合で評価する。ゲート通過後の数字も併記するが、
**規則の優劣は固定母集団の方で判断する**。

## ランク別に集計する（2026-08-05 ユーザー指摘）

7S / 7A / 7B は母集団の構造が違うため合算してはいけない。
- 7S/7A: `wt_overlap_n ∈ {0,1}`＝軸2車のうち公式印は最大1車。**印なし軸が必ず存在する**
- 7B: `wt_overlap_n == 2`＝軸2車が◎◯そのもの。**印なし軸が存在しない**

したがって「軸2の精度不足」の議論が成立するのは 7S/7A のみ。7B は軸が市場と一致して
いる前提のランクで、改善余地は軸選定ではなく相手の絞り方（[[keirin_7b_dominant_trio_candidate_2026_08_05]]）にある。

## 測ること

### 1) 印の内訳（◎/◯/△/×/無印）별 3着内率
前回の `exp_axis2_mark_accuracy.py` は ◎◯ のみを「印あり」とし、△・×・無印を
まとめて「印なし」にしていた。ここでは5分類に割る。

### 2) 二軸同時の分解
- 軸1単独 3着内率 / 軸2単独 3着内率 / 両方3着内率
- **独立ならの期待値 P1×P2 と実測の差**（7車で3枠を奪い合うため負の相関が出るはず）
- 条件付き P(軸2∈top3 | 軸1∈top3) と P(軸2∈top3 | 軸1∉top3)
- 外し方の内訳（軸1のみ / 軸2のみ / 両方外し）

### 3) 規則の入れ替え比較（固定母集団・両方3着内率が主指標）
軸1側・軸2側それぞれの規則を差し替え、どちらを直すと同時率が伸びるかを見る。

⚠️ オッズ非依存（選出は確率のみ）。DB書き込みなし。
honest: 窓ごとに学習しなおす walk-forward 4窓。

使い方:
    python scripts/exp_axis_rule_decomposition.py [--windows w1,w2,w3,w4]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7b_order_disagree, rank_7s_field_entropy, rank_7s_wt_overlap_n,
)

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
    "w3": ("2025-10-01", "2025-12-31"),
    "w4": ("2025-07-01", "2025-09-30"),
}
SEEDS = [42, 101, 202, 303, 404]
MARK_LABEL = {1: "◎", 2: "◯", 3: "△", 4: "×"}
W2 = RANK_AXIS2_BAD_WEIGHT


CACHE_DIR = REPO / "data" / "exp_cache"


def window_preds(df, tf, tt):
    """窓の予測(p3/pw/bad)を返す。キャッシュがあれば再学習しない。

    同じ4窓を使う実験を繰り返すため（軸規則の掃引・7Bの閾値検証など）、
    1窓あたり15分の再学習を毎回やり直さないようにする。キャッシュキーに
    特徴量数と学習開始日を含め、特徴量セットが変わったら作り直される。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{TRAIN_FROM}.pkl"
    if cache.exists():
        print(f"  [cache] {cache.name} を利用（再学習なし）", flush=True)
        return pd.read_pickle(cache)
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < tf)]
    test = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)]
    print(f"  学習中 train {len(train):,} / test {len(test):,} ...", flush=True)
    out = test[["race_key", "frame_no"]].copy()
    out["pp3"] = fit_predict(train, test, TARGET_COL_WT)
    out["ppw"] = fit_predict(train, test, "win_flag")
    out["pbad"] = fit_predict(train, test, "bad6")
    out.to_pickle(cache)
    print(f"  [cache] {cache.name} を保存", flush=True)
    return out


def fit_predict(train, test, target):
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(train[FEATURE_COLS_WT], train[target])
        preds.append(m.predict_proba(test[FEATURE_COLS_WT])[:, 1])
    return np.mean(preds, axis=0)


# ---- 軸1の候補規則（レース r → frame_no）--------------------------------
def a1_pw(r):                 # 現行
    return max(r["pw"], key=lambda f: r["pw"][f])


def a1_p3(r):
    return max(r["p3"], key=lambda f: r["p3"][f])


def a1_pw_bad(r):             # 1着率にも大敗ペナルティ（w=0.3）
    zw, zb = _race_zscore(r["pw"]), _race_zscore(r["bad"])
    return max(zw, key=lambda f: zw[f] - W2 * zb[f])


def a1_honmei(r):             # 公式◎
    return r["hon"]


# ---- 軸2の候補規則（レース r, 軸1 → frame_no）---------------------------
def a2_p3_bad(r, a1):         # 現行
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
    c = [f for f in r["p3"] if f != a1]
    return max(c, key=lambda f: zp[f] - W2 * zb[f]) if c else None


def a2_p3(r, a1):
    c = [f for f in r["p3"] if f != a1]
    return max(c, key=lambda f: r["p3"][f]) if c else None


def a2_pw(r, a1):             # 1着率の2番手
    c = [f for f in r["pw"] if f != a1]
    return max(c, key=lambda f: r["pw"][f]) if c else None


def a2_bad_only(r, a1):       # 大敗しにくさだけで選ぶ
    c = [f for f in r["bad"] if f != a1]
    return min(c, key=lambda f: r["bad"][f]) if c else None


A1_RULES = {"pw最上位(現行)": a1_pw, "p3最上位": a1_p3,
            "z(pw)-0.3z(bad)": a1_pw_bad, "公式◎": a1_honmei}
A2_RULES = {"z(p3)-0.3z(bad)(現行)": a2_p3_bad, "p3最上位": a2_p3,
            "pw2番手": a2_pw, "bad最小": a2_bad_only}


def classify_rank(r, a1, a2):
    """本番と同じランク判定。該当しなければ None。"""
    ov = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
    if ov == 2:
        return "7B" if rank_7b_order_disagree(r["pw"], r["hon"]) is True else None
    if ov not in (0, 1):
        return None
    n_fail = ((r["p3"][a1] + r["p3"][a2] > RANK_7S_AXIS_SUM_MAX)
              + (rank_7s_field_entropy(r["p3"]) > RANK_7S_ENTROPY_MAX))
    return "7S" if n_fail == 0 else ("7A" if n_fail == 1 else None)


def run_window(df, tf, tt, acc):
    print(f"\n######## 窓 test={tf}〜{tt} ########", flush=True)
    preds = window_preds(df, tf, tt)
    t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
        preds, on=["race_key", "frame_no"], how="inner")

    races = []
    for rk, g in t.groupby("race_key"):
        if len(g) != 7:
            continue
        fo = {int(x.frame_no): (int(x.finish_order)
                                if x.finish_order is not None and x.finish_order == x.finish_order
                                else 0) for x in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue
        mark = {int(x.frame_no): x.prediction_mark for x in g.itertuples(index=False)}
        races.append({
            "top3": top3, "mark": mark,
            "hon": next((f for f, m in mark.items() if m == 1), None),
            "tai": next((f for f, m in mark.items() if m == 2), None),
            "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
            "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
            "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)},
        })
    print(f"  固定母集団（7車・結果確定）: {len(races)} レース")

    # ---- 1) 印の内訳 + 2) 二軸同時の分解（現行規則・ランク別）------------
    st = defaultdict(lambda: dict(n=0, n1=0, n2=0, both=0, only1=0, only2=0, neither=0))
    for r in races:
        a1 = a1_pw(r)
        a2 = a2_p3_bad(r, a1)
        rk_lbl = classify_rank(r, a1, a2) or "対象外"
        i1, i2 = a1 in r["top3"], a2 in r["top3"]
        for scope in ("ALL", rk_lbl):
            d = st[scope]
            d["n"] += 1; d["n1"] += i1; d["n2"] += i2
            d["both"] += i1 and i2
            d["only1"] += i1 and not i2
            d["only2"] += i2 and not i1
            d["neither"] += not i1 and not i2
            for lbl, a in (("軸1", a1), ("軸2", a2)):
                mk = MARK_LABEL.get(r["mark"].get(a), "無印")
                k = f"MK:{scope}/{lbl}/{mk}"
                acc.setdefault(k, [0, 0])
                acc[k][0] += 1
                acc[k][1] += int(a in r["top3"])
    for scope, d in st.items():
        acc.setdefault("J:" + scope, []).append(
            {k: (v / d["n"] if k != "n" else v) for k, v in d.items()})
    n1 = n2 = both = only1 = only2 = neither = 0
    for r in []:
        pass

    # ---- 3) 規則の入れ替え（固定母集団 + ゲート後を併記）----------------
    for l1, f1 in A1_RULES.items():
        for l2, f2 in A2_RULES.items():
            lbl = f"軸1={l1} / 軸2={l2}"
            cnt = defaultdict(lambda: [0, 0])
            for r in races:
                a1 = f1(r)
                if a1 is None:
                    continue
                a2 = f2(r, a1)
                if a2 is None or a2 == a1:
                    continue
                ok = {a1, a2} <= r["top3"]
                for scope in ("ALL", classify_rank(r, a1, a2) or "対象外"):
                    cnt[scope][0] += 1
                    cnt[scope][1] += ok
            acc.setdefault("V:" + lbl, []).append(
                {sc: dict(n=v[0], both=100 * v[1] / v[0] if v[0] else 0)
                 for sc, v in cnt.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    args = ap.parse_args()
    max_to = max(t for _, t in WINDOWS.values())
    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    print(f"7車立てに限定: {len(df):,}行")

    acc: dict = {}
    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt, acc)

    SCOPES = ("ALL", "7S", "7A", "7B")
    print("\n" + "=" * 92)
    print("【1】現行規則で選ばれた軸の、公式印の内訳と3着内率（4窓合算・ランク別）")
    for sc in SCOPES:
        keys = {k: v for k, v in acc.items() if k.startswith(f"MK:{sc}/")}
        if not keys:
            continue
        print(f"\n  ── {sc}")
        for lbl in ("軸1", "軸2"):
            tot = sum(v[0] for k, v in keys.items() if f"/{lbl}/" in k)
            if not tot:
                continue
            cells = []
            for mk in ("◎", "◯", "△", "×", "無印"):
                k = f"MK:{sc}/{lbl}/{mk}"
                if k in keys:
                    n, ok = keys[k]
                    cells.append(f"{mk} {100*n/tot:4.1f}%→3着内{100*ok/n:5.1f}%(n={n})")
            print(f"     {lbl}: " + "  ".join(cells))

    print("\n【2】二軸同時の分解（現行規則・窓平均・ランク別）")
    print(f"  {'区分':<8}{'n':>6}{'軸1':>8}{'軸2':>8}{'両方':>8}{'独立仮定':>10}{'差':>8}"
          f"{'軸1のみ':>9}{'軸2のみ':>9}{'両方外し':>9}")
    for sc in SCOPES:
        v = acc.get("J:" + sc)
        if not v:
            continue
        m = {x: float(np.mean([s2[x] for s2 in v])) for x in
             ("n", "n1", "n2", "both", "only1", "only2", "neither")}
        ind = m["n1"] * m["n2"]
        print(f"  {sc:<8}{m['n']:>6.0f}{100*m['n1']:>7.1f}%{100*m['n2']:>7.1f}%"
              f"{100*m['both']:>7.1f}%{100*ind:>9.1f}%{100*(m['both']-ind):>+7.1f}pt"
              f"{100*m['only1']:>8.1f}%{100*m['only2']:>8.1f}%{100*m['neither']:>8.1f}%")
    v = acc.get("J:ALL")
    if v:
        m = {x: float(np.mean([s2[x] for s2 in v])) for x in ("n1", "both", "only2")}
        pc = m["both"] / m["n1"] if m["n1"] else 0
        pu = m["only2"] / (1 - m["n1"]) if m["n1"] < 1 else 0
        print(f"\n  P(軸2∈top3 | 軸1∈top3) = {100*pc:.1f}%  vs  "
              f"P(軸2∈top3 | 軸1∉top3) = {100*pu:.1f}%")

    print("\n【3】規則の入れ替え（両方3着内率・★母集団を固定した ALL で判断する）")
    rows = []
    for k, v in acc.items():
        if not k.startswith("V:"):
            continue
        mm = {}
        for sc in SCOPES:
            vals = [s2[sc]["both"] for s2 in v if sc in s2]
            ns = [s2[sc]["n"] for s2 in v if sc in s2]
            mm[sc] = (float(np.mean(vals)) if vals else 0.0,
                      float(np.mean(ns)) if ns else 0.0)
        rows.append((k[2:], mm, [s2["ALL"]["both"] for s2 in v if "ALL" in s2]))
    rows.sort(key=lambda x: -x[1]["ALL"][0])
    print(f"  {'案':<40}" + "".join(f"{sc:>16}" for sc in SCOPES))
    for lbl, mm, _ in rows:
        print(f"  {lbl:<40}"
              + "".join(f"{mm[sc][0]:>9.1f}%(n{mm[sc][1]:4.0f})" for sc in SCOPES))
    print("\n  窓別（ALL の両方3着内・符号反転の確認／上位5案）")
    for lbl, mm, per in rows[:5]:
        print(f"  {lbl:<40}" + "  ".join(f"{x:5.1f}%" for x in per))


if __name__ == "__main__":
    main()
