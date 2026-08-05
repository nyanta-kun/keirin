"""7B の絞り込み条件を確認窓で一度きり検証する（2026-08-05・open_tasks I-13）。

## 背景

7B は 軸2車が両方3着内になる率 48.8% に対し的中 25.3%。**軸が揃ったのに落とす
23.4pt のうち 20.1pt が「意図的に除外した WT△ が3着に来た」**（`exp_7b_miss_
decomposition.py`）。ただし △ を買い目へ戻すと的中は +20pt 跳ねるが ROI は
75.3→72.5% に落ちる（配当が消える）。つまり **△除外は設計どおり**で、
改善余地は「△を除外して**良いレースだけ**を選ぶ」側にある。

## 持ち込む候補（掃引窓 2025-07〜2026-07 で作成）

いずれも「△が弱い＝除外しても3着に来にくい」を測る量。オッズ非依存
（朝の入稿時点で確定している）。

| 候補 | 掃引窓 ROI | 窓別(w1 w2 w3 w4) |
|---|---|---|
| `△と4位の差 <= q30`（相対的な抜け具合） | 83.0% | 100.5 91.0 66.1 74.4 |
| `p3[△] <= q30`（絶対的な強さ） | 79.6% | 93.4 88.5 70.5 65.9 |
| 基準（7B全件） | 75.3% | 86.6 76.0 65.4 73.3 |

⚠️ **どちらも掃引窓の時点で2窓が75%割れ**しており、4窓一貫していない。
`exp_7s7a_threshold_review` の実測では掃引窓の改善幅は確認窓でおよそ 1/3 に
縮むか符号が反転する。したがって本検証は「効くはず」ではなく
**「掃引窓で見えた差が確認窓に残らないことを確かめる」**姿勢で読むこと。

さらに掃引窓では **w3(2025-10〜12) だけどの条件でも 45〜74% と沈む**（基準65.4%）。
どの候補も w1/w2 の伸びで平均を作っており、w3 を直していない。これは
「7Bが効かない時期があり、候補はそれを説明できていない」ことを意味する。

## 手順の約束（`exp_7a_exclusion_confirm.py` と同じ）

- **閾値は掃引窓の分位点を絶対値として算出し、確認窓では動かさない**
  （本スクリプトが掃引窓から自動計算して持ち込む。転記ミス防止）。
  確認窓で分位を取り直すと「その窓に合わせた閾値」になり検証にならない。
- 確認窓は 2024-07〜2025-06（掃引に一度も使っていない4窓）。
  学習は各窓の開始日より前のみ。
- 窓別の符号一貫性を必ず見る（**平均は窓別の反転を隠す**）。
- 対照として同一ライン（7SSで採用・7Bでは掃引窓で無効だった）も並べる。

⚠️ 掃引窓と確認窓で TRAIN_FROM が違う（前者2024-04-01・後者2022-12-01）。
   キャッシュの都合だが、どちらも「学習は窓開始日より前のみ」は満たしている。

DB書き込みなし。予測はキャッシュ利用（数分で完了）。

使い方:
    python scripts/exp_7b_exclusion_confirm.py
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt,
)
from src.strategy_wt import (  # noqa: E402
    RANK_AXIS2_BAD_WEIGHT, _race_zscore, rank_7b_order_disagree,
    rank_7b_select_legs, rank_7s_wt_overlap_n, rank_7ss_same_line,
)

STAKE = 100
SWEEP = {"w1": ("2026-04-13", "2026-07-15", 94), "w2": ("2026-01-01", "2026-04-12", 102),
         "w3": ("2025-10-01", "2025-12-31", 92), "w4": ("2025-07-01", "2025-09-30", 92)}
SWEEP_TRAIN_FROM = "2024-04-01"
CONFIRM = {"c1": ("2025-04-01", "2025-06-30", 91), "c2": ("2025-01-01", "2025-03-31", 90),
           "c3": ("2024-10-01", "2024-12-31", 92), "c4": ("2024-07-01", "2024-09-30", 92)}
CONFIRM_TRAIN_FROM = "2022-12-01"
CACHE_DIR = REPO / "data" / "exp_cache"

# 掃引窓で候補化した分位点（ここを動かさないことが検証の前提）。
QUANTILES = (0.30, 0.50)


def cached_preds(tf, tt, train_from):
    p = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{train_from}.pkl"
    if not p.exists():
        raise SystemExit(f"[FATAL] 予測キャッシュがありません: {p}\n"
                         f"  先に該当窓の掃引/確認スクリプトを実行してください。")
    return pd.read_pickle(p)


def load_trio(keys):
    out = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            ch = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    v = float(od)
                except (TypeError, ValueError):
                    continue
                if 0 < v < 9999:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                    if len(p) == 3:
                        out[rk][p] = v
    return out


def build(df, spec, train_from):
    """本番 7B と同一の母集団・軸・買い目を再現して窓ごとに返す。"""
    per = []
    for w, (tf, tt, days) in spec.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, train_from), on=["race_key", "frame_no"], how="inner")
        races = []
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            fo = {int(x.frame_no): (int(x.finish_order)
                                    if x.finish_order == x.finish_order
                                    and x.finish_order is not None else 0)
                  for x in g.itertuples(index=False)}
            top3 = {f for f, v in fo.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            mk = {int(x.frame_no): x.prediction_mark for x in g.itertuples(index=False)}
            lg = {int(x.frame_no): x.line_group for x in g.itertuples(index=False)}
            r = {"rk": rk, "top3": top3,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in g.itertuples(index=False)},
                 "pw": {int(x.frame_no): float(x.ppw) for x in g.itertuples(index=False)},
                 "bad": {int(x.frame_no): float(x.pbad) for x in g.itertuples(index=False)}}
            # 軸選定は本番と同一（軸1=1着率最上位 / 軸2=z(3着内率)-0.3*z(大敗率)）
            a1 = max(r["pw"], key=lambda f: r["pw"][f])
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            cand = [f for f in r["p3"] if f != a1]
            if not cand:
                continue
            a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
            # 7B の母集団: 軸2車がWT◎◯と完全一致(overlap==2) ∧ 順序不一致
            if rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"]) != 2:
                continue
            if rank_7b_order_disagree(r["pw"], r["hon"]) is not True:
                continue
            r["a1"], r["a2"] = a1, a2
            others = sorted(set(r["p3"]) - {a1, a2})
            r["others"] = others
            r["legs3"] = rank_7b_select_legs(others, r["p3"], r["ana"])
            r["same_line"] = rank_7ss_same_line(a1, a2, lg)
            # 候補指標: △の「絶対的な強さ」と「4位に対する抜け具合」
            rest_no_ana = [f for f in others if f != r["ana"]]
            r["p3ana"] = r["p3"].get(r["ana"], 0.0)
            r["gap_ana"] = (r["p3ana"] - max(r["p3"][f] for f in rest_no_ana)
                            if rest_no_ana and r["ana"] is not None else 0.0)
            races.append(r)
        trio = load_trio(sorted({r["rk"] for r in races}))
        per.append((w, [r for r in races if trio.get(r["rk"])], trio, days))
    return per


def settle(r, board, legs=None):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in (r["legs3"] if legs is None else legs)
            if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def main():
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=CONFIRM_TRAIN_FROM,
                                            max_date="2026-07-15"))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    print("\n掃引窓から閾値を算出（確認窓では動かさない）...", flush=True)
    sw = build(df, SWEEP, SWEEP_TRAIN_FROM)
    print(f"  掃引窓 7B母集団 {sum(len(rs) for _, rs, _, _ in sw)} レース")
    THR: dict[tuple[str, float], float] = {}
    for key, nm in (("gap_ana", "△と4位の差"), ("p3ana", "p3[△]")):
        v = np.concatenate([[r[key] for r in rs] for _, rs, _, _ in sw])
        for q in QUANTILES:
            THR[(key, q)] = float(np.quantile(v, q))
            print(f"  {nm} 下位{int(q*100)}% → 閾値 {THR[(key, q)]:.4f}")

    print("\n確認窓を構築 ...", flush=True)
    cw = build(df, CONFIRM, CONFIRM_TRAIN_FROM)
    print(f"  確認窓 7B母集団 {sum(len(rs) for _, rs, _, _ in cw)} レース")

    def evaluate(per_window, pred, legs_fn=None):
        per = []
        for w, races, trio, days in per_window:
            rows = []
            for r in races:
                if not pred(r):
                    continue
                s = settle(r, trio[r["rk"]], legs_fn(r) if legs_fn else None)
                if s:
                    rows.append(s)
            if not rows:
                continue
            bet = sum(x[0] for x in rows); ret = sum(x[1] for x in rows)
            h = [x for x in rows if x[2]]
            per.append(dict(n=len(rows), per_day=len(rows) / days,
                            hit=100 * len(h) / len(rows),
                            roi=100 * ret / bet if bet else 0,
                            med=statistics.median([x[1] / x[0] for x in h]) if h else 0))
        if not per:
            return None, []
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}, \
               [p["roi"] for p in per]

    def show(per_window, lbl, pred, legs_fn=None, width=32):
        m, per = evaluate(per_window, pred, legs_fn)
        if not m:
            print(f"  {lbl:<{width}} 該当なし")
            return
        flag = "✓" if len(per) == 4 and all(x >= 75 for x in per) else " "
        print(f"  {lbl:<{width}}{m['n']:>6.0f}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%{m['med']:>7.2f}倍  {flag} "
              + " ".join(f"{x:5.1f}" for x in per))

    def table(per_window, title, win_labels):
        print("\n" + "=" * 104)
        print(title)
        print(f"  {'条件':<32}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}{'中央値':>8}"
              f"     窓別ROI({win_labels})")
        show(per_window, "基準（7B全件）", lambda r: True)
        for key, nm in (("gap_ana", "△と4位の差"), ("p3ana", "p3[△]")):
            for q in QUANTILES:
                t = THR[(key, q)]
                show(per_window, f"{nm} <= {t:.3f}（掃引窓 下位{int(q*100)}%）",
                     lambda r, k=key, v=t: r[k] <= v)
        print("  ── 対照（7SSで採用・7Bでは掃引窓で無効だった条件）")
        show(per_window, "同一ライン", lambda r: r["same_line"])
        show(per_window, "別ライン", lambda r: not r["same_line"])
        print("  ── 対照（買い目の作り方・絞り込みなし）")
        show(per_window, "△を戻して4点", lambda r: True,
             legs_fn=lambda r: (r["legs3"] + [r["ana"]]) if r["ana"] is not None
             else r["legs3"])

    table(sw, "【参考】掃引窓 2025-07〜2026-07（候補を作った窓・再掲）", "w1 w2 w3 w4")
    table(cw, "【本番】確認窓 2024-07〜2025-06（閾値は掃引窓の絶対値で固定・一度きり）",
          "c1 c2 c3 c4")

    print("\n  ✓ = 4窓すべてで ROI>=75%")
    print("  ※ 採否は確認窓の窓別一貫性で判断する。平均だけを見ないこと。")

    # ---- 確認窓のブートストラップ（n=190 程度なので差が偶然かを必ず確認する） ----
    # レース単位で復元抽出する。基準と絞り込み後は**同じレース集合の部分集合**なので、
    # 同じリサンプルで両方を再計算して差を取る（paired）。
    print("\n" + "=" * 104)
    print("【確認窓のブートストラップ】レース単位 paired 復元抽出 2,000回")
    rows_all = []
    for w, races, trio, days in cw:
        for r in races:
            s = settle(r, trio[r["rk"]])
            if s:
                rows_all.append((s, r["gap_ana"], r["p3ana"], r["same_line"]))
    print(f"  対象 {len(rows_all)} レース（確認窓4窓 合算）")

    def roi_of(sel):
        bet = sum(x[0][0] for x in sel); ret = sum(x[0][1] for x in sel)
        return 100 * ret / bet if bet else float("nan")

    rng = np.random.default_rng(20260805)
    idx_all = np.arange(len(rows_all))
    for nm, keep in (
        (f"△と4位の差 <= {THR[('gap_ana', 0.30)]:.3f}",
         lambda x: x[1] <= THR[("gap_ana", 0.30)]),
        (f"p3[△] <= {THR[('p3ana', 0.30)]:.3f}",
         lambda x: x[2] <= THR[("p3ana", 0.30)]),
        ("同一ライン", lambda x: x[3]),
    ):
        diffs, rois = [], []
        for _ in range(2000):
            samp = [rows_all[i] for i in rng.choice(idx_all, len(idx_all), replace=True)]
            sub = [x for x in samp if keep(x)]
            if not sub:
                continue
            rois.append(roi_of(sub))
            diffs.append(roi_of(sub) - roi_of(samp))
        lo, hi = np.percentile(rois, [2.5, 97.5])
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        sign = "有意" if dlo > 0 else ("負に有意" if dhi < 0 else "有意差なし")
        print(f"  {nm:<28} ROI {np.mean(rois):6.1f}% [{lo:5.1f}, {hi:5.1f}]"
              f"   基準との差 {np.mean(diffs):+6.1f}pt [{dlo:+6.1f}, {dhi:+6.1f}]  {sign}")
    print("\n  ※ 差のCIが0をまたぐなら「掃引窓で見えた改善は確認窓では確認できなかった」と読む。")


if __name__ == "__main__":
    main()
