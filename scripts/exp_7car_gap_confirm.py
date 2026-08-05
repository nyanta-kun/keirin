"""7車の空白から絞った3候補を確認窓で一度きり検証する（2026-08-05・7B代替）。

センサス `exp_7car_coverage_census.py`（掃引窓 2025-07〜2026-07）で作った候補を、
**掃引に一度も使っていない確認窓 2024-07〜2025-06** で検証する。

## 持ち込む候補（3件のみ。約90セル見た中から機序・n・窓の散らばりで絞った）

| # | 候補 | 買い目 | 掃引窓 |
|---|---|---|---|
| 1 | **空白3 × (決勝 ∨ 準決勝)** | △除外3点 | 7.63件/日・ROI 84.0%相当（決勝87.1 / 準決勝82.8・両方4窓✓） |
| 2 | **空白3 全件**（土台確認） | △除外3点 | 48.53件/日・ROI 76.9%（窓別 76.7/77.3/74.7/78.9＝±1.3pt） |
| 3 | **空白1 × 逃げ型0-1人** | 総流し5点 | 0.53件/日・ROI 104.4%（4窓✓だが n=201・符号反転の懸念あり） |

空白3 = overlap==2 ∧ **order一致**（モデル1位=WT◎＝市場と完全合意）。7車の82%が
overlap==2 で、その92%がここ。7B（order不一致・3.97件/日）の隣にある48.53件/日。

## 持ち込まないもの（掃引窓で棄却）

- P1/P2 軸2車のライン先頭・ライン内位置 … 空白1で ROI 87.2% に見えるが窓別
  94.4/131.2/**59.3/64.0** と2窓が壁割れ＝不安定
- P4 ◎○が同一ラインか … **仮説と逆**（同一77.4% > 別76.0%）かつ差 +1.4pt で無意味
- P5/P6 ライン数・P7 級班・P9 バンク周長 … 機序が弱く効果量も小さい

## 読み方の約束

- **掃引窓は楽観的に出る**。実績: 7S 92.4%→84.4% / 7SS 91.8%→85.9%＝**6〜8pt 縮む**。
  確認窓の絶対値がこの縮み幅の範囲に収まるかを見る
- **窓別の符号一貫性を必ず見る**（平均は反転を隠す）
- **ブートストラップ必須**。2026-08-05 の7Bで「4窓すべて改善」でも有意差なしだった
  （n=190で ±20pt の CI）。**窓別一貫性は反転の検出には使えるが効果の立証には使えない**

DB書き込みなし。予測はキャッシュ利用。

使い方:
    python scripts/exp_7car_gap_confirm.py
"""
from __future__ import annotations

import re
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
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX, RANK_AXIS2_BAD_WEIGHT,
    _race_zscore, rank_7b_order_disagree, rank_7b_select_legs,
    rank_7s_field_entropy, rank_7s_wt_overlap_n, rank_7ss_same_line,
)

STAKE = 100
SWEEP = {"w1": ("2026-04-13", "2026-07-15", 94), "w2": ("2026-01-01", "2026-04-12", 102),
         "w3": ("2025-10-01", "2025-12-31", 92), "w4": ("2025-07-01", "2025-09-30", 92)}
SWEEP_TRAIN_FROM = "2024-04-01"
CONFIRM = {"c1": ("2025-04-01", "2025-06-30", 91), "c2": ("2025-01-01", "2025-03-31", 90),
           "c3": ("2024-10-01", "2024-12-31", 92), "c4": ("2024-07-01", "2024-09-30", 92)}
CONFIRM_TRAIN_FROM = "2022-12-01"
CACHE_DIR = REPO / "data" / "exp_cache"

FINALS = ("決勝", "準決勝")


def cached_preds(tf, tt, train_from):
    p = CACHE_DIR / f"wf_preds_{tf}_{tt}_f{len(FEATURE_COLS_WT)}_{train_from}.pkl"
    if not p.exists():
        raise SystemExit(f"[FATAL] 予測キャッシュがありません: {p}")
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


def _truthy(v) -> bool:
    if v is None or v != v:
        return False
    if isinstance(v, str):
        return v not in ("", "0", "false", "False", "None")
    return bool(v)


def build(df, spec, train_from):
    races = []
    days_total = sum(d for _, _, d in spec.values())
    for w, (tf, tt, _days) in spec.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, train_from), on=["race_key", "frame_no"], how="inner")
        for rk, g in t.groupby("race_key"):
            if len(g) != 7:
                continue
            rows = list(g.itertuples(index=False))
            fo = {int(x.frame_no): (int(x.finish_order)
                                    if x.finish_order == x.finish_order
                                    and x.finish_order is not None else 0) for x in rows}
            top3 = {f for f, v in fo.items() if 1 <= v <= 3}
            if len(top3) != 3:
                continue
            mk = {int(x.frame_no): x.prediction_mark for x in rows}
            lg = {int(x.frame_no): x.line_group for x in rows}
            r = {"rk": rk, "w": w, "top3": top3,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in rows},
                 "pw": {int(x.frame_no): float(x.ppw) for x in rows},
                 "bad": {int(x.frame_no): float(x.pbad) for x in rows},
                 "race_type": str(rows[0].race_type),
                 "n_front": sum(1 for x in rows if _truthy(x.front_runner))}
            a1 = max(r["pw"], key=lambda f: r["pw"][f])
            zp, zb = _race_zscore(r["p3"]), _race_zscore(r["bad"])
            cand = [f for f in r["p3"] if f != a1]
            if not cand:
                continue
            a2 = max(cand, key=lambda f: zp[f] - RANK_AXIS2_BAD_WEIGHT * zb[f])
            r["a1"], r["a2"] = a1, a2
            r["ov"] = rank_7s_wt_overlap_n(a1, a2, r["hon"], r["tai"])
            r["asum"] = r["p3"][a1] + r["p3"][a2]
            r["ent"] = rank_7s_field_entropy(r["p3"])
            r["axis_ok"] = r["asum"] <= RANK_7S_AXIS_SUM_MAX
            r["ent_ok"] = r["ent"] <= RANK_7S_ENTROPY_MAX
            r["same_line"] = rank_7ss_same_line(a1, a2, lg)
            r["order_dis"] = rank_7b_order_disagree(r["pw"], r["hon"])
            others = sorted(set(r["p3"]) - {a1, a2})
            r["others"] = others
            r["legs3"] = rank_7b_select_legs(others, r["p3"], r["ana"])
            races.append(r)
    trio = load_trio(sorted({r["rk"] for r in races}))
    return [r for r in races if trio.get(r["rk"])], trio, days_total


def settle(r, board, legs):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in legs if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


# ---- 空白の定義（センサスと同一・ここを動かさないことが検証の前提） ----
def is_gap1(r):   # overlap∈{0,1} ∧ axis_ok ∧ entropy不合格 ∧ 別ライン
    return r["ov"] in (0, 1) and r["axis_ok"] and not r["ent_ok"] and not r["same_line"]


def is_gap3(r):   # overlap==2 ∧ order一致
    return r["ov"] == 2 and r["order_dis"] is not True


def main():
    print("データ読み込み ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=CONFIRM_TRAIN_FROM,
                                            max_date="2026-07-15"))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    sets = {}
    for nm, spec, tfrom in (("掃引窓 2025-07〜2026-07", SWEEP, SWEEP_TRAIN_FROM),
                            ("確認窓 2024-07〜2025-06", CONFIRM, CONFIRM_TRAIN_FROM)):
        print(f"{nm} を構築 ...", flush=True)
        sets[nm] = build(df, spec, tfrom)
        print(f"  7車・オッズ有り {len(sets[nm][0])} レース")

    def ev(races, trio, days, sel_fn, mode):
        per_w, rows_all = defaultdict(list), []
        for r in races:
            if not sel_fn(r):
                continue
            legs = r["others"] if mode == "full" else r["legs3"]
            s = settle(r, trio[r["rk"]], legs)
            if s:
                per_w[r["w"]].append(s); rows_all.append(s)
        if not rows_all:
            return None
        rois = []
        for w in sorted(per_w):
            rw = per_w[w]
            b = sum(x[0] for x in rw); rt = sum(x[1] for x in rw)
            rois.append(100 * rt / b if b else 0)
        bet = sum(x[0] for x in rows_all); ret = sum(x[1] for x in rows_all)
        h = [x for x in rows_all if x[2]]
        return dict(n=len(rows_all), per_day=len(rows_all) / days,
                    hit=100 * len(h) / len(rows_all),
                    roi=100 * ret / bet if bet else 0, rois=rois)

    CANDS = [
        ("空白3 全件（土台）", is_gap3, "three"),
        ("★空白3 × (決勝∨準決勝)",
         lambda r: is_gap3(r) and r["race_type"] in FINALS, "three"),
        ("  内訳: 空白3 × 決勝",
         lambda r: is_gap3(r) and r["race_type"] == "決勝", "three"),
        ("  内訳: 空白3 × 準決勝",
         lambda r: is_gap3(r) and r["race_type"] == "準決勝", "three"),
        ("  対照: 空白3 × 予選・一般系",
         lambda r: is_gap3(r) and r["race_type"] not in FINALS, "three"),
        ("空白1 全件（土台）", is_gap1, "full"),
        ("★空白1 × 逃げ型0-1人",
         lambda r: is_gap1(r) and r["n_front"] <= 1, "full"),
        ("  対照: 空白1 × 逃げ型2人以上",
         lambda r: is_gap1(r) and r["n_front"] >= 2, "full"),
    ]

    for nm in sets:
        races, trio, days = sets[nm]
        print("\n" + "=" * 104)
        print(f"【{nm}】（{days}日）")
        print(f"  {'候補':<30}{'n':>7}{'件/日':>8}{'的中':>9}{'ROI':>9}"
              f"     窓別ROI")
        for lbl, fn, mode in CANDS:
            m = ev(races, trio, days, fn, mode)
            if not m:
                print(f"  {lbl:<30} 該当なし")
                continue
            flag = "✓" if len(m["rois"]) == 4 and all(x >= 75 for x in m["rois"]) else " "
            print(f"  {lbl:<30}{m['n']:>7}{m['per_day']:>8.2f}{m['hit']:>8.1f}%"
                  f"{m['roi']:>8.1f}%  {flag} " + " ".join(f"{x:5.1f}" for x in m["rois"]))

    # ---- ブートストラップ（確認窓・各土台に対する候補の上乗せ分） ----
    races, trio, days = sets["確認窓 2024-07〜2025-06"]
    print("\n" + "=" * 104)
    print("【確認窓のブートストラップ】レース単位 paired 復元抽出 2,000回")
    rng = np.random.default_rng(20260805)
    for base_lbl, base_fn, mode, subs in (
        ("空白3", is_gap3, "three",
         [("× (決勝∨準決勝)", lambda r: r["race_type"] in FINALS),
          ("× 決勝のみ", lambda r: r["race_type"] == "決勝"),
          ("× 準決勝のみ", lambda r: r["race_type"] == "準決勝")]),
        ("空白1", is_gap1, "full",
         [("× 逃げ型0-1人", lambda r: r["n_front"] <= 1)]),
    ):
        pool = []
        for r in races:
            if not base_fn(r):
                continue
            legs = r["others"] if mode == "full" else r["legs3"]
            s = settle(r, trio[r["rk"]], legs)
            if s:
                pool.append((s, r))
        if not pool:
            continue
        print(f"\n  ■ 土台 {base_lbl}（n={len(pool)}）")

        def roi_of(sel):
            b = sum(x[0][0] for x in sel); rt = sum(x[0][1] for x in sel)
            return 100 * rt / b if b else float("nan")

        idx = np.arange(len(pool))
        for slbl, sfn in subs:
            rois, diffs = [], []
            for _ in range(2000):
                samp = [pool[i] for i in rng.choice(idx, len(idx), replace=True)]
                sub = [x for x in samp if sfn(x[1])]
                if not sub:
                    continue
                rois.append(roi_of(sub))
                diffs.append(roi_of(sub) - roi_of(samp))
            if not rois:
                print(f"    {slbl:<20} 該当なし")
                continue
            lo, hi = np.percentile(rois, [2.5, 97.5])
            dlo, dhi = np.percentile(diffs, [2.5, 97.5])
            sign = "有意" if dlo > 0 else ("負に有意" if dhi < 0 else "有意差なし")
            print(f"    {slbl:<20} ROI {np.mean(rois):6.1f}% [{lo:5.1f}, {hi:5.1f}]"
                  f"   土台との差 {np.mean(diffs):+6.1f}pt [{dlo:+6.1f}, {dhi:+6.1f}]  {sign}")

        # 土台そのものが控除率75%を越えるか
        boot = [roi_of([pool[i] for i in rng.choice(idx, len(idx), replace=True)])
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"    {'（土台のROI）':<20} ROI {np.mean(boot):6.1f}% [{lo:5.1f}, {hi:5.1f}]"
              f"   → 75%を {'上回る' if lo > 75 else '上回るとは言えない'}")

    print("\n  ✓ = 4窓すべてで ROI>=75%")
    print("  ※ 掃引窓は楽観的（7S 92.4→84.4 / 7SS 91.8→85.9＝6〜8pt縮む）。"
          "確認窓の絶対値で判断すること。")


if __name__ == "__main__":
    main()
