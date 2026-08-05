"""7車レースの被覆センサス — 現行ランクが取りこぼしている空白の実測（2026-08-05）。

ユーザー依頼「7SS/7S/7A で対象としていないレース・パターンで再現性の高そうな
ものを想定・検証」の第1段階。**候補を作るための記述統計**であり、採否は決めない。

## 現行の被覆マップ（`src/strategy_wt.py` の定義から導出）

    overlap = rank_7s_wt_overlap_n(軸1, 軸2, WT◎, WT○)   # 軸2車と印の重なり数
    axis_ok = axis_sum <= RANK_7S_AXIS_SUM_MAX(1.40)
    ent_ok  = entropy  <= RANK_7S_ENTROPY_MAX(1.8329)

    overlap∈{0,1} ∧  axis_ok ∧  ent_ok                → 7S
    overlap∈{0,1} ∧ ¬axis_ok ∧  ent_ok                → 7A
    overlap∈{0,1} ∧  axis_ok ∧ ¬ent_ok ∧  same_line   → 7SS
    overlap∈{0,1} ∧  axis_ok ∧ ¬ent_ok ∧ ¬same_line   → 空白1  ★最大の空白
    overlap∈{0,1} ∧ ¬axis_ok ∧ ¬ent_ok                → 空白2  ★未測定
    overlap==2    ∧  order_disagree                    → 7B（2026-08-05に自動投稿OFF）
    overlap==2    ∧ ¬order_disagree                    → 空白3  ★市場と完全合意
    overlap is None                                    → 対象外（WTマーク欠損）

## 測ること

1. 各セルの 件数/日・的中・ROI（買い目は 7S/7A/7SS と同じ三連複 軸2車+総流し5点。
   overlap==2 のセルは 7B と同じ △除外3点 も併記する）
2. 空白1 / 空白2 / 空白3 の中を、**WT印とオッズが構造的に encode していない量**で
   分解する（P1〜P8）。7SS が効いたのは `line_group` がまさにそれだったため。

   P1 別ラインでも両方がライン先頭か   is_line_leader
   P2 軸2車のライン内役割の組み合わせ  line_pos
   P3 先行争いの構造（逃げ型の人数）   front_runner / style
   P4 ◎○が同一ラインか（印の側）      line_group × prediction_mark
   P5 全員単騎戦 / P6 2分戦            n_lines
   P7 級班混合戦                        player_class
   P8 開催段階                          race_type

⚠️ **掃引窓（2025-07〜2026-07）のみで実行する。** 確認窓（2024-07〜2025-06）は
   候補を絞ってから一度きり使うため、ここでは一切参照しない。
⚠️ ここで見える差は多重比較で必ず膨らむ。**採否は確認窓＋ブートストラップで決める**
   （2026-08-05 の 7B で「4窓すべて改善」でも有意差なしだった教訓）。

DB書き込みなし。予測はキャッシュ利用。

使い方:
    python scripts/exp_7car_coverage_census.py
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
CACHE_DIR = REPO / "data" / "exp_cache"
TOTAL_DAYS = sum(d for _, _, d in SWEEP.values())


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


def build(df):
    """掃引窓の7車レースを、分類に必要な属性をすべて付けて返す。"""
    races = []
    for w, (tf, tt, days) in SWEEP.items():
        t = df[(df["race_date"] >= tf) & (df["race_date"] <= tt)].merge(
            cached_preds(tf, tt, SWEEP_TRAIN_FROM),
            on=["race_key", "frame_no"], how="inner")
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
            lp = {int(x.frame_no): x.line_pos for x in rows}
            ll = {int(x.frame_no): _truthy(x.is_line_leader) for x in rows}
            r = {"rk": rk, "w": w, "top3": top3,
                 "hon": next((f for f, m in mk.items() if m == 1), None),
                 "tai": next((f for f, m in mk.items() if m == 2), None),
                 "ana": next((f for f, m in mk.items() if m == 3), None),
                 "p3": {int(x.frame_no): float(x.pp3) for x in rows},
                 "pw": {int(x.frame_no): float(x.ppw) for x in rows},
                 "bad": {int(x.frame_no): float(x.pbad) for x in rows},
                 "lg": lg, "lp": lp, "ll": ll,
                 "n_lines": int(rows[0].n_lines) if rows[0].n_lines == rows[0].n_lines else 0,
                 "race_type": str(rows[0].race_type),
                 "grade": str(rows[0].grade),
                 "bank": (float(rows[0].bank_length)
                          if rows[0].bank_length == rows[0].bank_length else 0.0),
                 "n_front": sum(1 for x in rows if _truthy(x.front_runner)),
                 "n_class": len({str(x.player_class) for x in rows}),
                 }
            # 軸選定（本番と同一の3ヘッド）
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
            # ---- 分解軸 ----
            r["both_leader"] = ll.get(a1, False) and ll.get(a2, False)   # P1
            r["n_leader"] = int(ll.get(a1, False)) + int(ll.get(a2, False))
            def _pos(f):                                                  # P2
                v = lp.get(f)
                return int(v) if v == v and v is not None else 0
            r["pos_pair"] = tuple(sorted((_pos(a1), _pos(a2))))
            # P4: WT◎○が同一ラインか
            r["mark_same_line"] = rank_7ss_same_line(r["hon"], r["tai"], lg) \
                if (r["hon"] is not None and r["tai"] is not None) else False
            races.append(r)
    trio = load_trio(sorted({r["rk"] for r in races}))
    return [r for r in races if trio.get(r["rk"])], trio


def settle(r, board, legs):
    a1, a2 = r["a1"], r["a2"]
    legs = [x for x in legs if frozenset({a1, a2, x}) in board]
    if not legs:
        return None
    rest = r["top3"] - {a1, a2}
    hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 and next(iter(rest)) in legs
    ret = (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0
    return len(legs) * STAKE, ret, hit


def main():
    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t, _ in SWEEP.values())
    df = build_features_wt(load_raw_data_wt(min_date=SWEEP_TRAIN_FROM, max_date=max_to))
    fo_ = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo_ >= 6) & (fo_ >= 1)).astype(int)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()

    print("掃引窓を構築 ...", flush=True)
    races, trio = build(df)
    print(f"  7車・オッズ有り: {len(races)} レース（{TOTAL_DAYS}日）\n")

    def ev(sel, mode="full"):
        """mode: full=総流し5点 / three=△除外3点"""
        per_w, rows_all = defaultdict(list), []
        for r in sel:
            legs = r["others"] if mode == "full" else r["legs3"]
            s = settle(r, trio[r["rk"]], legs)
            if s:
                per_w[r["w"]].append(s); rows_all.append(s)
        if not rows_all:
            return None
        rois = []
        for w in SWEEP:
            rw = per_w.get(w) or []
            if rw:
                b = sum(x[0] for x in rw); rt = sum(x[1] for x in rw)
                rois.append(100 * rt / b if b else 0)
        bet = sum(x[0] for x in rows_all); ret = sum(x[1] for x in rows_all)
        h = [x for x in rows_all if x[2]]
        return dict(n=len(rows_all), per_day=len(rows_all) / TOTAL_DAYS,
                    hit=100 * len(h) / len(rows_all),
                    roi=100 * ret / bet if bet else 0, rois=rois)

    def show(lbl, sel, mode="full", width=38, indent=""):
        m = ev(sel, mode)
        if not m:
            print(f"{indent}  {lbl:<{width}} 該当なし")
            return
        flag = "✓" if len(m["rois"]) == 4 and all(x >= 75 for x in m["rois"]) else " "
        print(f"{indent}  {lbl:<{width}}{m['n']:>6}{m['per_day']:>7.2f}{m['hit']:>8.1f}%"
              f"{m['roi']:>8.1f}%  {flag} " + " ".join(f"{x:5.1f}" for x in m["rois"]))

    HDR = (f"  {'区分':<38}{'n':>6}{'件/日':>7}{'的中':>9}{'ROI':>9}"
           f"     窓別ROI(w1 w2 w3 w4)")

    # ---- 1) 被覆マップ ------------------------------------------------
    o01 = [r for r in races if r["ov"] in (0, 1)]
    o2 = [r for r in races if r["ov"] == 2]
    onone = [r for r in races if r["ov"] is None]
    cells = {
        "7S":    [r for r in o01 if r["axis_ok"] and r["ent_ok"]],
        "7A":    [r for r in o01 if not r["axis_ok"] and r["ent_ok"]],
        "7SS":   [r for r in o01 if r["axis_ok"] and not r["ent_ok"] and r["same_line"]],
        "空白1 (E群・別ライン)":
                 [r for r in o01 if r["axis_ok"] and not r["ent_ok"] and not r["same_line"]],
        "空白2 (両ゲート不合格)":
                 [r for r in o01 if not r["axis_ok"] and not r["ent_ok"]],
    }
    cells7b = [r for r in o2 if r["order_dis"] is True]
    cells_gap3 = [r for r in o2 if r["order_dis"] is not True]

    print("=" * 108)
    print(f"【1】被覆マップ（買い目=三連複 軸2車+総流し5点。掃引窓 {TOTAL_DAYS}日）")
    print(HDR)
    for k, v in cells.items():
        show(k, v)
    print("  ── overlap==2（7Bの土俵）")
    show("7B（order不一致・△除外3点）", cells7b, "three")
    show("7B（同・総流し5点で比較）", cells7b, "full")
    show("空白3（order一致・△除外3点）", cells_gap3, "three")
    show("空白3（同・総流し5点）", cells_gap3, "full")
    print("  ── 参考")
    show("overlap=None（マーク欠損）", onone)
    print(f"\n  母集団シェア: overlap∈0,1 {len(o01)} / ==2 {len(o2)} / None {len(onone)}"
          f"  （7車全体 {len(races)}）")

    # ---- 2) 空白の分解 ------------------------------------------------
    def breakdown(name, sel, mode="full"):
        if not sel:
            print(f"\n■ {name}: 該当なし")
            return
        print("\n" + "=" * 108)
        print(f"■ {name}  （n={len(sel)} / {len(sel)/TOTAL_DAYS:.2f}件_日）")
        print(HDR)
        show("基準（この空白の全件）", sel, mode)
        print("  ── P1 別ラインでも両方がライン先頭か")
        for lbl, fn in (("軸2車とも line_leader", lambda r: r["n_leader"] == 2),
                        ("片方だけ leader", lambda r: r["n_leader"] == 1),
                        ("どちらも leader でない", lambda r: r["n_leader"] == 0)):
            show(lbl, [r for r in sel if fn(r)], mode)
        print("  ── P2 軸2車のライン内位置の組み合わせ（多い順に上位5）")
        cnt = defaultdict(int)
        for r in sel:
            cnt[r["pos_pair"]] += 1
        for pp, _n in sorted(cnt.items(), key=lambda kv: -kv[1])[:5]:
            show(f"line_pos {pp}", [r for r in sel if r["pos_pair"] == pp], mode)
        print("  ── P3 先行争いの構造（逃げ型の人数）")
        for lbl, fn in (("逃げ型 0-1人", lambda r: r["n_front"] <= 1),
                        ("逃げ型 2人", lambda r: r["n_front"] == 2),
                        ("逃げ型 3人以上", lambda r: r["n_front"] >= 3)):
            show(lbl, [r for r in sel if fn(r)], mode)
        print("  ── P4 WT◎○が同一ラインか（印の側）")
        show("◎○が同一ライン", [r for r in sel if r["mark_same_line"]], mode)
        show("◎○が別ライン", [r for r in sel if not r["mark_same_line"]], mode)
        print("  ── P5/P6 ライン数")
        for lbl, fn in (("n_lines=2（2分戦）", lambda r: r["n_lines"] == 2),
                        ("n_lines=3（標準）", lambda r: r["n_lines"] == 3),
                        ("n_lines=4", lambda r: r["n_lines"] == 4),
                        ("n_lines>=5（単騎多数）", lambda r: r["n_lines"] >= 5)):
            show(lbl, [r for r in sel if fn(r)], mode)
        print("  ── P7 級班の混合度")
        for lbl, fn in (("単一級班", lambda r: r["n_class"] == 1),
                        ("2級班混合", lambda r: r["n_class"] == 2),
                        ("3級班以上", lambda r: r["n_class"] >= 3)):
            show(lbl, [r for r in sel if fn(r)], mode)
        print("  ── P8 開催段階")
        for lbl, keys in (("予選系", ("予選", "チャレンジ予選", "特予選")),
                          ("準決勝", ("準決勝",)),
                          ("決勝", ("決勝",)),
                          ("一般・特選系", ("一般", "特選", "初特選"))):
            show(lbl, [r for r in sel if r["race_type"] in keys], mode)
        print("  ── P9 バンク周長（参考・7車では過去に否定）")
        for lbl, fn in (("333m系 (<360)", lambda r: 0 < r["bank"] < 360),
                        ("400m系", lambda r: 360 <= r["bank"] < 460),
                        ("500m系 (>=460)", lambda r: r["bank"] >= 460)):
            show(lbl, [r for r in sel if fn(r)], mode)

    breakdown("空白1 (overlap∈{0,1} ∧ axis_ok ∧ entropy不合格 ∧ 別ライン)",
              cells["空白1 (E群・別ライン)"])
    breakdown("空白2 (overlap∈{0,1} ∧ 両ゲート不合格)",
              cells["空白2 (両ゲート不合格)"])
    breakdown("空白3 (overlap==2 ∧ order一致・△除外3点)", cells_gap3, "three")

    print("\n" + "=" * 108)
    print("  ✓ = 掃引窓4窓すべてで ROI>=75%")
    print("  ⚠️ ここは候補を作る窓。多重比較で必ず膨らむ。採否は確認窓"
          "（2024-07〜2025-06）＋ブートストラップで決めること。")


if __name__ == "__main__":
    main()
