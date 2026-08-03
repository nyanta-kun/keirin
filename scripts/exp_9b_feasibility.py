"""【読み取り専用・否定結果の記録】9車立てへ 7B の設計を移植できるか（9B）の検証。

2026-08-03 実施。**結論: 不採用（ユーザー判断 (b) 現状維持）。**

## 背景

7B（◎◯一致だが順序・相手で市場と不一致・三連複3点）を7車立てで新設した直後、
同じ設計を9車立てへ広げられるかというユーザー要望を受けて検証した。

## 結論と根拠（全期間 2024-01-05〜2026-08-02・521日・5,677候補）

| | 7B(採用済) | 9B(本検証) |
|---|---|---|
| n | 5,624 | **556** |
| 件/日 | 5.95 | **1.07** |
| 的中率 | 26.5% | 21.0% |
| ROI | 75.9% | **82.1%** |
| ガミ率 | 8.1% | 4.3% |
| 月次ROI標準偏差 | **14.3** | **62.0** |
| 月次ROI 60%割れ | **0/32ヶ月** | **10/32ヶ月** |

ROI 自体は 7B を上回るが、以下4点から不採用とした:

1. **大穴依存**: 最高配当3本を除くと 73.5% と控除率75%を割る（0本82.1 → 1本78.4 →
   3本73.5 → 5本69.4 → 10本60.4）。7B は10本除外でも 72.6% を保つ。
   9B は母集団の性質ではなくごく少数の高配当で成立している。
2. **年次のブレ**: 2024 82.2% / 2025 72.5% / 2026 98.0%（7B は 78.6/78.8 とほぼ一定）。
3. **導入動機が無い**: 7B は「7S/7A の overlap∈{0,1} が 18〜23% まで枯渇し増枠余地
   ゼロ」だから作った。**9車の overlap∈{0,1} は 46.2%（0:3.5% + 1:42.7%）で枯渇して
   いない**。9S/9A の母集団は健在。
4. **既存 9S/9A の方が母数で有利**: overlap01 総流し7点は 3.96件/日・ROI 71.0%・
   最高247.1倍で、件数は約4倍。

## 副次的な発見（今後に使える）

**△(ana) を相手から外す効果は9車でも成立する**: K=3 で ROI 69.8% → 82.1%（+12.3pt）、
20倍超の的中 7本 → 18本。「市場と一致した相手を買うと配当が消える」という 7B の
中核仮説は車数に依存しない性質である可能性が高い。
→ 未検証の派生案: **9S/9A の総流し7点から △ を外して6点にする**（母集団を減らさず
   配当を改善できるか）。本検証では実施していない。

## 使い方

    # ① 月次凍結vintageで9車候補キャッシュを作る（32ヶ月・約60分）
    PYTHONPATH=. .venv/bin/python scripts/exp_9b_feasibility.py --build-cache /tmp/rich9
    # ② 集計
    PYTHONPATH=. .venv/bin/python scripts/exp_9b_feasibility.py --analyze /tmp/rich9

DB書き込みなし。
"""
from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    rank_7s_field_entropy, rank_7s_select_axis, rank_7s_wt_mark3_overlap_n,
    rank_7s_wt_overlap_n,
)
from src.wt_vintage_config import monthly_windows

from scripts.backfill_7s_rank_wt import _load_trio_boards

N_CAR = 9
STAKE = 100


def build_cache(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for date_from, date_to, eval_model, win_model in monthly_windows():
        tag = date_from[:7]
        dst = out_dir / f"{tag}.pkl"
        if dst.exists():
            print(f"[skip] {tag}", flush=True)
            continue
        print(f"[build] {tag}", flush=True)
        model, wmodel = load_model(eval_model), load_model(win_model)
        df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
        rows: list[dict] = []
        if not df.empty:
            with get_connection() as c:
                ne_map = dict(c.execute(
                    "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
                    (date_from, date_to)))
                date_map = dict(c.execute(
                    "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
                    (date_from, date_to)))
                rks = [rk for rk, ne in ne_map.items() if ne and int(ne) == N_CAR]
                fins: dict[str, list] = {}
                marks: dict[str, dict[int, int]] = {}
                for i in range(0, len(rks), 900):
                    chunk = rks[i:i + 900]
                    q = ("SELECT race_key, frame_no, finish_order, prediction_mark "
                         "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
                    for rk, fno, fo, pmv in c.execute(q, chunk):
                        if fo is not None and fo >= 1:
                            fins.setdefault(rk, []).append((fo, int(fno)))
                        if pmv is not None:
                            marks.setdefault(rk, {})[int(fno)] = int(pmv)
            df = df[df["race_key"].isin(set(rks))].copy()
            if not df.empty:
                X = prepare_X(df)
                df["pred_prob"] = model.predict_proba(X)[:, 1]
                df["pred_win"] = wmodel.predict_proba(X)[:, 1]
                trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
                pm = _load_payouts_wt(df["race_key"].unique().tolist())
                for rk, g in df.groupby("race_key"):
                    if ne_map.get(rk) != N_CAR or len(g) != N_CAR:
                        continue
                    trio = trio_bd.get(rk)
                    if not trio:
                        continue
                    board = set()
                    for k in trio:
                        board |= set(k)
                    if len(board) != N_CAR:
                        continue
                    fin = sorted(fins.get(rk, []))
                    if len(fin) < 3:
                        continue
                    wp = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
                    tp = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
                    sel = rank_7s_select_axis(wp, tp)
                    if sel is None:
                        continue
                    a1, a2, asum = sel
                    if a1 not in board or a2 not in board:
                        continue
                    others = sorted(board - {a1, a2})
                    if len(others) != N_CAR - 2:
                        continue
                    mk = marks.get(rk, {})
                    h = next((f for f, v in mk.items() if v == 1), None)
                    t = next((f for f, v in mk.items() if v == 2), None)
                    an = next((f for f, v in mk.items() if v == 3), None)
                    actual = frozenset(f for _, f in fin[:3])
                    rows.append({
                        "race_key": rk, "race_date": date_map.get(rk, ""),
                        "axis1": a1, "axis2": a2, "axis_sum": asum,
                        "entropy": rank_7s_field_entropy(tp), "others": others,
                        "top3_probs": tp, "win_probs": wp,
                        "trio_legs": {x: trio.get(frozenset({a1, a2, x})) for x in others},
                        "actual_top3": tuple(sorted(actual)),
                        "trio_pay": pm.get(rk, {}).get(("trio", actual), 0),
                        "wt_overlap_n": rank_7s_wt_overlap_n(a1, a2, h, t),
                        "wt_mark3_overlap_n": rank_7s_wt_mark3_overlap_n(a1, a2, h, t, an),
                        "wt_marks": {"honmei": h, "taikou": t, "ana": an},
                    })
        with open(dst, "wb") as f:
            pickle.dump(rows, f)
        print(f"[done] {tag} {len(rows)}", flush=True)


def _legs(c: dict, k: int, drop_ana: bool) -> list[int]:
    r = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    if drop_ana:
        ana = c["wt_marks"].get("ana")
        r = [x for x in r if x != ana]
    return [x for x in r[:k] if c["trio_legs"].get(x) is not None]


def _score(cands: list[dict], k: int, drop: bool, days: int) -> dict:
    bet = ret = hit = gami = 0
    pays: list[int] = []
    for c in cands:
        lg = _legs(c, k, drop)
        if not lg:
            continue
        stake = len(lg) * STAKE
        bet += stake
        top3 = set(c["actual_top3"])
        rest = top3 - {c["axis1"], c["axis2"]}
        if len(top3 & {c["axis1"], c["axis2"]}) == 2 and len(rest) == 1 and rest.pop() in lg:
            hit += 1
            ret += c["trio_pay"]
            pays.append(c["trio_pay"])
            if c["trio_pay"] < stake:
                gami += 1
    n = len(cands)
    return {"n": n, "per": n / days if days else 0,
            "hit": 100 * hit / n if n else 0, "roi": 100 * ret / bet if bet else 0,
            "gami": 100 * gami / hit if hit else 0,
            "med": statistics.median(pays) / 100 if pays else 0,
            "n20": sum(1 for p in pays if p >= 2000),
            "mx": max(pays) / 100 if pays else 0, "pays": pays, "bet": bet}


def analyze(cache: Path) -> None:
    allc: list[dict] = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            allc += pickle.load(f)
    days = len({c["race_date"] for c in allc})
    for c in allc:
        tw = max(c["win_probs"], key=lambda f: c["win_probs"][f])
        c["dis"] = tw != c["wt_marks"].get("honmei")

    print(f"全期間 {len(allc)}件 / {days}日 = {len(allc)/days:.2f}件/日")
    cnt: dict = defaultdict(int)
    for c in allc:
        cnt[c["wt_overlap_n"]] += 1
    for k in (0, 1, 2, None):
        print(f"  overlap={k}: {cnt[k]:5d} ({100*cnt[k]/len(allc):5.1f}%)")

    o2d = [c for c in allc if c["wt_overlap_n"] == 2 and c["dis"]]
    o01 = [c for c in allc if c["wt_overlap_n"] in (0, 1)]

    def msd(cands, k, drop):
        bm = defaultdict(list)
        for c in cands:
            bm[c["race_date"][:7]].append(c)
        rois = [_score(v, k, drop, 1)["roi"] for v in bm.values() if _score(v, k, drop, 1)["n"]]
        return (statistics.pstdev(rois) if len(rois) > 1 else 0,
                sum(1 for r in rois if r < 60), len(rois))

    print(f"\n{'条件':34} {'n':>5} {'件/日':>6} {'的中':>6} {'ROI':>7} {'ガミ':>6} "
          f"{'中央値':>7} {'20倍+':>5} {'月σ':>6} {'<60%':>6}")
    for lbl, cs, k, d in ([("参考:overlap01 総流し7点", o01, 7, False)]
                          + [(f"9B案 △除外 K={k}", o2d, k, True) for k in (7, 5, 4, 3, 2)]
                          + [(f"（参考）△除外なし K={k}", o2d, k, False) for k in (5, 3)]):
        r = _score(cs, k, d, days)
        s, lo, nm = msd(cs, k, d)
        print(f"{lbl:34} {r['n']:5d} {r['per']:6.2f} {r['hit']:5.1f}% {r['roi']:6.1f}% "
              f"{r['gami']:5.1f}% {r['med']:6.1f}倍 {r['n20']:5d} {s:6.1f} {lo:3d}/{nm}")

    print("\n=== 大穴依存（9B案 K=3・△除外）===")
    r = _score(o2d, 3, True, days)
    ps = sorted(r["pays"], reverse=True)
    for d in (0, 1, 3, 5, 10):
        print(f"  最高配当{d:2d}本除外: ROI {100*(sum(ps)-sum(ps[:d]))/r['bet']:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", metavar="DIR")
    ap.add_argument("--analyze", metavar="DIR")
    args = ap.parse_args()
    if args.build_cache:
        build_cache(Path(args.build_cache))
    if args.analyze:
        analyze(Path(args.analyze))
    if not args.build_cache and not args.analyze:
        ap.error("--build-cache か --analyze のいずれかを指定してください")


if __name__ == "__main__":
    main()
