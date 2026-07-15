"""波乱見込みレース×人気薄（穴選手）のROI検証（実精算方式・2026-07-16）。

前段の傾向分析（exp_upset_dark_riders_wt.py・2窓一貫）:
  波乱見込みレースの市場4-7位のうち「モデル3位内 ∧ ライン先頭/番手」は3着内43-46%。
本ハーネスはこの穴選手を絡めた各券種のROIを検証する。

母集団（発走前判定のみ）:
  7車 ∧ 盤面7車（欠車なし） ∧ 指数エントロピー≥窓内Q3 ∧ 盤面min三連複オッズ≥窓内Q3
穴選手（dark）:
  市場評価4-7位（盤面Σ1/trioオッズ順位） ∧ モデル指数3位内 ∧ ライン内位置 先頭or番手（単騎含む）
精算: 実精算（落車・失格は外れ計上）。ワイドは収録オッズ=レンジ下限（保守的）。

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_dark_rider_roi_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.database import get_connection

STAKE = 100


def load_pair_odds(race_keys):
    """quinella（二車複）/ quinellaPlace（ワイド・レンジ下限）を返す。"""
    q = defaultdict(dict)
    wide = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            rows = c.execute(
                "SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                "WHERE bet_type IN ('quinella','quinellaPlace') AND race_key IN (%s)"
                % ",".join("?" * len(chunk)), chunk).fetchall()
            for rk, bt, comb, od in rows:
                if od is None or not (0 < float(od) < 9000):
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                except ValueError:
                    continue
                if len(parts) != 2:
                    continue
                (q if bt == "quinella" else wide)[rk][parts] = float(od)
    return q, wide


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) == 7})].copy()
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    trio_bd, _ = E.load_boards(df["race_key"].unique().tolist())
    q_bd, w_bd = load_pair_odds(df["race_key"].unique().tolist())

    races = []
    for rk, g in df.groupby("race_key"):
        trio = trio_bd.get(rk, {})
        board = set()
        for combo in trio:
            board |= set(combo)
        if len(board) != 7 or len(g) != 7:
            continue  # 欠車あり（発走前に判明）は対象外
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        p = g["pred_prob"].to_numpy()
        total = p.sum()
        if total <= 0:
            continue
        s = p / total
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        odds_all = np.array([v for v in trio.values() if 0 < v < 9000])
        if not len(odds_all):
            continue
        mto = float(odds_all.min())

        qi = {}
        for combo, ov in trio.items():
            if 0 < ov < 9000:
                for fno in combo:
                    qi[fno] = qi.get(fno, 0.0) + 1.0 / ov
        mkt_rank = {fno: r + 1 for r, (fno, _) in
                    enumerate(sorted(qi.items(), key=lambda x: -x[1]))}

        # 結果（実精算: 3完走未満=結果なしはスキップ・落車失格は着外=外れ）
        fo = g["finish_order"]
        fin = g[fo.notna() & (fo >= 1)].sort_values("finish_order")
        fins = fin["frame_no"].astype(int).tolist()
        if len(fins) < 3:
            continue
        top3 = frozenset(fins[:3])
        top2 = frozenset(fins[:2])

        frames = g["frame_no"].astype(int).tolist()
        darks = []
        for i, row in g.iterrows():
            fno = int(row["frame_no"])
            lpos = int(row["line_pos"]) if row["line_pos"] == row["line_pos"] else 0
            lsize = int(row["line_size"]) if row["line_size"] == row["line_size"] else 1
            if (mkt_rank.get(fno, 7) >= 4 and (i + 1) <= 3
                    and (lsize == 1 or lpos in (1, 2))):
                darks.append(fno)
        races.append({
            "rk": rk, "entropy": ent, "mto": mto,
            "m1": frames[0], "m2": frames[1],
            "mkt_top2": [fno for fno, r in sorted(mkt_rank.items(), key=lambda x: x[1])[:2]],
            "darks": darks, "board": board,
            "top3": top3, "top2": top2,
            "trio": trio, "quinella": q_bd.get(rk, {}), "wide": w_bd.get(rk, {}),
        })
    return races


def _pay(odds):
    return (int(odds * 100) // 10 * 10) * (STAKE // 100)


def bets_for(r, strategy):
    """(bet_key, odds, hit) のリスト。オッズ未収録目は購入不可としてスキップ。"""
    out = []
    for d in r["darks"]:
        if strategy == "wide_all":       # ワイド: dark×全車（6点）
            pairs = [frozenset({d, o}) for o in r["board"] if o != d]
        elif strategy == "wide_m12":     # ワイド: dark×モデル1,2位
            pairs = [frozenset({d, m}) for m in (r["m1"], r["m2"]) if m != d]
        elif strategy == "wide_mkt12":   # ワイド: dark×市場1,2位
            pairs = [frozenset({d, m}) for m in r["mkt_top2"] if m != d]
        elif strategy == "quinella_m12":  # 二車複: dark×モデル1,2位
            pairs = [frozenset({d, m}) for m in (r["m1"], r["m2"]) if m != d]
        elif strategy == "trio_dm1":     # 三連複: dark+モデル1位の2車軸→全流し
            if d == r["m1"]:
                continue
            for o in r["board"]:
                if o in (d, r["m1"]):
                    continue
                c3 = frozenset({d, r["m1"], o})
                ov = r["trio"].get(c3)
                if ov:
                    out.append((c3, ov, c3 == r["top3"]))
            continue
        else:
            raise ValueError(strategy)
        book = r["wide"] if strategy.startswith("wide") else r["quinella"]
        tgt = r["top3"] if strategy.startswith("wide") else r["top2"]
        for pr in pairs:
            ov = book.get(pr)
            if ov:
                out.append((pr, ov, pr <= tgt))
    return out


STRATEGIES = [
    ("ワイド dark×全車",      "wide_all"),
    ("ワイド dark×モデル1,2位", "wide_m12"),
    ("ワイド dark×市場1,2位",  "wide_mkt12"),
    ("二車複 dark×モデル1,2位", "quinella_m12"),
    ("三連複 dark+モデル1位軸流し", "trio_dm1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（波乱見込み×穴選手 ROI検証・実精算・ワイドは下限オッズ）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect(model, f, t)
        ents = np.array([r["entropy"] for r in races])
        mtos = np.array([r["mto"] for r in races])
        eq3, mq3 = np.percentile(ents, 75), np.percentile(mtos, 75)
        sel = [r for r in races if r["entropy"] >= eq3 and r["mto"] >= mq3 and r["darks"]]
        days = len({r["rk"][:8] for r in races}) or 1
        print(f"\n===== {f} 〜 {t}（7車{len(races)}R → 波乱見込み∧穴選手あり {len(sel)}R / "
              f"{len(sel)/days:.1f}R/日） =====")
        print(f"  {'戦略':<22} {'R数':>4} {'点数':>5} {'的中':>4} {'的中率':>6} {'投資':>8} {'払戻':>8} {'ROI':>7}  CI95%")
        for label, st in STRATEGIES:
            per_race = []
            nb = nh = b = pp = 0
            for r in sel:
                bets = bets_for(r, st)
                if not bets:
                    continue
                rb = len(bets) * STAKE
                rp = sum(_pay(ov) for _, ov, hit in bets if hit)
                nb += len(bets)
                nh += sum(1 for *_, hit in bets if hit)
                b += rb
                pp += rp
                per_race.append((rb, rp))
            if not b:
                print(f"  {label:<22} {'—':>4}")
                continue
            arr_b = np.array([x[0] for x in per_race]); arr_p = np.array([x[1] for x in per_race])
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(arr_b), size=(2000, len(arr_b)))
            rois = arr_p[idx].sum(axis=1) / arr_b[idx].sum(axis=1)
            lo, hi = np.percentile(rois, [2.5, 97.5])
            print(f"  {label:<22} {len(per_race):>4} {nb:>5} {nh:>4} {nh/nb:>6.1%} "
                  f"{b:>8,} {pp:>8,} {pp/b:>6.1%}  [{lo:.0%},{hi:.0%}]")


if __name__ == "__main__":
    main()
