"""穴選手と「連携して」3着内に来る相方の条件分析（ペア単位・実精算・2026-07-16）。

穴 = 波乱見込みレースの 市場4-7位 ∧ モデル3位内 ∧ ライン先頭/番手（or 単騎）。
穴と他の全6車のペアについて、関係性特徴ごとに
  軸成立率 = P(穴と相方がともに3着内) と 三連複2車軸全流し(5点)ROI を計測する。

特徴（ペア関係）:
  T1 相方の位置関係（同ライン先頭/番手/3番手+・別ライン先頭/番手/3番手+・相方単騎）
  T2 穴の位置 × 相方の関係
  T3 相方の脚質 × 同/別ライン
  T4 穴の脚質 × 相方の脚質
  T5 相方の得点順位帯 × 同/別ライン
  T6 相方のモデル順位帯 × 同/別ライン

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_dark_pair_features_wt.py \
      --model lgbm_wt_2026h1_eval --windows 2026-04-01:2026-06-30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.database import get_connection

STAKE = 100


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

    races = []
    for rk, g in df.groupby("race_key"):
        trio = trio_bd.get(rk, {})
        board = set()
        for combo in trio:
            board |= set(combo)
        if len(board) != 7 or len(g) != 7:
            continue
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

        fo = g["finish_order"]
        fin = g[fo.notna() & (fo >= 1)].sort_values("finish_order")
        fins = fin["frame_no"].astype(int).tolist()
        if len(fins) < 3:
            continue
        top3 = frozenset(fins[:3])

        rp = g["race_point"].to_numpy(dtype=float)
        rp_order = (-np.nan_to_num(rp, nan=-1e9)).argsort()
        rp_rank = np.empty(7, dtype=int)
        rp_rank[rp_order] = np.arange(1, 8)

        riders = {}
        for i, row in g.iterrows():
            fno = int(row["frame_no"])
            riders[fno] = {
                "model_rank": i + 1,
                "mkt_rank": mkt_rank.get(fno, 7),
                "rp_rank": int(rp_rank[i]),
                "lg": int(row["line_group"]) if row["line_group"] == row["line_group"] else None,
                "lpos": int(row["line_pos"]) if row["line_pos"] == row["line_pos"] else 0,
                "lsize": int(row["line_size"]) if row["line_size"] == row["line_size"] else 1,
                "style": row["style"] if isinstance(row["style"], str) else "?",
            }
        darks = [fno for fno, r in riders.items()
                 if r["mkt_rank"] >= 4 and r["model_rank"] <= 3
                 and (r["lsize"] == 1 or r["lpos"] in (1, 2))]
        races.append({"rk": rk, "entropy": ent, "mto": mto, "riders": riders,
                      "darks": darks, "board": board, "top3": top3, "trio": trio})
    return races


def pos_label(r):
    if r["lsize"] == 1:
        return "単騎"
    return {1: "先頭", 2: "番手"}.get(r["lpos"], "3番手+")


def pair_tables(sel):
    """ペア列挙 → {table_name: {bin: [n, pair_hit, bets, bet_amt, pay]}}"""
    tabs = {k: {} for k in ("T1関係", "T2穴位置×関係", "T3相方脚質×同別", "T4脚質組合せ",
                            "T5相方得点×同別", "T6相方モデル×同別")}

    def add(tab, key, hit, bet, pay, nbets):
        e = tabs[tab].setdefault(key, [0, 0, 0, 0, 0])
        e[0] += 1
        e[1] += 1 if hit else 0
        e[2] += nbets
        e[3] += bet
        e[4] += pay

    for r in sel:
        for d in r["darks"]:
            dr = r["riders"][d]
            for o in r["board"]:
                if o == d:
                    continue
                orr = r["riders"][o]
                same = (dr["lg"] is not None and orr["lg"] is not None
                        and dr["lg"] == orr["lg"])
                opos = pos_label(orr)
                rel = ("同L" + opos) if same else (
                    "相方単騎" if orr["lsize"] == 1 else "別L" + opos)
                hit = {d, o} <= r["top3"]
                # 三連複2車軸全流し
                bet = pay = nbets = 0
                for t in r["board"]:
                    if t in (d, o):
                        continue
                    c3 = frozenset({d, o, t})
                    ov = r["trio"].get(c3)
                    if not ov:
                        continue
                    nbets += 1
                    bet += STAKE
                    if c3 == r["top3"]:
                        pay += int(ov * 100) // 10 * 10

                add("T1関係", rel, hit, bet, pay, nbets)
                add("T2穴位置×関係", f"穴{pos_label(dr)}∧{rel}", hit, bet, pay, nbets)
                add("T3相方脚質×同別", ("同L" if same else "別L") + f"×{orr['style']}",
                    hit, bet, pay, nbets)
                add("T4脚質組合せ", f"穴{dr['style']}×相方{orr['style']}", hit, bet, pay, nbets)
                rp_b = "得点1-2" if orr["rp_rank"] <= 2 else ("得点3-4" if orr["rp_rank"] <= 4 else "得点5+")
                add("T5相方得点×同別", ("同L" if same else "別L") + f"×{rp_b}", hit, bet, pay, nbets)
                mr_b = "M1-2" if orr["model_rank"] <= 2 else ("M3-4" if orr["model_rank"] <= 4 else "M5+")
                add("T6相方モデル×同別", ("同L" if same else "別L") + f"×{mr_b}", hit, bet, pay, nbets)
    return tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（穴×相方ペアの連携条件・実精算）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect(model, f, t)
        ents = np.array([r["entropy"] for r in races])
        mtos = np.array([r["mto"] for r in races])
        eq3, mq3 = np.percentile(ents, 75), np.percentile(mtos, 75)
        sel = [r for r in races if r["entropy"] >= eq3 and r["mto"] >= mq3 and r["darks"]]
        print(f"\n===== {f} 〜 {t}（波乱見込み∧穴あり {len(sel)}R） =====")
        tabs = pair_tables(sel)
        for tab, bins in tabs.items():
            print(f"  ◆ {tab}   （n=ペア数 / 軸成立率 / 三連複流しROI）")
            for k in sorted(bins, key=lambda x: -bins[x][1] / max(bins[x][0], 1)):
                n, h, nb, b, pp = bins[k]
                if n < 60 or not b:
                    continue
                print(f"    {k:<20} n={n:>5}  成立={h/n:>5.1%}  ROI={pp/b:>6.1%}")


if __name__ == "__main__":
    main()
