"""穴選手＋2車目軸の選定検証（三連複2車軸全流し・実精算・2026-07-16）。

前提（exp_dark_rider_roi_wt.py）: 波乱見込みレースの穴選手
（市場4-7位∧モデル3位内∧先頭/番手）は3着内43-46%。モデル1位を2車目軸に
固定した三連複流しは ROI 83-87% で不成立 → 2車目軸の選び方を網羅比較する。

2車目軸の候補:
  m1     モデル指数1位            m2     モデル指数2位
  mkt1   市場評価1位              rp1    競走得点1位
  mate   穴と同ラインの相方（穴が番手→先頭 / 先頭→番手。単騎は対象外）
  dark2  もう1人の穴選手（2人以上該当時のみ）

出力: 軸2車成立率（両者3着内）・三連複2車軸全流し（5点）ROI・CI。

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_dark_axis2_wt.py \
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

        frames = g["frame_no"].astype(int).tolist()
        rp = g["race_point"].to_numpy(dtype=float)
        rp1 = int(g["frame_no"].iloc[int(np.nanargmax(rp))]) if not np.isnan(rp).all() else None

        # ライン構成: group → [(line_pos, fno)]
        lines = {}
        for i, row in g.iterrows():
            lgv = row["line_group"]
            if lgv != lgv:
                continue
            lines.setdefault(int(lgv), []).append(
                (int(row["line_pos"]) if row["line_pos"] == row["line_pos"] else 9,
                 int(row["frame_no"])))

        darks = []
        for i, row in g.iterrows():
            fno = int(row["frame_no"])
            lpos = int(row["line_pos"]) if row["line_pos"] == row["line_pos"] else 0
            lsize = int(row["line_size"]) if row["line_size"] == row["line_size"] else 1
            if (mkt_rank.get(fno, 7) >= 4 and (i + 1) <= 3
                    and (lsize == 1 or lpos in (1, 2))):
                lgv = row["line_group"]
                mate = None
                if lgv == lgv and lsize >= 2:
                    members = sorted(lines.get(int(lgv), []))
                    want = 1 if lpos == 2 else 2
                    for mp, mf in members:
                        if mp == want and mf != fno:
                            mate = mf
                            break
                darks.append({"fno": fno, "mate": mate})
        races.append({
            "rk": rk, "entropy": ent, "mto": mto,
            "m1": frames[0], "m2": frames[1],
            "mkt1": min(mkt_rank, key=mkt_rank.get),
            "rp1": rp1,
            "darks": darks, "board": board, "top3": top3, "trio": trio,
        })
    return races


def axis2_of(r, d, kind):
    if kind == "m1":
        return r["m1"]
    if kind == "m2":
        return r["m2"]
    if kind == "mkt1":
        return r["mkt1"]
    if kind == "rp1":
        return r["rp1"]
    if kind == "mate":
        return d["mate"]
    if kind == "dark2":
        others = [x["fno"] for x in r["darks"] if x["fno"] != d["fno"]]
        return others[0] if others else None
    raise ValueError(kind)


AXES = [("モデル1位", "m1"), ("モデル2位", "m2"), ("市場1位", "mkt1"),
        ("得点1位", "rp1"), ("同ライン相方", "mate"), ("穴2車併存", "dark2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（穴＋2車目軸の三連複2車軸全流し・実精算）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect(model, f, t)
        ents = np.array([r["entropy"] for r in races])
        mtos = np.array([r["mto"] for r in races])
        eq3, mq3 = np.percentile(ents, 75), np.percentile(mtos, 75)
        sel = [r for r in races if r["entropy"] >= eq3 and r["mto"] >= mq3 and r["darks"]]
        n_dark = sum(len(r["darks"]) for r in sel)
        dhit = sum(1 for r in sel for d in r["darks"] if d["fno"] in r["top3"])
        days = len({r["rk"][:8] for r in races}) or 1
        print(f"\n===== {f} 〜 {t}（波乱見込み∧穴あり {len(sel)}R / 穴{n_dark}人 / "
              f"穴の3着内率 {dhit/n_dark:.1%} / {len(sel)/days:.1f}R/日） =====")
        print(f"  {'2車目軸':<12} {'R数':>4} {'軸成立率':>7} {'点数':>5} {'的中':>4} {'投資':>8} {'払戻':>8} {'ROI':>7}  CI95%")
        for label, kind in AXES:
            per_race = []
            n_pair = n_pairhit = nb = nh = b = pp = 0
            for r in sel:
                rb = rp_ = 0
                for d in r["darks"]:
                    a2 = axis2_of(r, d, kind)
                    if a2 is None or a2 == d["fno"]:
                        continue
                    n_pair += 1
                    if {d["fno"], a2} <= r["top3"]:
                        n_pairhit += 1
                    for o in r["board"]:
                        if o in (d["fno"], a2):
                            continue
                        c3 = frozenset({d["fno"], a2, o})
                        ov = r["trio"].get(c3)
                        if not ov:
                            continue
                        nb += 1
                        rb += STAKE
                        if c3 == r["top3"]:
                            pay = (int(ov * 100) // 10 * 10)
                            nh += 1
                            rp_ += pay
                if rb:
                    per_race.append((rb, rp_))
                    b += rb
                    pp += rp_
            if not b:
                print(f"  {label:<12} {'—':>4}")
                continue
            arr_b = np.array([x[0] for x in per_race]); arr_p = np.array([x[1] for x in per_race])
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(arr_b), size=(2000, len(arr_b)))
            rois = arr_p[idx].sum(axis=1) / arr_b[idx].sum(axis=1)
            lo, hi = np.percentile(rois, [2.5, 97.5])
            print(f"  {label:<12} {len(per_race):>4} {n_pairhit/n_pair:>7.1%} {nb:>5} {nh:>4} "
                  f"{b:>8,} {pp:>8,} {pp/b:>6.1%}  [{lo:.0%},{hi:.0%}]")


if __name__ == "__main__":
    main()
