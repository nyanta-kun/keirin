"""三連単 1着固定フォーメーション検証（2026-07-10）

フォーメーション: 1着=指数1位固定 / 2着=指数2,3位 / 3着=全通り
  → 点数 = 2 × (n_riders - 2)。的中条件 = 「1着=指数1位 ∧ 2着∈{指数2,3位}」

購入レース条件のスイープ:
  - gap12_min: 1位と2位以下の離れ（0.10〜0.25）
  - gap34_min: 2,3位と4位以下の離れ（2着候補の質）
  - gami_min:  購入全目の三連単最安オッズ下限（レース単位・0=なし）
  - riders:    7車のみ / 7+車

評価: 2025年（lgbm_wt_train_only・11-12月=真OOS）+ 2026-06（june_eval・真OOS）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

CAND_GAP12 = 0.07


def load_trifecta_board(race_keys):
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trifecta' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None and 0 < float(od) < 90000:
                    try:
                        a, b, cc = (int(x) for x in re.split(r"[-=→]", str(comb)))
                        board[rk][(a, b, cc)] = float(od)
                    except ValueError:
                        pass
    return board


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) >= 7})].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    board = load_trifecta_board(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 5:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < CAND_GAP12:
            continue
        gap34 = p[2] - p[3] if n >= 4 else 0.0
        frames = g["frame_no"].astype(int).tolist()
        r1, r2, r3 = frames[0], frames[1], frames[2]

        # 実着順 1-2-3着（着順が揃わないレースは除外）
        fin = {}
        for _, row in g.iterrows():
            fo = int(row["finish_order"])
            if fo in (1, 2, 3):
                fin[fo] = int(row["frame_no"])
        if len(fin) < 3:
            continue

        bd = board.get(rk, {})
        # 買い目 = (r1, s, t)  s∈{r2,r3}, t=その他全部
        combos = []
        for s in (r2, r3):
            for t in frames:
                if t in (r1, s):
                    continue
                key = (r1, s, t)
                if key in bd:
                    combos.append((key, bd[key]))
        if not combos:
            continue

        rows.append({
            "rk": rk, "n": n, "gap12": gap12, "gap34": gap34,
            "combos": combos,
            "actual": (fin[1], fin[2], fin[3]),
        })
    return rows


def evaluate(rows, gap12_min, gap34_min, gami_min, riders7_only):
    n = h = b = pp = 0
    for r in rows:
        if riders7_only and r["n"] != 7:
            continue
        if r["gap12"] < gap12_min or r["gap34"] < gap34_min:
            continue
        odds = [o for _, o in r["combos"]]
        if gami_min > 0 and min(odds) < gami_min:
            continue
        pay = 0
        for key, o in r["combos"]:
            if key == r["actual"]:
                pay = int(o * 100)
                break
        n += 1
        h += 1 if pay > 0 else 0
        b += len(r["combos"]) * 100
        pp += pay
    return n, h, b, pp


def sweep(rows, label, days):
    print(f"\n===== {label}（候補{len(rows)}R / {days}日） =====")
    print(f"{'gap12':>6} {'gap34':>6} {'gami':>5} {'車':>4} {'R数':>5} {'R/日':>5} "
          f"{'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>7}")
    out = []
    for g12 in (0.10, 0.15, 0.20, 0.25):
        for g34 in (0.0, 0.02, 0.04):
            for gami in (0, 5, 7, 10):
                for r7 in (False, True):
                    n, h, b, pp = evaluate(rows, g12, g34, gami, r7)
                    if n < 30:
                        continue
                    out.append((g12, g34, gami, r7, n, h, b, pp))
    for g12, g34, gami, r7, n, h, b, pp in sorted(out, key=lambda x: -(x[7] / x[6])):
        print(f"{g12:>6.2f} {g34:>6.2f} {gami:>5} {'7のみ' if r7 else '7+':>4} "
              f"{n:>5} {n/days:>5.1f} {h/n:>6.1%} {b:>9,} {pp:>9,} {pp/b:>6.1%}")


def main():
    print("2025年: lgbm_wt_train_only（11-12月=真OOS）", flush=True)
    model = load_model("lgbm_wt_train_only")
    rows25 = collect(model, "2025-01-01", "2025-12-31")
    sweep(rows25, "2025年", 365)

    print("\n2026-06: lgbm_wt_june_eval（真OOS）", flush=True)
    model2 = load_model("lgbm_wt_june_eval")
    rows26 = collect(model2, "2026-06-01", "2026-06-19")
    sweep(rows26, "2026-06", 19)


if __name__ == "__main__":
    main()
