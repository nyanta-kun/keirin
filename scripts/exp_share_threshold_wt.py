"""占有率（シェア）ベースの軸選定条件の検証（実精算方式・2026-07-16）。

ユーザー仮説: gap（確率の絶対差）ではなく「全出走車に対する1位の占有率が高く、
かつ2位占有率と一定以上離れている」レースは入着率が上がるのではないか。

定義（レースごと・盤面掲載車のみ）:
  share_i = pred_prob_i / Σ pred_prob（全掲載車）
  s1 = 1位の占有率 / s2 = 2位の占有率 / sep = s1 - s2（占有率ポイント差）
  ※ pred_prob は P(3着内) のためレース合計≈300%相当 → s1 の理論上限 ≈ 33%

計測（グリッド全数）:
  - axis_hit: 指数1位・2位がともに3着内（=SS的中条件）
  - p1_top3 : 指数1位が3着内（単独入着率）
  - ROI     : 現行オッズゲート（min全目≥7・非選抜）適用後の三連複全目100円/点

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_share_threshold_wt.py \
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
from src.strategy_wt import ss_policy


def collect_shares(model, date_from, date_to):
    """レースごとに占有率・結果・オッズ・種別を返す（実精算方式・盤面ランキング）。"""
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rt_map = dict(c.execute(
            "SELECT race_key, race_type FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) == 7})].copy()
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    trio_bd, _ = E.load_boards(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        board = set()
        for combo in trio_bd.get(rk, {}):
            board |= set(combo)
        if not board:
            continue
        g = g[g["frame_no"].astype(int).isin(board)]
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 5:
            continue
        p = g["pred_prob"].to_numpy()
        total = p.sum()
        if total <= 0:
            continue
        s = p / total
        fin = {}
        for _, row in g.iterrows():
            fo = row["finish_order"]
            if fo != fo or fo is None:
                continue
            fo = int(fo)
            if fo in (1, 2, 3):
                fin[fo] = int(row["frame_no"])
        if len(fin) < 3:
            continue
        frames = g["frame_no"].astype(int).tolist()
        top3 = frozenset(fin.values())
        rows.append({
            "rk": rk,
            "s1": float(s[0]), "s2": float(s[1]),
            "sep": float(s[0] - s[1]),
            "p1": frames[0], "p2": frames[1],
            "frames": frames, "top3": top3,
            "trio": trio_bd.get(rk, {}),
            "race_type": rt_map.get(rk),
        })
    return rows


def eval_cell(rows, s1_min, sep_min):
    """セル該当レースの (n, axis_hit, p1_top3, gated_n, gated_hit, bet, pay)。"""
    n = ah = p1h = gn = gh = b = pp = 0
    for r in rows:
        if r["s1"] < s1_min or r["sep"] < sep_min:
            continue
        n += 1
        axis_hit = {r["p1"], r["p2"]} <= r["top3"]
        ah += 1 if axis_hit else 0
        p1h += 1 if r["p1"] in r["top3"] else 0
        # 現行オッズゲート適用後のROI（三連複全目100円/点）
        if ss_policy(r["race_type"])[0]:
            continue
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < E.SS_GAMI:
            continue
        pay = 0
        for t, o in legs.items():
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = int(o * 100) // 10 * 10
                break
        gn += 1
        gh += 1 if pay > 0 else 0
        b += len(legs) * 100
        pp += pay
    return n, ah, p1h, gn, gh, b, pp


S1_GRID = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
SEP_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（占有率グリッド検証・実精算）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = collect_shares(model, f, t)
        days = len({r["rk"][:8] for r in rows}) or 1
        s1s = np.array([r["s1"] for r in rows])
        seps = np.array([r["sep"] for r in rows])
        print(f"\n===== {f} 〜 {t}（7車 {len(rows)}R / {days}日） =====")
        print(f"  s1分布: p50={np.percentile(s1s,50):.1%} p75={np.percentile(s1s,75):.1%} "
              f"p90={np.percentile(s1s,90):.1%} p99={np.percentile(s1s,99):.1%} max={s1s.max():.1%}")
        print(f"  sep分布: p50={np.percentile(seps,50):.1%} p90={np.percentile(seps,90):.1%} max={seps.max():.1%}")

        print(f"\n  ◆ 軸2車3着内率（=SS的中条件） n / axis_hit% / p1単独3着内%")
        hdr = "  s1\\sep " + "".join(f"{s:>16.0%}" for s in SEP_GRID)
        print(hdr)
        for s1m in S1_GRID:
            cells = []
            for sm in SEP_GRID:
                n, ah, p1h, *_ = eval_cell(rows, s1m, sm)
                cells.append(f"{n:>5}/{ah/n:>5.0%}/{p1h/n:>4.0%}" if n else f"{'—':>16}")
            print(f"  {s1m:>5.0%}  " + "".join(f"{c:>16}" for c in cells))

        print(f"\n  ◆ 現行オッズゲート（min全目≥7・非選抜）適用後の三連複ROI  n / hit% / ROI")
        print(hdr)
        for s1m in S1_GRID:
            cells = []
            for sm in SEP_GRID:
                _, _, _, gn, gh, b, pp = eval_cell(rows, s1m, sm)
                cells.append(f"{gn:>4}/{gh/gn:>4.0%}/{pp/b:>5.0%}" if gn and b else f"{'—':>16}")
            print(f"  {s1m:>5.0%}  " + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
