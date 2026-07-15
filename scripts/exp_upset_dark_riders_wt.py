"""波乱見込みレースで3着以内に来る人気薄選手の傾向分析（実精算方式・2026-07-16）。

母集団レース: クリーン7車完走（欠車・落車・失格なし）のうち波乱見込み
  = 指数エントロピー ≥ 窓内Q3 ∧ 盤面min三連複オッズ ≥ 窓内Q3
    （exp_upset_trio30_wt.py で2窓一貫・波乱率 18%→21-22% の混戦シグナル）

対象選手: 市場人気薄 = 最終盤面Σ(1/trioオッズ) 由来の市場評価順位 4〜7位
目的変数: 3着以内（入着）

特徴: ライン内位置 / 脚質 / 得点順位 / モデル指数順位（市場との乖離） /
      所属ラインの強さ（占有率順位）/ ライン人数 / 級班

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_upset_dark_riders_wt.py \
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


def collect_riders(model, date_from, date_to):
    """クリーン7車完走レースのレース単位混戦指標 + 選手単位特徴を返す。"""
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
        fo = g["finish_order"]
        if fo.isna().any() or (fo < 1).any():
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        p = g["pred_prob"].to_numpy()
        total = p.sum()
        if total <= 0:
            continue
        s = p / total
        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        odds_all = np.array([v for v in trio.values() if v and 0 < v < 9000])
        if not len(odds_all):
            continue
        mto = float(odds_all.min())

        # 市場評価順位（Σ1/odds 大 = 人気）
        qi = {}
        for combo, ov in trio.items():
            if ov and 0 < ov < 9000:
                for fno in combo:
                    qi[fno] = qi.get(fno, 0.0) + 1.0 / ov
        mkt_rank = {fno: r + 1 for r, (fno, _) in
                    enumerate(sorted(qi.items(), key=lambda x: -x[1]))}

        # ライン占有率順位
        lg = g["line_group"].to_numpy()
        line_share = {}
        for i in range(7):
            key = int(lg[i]) if lg[i] == lg[i] else -(i + 1)
            line_share[key] = line_share.get(key, 0.0) + float(s[i])
        line_rank = {k: r + 1 for r, (k, _) in
                     enumerate(sorted(line_share.items(), key=lambda x: -x[1]))}

        # 得点順位
        rp = g["race_point"].to_numpy(dtype=float)
        rp_order = (-np.nan_to_num(rp, nan=-1e9)).argsort()
        rp_rank = np.empty(7, dtype=int)
        rp_rank[rp_order] = np.arange(1, 8)

        riders = []
        for i, row in g.iterrows():
            fno = int(row["frame_no"])
            lkey = int(lg[i]) if lg[i] == lg[i] else -(i + 1)
            riders.append({
                "fno": fno,
                "top3": 1 if int(row["finish_order"]) <= 3 else 0,
                "mkt_rank": mkt_rank.get(fno, 7),
                "model_rank": i + 1,
                "rp_rank": int(rp_rank[i]),
                "line_pos": int(row["line_pos"]) if row["line_pos"] == row["line_pos"] else 0,
                "line_size": int(row["line_size"]) if row["line_size"] == row["line_size"] else 1,
                "line_rank": line_rank.get(lkey, 9),
                "style": row["style"] if isinstance(row["style"], str) else "?",
                "pclass": row["player_class"] if isinstance(row["player_class"], str) else "?",
            })
        races.append({"rk": rk, "entropy": ent, "mto": mto, "riders": riders})
    return races


def rate_table(riders, key_fn, label):
    """bin → (n, top3率)。"""
    bins = {}
    for r in riders:
        k = key_fn(r)
        if k is None:
            continue
        n, h = bins.get(k, (0, 0))
        bins[k] = (n + 1, h + r["top3"])
    print(f"  {label}")
    for k in sorted(bins):
        n, h = bins[k]
        if n >= 30:
            print(f"    {k:<16} n={n:>5}  3着内率 {h/n:>6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（波乱見込みレースの人気薄入着傾向）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        races = collect_riders(model, f, t)
        ents = np.array([r["entropy"] for r in races])
        mtos = np.array([r["mto"] for r in races])
        eq3, mq3 = np.percentile(ents, 75), np.percentile(mtos, 75)
        sel = [r for r in races if r["entropy"] >= eq3 and r["mto"] >= mq3]
        dark = [x for r in sel for x in r["riders"] if x["mkt_rank"] >= 4]
        base = np.mean([x["top3"] for x in dark]) if dark else 0
        print(f"\n===== {f} 〜 {t}（クリーン7車 {len(races)}R → 波乱見込み {len(sel)}R / "
              f"人気薄(市場4-7位) {len(dark)}人 / 基準3着内率 {base:.1%}） =====")

        rate_table(dark, lambda r: f"市場{r['mkt_rank']}位", "◆ 市場人気順位別（参考・基準の内訳）")
        rate_table(dark, lambda r: f"モデル{min(r['model_rank'],5)}位" + ("+" if r['model_rank']>=5 else ""),
                   "◆ モデル指数順位（市場4-7位のうちモデルが評価している選手）")
        rate_table(dark, lambda r: f"得点{min(r['rp_rank'],5)}位" + ("+" if r['rp_rank']>=5 else ""),
                   "◆ 競走得点順位")
        rate_table(dark, lambda r: {1: "先頭", 2: "番手", 3: "3番手+"}.get(min(r["line_pos"], 3), None)
                   if r["line_size"] >= 2 else "単騎",
                   "◆ ライン内位置")
        rate_table(dark, lambda r: f"ライン{min(r['line_rank'],3)}位" + ("+" if r['line_rank']>=3 else "")
                   if r["line_size"] >= 2 else "単騎",
                   "◆ 所属ラインの強さ（占有率順位）")
        rate_table(dark, lambda r: r["style"], "◆ 脚質")
        rate_table(dark, lambda r: r["pclass"], "◆ 級班")

        # 有望クロス: モデル評価×ライン内位置 / 番手×強ライン
        rate_table(dark, lambda r: ("モデル3位内∧" if r["model_rank"] <= 3 else "モデル4位下∧")
                   + ({1: "先頭", 2: "番手"}.get(r["line_pos"], "3番手+") if r["line_size"] >= 2 else "単騎"),
                   "◆ クロス: モデル評価 × ライン内位置")
        rate_table(dark, lambda r: ("強ライン(1位)" if r["line_rank"] == 1 else "他ライン")
                   + {1: "先頭", 2: "番手"}.get(r["line_pos"], "3番手+")
                   if r["line_size"] >= 2 else None,
                   "◆ クロス: ライン強さ × ライン内位置")


if __name__ == "__main__":
    main()
