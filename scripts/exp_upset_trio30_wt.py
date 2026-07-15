"""波乱レース（三連複30倍以上）の発走前傾向分析（実精算方式・2026-07-16）。

母集団: 7車ちょうど ∧ 欠車・落車・失格なし（盤面7車 ∧ 完走7車）のクリーンレース。
目的変数: 的中三連複（実際の1-2-3着の組合せ）の最終オッズ ≥ 30倍 = 波乱。

発走前に計算可能な特徴量を網羅的に比較する:
  [指数系]   s1(1位占有率) / sep12(1-2位占有率差) / pred_sd / エントロピー / top3_sum / gap23
  [ライン系] ライン数 / 最大ライン占有率(メンバー占有率和) / ライン占有率差(1位-2位ライン) /
             軸2車同一ライン / 全単騎
  [得点系]   競走得点SD / 上位2平均-残り平均(gap2r)
  [市場系]   盤面min三連複オッズ / 10倍未満組合せ数 / 市場的中集中度(Σ1/odds由来の1位車シェア)

出力: 各特徴の四分位別 波乱率 + AUC（2窓で方向一致するものが本物）。

使い方（ローカルSQLite・KEIRIN_DB_URL は設定しないこと）:
  PYTHONPATH=. .venv/bin/python scripts/exp_upset_trio30_wt.py \
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

UPSET_ODDS = 30.0


def collect_clean(model, date_from, date_to):
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

    rows = []
    for rk, g in df.groupby("race_key"):
        board = set()
        for combo in trio_bd.get(rk, {}):
            board |= set(combo)
        if len(board) != 7 or len(g) != 7:
            continue  # 欠車あり or データ欠損
        fo = g["finish_order"]
        if fo.isna().any() or (fo < 1).any():
            continue  # 落車・失格・棄権あり → 除外（クリーン完走レースのみ）
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        p = g["pred_prob"].to_numpy()
        total = p.sum()
        if total <= 0:
            continue
        s = p / total

        frames = g["frame_no"].astype(int).tolist()
        fin = g.sort_values("finish_order")["frame_no"].astype(int).tolist()
        top3 = frozenset(fin[:3])
        trio = trio_bd.get(rk, {})
        win_odds = trio.get(top3)
        if not win_odds:
            continue

        # ライン系: line_group ごとの占有率和
        lg = g["line_group"].to_numpy()
        line_shares = {}
        for i in range(7):
            key = int(lg[i]) if lg[i] == lg[i] else -(i + 1)  # 欠損は単独扱い
            line_shares[key] = line_shares.get(key, 0.0) + float(s[i])
        ls_sorted = sorted(line_shares.values(), reverse=True)
        n_lines = len(line_shares)
        axis_same_line = (lg[0] == lg[0] and lg[1] == lg[1] and int(lg[0]) == int(lg[1]))

        # 得点系
        rp = g["race_point"].to_numpy(dtype=float)
        rp_valid = rp[~np.isnan(rp)]
        if len(rp_valid) >= 5:
            rp_sd = float(np.std(rp_valid))
            rv = np.sort(rp_valid)[::-1]
            gap2r = float(rv[:2].mean() - rv[2:].mean())
        else:
            rp_sd = gap2r = np.nan

        # 市場系（最終盤面）
        odds_all = np.array(list(trio.values()), dtype=float)
        qi = {}
        for combo, ov in trio.items():
            if ov and 0 < ov < 9000:
                for fno in combo:
                    qi[fno] = qi.get(fno, 0.0) + 1.0 / ov
        qsum = sum(qi.values()) or 1.0
        mkt_s1 = max(qi.values()) / qsum if qi else np.nan

        ent = float(-(s * np.log(np.clip(s, 1e-9, None))).sum())
        rows.append({
            "rk": rk,
            "upset": 1 if win_odds >= UPSET_ODDS else 0,
            "win_odds": float(win_odds),
            # 指数系
            "s1": float(s[0]), "sep12": float(s[0] - s[1]),
            "pred_sd": float(np.std(p)), "entropy": ent,
            "top3_sum": float(p[:3].sum()), "gap23_pt": float((p[1] - p[2]) * 100),
            # ライン系
            "n_lines": float(n_lines),
            "max_line_share": float(ls_sorted[0]),
            "line_share_gap": float(ls_sorted[0] - ls_sorted[1]) if n_lines >= 2 else np.nan,
            "axis_same_line": 1.0 if axis_same_line else 0.0,
            # 得点系
            "score_sd": rp_sd, "score_gap2r": gap2r,
            # 市場系
            "min_trio_odds": float(odds_all.min()),
            "n_combos_lt10": float((odds_all < 10).sum()),
            "mkt_s1": float(mkt_s1),
        })
    return rows


def auc(vals, ys):
    """rank AUC（波乱=1を高値側とみなす向き）。"""
    v = np.asarray(vals, dtype=float)
    y = np.asarray(ys, dtype=int)
    m = ~np.isnan(v)
    v, y = v[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    order = np.argsort(v)
    ranks = np.empty(len(v)); ranks[order] = np.arange(1, len(v) + 1)
    pos = y == 1
    return (ranks[pos].mean() - (pos.sum() + 1) / 2) / (~pos).sum()


FEATURES = [
    "s1", "sep12", "pred_sd", "entropy", "top3_sum", "gap23_pt",
    "n_lines", "max_line_share", "line_share_gap", "axis_same_line",
    "score_sd", "score_gap2r",
    "min_trio_odds", "n_combos_lt10", "mkt_s1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（波乱=的中三連複{UPSET_ODDS:.0f}倍以上・クリーン7車完走のみ）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = collect_clean(model, f, t)
        y = np.array([r["upset"] for r in rows])
        base = y.mean() if len(y) else 0
        print(f"\n===== {f} 〜 {t}（クリーン7車 {len(rows)}R / 波乱率 {base:.1%} / "
              f"中央値配当 {np.median([r['win_odds'] for r in rows]):.1f}倍） =====")
        print(f"  {'特徴':<16} {'AUC':>6}  Q1(低)   Q2     Q3     Q4(高)   （四分位別の波乱率）")
        for feat in FEATURES:
            v = np.array([r[feat] for r in rows], dtype=float)
            m = ~np.isnan(v)
            vv, yy = v[m], y[m]
            if len(vv) < 100 or len(np.unique(vv)) < 3:
                continue
            a = auc(vv, yy)
            qs = np.percentile(vv, [25, 50, 75])
            rates = []
            for lo, hi in [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]:
                sel = (vv > lo) & (vv <= hi) if hi != np.inf else (vv > lo)
                rates.append(yy[sel].mean() if sel.sum() else np.nan)
            print(f"  {feat:<16} {a:>6.3f}  " + "  ".join(f"{r:>5.1%}" for r in rates))


if __name__ == "__main__":
    main()
