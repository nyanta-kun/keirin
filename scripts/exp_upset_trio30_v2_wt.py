"""波乱レース（三連複30倍以上）の発走前傾向分析 v2（2026-07-26）。

exp_upset_trio30_wt.py（2026-07-16）の再検証＋拡張版。
オリジナル版は lgbm_wt_2026h1_eval（40特徴）で学習済みだが、現行 feature_wt.py は
48特徴に増えており shape mismatch で実行不能になっていたため、現行スキーマに
対応した lgbm_wt_eval / lgbm_wt_win_eval（test_from=2026-04-24 の正真OOSモデル）
に切り替えて再現し、下記の新規特徴を追加検証する:

  [新規: 単勝率/複勝率のばらつき] win_rate_sd / place_rate_sd
    （first_rate_norm / third_rate_norm の7車内SD。公式発表の1着率/3着内率）
  [新規: 単勝率/複勝率の集中度] win_rate_gap2r / place_rate_gap2r（上位2平均-残り平均）
  [新規: 自己ローリング実績のばらつき] win_3m_sd / top3_3m_sd（過去90日実績勝率/複勝率の7車内SD）
  [新規: 抜け出し/展開] n_senko（レース内の逃げ・先行人数。0人=展開不分明で波乱、という
    既存の n_senko 特徴を波乱=三連複30倍予測に転用）
  [新規: 平均との差] top_score_z（得点最上位車の得点zスコア=場の平均からの突出度。
    低い(突出していない)ほど拮抗＝波乱、という仮説）
  [既存参考] score_std は feature_wt.py の生成値をそのまま再利用（再計算しない）

母集団・的中定義・AUC定義は元スクリプトと同一（7車ちょうど・欠車/落車/失格なしの
クリーンレース、trio最終オッズ>=30倍=波乱）。

使い方:
  KEIRIN_DB_URL=... PYTHONPATH=. .venv/bin/python scripts/exp_upset_trio30_v2_wt.py \
      --model lgbm_wt_eval --win-model lgbm_wt_win_eval \
      --windows 2026-04-24:2026-06-10 2026-06-11:2026-07-25
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


def collect_clean(model, win_model, date_from, date_to):
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
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd, _ = E.load_boards(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        board = set()
        for combo in trio_bd.get(rk, {}):
            board |= set(combo)
        if len(board) != 7 or len(g) != 7:
            continue
        fo = g["finish_order"]
        if fo.isna().any() or (fo < 1).any():
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        p = g["pred_prob"].to_numpy()
        pw = g["pred_win"].to_numpy()
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

        # ライン系
        lg = g["line_group"].to_numpy()
        line_shares = {}
        for i in range(7):
            key = int(lg[i]) if lg[i] == lg[i] else -(i + 1)
            line_shares[key] = line_shares.get(key, 0.0) + float(s[i])
        ls_sorted = sorted(line_shares.values(), reverse=True)
        n_lines = len(line_shares)
        axis_same_line = (lg[0] == lg[0] and lg[1] == lg[1] and int(lg[0]) == int(lg[1]))

        # 得点系（既存カラムをそのまま利用）
        rp = g["race_point"].to_numpy(dtype=float)
        rp_valid = rp[~np.isnan(rp)]
        if len(rp_valid) >= 5:
            rv = np.sort(rp_valid)[::-1]
            gap2r = float(rv[:2].mean() - rv[2:].mean())
        else:
            gap2r = np.nan
        score_std = float(g["score_std"].iloc[0]) if "score_std" in g.columns else np.nan
        score_z = g["score_z"].to_numpy(dtype=float) if "score_z" in g.columns else np.full(7, np.nan)
        top_score_z = float(np.nanmax(score_z)) if np.isfinite(score_z).any() else np.nan

        # 新規: 単勝率/複勝率（公式first_rate/third_rate）のばらつき・集中度
        wrate = g["first_rate_norm"].to_numpy(dtype=float) if "first_rate_norm" in g.columns else np.full(7, np.nan)
        prate = g["third_rate_norm"].to_numpy(dtype=float) if "third_rate_norm" in g.columns else np.full(7, np.nan)
        win_rate_sd = float(np.nanstd(wrate)) if np.isfinite(wrate).any() else np.nan
        place_rate_sd = float(np.nanstd(prate)) if np.isfinite(prate).any() else np.nan

        def _gap2r(vals):
            v = vals[~np.isnan(vals)]
            if len(v) < 5:
                return np.nan
            vs = np.sort(v)[::-1]
            return float(vs[:2].mean() - vs[2:].mean())

        win_rate_gap2r = _gap2r(wrate)
        place_rate_gap2r = _gap2r(prate)

        # 新規: 自己ローリング実績（win_3m/top3_3m）のばらつき
        w3 = g["win_3m"].to_numpy(dtype=float) if "win_3m" in g.columns else np.full(7, np.nan)
        t3 = g["top3_3m"].to_numpy(dtype=float) if "top3_3m" in g.columns else np.full(7, np.nan)
        win_3m_sd = float(np.nanstd(w3)) if np.isfinite(w3).any() else np.nan
        top3_3m_sd = float(np.nanstd(t3)) if np.isfinite(t3).any() else np.nan

        # 新規: 展開（逃げ人数）
        n_senko = float(g["n_senko"].iloc[0]) if "n_senko" in g.columns else np.nan

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
            # 指数系（既存）
            "s1": float(s[0]), "sep12": float(s[0] - s[1]),
            "pred_sd": float(np.std(p)), "entropy": ent,
            "top3_sum": float(p[:3].sum()), "gap23_pt": float((p[1] - p[2]) * 100),
            # ライン系（既存）
            "n_lines": float(n_lines),
            "max_line_share": float(ls_sorted[0]),
            "line_share_gap": float(ls_sorted[0] - ls_sorted[1]) if n_lines >= 2 else np.nan,
            "axis_same_line": 1.0 if axis_same_line else 0.0,
            # 得点系（既存＋新規）
            "score_sd": score_std, "score_gap2r": gap2r,
            "top_score_z": top_score_z,
            # 市場系（既存）
            "min_trio_odds": float(odds_all.min()),
            "n_combos_lt10": float((odds_all < 10).sum()),
            "mkt_s1": float(mkt_s1),
            # === 新規 ===
            "win_rate_sd": win_rate_sd, "place_rate_sd": place_rate_sd,
            "win_rate_gap2r": win_rate_gap2r, "place_rate_gap2r": place_rate_gap2r,
            "win_3m_sd": win_3m_sd, "top3_3m_sd": top3_3m_sd,
            "n_senko": n_senko,
        })
    return rows


def auc(vals, ys):
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
    "score_sd", "score_gap2r", "top_score_z",
    "min_trio_odds", "n_combos_lt10", "mkt_s1",
    "win_rate_sd", "place_rate_sd", "win_rate_gap2r", "place_rate_gap2r",
    "win_3m_sd", "top3_3m_sd", "n_senko",
]

NEW_FEATURES = {
    "win_rate_sd", "place_rate_sd", "win_rate_gap2r", "place_rate_gap2r",
    "win_3m_sd", "top3_3m_sd", "n_senko", "top_score_z",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--win-model", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}/{args.win_model}（波乱=的中三連複{UPSET_ODDS:.0f}倍以上・クリーン7車完走のみ）", flush=True)
    model = load_model(args.model)
    win_model = load_model(args.win_model)

    all_windows = []
    for w in args.windows:
        f, t = w.split(":")
        rows = collect_clean(model, win_model, f, t)
        all_windows.append((f, t, rows))
        y = np.array([r["upset"] for r in rows])
        base = y.mean() if len(y) else 0
        print(f"\n===== {f} 〜 {t}（クリーン7車 {len(rows)}R / 波乱率 {base:.1%} / "
              f"中央値配当 {np.median([r['win_odds'] for r in rows]):.1f}倍） =====")
        print(f"  {'特徴':<18} {'AUC':>6}  Q1(低)   Q2     Q3     Q4(高)   （四分位別の波乱率）  {'[新規]' if False else ''}")
        for feat in FEATURES:
            v = np.array([r[feat] for r in rows], dtype=float)
            m = ~np.isnan(v)
            vv, yy = v[m], y[m]
            if len(vv) < 50 or len(np.unique(vv)) < 3:
                print(f"  {feat:<18} {'n/a':>6}  (有効データ不足 n={len(vv)})" + ("  ← 新規" if feat in NEW_FEATURES else ""))
                continue
            a = auc(vv, yy)
            qs = np.percentile(vv, [25, 50, 75])
            rates = []
            for lo, hi in [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]:
                sel = (vv > lo) & (vv <= hi) if hi != np.inf else (vv > lo)
                rates.append(yy[sel].mean() if sel.sum() else np.nan)
            mark = "  ← 新規" if feat in NEW_FEATURES else ""
            print(f"  {feat:<18} {a:>6.3f}  " + "  ".join(f"{r:>5.1%}" for r in rates) + mark)

    if len(all_windows) >= 2:
        print("\n===== 2窓一致チェック（AUC>0.55 かつ 両窓で同方向のみ「本物」候補） =====")
        for feat in FEATURES:
            aucs = []
            for f, t, rows in all_windows:
                y = np.array([r["upset"] for r in rows])
                v = np.array([r[feat] for r in rows], dtype=float)
                aucs.append(auc(v, y))
            aucs = np.array(aucs)
            if np.all(np.isnan(aucs)):
                continue
            consistent = np.all(aucs > 0.55) or np.all(aucs < 0.45)
            mark = "  ← 新規" if feat in NEW_FEATURES else ""
            flag = " ✅一致" if consistent else ""
            print(f"  {feat:<18} " + " / ".join(f"{a:.3f}" for a in aucs) + flag + mark)


if __name__ == "__main__":
    main()
