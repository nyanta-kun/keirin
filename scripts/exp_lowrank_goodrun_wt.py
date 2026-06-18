"""≤6車: 指数下位（モデルランク4位以下）が好走する条件の網羅的残差分析（doc13 P4-3 の一般化）。

問い: モデルがレース内 pred_prob ランキングで4位以下に置いた選手が top3 に入る条件は何か。
doc13 で「指数下位の単騎逃げ」の過小評価（残差 TR+7.1%/TE+27.6%）が1件見つかっており、
その全次元走査（1次元のみ・組合せ爆発させない）。

設計（事前指定・docs/analysis/13 の判定様式を踏襲・doc18 セマンティクス準拠）:
  - ≤6車は出走表基準。ランキングは全エントリー（欠車含む・_apply_pred_prob_wt=G01修正版）。
    好走 = finish_order between(1,3)。0=欠車は着外。
  - model = lgbm_wt_eval（2023-07〜2026-02学習・2026-03以降OOS）。週次再学習 lgbm_wt は
    リークのため不使用。
  - TR = 2024-01〜2026-02（in-train＝残差は保守側に出る）/ TE = 2026-03〜（OOS）。
    判定は TR/TE 同方向のみ採用。
  - 残差2系統:
      residual_model  = 実top3率 − mean(isotonic較正済 pred_prob)   …モデルの過小評価
      residual_market = 実top3率 − mean(市場示唆P(top3))             …市場の過小評価
    市場示唆P(top3)は trio 全盤面から逆算: q_i = Σ_{iを含む組} 1/odds, P = 3q/Σq
    （exp_fav_gap_disagree_wt.py と同ロジック・最終オッズ＝確定直前の市場評価）。
  - isotonic 較正は TR の rank4+ 選手（市場カバレッジあり標本）で fit → 全期間 transform。
  - 判定3分類（セルごと・Wilson 95%CI）:
      A = モデル・市場とも過小評価（TRで実績CI下限が双方の平均予測を上回り・TE同方向）
      B = モデル過小評価だが市場は織込済（較正ギャップのみ・エッジではない）
      C = ノイズ（CI跨ぎ / TR-TE方向不一致 / n不足）
  - 多重比較: セル数と「期待される偽陽性数」を出力。
  - 本スクリプトは記述的残差分析であり ROI/購入の主張はしない。市場示唆は最終オッズ由来。

usage: .venv/bin/python3 scripts/exp_lowrank_goodrun_wt.py [--from 2023-07-01] [--to 2026-06-12]
       （--from はローリング特徴のウォームアップを含むデータロード開始日。
         TR/TE 境界は --train-from/--train-to/--test-from で指定）
"""
import sys
import math
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt
from src.database import get_connection
from exp_segment_first_wt import load_boards

# 級班の強さ（レース内相対用）。cls4=ガールズは別カテゴリ（レース内全員同班）→ None。
CLASS_STRENGTH = {"SS": 6, "S1": 5, "S2": 4, "cls1": 4, "A1": 3, "A2": 2, "A3": 1, "B": 0}

MIN_TR, MIN_TE = 100, 15      # 判定に使う最小標本（doc13 P4-3 が TR244/TE20 だった水準を参考）


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def market_top3_probs(board: dict, frames: set) -> dict | None:
    """trio 盤面から各車の市場P(top3)を逆算。全 frame をカバーできなければ None。"""
    q: dict[int, float] = {}
    n_combo = 0
    for combo, ov in board.items():
        if ov is None or ov <= 0 or ov >= 9000:
            continue
        n_combo += 1
        for f in combo:
            q[f] = q.get(f, 0.0) + 1.0 / ov
    if n_combo < 4 or not frames <= set(q):
        return None
    tot = sum(q.values())
    return {f: 3.0 * v / tot for f, v in q.items()}


def _rel(vals: list, idx: int, hi_lab: str, lo_lab: str) -> str | None:
    """レース内相対（最大/最小/中間）。NaN や全員同値は None。"""
    v = vals[idx]
    ok = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if v is None or (isinstance(v, float) and math.isnan(v)) or len(ok) < 2 or min(ok) == max(ok):
        return None
    if v == max(ok):
        return hi_lab
    if v == min(ok):
        return lo_lab
    return "mid"


def collect(load_from: str, to: str) -> list[dict]:
    model = load_model("lgbm_wt_eval")
    print(f"loading & building features ({load_from}〜{to}, 全エントリー)...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=load_from, max_date=to))
    df = _apply_pred_prob_wt(model, df)          # G01修正版: 全エントリーに pred_prob 付与

    # ≤6車は出走表基準（doc18 バイアス②）・rank4 が存在する n>=4 のみ・結果確定のみ
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[(sz >= 4) & (sz <= 6)].index)]
    done = df.groupby("race_key")["finish_order"].apply(lambda s: s.between(1, 3).sum() >= 3)
    df = df[df["race_key"].isin(done[done].index)].copy()

    with get_connection() as c:
        day_idx = dict(c.execute("SELECT race_key, day_index FROM wt_races"))
        vinfo = {v: (sl, cd) for v, sl, cd in c.execute(
            "SELECT venue_code, straight_len, cant_deg FROM venue_info")}
    print(f"  races(≤6車・結果確定): {df['race_key'].nunique():,} → loading trio boards...",
          flush=True)
    trio_b, _, _ = load_boards(df["race_key"].unique().tolist())

    riders = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        styles = [str(s) if isinstance(s, str) and s else "?" for s in g["style"]]
        n_nige = styles.count("逃")
        top3_sum, gap12, spread = p[0] + p[1] + p[2], p[0] - p[1], p[0] - p[3]
        mkt = market_top3_probs(trio_b.get(rk, {}), set(fr))
        mrank = ({f: i + 1 for i, f in enumerate(sorted(fr, key=lambda f: -mkt[f]))}
                 if mkt else {})
        cls = [CLASS_STRENGTH.get(str(x)) for x in g["player_class"]]
        terms = [None if np.isnan(x) else float(x) for x in
                 g["term"].astype(float).tolist()]
        gears = [None if np.isnan(x) else float(x) for x in
                 g["gear_ratio"].astype(float).tolist()]
        sl, cd = vinfo.get(str(g["venue_id"].iloc[0]), (None, None))
        lsize = g["line_size"].fillna(1).astype(int).tolist()
        lpos = g["line_pos"].fillna(1).astype(int).tolist()
        fins = g["finish_order"].fillna(0).astype(int).tolist()
        names = g["name"].tolist()
        for i in range(3, n):                      # モデルランク 4位以下
            ok_cls = [x for x in cls if x is not None]
            cls_rel = None
            if cls[i] is not None and ok_cls:
                mx = max(ok_cls)
                cls_rel = ("best_strict" if cls[i] == mx and ok_cls.count(mx) == 1
                           else "best_tied" if cls[i] == mx else "below")
            riders.append({
                "race_key": rk, "date": g["race_date"].iloc[0], "n": n,
                "rank": i + 1, "frame": fr[i], "name": names[i],
                "pred": p[i], "mkt_p": (mkt[fr[i]] if mkt else None),
                "mkt_rank": mrank.get(fr[i]),
                "style": styles[i], "line_size": lsize[i], "line_pos": lpos[i],
                "cls_rel": cls_rel,
                "term_rel": _rel(terms, i, "newest", "veteran"),
                "gear_rel": _rel(gears, i, "gear_max", "gear_min"),
                "grade": str(g["grade"].iloc[0]),
                "day_index": day_idx.get(rk, 0),
                "bank_length": g["bank_length"].iloc[0],
                "is_indoor": int(g["is_indoor"].iloc[0] or 0),
                "straight": sl, "cant": cd,
                "top3_sum": top3_sum, "gap12": gap12, "spread": spread,
                "n_nige": n_nige,
                "dns": fins[i] == 0,
                "top3": 1 <= fins[i] <= 3,
                "finish": fins[i],
            })
    return riders


def make_cells(cuts: dict) -> list[tuple[str, str, callable]]:
    q30 = cuts["spread_q30"]
    t3, g12 = cuts["t3q"], cuts["g12q"]
    slq, cdq = cuts["slq"], cuts["cdq"]
    C: list[tuple[str, str, callable]] = []
    C += [("ランク", f"rank{k}", lambda r, k=k: r["rank"] == k) for k in (4, 5, 6)]
    C += [("脚質", s, lambda r, s=s: r["style"] == s) for s in ("逃", "両", "追")]
    C += [
        ("ライン役割", "単騎", lambda r: r["line_size"] == 1),
        ("ライン役割", "ライン先頭", lambda r: r["line_size"] >= 2 and r["line_pos"] == 1),
        ("ライン役割", "番手", lambda r: r["line_pos"] == 2),
        ("ライン役割", "3番手+", lambda r: r["line_pos"] >= 3),
        ("ライン規模", "2車", lambda r: r["line_size"] == 2),
        ("ライン規模", "3車+", lambda r: r["line_size"] >= 3),
        ("doc13再現", "単騎×逃", lambda r: r["line_size"] == 1 and r["style"] == "逃"),
        ("doc13再現", "P4-3 拮抗×逃1人",
         lambda r: r["style"] == "逃" and r["n_nige"] == 1 and r["spread"] < q30),
        ("級班相対", "レース内最上位(単独)", lambda r: r["cls_rel"] == "best_strict"),
        ("級班相対", "レース内最上位(同着)", lambda r: r["cls_rel"] == "best_tied"),
        ("級班相対", "最上位未満", lambda r: r["cls_rel"] == "below"),
        ("期相対", "レース内最若手(期最大)", lambda r: r["term_rel"] == "newest"),
        ("期相対", "レース内最古参(期最小)", lambda r: r["term_rel"] == "veteran"),
        ("ギア相対", "レース内最大ギア", lambda r: r["gear_rel"] == "gear_max"),
        ("ギア相対", "レース内最小ギア", lambda r: r["gear_rel"] == "gear_min"),
        ("バンク", "333系(≤350m)", lambda r: (r["bank_length"] or 0) and r["bank_length"] <= 350),
        ("バンク", "400m", lambda r: r["bank_length"] == 400),
        ("バンク", "500m", lambda r: r["bank_length"] == 500),
        ("バンク", "ドーム", lambda r: r["is_indoor"] == 1),
    ]
    C += [("直線長", lab, lambda r, a=a, b=b: r["straight"] is not None and a <= r["straight"] < b)
          for lab, a, b in [("短(T1)", -1, slq[0]), ("中(T2)", slq[0], slq[1]),
                            ("長(T3)", slq[1], 999)]]
    C += [("カント", lab, lambda r, a=a, b=b: r["cant"] is not None and a <= r["cant"] < b)
          for lab, a, b in [("緩(T1)", -1, cdq[0]), ("中(T2)", cdq[0], cdq[1]),
                            ("急(T3)", cdq[1], 999)]]
    C += [
        ("車立て", "5車", lambda r: r["n"] == 5),
        ("車立て", "6車", lambda r: r["n"] == 6),
        ("グレード", "A級", lambda r: r["grade"] == "A級"),
        ("グレード", "S級", lambda r: r["grade"] == "S級"),
        ("グレード", "L級", lambda r: r["grade"] == "L級"),
        ("開催日次", "1日目", lambda r: r["day_index"] == 1),
        ("開催日次", "2日目", lambda r: r["day_index"] == 2),
        ("開催日次", "3日目+", lambda r: r["day_index"] >= 3),
    ]
    C += [("top3_sum帯", lab, lambda r, a=a, b=b: a <= r["top3_sum"] < b)
          for lab, a, b in [("Q1(波乱)", -9, t3[0]), ("Q2", t3[0], t3[1]),
                            ("Q3", t3[1], t3[2]), ("Q4(堅い)", t3[2], 9)]]
    C += [("gap12帯", lab, lambda r, a=a, b=b: a <= r["gap12"] < b)
          for lab, a, b in [("Q1(拮抗)", -9, g12[0]), ("Q2", g12[0], g12[1]),
                            ("Q3", g12[1], g12[2]), ("Q4(抜け)", g12[2], 9)]]
    C += [
        ("市場不一致", "市場が2+上に評価",
         lambda r: r["mkt_rank"] is not None and r["mkt_rank"] <= r["rank"] - 2),
        ("市場不一致", "市場はtop3級(mkt_rank≤3)",
         lambda r: r["mkt_rank"] is not None and r["mkt_rank"] <= 3),
        ("市場不一致", "市場も下位(mkt_rank≥4)",
         lambda r: r["mkt_rank"] is not None and r["mkt_rank"] >= 4),
    ]
    return C


def eval_cell(tr: list, te: list, fn) -> dict:
    out = {}
    for per, rows in (("TR", tr), ("TE", te)):
        sub = [r for r in rows if fn(r)]
        if not sub:
            out[per] = None
            continue
        k = sum(r["top3"] for r in sub)
        lo, hi = wilson(k, len(sub))
        out[per] = {
            "n": len(sub), "act": k / len(sub), "lo": lo, "hi": hi,
            "pred": float(np.mean([r["pred_cal"] for r in sub])),
            "mkt": float(np.mean([r["mkt_p"] for r in sub])),
        }
    return out


def judge(o: dict) -> str:
    tr, te = o.get("TR"), o.get("TE")
    if not tr or not te or tr["n"] < MIN_TR or te["n"] < MIN_TE:
        return "C(n)"
    rm_te = te["act"] - te["pred"]
    rk_te = te["act"] - te["mkt"]
    sig_model_tr = tr["lo"] > tr["pred"]       # TRで実績CI下限がモデル予測平均を上回る
    sig_mkt_tr = tr["lo"] > tr["mkt"]          # 同・市場示唆平均を上回る
    if not (sig_model_tr and rm_te > 0):
        return "C"
    if sig_mkt_tr and rk_te > 0:
        return "A"
    return "B"


def fmt_period(s: dict | None) -> str:
    if not s:
        return "    0"
    return (f"{s['n']:>6} {s['act']:>6.1%}[{s['lo']:>5.1%},{s['hi']:>5.1%}]"
            f" 予{s['pred']:>6.1%}({s['act']-s['pred']:>+6.1%})"
            f" 市{s['mkt']:>6.1%}({s['act']-s['mkt']:>+6.1%})")


def calib_table(rows: list, label: str, use_mkt: bool):
    bins = [0, .05, .08, .12, .16, .20, .25, .30, .40, 1.01]
    print(f"  ── {label} ──")
    head = f"  {'pred帯':<12}{'n':>7}{'mean(pred)':>11}{'実top3率':>9}{'残差':>8}"
    if use_mkt:
        head += f"{'mean(市場)':>11}{'対市場残差':>10}"
    print(head)
    for a, b in zip(bins, bins[1:]):
        sub = [r for r in rows if a <= r["pred"] < b]
        if len(sub) < 30:
            continue
        act = np.mean([r["top3"] for r in sub])
        pr = np.mean([r["pred"] for r in sub])
        line = f"  [{a:.2f},{b:.2f}){len(sub):>7,}{pr:>11.3f}{act:>9.1%}{act-pr:>+8.1%}"
        if use_mkt:
            mk = np.mean([r["mkt_p"] for r in sub])
            line += f"{mk:>11.3f}{act-mk:>+10.1%}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="load_from", default="2023-07-01",
                    help="データロード開始（ローリング特徴ウォームアップ込み）")
    ap.add_argument("--to", default="2026-06-12", help="TE終了日")
    ap.add_argument("--train-from", default="2024-01-01")
    ap.add_argument("--train-to", default="2026-02-28")
    ap.add_argument("--test-from", default="2026-03-01")
    args = ap.parse_args()

    riders = collect(args.load_from, args.to)
    full_tr = [r for r in riders if args.train_from <= r["date"] <= args.train_to]
    full_te = [r for r in riders if args.test_from <= r["date"] <= args.to]
    tr = [r for r in full_tr if r["mkt_p"] is not None]      # 主標本=市場カバレッジあり
    te = [r for r in full_te if r["mkt_p"] is not None]

    print(f"\n{'='*118}")
    print(f" ≤6車 指数下位(rank4+)の好走条件 残差分析  model=lgbm_wt_eval（TR=in-train/TE=OOS）")
    print(f" TR {args.train_from}〜{args.train_to}: 全{len(full_tr):,}人 → 市場あり主標本 {len(tr):,}人"
          f"（欠車率 {np.mean([r['dns'] for r in tr]):.1%}）")
    print(f" TE {args.test_from}〜{args.to}: 全{len(full_te):,}人 → 市場あり主標本 {len(te):,}人")
    print(f" 市場示唆=trio最終オッズ逆算（上限的な市場評価・実購入の主張はしない）")
    print(f"{'='*118}")

    # ---- Part1: 較正の全体像（rank4+ の pred_prob は top3 確率として較正されているか）----
    print(f"\n【Part1】rank4+ の較正（生 pred_prob・isotonic 前）")
    calib_table(tr, "TR 主標本(市場あり)", use_mkt=True)
    calib_table(te, "TE 主標本(市場あり)", use_mkt=True)
    calib_table(full_tr, "TR 全標本(欠車レース含む・市場列なし)", use_mkt=False)

    # isotonic 較正（TR 主標本で fit → 全行 transform）
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit([r["pred"] for r in tr], [r["top3"] for r in tr])
    for r in riders:
        r["pred_cal"] = float(iso.predict([r["pred"]])[0])
    g_act = np.mean([r["top3"] for r in tr])
    print(f"\n  isotonic較正: TR主標本 fit。全体 実top3率={g_act:.3f} / "
          f"mean(pred生)={np.mean([r['pred'] for r in tr]):.3f} / "
          f"mean(pred較正後)={np.mean([r['pred_cal'] for r in tr]):.3f} / "
          f"mean(市場示唆)={np.mean([r['mkt_p'] for r in tr]):.3f}")

    # ---- カット値（TR レース単位）----
    seen, sp, t3s, g12s = set(), [], [], []
    for r in tr:
        if r["race_key"] in seen:
            continue
        seen.add(r["race_key"])
        sp.append(r["spread"]); t3s.append(r["top3_sum"]); g12s.append(r["gap12"])
    sls = [r["straight"] for r in tr if r["straight"] is not None]
    cds = [r["cant"] for r in tr if r["cant"] is not None]
    cuts = {
        "spread_q30": float(np.quantile(sp, 0.30)),
        "t3q": [float(x) for x in np.quantile(t3s, [.25, .5, .75])],
        "g12q": [float(x) for x in np.quantile(g12s, [.25, .5, .75])],
        "slq": [float(x) for x in np.quantile(sls, [1 / 3, 2 / 3])],
        "cdq": [float(x) for x in np.quantile(cds, [1 / 3, 2 / 3])],
    }
    print(f"  cuts: spread_q30={cuts['spread_q30']:.3f} top3_sum四分位={[round(x,3) for x in cuts['t3q']]}"
          f" gap12四分位={[round(x,3) for x in cuts['g12q']]}"
          f" 直線三分位={[round(x,1) for x in cuts['slq']]} カント三分位={[round(x,2) for x in cuts['cdq']]}")

    # ---- Part2: 全セル走査 ----
    cells = make_cells(cuts)
    print(f"\n【Part2】セル走査（{len(cells)}セル・較正済predで残差・Wilson95%CI）")
    print(f"  判定: A=モデル・市場とも過小評価 / B=モデルのみ(市場織込済) / C=ノイズ・n不足")
    print(f"  期待偽陽性: TRのCI下限超え(一側α≈2.5%)×TE同方向(×0.5) ≈ {len(cells)*0.025*0.5:.1f}セル"
          f"（A判定はさらに市場側TR有意が必要・モデル/市場残差は相関するため上限値）")
    print(f"\n  {'セル':<34}{'判':>3}  {'n':>6} {'実top3率[95%CI]':<22}{'モデル較正(残差)':<17}{'市場(残差)'}")
    results = []
    for grp, lab, fn in cells:
        o = eval_cell(tr, te, fn)
        j = judge(o)
        results.append((grp, lab, fn, o, j))
        name = f"[{grp}] {lab}"
        print(f"  {name:<34}{j:>3}  TR {fmt_period(o['TR'])}")
        print(f"  {'':<34}{'':>3}  TE {fmt_period(o['TE'])}")

    # ---- Part3: 判定A の詳細（TE の的中例）----
    a_cells = [x for x in results if x[4] == "A"]
    print(f"\n【Part3】判定A = {len(a_cells)}セル（モデル・市場とも過小評価）")
    for grp, lab, fn, o, _ in a_cells:
        print(f"\n  ◆ [{grp}] {lab}")
        for per, s in (("TR", o["TR"]), ("TE", o["TE"])):
            print(f"    {per}: n={s['n']} 実{s['act']:.1%} [{s['lo']:.1%},{s['hi']:.1%}]"
                  f" モデル較正{s['pred']:.1%}(残差{s['act']-s['pred']:+.1%})"
                  f" 市場{s['mkt']:.1%}(残差{s['act']-s['mkt']:+.1%})")
        ex = [r for r in te if fn(r) and r["top3"]][:3]
        for r in ex:
            print(f"    例: {r['date']} {r['race_key']} 枠{r['frame']} {r['name']}"
                  f" rank{r['rank']} pred較正{r['pred_cal']:.2f} 市場{r['mkt_p']:.2f}"
                  f" → {r['finish']}着")

    # ---- Part4: doc13 P4-3 再現の明示 ----
    print(f"\n【Part4】doc13 P4-3（拮抗×逃1人×rank4+）再現確認")
    for grp, lab, fn, o, j in results:
        if grp == "doc13再現":
            print(f"  {lab}: 判定{j}")
            for per, s in (("TR", o["TR"]), ("TE", o["TE"])):
                if s:
                    print(f"    {per}: n={s['n']} 実{s['act']:.1%} モデル較正{s['pred']:.1%}"
                          f"(残差{s['act']-s['pred']:+.1%}) 市場{s['mkt']:.1%}"
                          f"(残差{s['act']-s['mkt']:+.1%})")
    n_b = sum(1 for x in results if x[4] == "B")
    n_c = sum(1 for x in results if x[4].startswith("C"))
    print(f"\n  集計: {len(cells)}セル → A {len(a_cells)} / B {n_b} / C {n_c}")
    print("  注意: 本分析は記述的残差。判定Aがあっても「買える」の主張ではない"
          "（ROI検証は3期間プロトコル・live実測が別途必要）。")


if __name__ == "__main__":
    main()
