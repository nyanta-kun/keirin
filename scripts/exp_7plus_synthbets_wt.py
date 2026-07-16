"""7+車 S/Aランク「全相手流し」の合成オッズフィルター効果検証

現行 S/A ランク（7車以上・全相手流し）の問題:
  ・gami = min_odds ≥ 5.0 しかチェックしていない
  ・出走台数が多いほど購入点数が増え、同じ gami でも合成オッズが下がる
    例: 5点流し gami=5x → 合成(最悪)= 5/5 = 1.0 (収支±0)
        6点流し gami=5x → 合成(最悪)= 5/6 = 0.83 (赤字確定!)
        7点流し gami=5x → 合成(最悪)= 5/7 = 0.71 (大赤字確定!)

定義する指標:
  gami      = 全購入レグの最安オッズ（現行フィルター）
  syn_min   = gami / n_bets  … 最悪ケースROI（最安レグが的中した場合）
  syn_harm  = 1 / Σ(1/oᵢ)   … 調和平均型合成オッズ（既存 exp_gami_synthetic_wt.py と同方式）
  avg_eff   = Σoᵢ/n / n      … 平均オッズ÷点数（算術平均ベース期待効率）

検証設計（doc18 本番忠実・3バイアス回避）:
  ・全エントリー（DNS含む）でランキング（バイアス①）
  ・7+車判定は出走表ベース（バイアス②）
  ・DNS軸=レース無効 / DNS相手=該当点のみ除外（バイアス③）
  ・モデル: lgbm_wt_eval（評価期間外での学習・週次リークなし）
  ・払戻: 最終オッズ（実運用ROIの上限値）

出力:
  A. レース件数・点数分布（n_bets別・合成オッズ分布）
  B. 現行(gami≥5)+各合成閾値スイープ → ROI/件数/的中率
  C. S/Aランク別の閾値効果比較
  D. 推奨閾値の決定基準

期間:
  TRAIN 2023-07-01〜2026-02-28
  TEST  2026-03-01〜2026-06-25
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _load_payouts_wt,
)
from roi_robustness_wt import roi_summary

MODEL     = "lgbm_wt_eval"
TRAIN_END = "2026-02-28"
TEST_START = "2026-03-01"
TEST_END   = "2026-06-25"

# S/A ランク判定閾値（bet-structure-guide.md 現行）
GAP12_S   = 0.10   # S: gap12 ≥ 0.10
GAP12_A   = 0.07   # A: gap12 ≥ 0.07 (A は [0.07, 0.10))
GAMI_MIN  = 5.0    # 現行 gami 足切り


def collect(date_from: str, date_to: str) -> list[dict]:
    """7+車の全相手流し S/A レースを収集し合成オッズを計算する。"""
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))

    # バイアス②: 出走表ベースで 7+車フィルタ（pred_prob付与前）
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes >= 7].index)].copy()

    # バイアス①: DNS含む全エントリーで pred_prob 付与
    df = _apply_pred_prob_wt(model, df)

    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    rows = []
    for rk, g in df.groupby("race_key"):
        # 全エントリー数（出走表基準）
        n_entries = len(g)
        if n_entries < 7:
            continue

        # DNS / 欠車の特定（バイアス③）
        dns_set = frozenset(g[g["finish_order"] == 0]["frame_no"].astype(int).tolist())

        # バイアス①: 全エントリー順でランキング
        g_sorted = g.sort_values("pred_prob", ascending=False)
        probs    = g_sorted["pred_prob"].tolist()
        frames   = g_sorted["frame_no"].astype(int).tolist()

        if len(probs) < 3:
            continue

        gap12 = probs[0] - probs[1]

        # S/Aランク判定
        if gap12 < GAP12_A:
            continue
        rank = "S" if gap12 >= GAP12_S else "A"

        p1, p2 = frames[0], frames[1]

        # バイアス③: 軸欠車 → レース無効
        if p1 in dns_set or p2 in dns_set:
            continue

        # 相手: top2以外の全車（DNS車は除外）
        opponents = [f for f in frames[2:] if f not in dns_set]
        if not opponents:
            continue

        # trio オッズ取得（全相手流し）
        rp = pm.get(rk, {})
        odds_list = []
        for opp in opponents:
            o = rp.get(("trio", frozenset((p1, p2, opp))))
            if o is not None and o > 0:
                odds_list.append(o / 100.0)

        # オッズが揃わない相手は除外（購入不可と同義）
        n_bets = len(odds_list)
        if n_bets == 0:
            continue

        # 現行 gami フィルタ
        gami = min(odds_list)
        if gami < GAMI_MIN:
            continue

        # 合成オッズ計算
        syn_min  = gami / n_bets
        syn_harm = 1.0 / sum(1.0 / o for o in odds_list)
        avg_eff  = (sum(odds_list) / n_bets) / n_bets

        # 的中判定
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3   = frozenset(fin["frame_no"].astype(int).tolist())
        pay    = 0
        hit    = top3 in [frozenset((p1, p2, opp)) for opp in opponents]
        if hit:
            pay = rp.get(("trio", top3), 0)

        rows.append({
            "rank":     rank,
            "n_bets":   n_bets,
            "n_entries": n_entries,
            "gami":     gami,
            "syn_min":  syn_min,
            "syn_harm": syn_harm,
            "avg_eff":  avg_eff,
            "gap12":    gap12,
            "pay":      float(pay),
            "bet":      n_bets * 100,
            "hit":      hit,
        })

    return rows


def _roi(rows: list[dict]):
    if not rows:
        return None, 0
    s = roi_summary([r["pay"] for r in rows], [r["bet"] for r in rows])
    return s, len(rows)


def _fmt(s, n):
    if s is None:
        return f"{'0':>5}R  --"
    return (f"{n:>5}R {s['roi']:>7.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]"
            f" 除{s['roi_ex_max']:>5.0%} 的{s['hit_rate']:>6.2%}")


def part_a(te: list[dict]):
    print(f"\n{'='*90}")
    print(f"  A. 件数・点数・合成オッズ分布（TEST {len(te)}R）")
    print(f"{'='*90}")
    nb_arr  = np.array([r["n_bets"]   for r in te])
    sm_arr  = np.array([r["syn_min"]  for r in te])
    sh_arr  = np.array([r["syn_harm"] for r in te])
    gm_arr  = np.array([r["gami"]     for r in te])

    print(f"  購入点数(n_bets): min={nb_arr.min():.0f} 中央={np.median(nb_arr):.1f} max={nb_arr.max():.0f} 平均={nb_arr.mean():.2f}")
    print(f"  gami(最安):       min={gm_arr.min():.2f} 中央={np.median(gm_arr):.2f} max={gm_arr.max():.2f}")
    print(f"  syn_min(最悪ROI): min={sm_arr.min():.2f} 中央={np.median(sm_arr):.2f} max={sm_arr.max():.2f}")
    print(f"  syn_harm(調和):   min={sh_arr.min():.2f} 中央={np.median(sh_arr):.2f} max={sh_arr.max():.2f}")
    print(f"\n  n_bets 分布:")
    print(f"    {'n_bets':>6}{'件数':>7}{'syn_min中央':>12}{'syn_harm中央':>13}{'ROI(TEST)':>12}")
    for nb in sorted(set(nb_arr.astype(int).tolist())):
        sub = [r for r in te if r["n_bets"] == nb]
        sm = np.median([r["syn_min"]  for r in sub])
        sh = np.median([r["syn_harm"] for r in sub])
        s, n = _roi(sub)
        roi_str = f"{s['roi']:>7.0%}" if s else "  --"
        print(f"    {nb:>6}{len(sub):>7}{sm:>12.2f}{sh:>13.2f}{roi_str:>12}")

    # gami ≥ 5 なのに syn_min < 1.0（赤字確定レース）の実態
    danger = [r for r in te if r["syn_min"] < 1.0]
    print(f"\n  ⚠ syn_min < 1.0（最安レグ的中でも赤字確定）: {len(danger)}R / {len(te)}R "
          f"= {len(danger)/len(te):.1%}")
    if danger:
        s_d, _ = _roi(danger)
        s_ok, _ = _roi([r for r in te if r["syn_min"] >= 1.0])
        if s_d:
            print(f"     syn_min<1.0 ROI: {s_d['roi']:.0%}   syn_min≥1.0 ROI: {s_ok['roi'] if s_ok else '--':.0%}")


def part_b(tr: list[dict], te: list[dict]):
    print(f"\n{'='*90}")
    print(f"  B. 合成オッズ閾値スイープ（現行=gami≥5 ベースに追加フィルタ）")
    print(f"{'='*90}")
    header = f"  {'条件':<30}{'TR':>28}{'TE':>46}"
    print(header)
    print(f"  {'-'*90}")

    def line(label, cond_fn):
        sr, nr = _roi([r for r in tr if cond_fn(r)])
        se, ne = _roi([r for r in te if cond_fn(r)])
        flag = ""
        if sr and se:
            if sr["roi"] > 1.0 and se["roi"] > 1.0 and ne >= 50:
                flag = "★再現"
            elif ne < 30:
                flag = "小標本"
        print(f"  {label:<30}{_fmt(sr, nr):>35}{_fmt(se, ne):>50}  {flag}")

    # 現行ベースライン
    line("[現行] gami≥5（追加なし）",  lambda r: True)
    print()

    # syn_min スイープ（最悪ケースROI）
    print("  ◆ syn_min = gami/n_bets（最悪ケースROI）フィルター追加:")
    for thr in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        line(f"  +syn_min≥{thr:.1f}", lambda r, t=thr: r["syn_min"] >= t)
    print()

    # syn_harm スイープ（調和平均型）
    print("  ◆ syn_harm = 1/Σ(1/oᵢ)（調和平均型）フィルター追加:")
    for thr in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        line(f"  +syn_harm≥{thr:.1f}", lambda r, t=thr: r["syn_harm"] >= t)
    print()

    # n_bets 直接フィルタ（点数制限）
    print("  ◆ n_bets（購入点数）直接制限:")
    for nb in [5, 4, 3]:
        line(f"  n_bets≤{nb}", lambda r, nb=nb: r["n_bets"] <= nb)
    print()

    # 組み合わせ: gami調整（台数連動）
    print("  ◆ gami/n_bets（点数連動動的ガミ閾値）:")
    for req in [1.0, 1.1, 1.2, 1.3, 1.5]:
        line(f"  gami/n_bets≥{req:.1f}(=syn_min≥{req})", lambda r, t=req: r["syn_min"] >= t)


def part_c(tr: list[dict], te: list[dict]):
    print(f"\n{'='*90}")
    print(f"  C. S/A ランク別の効果比較（syn_min フィルター）")
    print(f"{'='*90}")

    for rank in ["S", "A", "S+A"]:
        if rank == "S+A":
            sub_tr = tr
            sub_te = te
        else:
            sub_tr = [r for r in tr if r["rank"] == rank]
            sub_te = [r for r in te if r["rank"] == rank]

        print(f"\n  ◆ {rank}ランク（TEST: {len(sub_te)}R）")
        print(f"  {'条件':<28}{'TR':>28}{'TE':>46}")

        def line_r(label, cond_fn, src_tr, src_te):
            sr, nr = _roi([r for r in src_tr if cond_fn(r)])
            se, ne = _roi([r for r in src_te if cond_fn(r)])
            flag = "★再現" if (sr and se and sr["roi"] > 1.0 and se["roi"] > 1.0 and ne >= 30) else ""
            print(f"  {label:<28}{_fmt(sr, nr):>35}{_fmt(se, ne):>50}  {flag}")

        line_r("[現行] gami≥5",       lambda r: True, sub_tr, sub_te)
        for thr in [1.0, 1.1, 1.2, 1.3]:
            line_r(f"  +syn_min≥{thr:.1f}", lambda r, t=thr: r["syn_min"] >= t, sub_tr, sub_te)


def part_d(te: list[dict]):
    print(f"\n{'='*90}")
    print(f"  D. 採用閾値の決定基準サマリー（TEST {len(te)}R）")
    print(f"{'='*90}")

    best_n, best_roi, best_thr = 0, 0.0, None
    for thr in np.arange(0.5, 2.01, 0.1):
        sub = [r for r in te if r["syn_min"] >= thr]
        if not sub:
            continue
        s, n = _roi(sub)
        if s and s["roi"] > best_roi and n >= 30:
            best_roi, best_n, best_thr = s["roi"], n, thr

    print(f"  TEST で最高ROI（件数≥30）: syn_min ≥ {best_thr:.1f}  ROI={best_roi:.0%}  n={best_n}R")
    print(f"\n  判断基準:")
    print(f"    1. TR ROI > 100% かつ TE ROI > 100% → 再現あり")
    print(f"    2. 件数 ≥ 50R（TEST）→ 統計的信頼性")
    print(f"    3. 最大払戻除去ROI も > 100% → 単発頼りでない")
    print(f"    4. TE CI 下限 > 100% → 強い根拠")
    print(f"\n  ※ 最終オッズ上限値。朝→確定オッズドリフトで実運用はさらに下振れ。")
    print(f"     採否は picks_history(live実測)で前向きに確認すること。")


if __name__ == "__main__":
    print("collecting TRAIN...", flush=True)
    tr = collect("2023-07-01", TRAIN_END)
    print(f"  TRAIN: {len(tr)}R (S={sum(1 for r in tr if r['rank']=='S')} / A={sum(1 for r in tr if r['rank']=='A')})", flush=True)
    print("collecting TEST...", flush=True)
    te = collect(TEST_START, TEST_END)
    print(f"  TEST:  {len(te)}R (S={sum(1 for r in te if r['rank']=='S')} / A={sum(1 for r in te if r['rank']=='A')})", flush=True)

    part_a(te)
    part_b(tr, te)
    part_c(tr, te)
    part_d(te)
