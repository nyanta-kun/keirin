"""
人気上位2頭フィルター分析

AIの予想軸（1位・2位）が市場の1番人気・2番人気と一致するレースを除外した場合の
的中率・回収率を検証する。

人気順位は単勝オッズ未収集のため、quinella（2車複）の最低配当ペアで代用する。
(最低配当 = 市場が最も支持するペア ≈ 1番人気+2番人気)
"""
import sys
import pickle
import collections
from pathlib import Path

import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS, TARGET_COL
from src.database import get_connection

# ─── 設定 ────────────────────────────────────────────────────────────────────
HOLDOUT_FROM  = "2025-06-01"
HOLDOUT_TO    = "2026-02-28"
MODEL_PATH    = "data/models/lgbm.pkl"
MAX_RIDERS    = 6       # 6車立て以下
GAP12_MIN     = 0.06    # wave-picks 最低 gap12
SS_GAP12      = 0.15    # SS/S 閾値
SS_RATIO_MAX  = 1.3     # SS ratio 上限
UNIT          = 100     # 1点あたり掛け金 (円)
TRIFECTA_PTS  = 3       # 3連単 SS (3点)
TRIFECTA_BOX_PTS = 3    # 3連複 S/A (3点)
# ─────────────────────────────────────────────────────────────────────────────


def load_quinella_payouts(race_keys: list[str]) -> dict[str, dict[frozenset, int]]:
    """race_key → {frozenset({a,b}): payout} の辞書を返す"""
    if not race_keys:
        return {}
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        conn.row_factory = None
        rows = conn.execute(f"""
            SELECT race_key, combination, payout
            FROM odds
            WHERE race_key IN ({placeholders})
              AND bet_type = 'quinella'
              AND payout IS NOT NULL
        """, race_keys).fetchall()

    result: dict[str, dict] = collections.defaultdict(dict)
    for race_key, combo_str, payout in rows:
        parts = combo_str.split("=")
        if len(parts) == 2:
            key = frozenset([int(parts[0]), int(parts[1])])
            result[race_key][key] = payout
    return result


def load_result_payouts(race_keys: list[str]) -> dict:
    """race_key → {(bet_type, combination): payout} の辞書を返す"""
    if not race_keys:
        return {}
    placeholders = ",".join("?" * len(race_keys))
    with get_connection() as conn:
        conn.row_factory = None
        rows = conn.execute(f"""
            SELECT race_key, bet_type, combination, payout
            FROM odds
            WHERE race_key IN ({placeholders})
              AND payout IS NOT NULL
        """, race_keys).fetchall()

    result: dict = collections.defaultdict(dict)
    for race_key, bet_type, combo, payout in rows:
        result[race_key][(bet_type, combo)] = payout
    return result


def get_market_top2(race_key: str, q_map: dict) -> frozenset | None:
    """quinella最低配当ペア → 市場の1-2番人気ペア（代理）"""
    q = q_map.get(race_key)
    if not q:
        return None
    min_pair = min(q, key=lambda k: q[k])
    return min_pair


def classify_rank(gap12: float, ratio: float) -> str:
    if gap12 < GAP12_MIN:
        return "SKIP"
    if gap12 >= SS_GAP12 and ratio < SS_RATIO_MAX:
        return "SS"
    if gap12 >= SS_GAP12:
        return "S"
    return "A"


def eval_wave_picks(grp: pd.DataFrame, rank: str, payout_map: dict) -> tuple[bool, int]:
    """SS/S/A 戦略の的中判定と払戻額を返す。
    Returns (hit, payout)
    """
    grp = grp.sort_values("pred_prob", ascending=False)
    ranked = grp["frame_no"].tolist()
    race_key = grp["race_key"].iloc[0]
    race_payouts = payout_map.get(race_key, {})

    actual_order_df = grp[grp["finish_position"].isin([1, 2, 3])].sort_values("finish_position")
    if len(actual_order_df) < 3:
        return False, 0
    actual_order = tuple(actual_order_df["frame_no"].tolist())
    top3_set = frozenset(actual_order)

    pivot1, pivot2 = ranked[0], ranked[1]
    thirds = [r for r in ranked[2:5] if r not in (pivot1, pivot2)]

    if rank == "SS":
        # 3連単: pivot1→pivot2→{3rd} 3点
        combos = [(pivot1, pivot2, t) for t in thirds[:3]]
        for combo in combos:
            if combo == actual_order:
                pk = "-".join(map(str, actual_order))
                return True, race_payouts.get(("trifecta", pk), 0)
        return False, 0
    else:
        # 3連複: pivot1-pivot2-{3rd} 3点 (S/A 共通)
        combos = [frozenset([pivot1, pivot2, t]) for t in thirds[:3]]
        if top3_set in combos:
            pk = "=".join(map(str, sorted(top3_set)))
            return True, race_payouts.get(("trifecta_box", pk), 0)
        return False, 0


def main():
    print(f"Loading data {HOLDOUT_FROM} ~ {HOLDOUT_TO} ...")
    df_raw = load_raw_data(min_date=HOLDOUT_FROM, max_date=HOLDOUT_TO)
    df = build_features(df_raw)
    df = df[df["finish_position"].notna()].copy()

    print(f"Loading model: {MODEL_PATH}")
    model = pickle.load(open(MODEL_PATH, "rb"))

    # pred_prob 計算
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = pd.DataFrame(df[FEATURE_COLS].values, columns=FEATURE_COLS)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 6車以下フィルター
    race_sizes = df.groupby("race_key")["frame_no"].count()
    valid_races = race_sizes[race_sizes <= MAX_RIDERS].index
    df = df[df["race_key"].isin(valid_races)]

    all_race_keys = df["race_key"].unique().tolist()
    print(f"6車以下: {len(all_race_keys)}レース")

    # オッズデータ読み込み
    print("Loading odds data ...")
    q_map = load_quinella_payouts(all_race_keys)
    payout_map = load_result_payouts(all_race_keys)

    q_coverage = sum(1 for k in all_race_keys if k in q_map)
    print(f"Quinella coverage: {q_coverage}/{len(all_race_keys)} "
          f"({q_coverage/len(all_race_keys):.1%})")

    # ─── race_date取得 ──────────────────────────────────────────────────────
    race_date_map = df.groupby("race_key")["race_date"].first().to_dict()

    # ─── レース単位で分析 ──────────────────────────────────────────────────
    rows = []
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        ranked = grp["frame_no"].tolist()
        n = len(ranked)

        top1_prob = grp["pred_prob"].iloc[0]
        top2_prob = grp["pred_prob"].iloc[1] if n >= 2 else 0.0
        gap12 = top1_prob - top2_prob
        ratio = top1_prob / (3 / n)

        rank = classify_rank(gap12, ratio)
        if rank == "SKIP":
            continue

        ai_top2 = frozenset([ranked[0], ranked[1]])
        market_top2 = get_market_top2(race_key, q_map)

        # 市場と一致するか（quinellaデータなしはUnknown）
        if market_top2 is None:
            is_fav_match = None   # 判定不能
        else:
            is_fav_match = (ai_top2 == market_top2)

        hit, payout = eval_wave_picks(grp, rank, payout_map)

        n_pts = TRIFECTA_PTS if rank == "SS" else TRIFECTA_BOX_PTS
        bet_amount = n_pts * UNIT

        rows.append({
            "race_key":     race_key,
            "race_date":    race_date_map[race_key],
            "rank":         rank,
            "gap12":        gap12,
            "ratio":        ratio,
            "is_fav_match": is_fav_match,
            "hit":          hit,
            "payout":       payout,
            "bet_amount":   bet_amount,
        })

    df_res = pd.DataFrame(rows)
    print(f"\n総対象レース(gap12≥0.06, 6車以下): {len(df_res)}R")

    # ─── 低配当調整: 3連複で的中配当≤300円をハズレ扱い＋1点(100円)削減 ──────
    LOW_ODDS_THRESH = 300  # 3倍以下

    def apply_low_odds_adjustment(sub: pd.DataFrame) -> pd.DataFrame:
        """S/A の的中配当≤300円をハズレ扱いし、投資を100円削減して返す"""
        sub = sub.copy()
        # SS は 3連単のため対象外
        mask_sa  = sub["rank"].isin(["S", "A"])
        mask_low = sub["hit"] & (sub["payout"] <= LOW_ODDS_THRESH)
        adj_mask = mask_sa & mask_low
        sub.loc[adj_mask, "hit"]        = False
        sub.loc[adj_mask, "payout"]     = 0
        sub.loc[adj_mask, "bet_amount"] = sub.loc[adj_mask, "bet_amount"] - 100
        return sub

    # ─── 集計関数 ─────────────────────────────────────────────────────────
    def summarize(sub: pd.DataFrame, label: str, show_payout_detail: bool = False):
        if sub.empty:
            print(f"\n{label}: データなし")
            return

        # 期間日数（取引日数）
        n_days = sub["race_date"].nunique()
        n_total = len(sub)
        avg_per_day = n_total / n_days if n_days else 0

        print(f"\n{'='*80}")
        print(f"  {label}  ({n_total}R / {n_days}日 / 平均 {avg_per_day:.1f}R/日)")
        print(f"{'='*80}")
        print(f"  {'ランク':<5} {'件数':>6}  {'的中':>5}  {'的中率':>7}  "
              f"{'投資':>10}  {'回収':>10}  {'ROI':>7}  {'損益':>10}  {'平均配当':>8}")
        print(f"  {'-'*75}")

        total_bet = 0
        total_ret = 0
        total_hit = 0
        for rk in ["SS", "S", "A"]:
            s = sub[sub["rank"] == rk]
            if s.empty:
                continue
            bet = s["bet_amount"].sum()
            ret = s["payout"].sum()
            hits = int(s["hit"].sum())
            n = len(s)
            roi = ret / bet if bet > 0 else 0
            avg_pay = ret / hits if hits > 0 else 0
            print(f"  {rk:<5} {n:>6}  {hits:>5}  {hits/n:>7.1%}  "
                  f"{bet:>10,}  {ret:>10,}  {roi:>7.1%}  {ret-bet:>+10,}  {avg_pay:>8,.0f}円")
            total_bet += bet
            total_ret += ret
            total_hit += hits

            # S/A の配当分布詳細
            if show_payout_detail and rk in ("S", "A"):
                hit_rows = s[s["hit"]]
                if not hit_rows.empty:
                    p = hit_rows["payout"]
                    low = (p < 300).sum()
                    med = ((p >= 300) & (p < 600)).sum()
                    high = (p >= 600).sum()
                    print(f"        ↳ 配当分布: <300円={low}回({low/len(p):.0%})  "
                          f"300-600円={med}回({med/len(p):.0%})  "
                          f"≥600円={high}回({high/len(p):.0%})  "
                          f"中央値={p.median():.0f}円")

        print(f"  {'-'*75}")
        roi_all = total_ret / total_bet if total_bet > 0 else 0
        avg_pay_all = total_ret / total_hit if total_hit > 0 else 0
        print(f"  {'合計':<5} {n_total:>6}  {int(total_hit):>5}  "
              f"{total_hit/n_total:>7.1%}  "
              f"{total_bet:>10,}  {total_ret:>10,}  {roi_all:>7.1%}  "
              f"{total_ret-total_bet:>+10,}  {avg_pay_all:>8,.0f}円")
        print(f"{'='*80}")
        return avg_per_day

    # ─── 全体（現行） ─────────────────────────────────────────────────────
    summarize(df_res, "【全体 (現行)】", show_payout_detail=True)

    # quinellaデータありのみで比較
    df_with_q = df_res[df_res["is_fav_match"].notna()].copy()
    n_no_q = len(df_res) - len(df_with_q)
    if n_no_q > 0:
        print(f"\n  ※ quinella未収録のため判定不能: {n_no_q}R（以下は判定可能レースのみ）")

    # ─── 人気一致レース（除外対象） ───────────────────────────────────────
    df_fav = df_with_q[df_with_q["is_fav_match"] == True]
    summarize(df_fav, "【除外対象】AI軸2頭 = 市場1-2番人気", show_payout_detail=True)

    # ─── 人気不一致レース（除外後） ───────────────────────────────────────
    df_non_fav = df_with_q[df_with_q["is_fav_match"] == False]
    avg = summarize(df_non_fav, "【除外後】AI軸2頭 ≠ 市場1-2番人気", show_payout_detail=True)

    # ─── 月別 (除外後) ───────────────────────────────────────────────────
    if not df_non_fav.empty:
        print(f"\n{'='*70}")
        print("  【月別 (除外後 / ランク合計)】")
        print(f"{'='*70}")
        print(f"  {'月':<8} {'件数':>5}  {'的中':>4}  {'的中率':>7}  {'ROI':>7}  {'損益':>10}")
        print(f"  {'-'*55}")

        df_non_fav = df_non_fav.copy()
        df_non_fav["ym"] = df_non_fav["race_date"].str[:7]
        for ym, grp in df_non_fav.groupby("ym"):
            bet = grp["bet_amount"].sum()
            ret = grp["payout"].sum()
            hits = grp["hit"].sum()
            n = len(grp)
            roi = ret / bet if bet > 0 else 0
            n_days_m = grp["race_date"].nunique()
            avg_m = n / n_days_m if n_days_m else 0
            print(f"  {ym:<8} {n:>5}  {int(hits):>4}  {hits/n:>7.1%}  "
                  f"{roi:>7.1%}  {ret-bet:>+10,}  ({avg_m:.1f}R/日)")
        print(f"{'='*70}")

    # ─── ランク別詳細（除外後） ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  【ランク別詳細 — 全体 vs 除外後】")
    print(f"{'='*80}")
    print(f"  {'ランク':<5}  {'全体R':>6}  {'全体ROI':>8}  {'除外後R':>7}  "
          f"{'除外後ROI':>9}  {'除外率':>7}  {'avg配当(全)':>11}  {'avg配当(除外後)':>13}")
    print(f"  {'-'*75}")
    for rk in ["SS", "S", "A"]:
        g_all = df_res[df_res["rank"] == rk]
        g_nf  = df_non_fav[df_non_fav["rank"] == rk] if not df_non_fav.empty else pd.DataFrame()
        g_dq  = df_with_q[df_with_q["rank"] == rk]

        n_all = len(g_all)
        roi_all = g_all["payout"].sum() / g_all["bet_amount"].sum() if n_all else 0
        hits_all = int(g_all["hit"].sum())
        avg_pay_all = g_all["payout"].sum() / hits_all if hits_all else 0

        n_nf = len(g_nf)
        roi_nf = g_nf["payout"].sum() / g_nf["bet_amount"].sum() if n_nf else 0
        hits_nf = int(g_nf["hit"].sum()) if not g_nf.empty else 0
        avg_pay_nf = g_nf["payout"].sum() / hits_nf if hits_nf else 0

        n_dq = len(g_dq)
        excl_rate = 1 - n_nf / n_dq if n_dq else 0

        print(f"  {rk:<5}  {n_all:>6}  {roi_all:>8.1%}  {n_nf:>7}  "
              f"{roi_nf:>9.1%}  {excl_rate:>7.1%}  "
              f"{avg_pay_all:>9,.0f}円  {avg_pay_nf:>10,.0f}円")
    print(f"{'='*80}")

    # ─── S/A: 3連複配当の低配当フィルター分析 ──────────────────────────────
    print(f"\n{'='*80}")
    print("  【S/A 3連複: 的中時配当レベル別 分析】")
    print("  ※ 軸ペアが人気か否かと、実際の配当水準の関係を確認")
    print(f"{'='*80}")

    for rk in ["S", "A"]:
        g = df_with_q[df_with_q["rank"] == rk].copy()
        g_fav  = g[g["is_fav_match"] == True]
        g_nfav = g[g["is_fav_match"] == False]

        print(f"\n  ── {rk}ランク ({len(g)}R) ──")
        print(f"  {'カテゴリ':<18} {'件数':>5}  {'的中':>4}  {'的中率':>7}  "
              f"{'avg配当':>8}  {'中央値':>7}  {'ROI':>7}  "
              f"{'<300円%':>7}  {'≥600円%':>7}")
        print(f"  {'-'*75}")

        for label, sub in [("AI=市場(除外対象)", g_fav), ("AI≠市場(除外後)", g_nfav)]:
            if sub.empty:
                continue
            n = len(sub)
            hits = int(sub["hit"].sum())
            bet = sub["bet_amount"].sum()
            ret = sub["payout"].sum()
            roi = ret / bet if bet else 0
            hit_rows = sub[sub["hit"]]
            if hits > 0:
                avg_pay = hit_rows["payout"].mean()
                med_pay = hit_rows["payout"].median()
                low_pct = (hit_rows["payout"] < 300).mean()
                high_pct = (hit_rows["payout"] >= 600).mean()
            else:
                avg_pay = med_pay = low_pct = high_pct = 0
            print(f"  {label:<18} {n:>5}  {hits:>4}  {hits/n:>7.1%}  "
                  f"{avg_pay:>8,.0f}円  {med_pay:>7,.0f}円  {roi:>7.1%}  "
                  f"{low_pct:>7.1%}  {high_pct:>7.1%}")

    print(f"\n{'='*80}")
    print("  ※ <300円 = 投資(300円)を下回る的中（実質損失）、≥600円 = 高配当的中")

    # ─── 低配当調整適用 ────────────────────────────────────────────────────
    print(f"\n\n{'#'*80}")
    print("  【低配当調整 (3連複 的中配当≤300円 → ハズレ扱い＋1点削減)】")
    print(f"{'#'*80}")
    print("  ロジック: 的中しても配当≤300円 (3倍以下) の場合、その1点は「買わなかった」と仮定")
    print("           → bet_amount を100円削減、payout=0、hitをFalseに変換")

    df_adj       = apply_low_odds_adjustment(df_res.copy())
    df_adj_with_q = apply_low_odds_adjustment(df_with_q.copy())
    df_adj_fav   = apply_low_odds_adjustment(df_fav.copy())
    df_adj_nfav  = apply_low_odds_adjustment(df_non_fav.copy())

    # 調整による変化件数サマリ
    for rk in ["S", "A"]:
        orig = df_res[df_res["rank"] == rk]
        adj  = df_adj[df_adj["rank"] == rk]
        changed = (orig["hit"].values & ~adj["hit"].values).sum()
        print(f"\n  {rk}ランク: 低配当的中 → ハズレ変換 {changed}件 / {int(orig['hit'].sum())}件中")

    # ─── 集計: 調整後 全体 ─────────────────────────────────────────────────
    summarize(df_adj, "【調整後 全体】", show_payout_detail=False)

    # ─── 集計: 調整後 市場一致 ─────────────────────────────────────────────
    summarize(df_adj_fav,  "【調整後】AI軸2頭 = 市場1-2番人気", show_payout_detail=False)

    # ─── 集計: 調整後 市場不一致 ───────────────────────────────────────────
    summarize(df_adj_nfav, "【調整後】AI軸2頭 ≠ 市場1-2番人気", show_payout_detail=False)

    # ─── 月別 (調整後 全体) ────────────────────────────────────────────────
    if not df_adj.empty:
        print(f"\n{'='*80}")
        print("  【月別 (調整後全体 / S+A合計)】 ← SSは変更なし")
        print(f"{'='*80}")
        print(f"  {'月':<8} {'件数':>5}  {'的中':>4}  {'的中率':>7}  {'投資':>10}  "
              f"{'回収':>10}  {'ROI':>7}  {'損益':>10}  ({'/日'})")
        print(f"  {'-'*72}")

        df_adj_sa = df_adj[df_adj["rank"].isin(["S", "A"])].copy()
        df_adj_sa["ym"] = df_adj_sa["race_date"].str[:7]
        for ym, grp in df_adj_sa.groupby("ym"):
            bet  = grp["bet_amount"].sum()
            ret  = grp["payout"].sum()
            hits = int(grp["hit"].sum())
            n    = len(grp)
            roi  = ret / bet if bet else 0
            n_days_m = grp["race_date"].nunique()
            avg_m = n / n_days_m if n_days_m else 0
            print(f"  {ym:<8} {n:>5}  {hits:>4}  {hits/n:>7.1%}  {bet:>10,}  "
                  f"{ret:>10,}  {roi:>7.1%}  {ret-bet:>+10,}  ({avg_m:.1f}R/日)")
        bet_tot = df_adj_sa["bet_amount"].sum()
        ret_tot = df_adj_sa["payout"].sum()
        print(f"  {'合計':<8} {len(df_adj_sa):>5}  {int(df_adj_sa['hit'].sum()):>4}  "
              f"{df_adj_sa['hit'].mean():>7.1%}  {bet_tot:>10,}  "
              f"{ret_tot:>10,}  {ret_tot/bet_tot if bet_tot else 0:>7.1%}  "
              f"{ret_tot-bet_tot:>+10,}")
        print(f"{'='*80}")

    # ─── 調整前後 比較サマリ ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  【調整前後 比較サマリ (S/A のみ)】")
    print(f"{'='*80}")
    print(f"  {'カテゴリ':<22}  {'件数':>5}  {'調整前ROI':>9}  {'調整後ROI':>9}  "
          f"{'調整前損益':>10}  {'調整後損益':>10}")
    print(f"  {'-'*72}")

    for label, orig_df, adj_df in [
        ("全体(S+A)",     df_res[df_res["rank"].isin(["S","A"])],   df_adj[df_adj["rank"].isin(["S","A"])]),
        ("市場一致(S+A)", df_fav[df_fav["rank"].isin(["S","A"])],   df_adj_fav[df_adj_fav["rank"].isin(["S","A"])]),
        ("市場不一致(S+A)",df_non_fav[df_non_fav["rank"].isin(["S","A"])], df_adj_nfav[df_adj_nfav["rank"].isin(["S","A"])]),
    ]:
        n = len(orig_df)
        if n == 0:
            continue
        b_orig = orig_df["bet_amount"].sum()
        r_orig = orig_df["payout"].sum()
        b_adj  = adj_df["bet_amount"].sum()
        r_adj  = adj_df["payout"].sum()
        roi_orig = r_orig / b_orig if b_orig else 0
        roi_adj  = r_adj  / b_adj  if b_adj  else 0
        print(f"  {label:<22}  {n:>5}  {roi_orig:>9.1%}  {roi_adj:>9.1%}  "
              f"{r_orig-b_orig:>+10,}  {r_adj-b_adj:>+10,}")
    print(f"{'='*80}")
    print("  ※ 調整: 的中配当≤300円 → ハズレ扱い＋bet_amount -100円")


if __name__ == "__main__":
    main()
