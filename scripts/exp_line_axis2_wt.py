"""三連複 ライン軸2 + 指数下位除外 ROI 検証（doc51）

ユーザー提案:
  「２軸目に１軸目の同一ライン得点2位を入れ、
    指数一番下または２番目までは３列目から除外する」

戦略定義:
  S0: 現行（AI確率順: 軸1+軸2, 流し3名=3点）
  L1: 軸2=同一line_group内指数2位 + 流し全員から指数最下位1名除外
  L2: 軸2=同一line_group内指数2位 + 流し全員から指数下位2名除外
  R1: 軸2はAI順位2位のまま(変えず) + 流しから指数最下位1名除外のみ
  R2: R1 + 指数下位2名除外

フォールバック: pivot1のline_size=1（単独ライン）→ L1/L2の軸2はAI順位2位を使用

フィルタ: ≤6車, ガミ(S0ベース3点最安)≥5倍
TRAIN: 2023-07-01〜2025-06-30
VAL:   2025-07-01〜2026-02-28
HOLD:  2026-03-01〜2026-06-15
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X,
)
from src.database import get_connection
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS

GAMI_RACE_THRESHOLD = 5.0
MAX_RIDERS = 6


# ── データ準備 ────────────────────────────────────────────────────────────────

def _parse_combo(s):
    if isinstance(s, (list, tuple)):
        try:
            return frozenset(int(x) for x in s)
        except Exception:
            return None
    parts = re.split(r"[-=]", str(s))
    try:
        return frozenset(int(p) for p in parts)
    except Exception:
        return None


def load_all():
    df_raw = load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1])
    df = build_features_wt(df_raw)

    # line_group 等の raw カラムを保持（build_features_wt が落とす場合は raw から補完）
    need = ["line_group", "line_size", "line_pos", "is_line_leader", "race_point", "style"]
    for col in need:
        if col not in df.columns and col in df_raw.columns:
            df = df.merge(df_raw[["race_key", "frame_no", col]], on=["race_key", "frame_no"], how="left")

    with get_connection() as conn:
        ri = pd.read_sql("SELECT race_key, n_entries, grade FROM wt_races", conn)
        trio_df = pd.read_sql(
            "SELECT race_key, combination, odds_value FROM wt_odds WHERE bet_type='trio'", conn
        )

    df = df.merge(ri, on="race_key", how="left")

    print("  TRAIN 期間のみでリーク無しモデル学習中...", flush=True)
    fit = df[(df["race_date"] >= TRAIN[0]) & (df["race_date"] <= TRAIN[1])
             & (df["finish_order"] >= 1)]
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(prepare_X(fit).reset_index(drop=True), fit["top3_flag"].reset_index(drop=True).values)
    print(f"  学習完了 ({len(fit):,} 行)", flush=True)

    df = df.copy().reset_index(drop=True)
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # trio オッズマップ: {race_key: {frozenset: payout_yen}}
    trio_df["k"] = trio_df["combination"].apply(_parse_combo)
    trio_df = trio_df.dropna(subset=["k"])
    trio_map: dict[str, dict] = {}
    for r in trio_df.itertuples(index=False):
        trio_map.setdefault(r.race_key, {})[r.k] = int(round(r.odds_value * 100))

    # 実際の結果: {race_key: frozenset(top3 frames)}
    actual_trio = (
        df[df["finish_order"].between(1, 3)]
        .groupby("race_key")["frame_no"]
        .apply(lambda x: frozenset(x.astype(int).tolist()))
        .to_dict()
    )

    return df, trio_map, actual_trio


# ── 戦略評価コア ─────────────────────────────────────────────────────────────

def eval_strategies(df, trio_map, actual_trio):
    """戦略ごとの {period: [bet_result_rows]} を返す。"""

    def period_of(d):
        if TRAIN[0] <= d <= TRAIN[1]:
            return "TRAIN"
        if VAL_START <= d <= VAL_END:
            return "VAL"
        if HOLD[0] <= d <= HOLD[1]:
            return "HOLD"
        return None

    VAL_START, VAL_END = VAL[0], VAL[1]

    strats = ["S0", "L1", "L2", "R1", "R2"]
    records = {s: [] for s in strats}

    for race_key, grp in df.groupby("race_key"):
        period = period_of(grp["race_date"].iloc[0])
        if period is None:
            continue

        n_ent = int(grp["n_entries"].iloc[0])
        if n_ent > MAX_RIDERS:
            continue

        grp = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(grp) < 4:
            continue

        frames = grp["frame_no"].astype(int).tolist()
        pivot1 = frames[0]

        # race_point で降順ソート（指数ランキング用）
        grp_rp = grp.sort_values("race_point", ascending=False).reset_index(drop=True)
        frames_by_rp = grp_rp["frame_no"].astype(int).tolist()

        # ── S0: 現行（AI確率順）────────────────────────────────────────────────
        pivot2_s0 = frames[1]
        thirds_s0 = frames[2:5]  # AI rank 3,4,5

        # ガミチェック: S0 の3点の最安目 ≥ GAMI_RACE_THRESHOLD
        tmap = trio_map.get(race_key, {})
        gami_vals = []
        for t in thirds_s0:
            k = frozenset({pivot1, pivot2_s0, t})
            ov = tmap.get(k)
            if ov is not None and ov < 900000:
                gami_vals.append(ov / 100)
        if not gami_vals or min(gami_vals) < GAMI_RACE_THRESHOLD:
            continue

        # 実結果
        actual = actual_trio.get(race_key)
        if actual is None:
            continue

        # ── L: ライン軸2 ────────────────────────────────────────────────────────
        # pivot1 の line_group を取得
        p1_row = grp[grp["frame_no"] == pivot1].iloc[0]
        p1_line = p1_row["line_group"] if "line_group" in grp.columns else None

        pivot2_l = None
        if p1_line is not None and not pd.isna(p1_line):
            # 同一 line_group の他選手を race_point 降順で取り出す
            same_line = grp_rp[grp_rp["line_group"] == p1_line]["frame_no"].astype(int).tolist()
            same_line_excl = [f for f in same_line if f != pivot1]
            if same_line_excl:
                pivot2_l = same_line_excl[0]  # 同一ライン内指数2位(pivot1除く1位)

        if pivot2_l is None:
            pivot2_l = pivot2_s0  # フォールバック: AI順位2位

        # L1/L2 の流し候補: pivot1・pivot2_l を除く全員、race_point 降順
        pool_l = [f for f in frames_by_rp if f != pivot1 and f != pivot2_l]

        # ── R1/R2: 軸2固定（S0と同じ）+ 指数下位除外 ──────────────────────────
        pool_r = [f for f in frames_by_rp if f != pivot1 and f != pivot2_s0]

        def bet_records(pivot2, thirds, period, strategy):
            rows = []
            for t in thirds:
                k = frozenset({pivot1, pivot2, t})
                ov_yen = tmap.get(k)
                if ov_yen is None:
                    cost, ret = 100, 0
                else:
                    ov_yen = min(ov_yen, 899900)  # placeholder除外
                    cost = 100
                    ret = ov_yen if actual == k else 0
                rows.append({
                    "period": period,
                    "strategy": strategy,
                    "cost": cost,
                    "ret": ret,
                    "hit": int(ret > 0),
                })
            return rows

        # S0
        records["S0"].extend(bet_records(pivot2_s0, thirds_s0, period, "S0"))

        # L1: ライン軸2 + 指数最下位1名除外
        thirds_l1 = pool_l[:-1] if len(pool_l) > 1 else pool_l
        records["L1"].extend(bet_records(pivot2_l, thirds_l1, period, "L1"))

        # L2: ライン軸2 + 指数下位2名除外
        thirds_l2 = pool_l[:-2] if len(pool_l) > 2 else pool_l[:1]
        records["L2"].extend(bet_records(pivot2_l, thirds_l2, period, "L2"))

        # R1: AI軸2固定 + 指数最下位1名除外
        thirds_r1 = pool_r[:-1] if len(pool_r) > 1 else pool_r
        records["R1"].extend(bet_records(pivot2_s0, thirds_r1, period, "R1"))

        # R2: AI軸2固定 + 指数下位2名除外
        thirds_r2 = pool_r[:-2] if len(pool_r) > 2 else pool_r[:1]
        records["R2"].extend(bet_records(pivot2_s0, thirds_r2, period, "R2"))

    return records


# ── レポート印刷 ─────────────────────────────────────────────────────────────

def report(records):
    PERIODS = ["TRAIN", "VAL", "HOLD"]
    STRATS = ["S0", "L1", "L2", "R1", "R2"]
    LABELS = {
        "S0": "現行 3点  ",
        "L1": "L1:ライン軸2+下位1除外",
        "L2": "L2:ライン軸2+下位2除外",
        "R1": "R1:AI軸2固定+下位1除外",
        "R2": "R2:AI軸2固定+下位2除外",
    }

    print("\n" + "="*80)
    print("三連複 ライン軸2 + 指数下位除外 ROI 検証（doc51）")
    print("フィルタ: ≤6車 / ガミ(S0基準)≥5倍")
    print("="*80)
    print(f"{'戦略':<24}  {'期間':>6}  {'対象R':>7}  {'avg点':>6}  {'的中%':>7}  {'ROI':>8}")
    print("-"*70)

    for strat in STRATS:
        df_s = pd.DataFrame(records[strat])
        label = LABELS[strat]
        for period in PERIODS:
            sub = df_s[df_s["period"] == period]
            if sub.empty:
                continue
            # レース数は bet数 / avg_pts で計算
            total_cost = sub["cost"].sum()
            total_ret = sub["ret"].sum()
            n_bets = len(sub)
            n_hits = sub["hit"].sum()
            roi = total_ret / total_cost * 100 if total_cost > 0 else 0

            # レース数推定（点数平均で割る）
            # costは全て100円なので点数=n_bets / n_races
            # 同一期間のS0 n_bets = n_races * 3 → n_races = S0_bets/3
            flag = "★" if roi >= 100 else ""
            print(f"  {label:<22}  {period:>6}  {n_bets:>7}bets  {'---':>6}  {n_hits/n_bets*100:>6.1f}%  {roi:>7.1f}% {flag}")
        print()

    # レース数ベースの表（S0基準でレース数を計算）
    print("\n" + "="*80)
    print("【レース数・avg点ベース集計】")
    print(f"{'戦略':<24}  {'期間':>6}  {'R数':>6}  {'avg点':>6}  {'的中%':>7}  {'ROI':>8}")
    print("-"*70)

    # S0 のレース数を基準にする
    s0_df = pd.DataFrame(records["S0"])
    s0_races = {}
    for period in PERIODS:
        sub0 = s0_df[s0_df["period"] == period]
        # S0は3点固定なのでレース数 = bets / 3
        s0_races[period] = len(sub0) // 3

    for strat in STRATS:
        df_s = pd.DataFrame(records[strat])
        label = LABELS[strat]
        for period in PERIODS:
            sub = df_s[df_s["period"] == period]
            n_races = s0_races.get(period, 1)
            if sub.empty or n_races == 0:
                continue
            n_bets = len(sub)
            avg_pts = n_bets / n_races
            total_cost = sub["cost"].sum()
            total_ret = sub["ret"].sum()
            n_hits = sub["hit"].sum()
            hit_pct = n_hits / n_bets * 100
            roi = total_ret / total_cost * 100 if total_cost > 0 else 0
            flag = "★" if roi >= 100 else ""
            print(f"  {label:<22}  {period:>6}  {n_races:>6}R  {avg_pts:>5.1f}点  {hit_pct:>6.1f}%  {roi:>7.1f}% {flag}")
        print()

    # pivot2 変更の影響分析（L vs S の pivot2 が何割一致するか）
    print("\n" + "="*80)
    print("【補足: ライン軸2 vs AI軸2 の一致率・当たり外れ分析】")
    # L1 vs S0 で同じレース、同じ期間での比較
    s0_df = pd.DataFrame(records["S0"])
    l1_df = pd.DataFrame(records["L1"])
    print(f"  S0 VAL bets: {len(s0_df[s0_df['period']=='VAL'])}")
    print(f"  L1 VAL bets: {len(l1_df[l1_df['period']=='VAL'])}")


if __name__ == "__main__":
    print("三連複 ライン軸2 + 指数下位除外 ROI 検証 (doc51)")
    print("データ準備中...")
    df, trio_map, actual_trio = load_all()

    print("戦略評価中...", flush=True)
    records = eval_strategies(df, trio_map, actual_trio)

    report(records)
