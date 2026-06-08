"""統合「波乱/非本命ゲート」検証（winticket）

3タスクの収束結論「本命が堅いレースを見送り、本命が弱い=波乱レースを狙う」を
単一ゲートとして本番戦略(SS/S/A 3点)に統合し、現行 vs ゲートを train→OOS test で比較。
過学習回避のため最小シグナル(top2_sum)を主軸とし、重ねる価値(leader_gap/AI印除外)も併測。

ゲート: 各レースの top2_sum = pivot1_prob + pivot2_prob が閾値 T 未満のみ購入。
        （top2_sum 小 = 軸2頭への確率集中が弱い = 本命が割れている = 波乱余地）
すべて確定前情報のみ。オッズ不使用・朝7:00で算出可・直前まで不変。

実行:
  PYTHONPATH=. .venv/bin/python3 scripts/exp_upset_gate_wt.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import (
    _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt, _assign_tier,
)

TRAIN_FROM, TRAIN_TO = "2023-07-01", "2026-02-28"
TEST_FROM,  TEST_TO  = "2026-03-01", "2026-06-08"
MODEL = "lgbm_wt"
MAX_RIDERS = 6


def race_table(df: pd.DataFrame, payout_map: dict) -> pd.DataFrame:
    """レース単位に集約: tier / top2_sum / leader_gap / ai_chalk / hit / payout / bet。"""
    rows = []
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue
        probs = grp["pred_prob"].tolist()
        gap12 = probs[0] - probs[1]
        ratio = probs[0] / (3.0 / n)
        tier = _assign_tier(gap12, ratio)
        if tier is None:
            continue

        frames = grp["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:5]
        if not thirds:
            continue

        finished = grp[grp["finish_order"].between(1, 3)]
        top3_set = frozenset(finished["frame_no"].astype(int).tolist())
        if len(top3_set) < 3:
            continue
        actual_order = tuple(
            finished.sort_values("finish_order")["frame_no"].astype(int).tolist()
        )
        rp = payout_map.get(race_key, {})

        # ── ゲート用シグナル（確定前） ──
        top2_sum = probs[0] + probs[1]
        # leader_gap: pivot1 と pivot2 が別ラインなら先頭力差の代理として gap を採用
        lg = grp.set_index("frame_no")
        line_col = "line_group" if "line_group" in grp.columns else None
        if line_col is not None:
            try:
                same_line = (lg.loc[pivot1, line_col] == lg.loc[pivot2, line_col])
            except Exception:
                same_line = False
        else:
            same_line = False
        leader_gap = gap12 if not same_line else 0.0  # 別ライン頭同士=崩れ余地
        # AI印 chalk: 軸1が prediction_mark==1(本命) か
        ai_chalk = 0
        if "prediction_mark" in grp.columns:
            ai_chalk = int(grp.iloc[0].get("prediction_mark", 0) == 1)

        # 的中・払戻（本番3点）
        hit, payout = False, 0
        if tier == "SS":
            for t in thirds:
                if actual_order == (pivot1, pivot2, t):
                    payout = rp.get(("trifecta", (pivot1, pivot2, t)), 0); hit = True; break
        else:
            for t in thirds:
                combo = frozenset((pivot1, pivot2, t))
                if combo == top3_set:
                    payout = rp.get(("trio", combo), 0); hit = True; break

        rows.append({
            "race_key": race_key, "tier": tier,
            "top2_sum": top2_sum, "leader_gap": leader_gap, "ai_chalk": ai_chalk,
            "hit": hit, "payout": payout, "bet": 300,
        })
    return pd.DataFrame(rows)


def summarize(rt: pd.DataFrame, mask, label: str) -> dict:
    sub = rt[mask]
    bet = sub["bet"].sum(); ret = sub["payout"].sum()
    roi = ret / bet if bet else 0
    return {"label": label, "R": len(sub), "hit": int(sub["hit"].sum()),
            "hit_rate": (sub["hit"].mean() if len(sub) else 0),
            "ROI": roi, "pl": ret - bet}


def show(rt: pd.DataFrame, period: str):
    print(f"\n{'='*78}\n  【{period}】 対象 {len(rt)}R (SS/S/A)\n{'='*78}")
    # 現行（ゲートなし）
    base = summarize(rt, rt.index == rt.index, "現行(ゲートなし)")
    # top2_sum 閾値スイープ
    variants = [base]
    for T in [0.85, 0.80, 0.75, 0.70]:
        variants.append(summarize(rt, rt["top2_sum"] < T, f"top2_sum<{T}"))
    # 重ねる価値: top2_sum<0.80 ∧ 別ライン頭 / ∧ AI印本命除外
    variants.append(summarize(rt, (rt["top2_sum"] < 0.80) & (rt["leader_gap"] > 0),
                              "top2_sum<0.80 ∧ 別ライン頭"))
    variants.append(summarize(rt, (rt["top2_sum"] < 0.80) & (rt["ai_chalk"] == 0),
                              "top2_sum<0.80 ∧ AI印本命でない"))
    print(f"  {'条件':<26} {'R':>5} {'的中':>5} {'的中率':>7} {'ROI':>8} {'損益':>10}")
    print(f"  {'-'*70}")
    for v in variants:
        print(f"  {v['label']:<26} {v['R']:>5} {v['hit']:>5} {v['hit_rate']:>7.1%} "
              f"{v['ROI']:>8.1%} {v['pl']:>+10,}")
    # 層別（現行 vs top2_sum<0.80）
    print(f"\n  ── 層別 (現行 → top2_sum<0.80) ──")
    for t in ["SS", "S", "A"]:
        a = summarize(rt, rt["tier"] == t, t)
        b = summarize(rt, (rt["tier"] == t) & (rt["top2_sum"] < 0.80), t)
        print(f"    {t:<3} {a['R']:>4}R ROI {a['ROI']:>7.1%}  →  {b['R']:>4}R ROI {b['ROI']:>7.1%}")


def main():
    model = load_model(MODEL)
    out = {}
    for name, (f, t) in {"TRAIN": (TRAIN_FROM, TRAIN_TO), "TEST/OOS": (TEST_FROM, TEST_TO)}.items():
        print(f"\n[load] {name} {f}〜{t} ...")
        df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
        df = _apply_pred_prob_wt(model, df)
        df = _filter_by_n_riders(df, MAX_RIDERS)
        pm = _load_payouts_wt(df["race_key"].unique().tolist())
        rt = race_table(df, pm)
        out[name] = rt
        show(rt, f"{name}  {f}〜{t}")

    # 月別（TEST, top2_sum<0.80）
    rt = out["TEST/OOS"].copy()
    rt2 = rt[rt["top2_sum"] < 0.80].copy()
    if not rt2.empty:
        rt2["race_date"] = rt2["race_key"].str[:8]
        rt2["ym"] = rt2["race_date"].str[:6]
        print(f"\n{'='*78}\n  【TEST 月別 (top2_sum<0.80)】\n{'='*78}")
        print(f"  {'月':<8} {'R':>5} {'的中率':>7} {'ROI':>8} {'損益':>10}")
        for ym, g in rt2.groupby("ym"):
            bet = g["bet"].sum(); ret = g["payout"].sum()
            print(f"  {ym:<8} {len(g):>5} {g['hit'].mean():>7.1%} "
                  f"{ret/bet if bet else 0:>8.1%} {ret-bet:>+10,}")


if __name__ == "__main__":
    main()
