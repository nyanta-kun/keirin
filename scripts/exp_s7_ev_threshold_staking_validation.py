"""【S7本番実装前の最終検証】既存ゲートスタックに EV閾値配分を追加した場合の
honest全期間ROIを、月次凍結vintageモデルで検証する（2026-07-30）。

## 位置づけ（なぜこの検証が必須か）

[[keirin_staking_allocation_optimization_2026_07_30]]で「配分最適化（EV<1.0の
点を除外）はレース選択と組み合わせるとROI+10pt」を発見したが、その検証では
**軸選定に自前の S_jointpair 方式（pi×pj×lift 最大ペア）を使い、レース選択にも
自前の top3_sum_top2 指標を使った**。これは本番の `rank_7s_select_axis()`
（win_probs∩top3_probs重なり方式）や `rank_7s_daily_select()` の既存ゲート
（axis_sum<=1.3・entropy<=1.8329・wt_overlap等）とは異なる。

さらに本番の `S7_AXIS_SUM_MAX=1.3` は「axis_sum**高い**レース（三連複が
安くなりやすい極端な人気決着）を除外する」設計（`src/strategy_wt.py:305-321`
のコメント参照）であり、これは事前検証の「top2_prob_sum**高い**レースで
ratioが高い」という知見とは逆方向のフィルタである。両者は目的が異なる
（本番=配当下限の防御、事前検証=市場ミスプライシングのratio最大化）ため
数学的な矛盾ではないが、**本番の軸選定・既存ゲートと整合するかは別途確認が
必要**。

**本スクリプトは最も安全な変更（既存ゲート・軸選定は一切変更せず、
最終ステップにEV閾値配分だけを追加する）を、本番と同一の月次vintageモデル・
同一の軸選定・同一の既存ゲートスタックで検証する。**
`scripts/backfill_s7_rank_wt.py::build_rows()` と同一ロジックを踏襲し
（import可能な関数はそのまま再利用）、既存ゲート通過後の候補に対して
EV閾値フィルタ配分を追加した場合の的中率・ROIを計算する。

## 計算方法

各選出済みレースについて、35通り全体を our_prob（product×ライン相関lift、
正規化済み）で評価し、5点流しの各点の EV = our_prob × 実際の三連複オッズ
を計算。EV>=1.0の点だけに均等配分（他は0）。0点該当なら見送り。

honest: 月次凍結vintageモデル（`src.wt_vintage_config.monthly_windows()`）を
使用。ライン相関liftは各月の直前累積データ（TRAINに相当する期間）で
推定し、当月には適用のみ（リーク防止）。

DB書き込みなし・読み取り専用SELECTのみ。既存の picks_history には触れない。
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    RANK_7S_STAKE, rank_7s_evening_reselect, rank_7s_field_entropy,
    rank_7s_select_axis, rank_7s_wt_mark3_overlap_n, rank_7s_wt_overlap_n,
)
from src.wt_vintage_config import monthly_windows

from scripts.backfill_7s_rank_wt import _load_trio_boards

TOTAL_STAKE_PER_RACE = 500.0   # 5点×100円=500円（本番と同一の総投資額）


def line_bucket(bf, i, j):
    li, lj = bf[i].get("line_group"), bf[j].get("line_group")
    if li is None or lj is None:
        return "unknown"
    if li != lj:
        return "diff"
    pi, pj = bf[i].get("line_pos"), bf[j].get("line_pos")
    if pi is None or pj is None:
        return "same_other"
    a, b = sorted([int(pi), int(pj)])
    return {(1, 2): "same_12", (2, 3): "same_23", (1, 3): "same_13"}.get((a, b), "same_other")


def estimate_lifts(hist_races):
    """直近累積レース（当月より前）でライン相関liftを推定する。"""
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for bf, actual_top3 in hist_races:
        for i, j in combinations(sorted(bf), 2):
            b = line_bucket(bf, i, j)
            pi = bf[i]["p"]
            pj = bf[j]["p"]
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if i in actual_top3 and j in actual_top3:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


def race_our_probs(bf, lifts):
    frames = sorted(bf)
    p = {f: bf[f]["p"] for f in frames}
    raw = {}
    for tri in combinations(frames, 3):
        s = p[tri[0]] * p[tri[1]] * p[tri[2]]
        for x, y in combinations(tri, 2):
            s *= lifts.get(line_bucket(bf, x, y), 1.0)
        raw[frozenset(tri)] = s
    tot = sum(raw.values())
    if tot <= 0:
        return None
    return {k: v / tot for k, v in raw.items()}


def build_candidates_with_lineinfo(model_name, date_from, date_to, win_model_name):
    """backfill_s7_rank_wt.build_rows と同一ロジック＋line_group/line_posを追加保持。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    candidates = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        bf = {int(r.frame_no): {"p": float(r.pred_prob), "line_group": r.line_group,
                                "line_pos": r.line_pos} for r in g.itertuples(index=False)}
        sel = rank_7s_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = rank_7s_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)
        wt_overlap_n = rank_7s_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3_overlap_n = rank_7s_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        candidates.append({
            "race_key": rk, "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others, "trio": trio, "actual_top3": actual_top3, "bf": bf,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n,
        })
    return candidates, pm


def select_and_eval(candidates_by_day, pm, lifts, ev_hit_log=None):
    """rank_7s_evening_reselect() で本番と同一の選出をし、uniform/ev_threshold双方のROIを返す。

    ev_hit_log: Noneでなければ、ev_threshold_filterで的中した各点の
      (race_key, race_date, stake, odds, payout) をappendして頑健性検証に使う。
    """
    uni_bet = uni_ret = uni_hit = 0.0
    ev_bet = ev_ret = ev_hit = 0.0
    n_selected = n_ev_bet_races = 0

    for _d, day_cands in candidates_by_day.items():
        for c_ in rank_7s_evening_reselect(day_cands, [], set()):
            n_selected += 1
            axis1, axis2 = c_["axis1"], c_["axis2"]
            trio = c_["trio"]
            combos = []
            for x in c_["others"]:
                key = frozenset({axis1, axis2, x})
                if key in trio:
                    combos.append((x, key, trio[key]))
            if not combos:
                continue
            rk = c_["race_key"]
            actual_top3 = c_["actual_top3"]

            # ---- uniform（現行本番ロジック）----
            hit = actual_top3 in {k for _x, k, _o in combos}
            trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)
            uni_bet += len(combos) * RANK_7S_STAKE
            uni_hit += int(hit)
            if hit:
                uni_ret += trio_pay * RANK_7S_STAKE // 100

            # ---- ev_threshold_filter（追加提案）----
            probs = race_our_probs(c_["bf"], lifts)
            if probs is None:
                continue
            evs = []
            for x, key, odds in combos:
                p = probs.get(key)
                if p is None:
                    evs.append((key, odds, 0.0))
                    continue
                evs.append((key, odds, p * odds))
            qualify = [(key, odds) for key, odds, ev in evs if ev >= 1.0]
            if not qualify:
                continue
            n_ev_bet_races += 1
            stake_each = TOTAL_STAKE_PER_RACE / len(qualify)
            for key, odds in qualify:
                ev_bet += stake_each
                if key == actual_top3:
                    ev_hit += 1
                    pay = stake_each * odds
                    ev_ret += pay
                    if ev_hit_log is not None:
                        ev_hit_log.append((rk, c_["race_date"], stake_each, odds, pay))

    return {
        "n_selected": n_selected,
        "uniform": {"bet": uni_bet, "ret": uni_ret, "hit": uni_hit,
                    "roi": uni_ret / uni_bet * 100 if uni_bet else 0.0},
        "ev_threshold": {"bet": ev_bet, "ret": ev_ret, "hit": ev_hit, "n_bet_races": n_ev_bet_races,
                         "roi": ev_ret / ev_bet * 100 if ev_bet else 0.0},
    }


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}")

    hist_races = []          # ライン相関lift推定用の累積履歴（bf, actual_top3）
    ev_hit_log = []          # 頑健性検証用: ev_threshold的中の1件ごとの内訳
    totals = {"n_selected": 0,
              "uniform": {"bet": 0.0, "ret": 0.0, "hit": 0.0},
              "ev_threshold": {"bet": 0.0, "ret": 0.0, "hit": 0.0, "n_bet_races": 0}}

    for date_from, date_to, eval_model, win_model in windows:
        print(f"\n[main] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            print("  候補なし")
            continue

        # liftは「この月より前」の累積履歴のみで推定（リーク防止）
        lifts = estimate_lifts(hist_races) if hist_races else {}

        by_day = defaultdict(list)
        for c_ in candidates:
            by_day[c_["race_date"]].append(c_)

        res = select_and_eval(by_day, pm, lifts, ev_hit_log=ev_hit_log)
        for k in ("n_selected",):
            totals[k] += res[k]
        for scheme in ("uniform", "ev_threshold"):
            for k, v in res[scheme].items():
                if k == "roi":
                    continue
                totals[scheme][k] += v

        print(f"  選出: {res['n_selected']}R")
        print(f"  uniform      : bet={res['uniform']['bet']:,.0f} ret={res['uniform']['ret']:,.0f} "
              f"ROI={res['uniform']['roi']:.1f}%")
        print(f"  ev_threshold : bet_races={res['ev_threshold']['n_bet_races']} "
              f"bet={res['ev_threshold']['bet']:,.0f} ret={res['ev_threshold']['ret']:,.0f} "
              f"ROI={res['ev_threshold']['roi']:.1f}%")

        # この月の実績を履歴に追加（次月のlift推定に使う）
        for c_ in candidates:
            hist_races.append((c_["bf"], c_["actual_top3"]))

    print("\n" + "=" * 100)
    print("全期間合計（月次vintageモデル・honest walk-forward）")
    print("=" * 100)
    u, e = totals["uniform"], totals["ev_threshold"]
    u_roi = u["ret"] / u["bet"] * 100 if u["bet"] else 0.0
    e_roi = e["ret"] / e["bet"] * 100 if e["bet"] else 0.0
    print(f"選出レース数: {totals['n_selected']}R")
    print(f"uniform（現行本番）     : 投資{u['bet']:,.0f} 回収{u['ret']:,.0f} "
          f"的中{u['hit']:.0f} ROI {u_roi:.1f}%")
    print(f"ev_threshold_filter（提案）: 賭けレース{e['n_bet_races']} "
          f"投資{e['bet']:,.0f} 回収{e['ret']:,.0f} 的中{e['hit']:.0f} ROI {e_roi:.1f}%")
    print(f"\n差分: {e_roi - u_roi:+.1f}pt")

    print("\n" + "=" * 100)
    print("【頑健性検証】ev_threshold_filter的中の内訳（少数の大穴依存でないか）")
    print("=" * 100)
    if not ev_hit_log:
        print("的中0件")
    else:
        ev_hit_log.sort(key=lambda x: -x[4])
        total_ret = sum(x[4] for x in ev_hit_log)
        print(f"的中件数: {len(ev_hit_log)} / 総回収: {total_ret:,.0f}円")
        print(f"{'race_key':<22}{'日付':<12}{'stake':>8}{'odds':>9}{'payout':>10}{'累積寄与%':>10}")
        cum = 0.0
        for rk, d, stake, odds, pay in ev_hit_log:
            cum += pay
            print(f"{rk:<22}{d:<12}{stake:>8.1f}{odds:>9.1f}{pay:>10,.0f}{cum/total_ret*100:>9.1f}%")
        top1_share = ev_hit_log[0][4] / total_ret * 100
        top3_share = sum(x[4] for x in ev_hit_log[:3]) / total_ret * 100
        print(f"\n最大1件が総回収に占める割合: {top1_share:.1f}%")
        print(f"上位3件が総回収に占める割合: {top3_share:.1f}%")
        # 最大1件を除外した場合のROI（外れ値依存の頑健性チェック）
        e_bet = totals["ev_threshold"]["bet"]
        ret_wo_top1 = total_ret - ev_hit_log[0][4]
        roi_wo_top1 = ret_wo_top1 / e_bet * 100 if e_bet else 0.0
        ret_wo_top3 = total_ret - sum(x[4] for x in ev_hit_log[:3])
        roi_wo_top3 = ret_wo_top3 / e_bet * 100 if e_bet else 0.0
        print(f"\n最大1件を除いた場合のROI: {roi_wo_top1:.1f}%（除く前: {e_roi:.1f}%）")
        print(f"上位3件を除いた場合のROI: {roi_wo_top3:.1f}%")
        odds_vals = sorted(x[3] for x in ev_hit_log)
        print(f"\n的中したodds分布: min={odds_vals[0]:.1f} 中央値={odds_vals[len(odds_vals)//2]:.1f} "
              f"max={odds_vals[-1]:.1f}")


if __name__ == "__main__":
    main()
