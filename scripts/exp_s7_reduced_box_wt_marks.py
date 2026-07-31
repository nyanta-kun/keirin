"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

さらに本スクリプトは S7_AXIS_SUM_MAX = 1.3 をハードコードしているが、
現行 src/strategy_wt.py:328 の S7_AXIS_SUM_MAX は 1.5 に変更済みであり、
本スクリプトの閾値定義は現行仕様と食い違っている（この点でも数値は再現不可）。

S7の3列目ボックスからWT印(◎/◯)を除外する「絞り込み買い」のhonest ROI検証
（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

ユーザー方針の具体化:
  現行S7の買い目 = 軸2車(axis1,axis2) + 残り5車ボックス = 5点(500円)。
  軸2車のうち一方がWT印(◎/◯)と重なる場合(wt_overlap_n==1)、「◎◯のどちらか
  一方のみ3着内」が真のボリュームゾーンという仮説に基づき、軸に含まれなかった
  もう一方のWT印馬をそもそも3列目候補から除外する → 4点(400円)買いに変更。
  軸2車がWT印と全く重ならない場合(wt_overlap_n==0、SS)は、◎◯両方とも3着内に
  こない前提でどちらも3列目候補から除外する → 3点(300円)買いに変更。

  wt_overlap_n==2(軸がWT印◎◯と完全一致)は現行通り除外対象のまま(変更なし)。

honest全期間(2024-01-01〜2026-07-28)・quarterly walk-forwardモデル
（`exp_s7_gate_staged_audit.py`と同一のQUARTERS）で候補プールを再構築し、
現行5点買い vs 絞り込み買い(4点/3点)のROIをTRAIN(2024-01-01〜2025-12-31)選定
→TEST(2026-01-01〜2026-07-28)検証で比較する。

注意: 絞り込み買いは必然的に的中率を下げうる（除外した馬が実際の3着馬だった
ケースを取りこぼす）。賭け金も同時に減るため、ROIが改善するかは「除外した
コンボが的中に占めていた割合」と「賭け金削減率」のどちらが勝るか次第。
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    S7_STAKE, s7_field_entropy, s7_select_axis, s7_wt_mark3_overlap_n, s7_wt_overlap_n,
)

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
    ("2026-04-13", "2026-07-28", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

S7_AXIS_SUM_MAX = 1.3
S7_ENTROPY_MAX = 1.8329
S7_MARK3_OVERLAP_MAX = 1

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20260728"

CACHE_PATH = (Path(__file__).resolve().parent.parent / "data" / "exp_cache"
              / "s7_reduced_box_candidates.pkl")


def _load_trio_boards(race_keys):
    import re
    trio = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio[rk][parts] = fv
    return trio


def build_candidates(model_name, date_from, date_to, win_model_name):
    """S7候補を構築し、絞り込み判定に必要な生の軸/WT印frame_noまで保持する。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
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

    out = []
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
        sel = s7_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = s7_field_entropy(top3_probs)
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
        wt_overlap_n = s7_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou)
        wt_mark3 = s7_wt_mark3_overlap_n(axis1, axis2, wt_honmei, wt_taikou, wt_ana)

        combo_odds = {}
        for x in others:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combo_odds[key] = trio[key]
        if not combo_odds:
            continue
        trio_pay = pm.get(rk, {}).get(("trio", actual_top3), 0)

        out.append({
            "race_key": rk, "axis1": axis1, "axis2": axis2, "others": others,
            "axis_sum": axis_sum, "entropy": entropy,
            "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3,
            "wt_honmei": wt_honmei, "wt_taikou": wt_taikou,
            "actual_top3": actual_top3, "combo_odds": combo_odds, "trio_pay": trio_pay,
        })
    return out


def build_or_load():
    if CACHE_PATH.exists():
        print(f"[cache] 読み込み: {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    all_cands = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cands = build_candidates(eval_model, date_from, date_to, win_model)
        print(f"[build]   candidates: {len(cands)}", flush=True)
        all_cands.extend(cands)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_cands, f)
    print(f"[cache] 保存: {CACHE_PATH}", flush=True)
    return all_cands


def d_filter(c):
    return (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
            and c["entropy"] <= S7_ENTROPY_MAX
            and (c["wt_mark3_overlap_n"] is not None
                 and c["wt_mark3_overlap_n"] <= S7_MARK3_OVERLAP_MAX))


def bet_original(c):
    """現行5点買い: axis1+axis2+others(5車ボックス)。"""
    combos = c["combo_odds"]
    n = len(combos)
    bet = n * S7_STAKE
    hit = c["actual_top3"] in combos
    pay = c["trio_pay"] * S7_STAKE // 100 if hit else 0
    return n, bet, hit, pay


def bet_reduced(c):
    """絞り込み買い: wt_overlap_n==1なら軸に含まれない側のWT印馬を、
    wt_overlap_n==0ならWT印2頭とも3列目候補から除外する。
    wt_overlap_n==2はD構成では既に除外されているため対象外(呼ばれない)。
    """
    exclude = set()
    if c["wt_overlap_n"] == 1:
        axis_set = {c["axis1"], c["axis2"]}
        marks = {c["wt_honmei"], c["wt_taikou"]} - {None}
        # 軸に含まれていない側のWT印馬を除外
        exclude = marks - axis_set
    elif c["wt_overlap_n"] == 0:
        exclude = {c["wt_honmei"], c["wt_taikou"]} - {None}

    combos = {k: v for k, v in c["combo_odds"].items()
              if not any(x in exclude for x in (k - {c["axis1"], c["axis2"]}))}
    n = len(combos)
    bet = n * S7_STAKE
    hit = c["actual_top3"] in combos
    pay = c["trio_pay"] * S7_STAKE // 100 if hit else 0
    return n, bet, hit, pay


def summarize(cands, bet_fn):
    n_race = 0
    total_n = total_bet = total_pay = total_hits = 0
    for c in cands:
        n, bet, hit, pay = bet_fn(c)
        if n == 0:
            continue
        n_race += 1
        total_n += n
        total_bet += bet
        total_pay += pay
        total_hits += int(hit)
    roi = total_pay / total_bet * 100 if total_bet else 0.0
    hitrate = total_hits / n_race * 100 if n_race else 0.0
    return n_race, total_hits, hitrate, total_bet, total_pay, roi


def main():
    print("候補プール構築中(既存キャッシュがあれば再利用)...")
    all_cands = build_or_load()
    print(f"[main] 全期間 raw candidates合計: {len(all_cands)}")

    d_sel = [c for c in all_cands if d_filter(c)]
    print(f"[main] D構成(現行本番定義): {len(d_sel)}件")

    train = [c for c in d_sel if TRAIN_FROM <= c["race_key"][:8] <= TRAIN_TO]
    test = [c for c in d_sel if TEST_FROM <= c["race_key"][:8] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    print("\n" + "=" * 78)
    print("D構成全体: 現行5点買い vs 絞り込み買い(4点/3点) 比較")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n1, h1, hr1, bet1, pay1, roi1 = summarize(data, bet_original)
        n2, h2, hr2, bet2, pay2, roi2 = summarize(data, bet_reduced)
        print(f"\n[{label}]")
        print(f"  現行5点買い     : n={n1} hit={hr1:.1f}% bet={bet1:,} pay={pay1:,} ROI={roi1:.1f}%")
        print(f"  絞り込み買い    : n={n2} hit={hr2:.1f}% bet={bet2:,} pay={pay2:,} ROI={roi2:.1f}%")

    print("\n" + "=" * 78)
    print("wt_overlap_n別の内訳（重なり1=4点化対象／重なり0=3点化対象）")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        for ov in (0, 1):
            sub = [c for c in data if c["wt_overlap_n"] == ov]
            n1, h1, hr1, bet1, pay1, roi1 = summarize(sub, bet_original)
            n2, h2, hr2, bet2, pay2, roi2 = summarize(sub, bet_reduced)
            tag = "SS(重なり0→3点化)" if ov == 0 else "S(重なり1→4点化)"
            print(f"  {tag:<20} 現行: n={n1:>5} hit={hr1:>5.1f}% ROI={roi1:>6.1f}%  |  "
                  f"絞り込み: n={n2:>5} hit={hr2:>5.1f}% ROI={roi2:>6.1f}%")


if __name__ == "__main__":
    main()
