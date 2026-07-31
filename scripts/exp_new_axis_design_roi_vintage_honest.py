"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

新軸設計(軸1=◎◯強い方・軸2=非マーク最強手)のROIを、DB格納pred_top3_pct/
pred_win_pctに依存せず、各四半期の凍結vintageモデルを直接再ロードして再計算する
（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

背景: `exp_new_axis_design_roi.py`/`exp_new_axis_gate_robustness.py`は
wt_entries.pred_top3_pct/pred_win_pctをDBから直接読み出して使ったが、
2026-04-13以降の"tail"期間はbackfill_index_pct_wt.pyという一度きりの
手動スクリプトでlgbm_wt_eval(週次で再学習されホールドアウト境界test_fromが
可動するモデル)により書き込まれたものであり、そのスクリプト実行時点(2026-07-19)
のtest_fromが実際に何だったか検証できないため、リーク混入の可能性を排除できない
（[[keirin_s7_foundational_rethink_2026_07_29]]参照）。

本スクリプトは`exp_s7_gate_staged_audit.py`/`rebuild_s7_walkforward_pg.py`と
同じ厳密な方式（各四半期の凍結vintageモデルを明示的にロードしてpredict_proba
し直す）で、新軸設計のROIを独立に計算し直す。DBのpred_top3_pct/pred_win_pct列は
一切参照しない。

軸1 = ◎◯のうちpred_win_pct(モデル再計算値)が高い方
軸2 = 非マーク5車のうちpred_top3_pct(モデル再計算値)最上位1車
gap1 = |pred_win_pct(◎)-pred_win_pct(◯)|
gap2 = 非マーク上位1位と2位のpred_top3_pct差
mark_sum = pred_top3_pct(◎)+pred_top3_pct(◯)
ゲート: mark_sum<=120 & gap1>=20 & gap2>=10（前回DB値ベースで選定した閾値）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

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
    # tail: 現行lgbm_wt_evalの2026-07-29時点test_fromは2026-04-27より後のはず。
    # 安全側に倒し、tail期間は2026-04-13〜2026-04-26のみをこのモデルで評価し、
    # それ以降(test_from以後の真のholdout区間)は別途確認する。
    ("2026-04-13", "2026-07-28", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

STAKE = 100
TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"

SEL_MARK_TH, SEL_GAP1_TH, SEL_GAP2_TH = 120, 20, 10


def _load_trio_boards(race_keys):
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
    df["pred_top3"] = model.predict_proba(X)[:, 1] * 100
    df["pred_win"] = win_model.predict_proba(X)[:, 1] * 100
    trio_bd = _load_trio_boards(df["race_key"].unique().tolist())

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
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        if wt_honmei is None or wt_taikou is None:
            continue
        row_by_frame = {int(r.frame_no): r for r in g.itertuples(index=False)}
        if wt_honmei not in row_by_frame or wt_taikou not in row_by_frame:
            continue
        h_win = float(row_by_frame[wt_honmei].pred_win)
        t_win = float(row_by_frame[wt_taikou].pred_win)
        h_top3 = float(row_by_frame[wt_honmei].pred_top3)
        t_top3 = float(row_by_frame[wt_taikou].pred_top3)
        mark_sum = h_top3 + t_top3
        gap1 = abs(h_win - t_win)
        axis1 = wt_honmei if h_win >= t_win else wt_taikou

        others_frames = [f for f in row_by_frame if f not in (wt_honmei, wt_taikou)]
        if len(others_frames) != 5:
            continue
        others_sorted = sorted(others_frames, key=lambda f: -float(row_by_frame[f].pred_top3))
        axis2 = others_sorted[0]
        gap2 = float(row_by_frame[others_sorted[0]].pred_top3) - float(row_by_frame[others_sorted[1]].pred_top3)

        box = sorted(set(row_by_frame.keys()) - {axis1, axis2})
        if len(box) != 5:
            continue
        combo_odds = {}
        for x in box:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combo_odds[key] = trio[key]
        if not combo_odds:
            continue

        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue
        actual_top3 = frozenset(fno for _, fno in fin[:3])
        hit = actual_top3 in combo_odds
        odds = combo_odds.get(actual_top3, 0)
        pay = int(odds * STAKE) if hit else 0
        bet = len(combo_odds) * STAKE

        out.append({
            "race_key": rk, "mark_sum": mark_sum, "gap1": gap1, "gap2": gap2,
            "hit": int(hit), "payout": pay, "bet": bet,
        })
    return out


def summarize(data):
    n = len(data)
    hits = sum(c["hit"] for c in data)
    bet = sum(c["bet"] for c in data)
    pay = sum(c["payout"] for c in data)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


def main():
    all_cands = []
    for date_from, date_to, eval_model, win_model in QUARTERS:
        print(f"[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cands = build_candidates(eval_model, date_from, date_to, win_model)
        print(f"[build]   candidates: {len(cands)}", flush=True)
        for c in cands:
            c["race_date"] = c["race_key"][:8]
        all_cands.extend(cands)

    print(f"\n[main] 全期間 raw candidates合計: {len(all_cands)}")

    train = [c for c in all_cands if TRAIN_FROM.replace("-", "") <= c["race_date"] <= TRAIN_TO.replace("-", "")]
    test = [c for c in all_cands if TEST_FROM.replace("-", "") <= c["race_date"] <= TEST_TO.replace("-", "")]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    print("\n" + "=" * 78)
    print("ゲートなしベースラインROI（vintage再計算・DB値不使用）")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n, hits, hitrate, bet, pay, roi = summarize(data)
        print(f"  [{label}] n={n} hit={hitrate:.1f}% ROI={roi:.1f}%")

    print("\n" + "=" * 78)
    print(f"選定ゲート: mark_sum<={SEL_MARK_TH} & gap1>={SEL_GAP1_TH} & gap2>={SEL_GAP2_TH}")
    print("=" * 78)

    def sel(data):
        return [c for c in data
                if c["mark_sum"] <= SEL_MARK_TH and c["gap1"] >= SEL_GAP1_TH
                and c["gap2"] >= SEL_GAP2_TH]

    for label, data in (("TRAIN", sel(train)), ("TEST", sel(test))):
        n, hits, hitrate, bet, pay, roi = summarize(data)
        mark = " ★100%超" if roi > 100 else ""
        print(f"  [{label}] n={n} hit={hitrate:.1f}% ROI={roi:.1f}%{mark}")

    print("\n" + "=" * 78)
    print("月次ROI推移（vintage再計算・DB値不使用）")
    print("=" * 78)
    sel_all = sel(train) + sel(test)
    by_month = defaultdict(list)
    for c in sel_all:
        ym = c["race_key"][:6]
        by_month[ym].append(c)
    print(f"{'年月':<10}{'n':>6}{'hit%':>8}{'ROI':>9}")
    for ym in sorted(by_month.keys()):
        data = by_month[ym]
        n, hits, hitrate, bet, pay, roi = summarize(data)
        mark = " ★" if roi > 100 else " ×"
        print(f"{ym:<10}{n:>6}{hitrate:>7.1f}%{roi:>8.1f}%{mark}")


if __name__ == "__main__":
    main()
