"""新軸設計(軸1=◎◯強い方・軸2=非マーク最強手)のROIを、月次凍結vintageモデル
（2026-07-29構築・書き込み保護済み・唯一の正本`src.wt_vintage_config`）で
再計算する（[[keirin_s7_foundational_rethink_2026_07_29]] /
[[keirin_wt_foundational_audit_2026_07_29]]）。

背景: 従来のDB格納値ベース(ROI139%/235%)とvintage再計算ベース(ROI76%/79%)の
大きな乖離は、四半期vintageモデル18本が2026-07-28にアドホック実験で無断
上書きされていたことが確定原因と判明した。今回構築した月次vintageモデル
（書き込み保護済み・全62モデルが2026-07-29に一度に構築され以後上書き不可）
を使い、この検証をクリーンな状態からやり直す。

軸1 = ◎◯のうちpred_win_pct(モデル再計算値)が高い方
軸2 = 非マーク5車のうちpred_top3_pct(モデル再計算値)最上位1車
gap1 = |pred_win_pct(◎)-pred_win_pct(◯)|
gap2 = 非マーク上位1位と2位のpred_top3_pct差
mark_sum = pred_top3_pct(◎)+pred_top3_pct(◯)
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.wt_vintage_config import monthly_windows

STAKE = 100
TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20261231"

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


def build_candidates(eval_model_name, date_from, date_to, win_model_name):
    model = load_model(eval_model_name)
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
    windows = monthly_windows()
    print(f"対象月数: {len(windows)}（{windows[0][0]}〜{windows[-1][1]}）")

    all_cands = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        cands = build_candidates(eval_model, date_from, date_to, win_model)
        print(f"[build]   candidates: {len(cands)}", flush=True)
        for c in cands:
            c["race_date"] = c["race_key"][:8]
        all_cands.extend(cands)

    print(f"\n[main] 全期間 raw candidates合計: {len(all_cands)}")

    train = [c for c in all_cands if TRAIN_FROM <= c["race_date"] <= TRAIN_TO]
    test = [c for c in all_cands if TEST_FROM <= c["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    print("\n" + "=" * 78)
    print("ゲートなしベースラインROI（月次vintageモデル・書き込み保護済み）")
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
    print("月次ROI推移")
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

    print("\n" + "=" * 78)
    print("閾値近傍でのROI感応度（頑健性確認）")
    print("=" * 78)
    print(f"{'mark<=':>8}{'gap1>=':>8}{'gap2>=':>8}{'TRAIN n/ROI':>16}{'TEST n/ROI':>16}")
    for mark_th in (110, 115, 120, 125, 130):
        for g1_th in (15, 20, 25):
            for g2_th in (5, 10, 15):
                tr = [c for c in train if c["mark_sum"] <= mark_th and c["gap1"] >= g1_th
                      and c["gap2"] >= g2_th]
                te = [c for c in test if c["mark_sum"] <= mark_th and c["gap1"] >= g1_th
                      and c["gap2"] >= g2_th]
                n1, h1, hr1, b1, p1, r1 = summarize(tr)
                n2, h2, hr2, b2, p2, r2 = summarize(te)
                if n1 < 100:
                    continue
                mark_disp = " ★★" if (r1 > 100 and r2 > 100) else (" ★" if r1 > 100 or r2 > 100 else "")
                print(f"{mark_th:>8}{g1_th:>8}{g2_th:>8}{n1:>8}/{r1:>6.1f}%{n2:>8}/{r2:>6.1f}%{mark_disp}")


if __name__ == "__main__":
    main()
