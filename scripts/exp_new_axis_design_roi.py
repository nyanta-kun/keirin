"""新設計(軸1=◎◯の強い方・軸2=非マーク最強手)のhonest ROI検証 +
race_point(競争得点)とpctの乖離("値ごろ"仮説)の検証（2026-07-29・
[[keirin_s7_foundational_rethink_2026_07_29]]）。

これまでの検証で確認済み:
  - 軸1(◎◯のうちpred_win_pctが高い方)選定精度: TRAIN69.1%/TEST67.4%
  - 軸2(非マーク5車のうちpred_top3_pct最上位)選定精度: TRAIN64.4-65.2%/TEST60.9-62.3%
  - gap(候補間の確率差)が精度と強く相関(拮抗=コイントス、突出=90%超)
  - 軸1・軸2の的中はほぼ独立

本スクリプトは、この新設計の実際の配当込みROIを計算する。LightGBMモデルの
再実行は不要（wt_entries.pred_top3_pct/pred_win_pctは既に格納済みの発走前
予測値をそのまま使う）。

買い目: axis1(◎◯の強い方) + axis2(非マーク5車の最強手) + 残り5車ボックス = 5点。
ゲート: mark_sum(both除外)・gap1(軸1確信度)・gap2(軸2確信度)をTRAIN選定→
TEST一度きり検証。

あわせてユーザー仮説「pctが高いが競争得点(race_point)が低い＝市場が過小評価
している可能性」を、実際の的中時配当と紐づけて検証する。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
STAKE = 100


def load_entries():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.pred_win_pct, e.race_point, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.n_entries = 7 AND e.pred_top3_pct IS NOT NULL "
            "AND r.race_date >= :from_date",
            {"from_date": TRAIN_FROM}).fetchall()
    return rows


def load_trio_boards(race_keys):
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


def main():
    print("データ読み込み中(2024-01-01〜)...")
    rows = load_entries()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)
    print(f"  レース数(候補): {len(by_race)}")

    race_keys = list(by_race.keys())
    trio_bd = load_trio_boards(race_keys)

    cands = []
    for rk, ents in by_race.items():
        if len(ents) != 7:
            continue
        race_date = str(ents[0]["race_date"])
        honmei = next((e for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((e for e in ents if e["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        others = [e for e in ents if e["frame_no"] not in (honmei["frame_no"], taikou["frame_no"])]
        if len(others) != 5:
            continue
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
            continue
        if honmei["pred_win_pct"] is None or taikou["pred_win_pct"] is None:
            continue
        if any(e["pred_top3_pct"] is None for e in others):
            continue

        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue

        h_win, t_win = float(honmei["pred_win_pct"]), float(taikou["pred_win_pct"])
        h_top3, t_top3 = float(honmei["pred_top3_pct"]), float(taikou["pred_top3_pct"])
        mark_sum = h_top3 + t_top3
        gap1 = abs(h_win - t_win)

        if h_win >= t_win:
            axis1, axis1_rp, axis1_pct = honmei["frame_no"], honmei["race_point"], h_top3
        else:
            axis1, axis1_rp, axis1_pct = taikou["frame_no"], taikou["race_point"], t_top3

        top3_ranked = sorted(others, key=lambda e: -float(e["pred_top3_pct"]))
        axis2 = top3_ranked[0]["frame_no"]
        axis2_rp = top3_ranked[0]["race_point"]
        axis2_pct = float(top3_ranked[0]["pred_top3_pct"])
        gap2 = float(top3_ranked[0]["pred_top3_pct"]) - float(top3_ranked[1]["pred_top3_pct"])

        box = sorted({e["frame_no"] for e in ents} - {axis1, axis2})
        if len(box) != 5:
            continue
        combo_odds = {}
        for x in box:
            key = frozenset({axis1, axis2, x})
            if key in trio:
                combo_odds[key] = trio[key]
        if not combo_odds:
            continue

        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        actual_top3 = frozenset(fno for _, fno in fin[:3])
        hit = actual_top3 in combo_odds
        odds = combo_odds.get(actual_top3, 0)
        pay = int(odds * STAKE) if hit else 0
        bet = len(combo_odds) * STAKE

        cands.append({
            "race_key": rk, "race_date": race_date, "mark_sum": mark_sum,
            "gap1": gap1, "gap2": gap2, "hit": int(hit), "payout": pay, "bet": bet,
            "axis1_rp": axis1_rp, "axis1_pct": axis1_pct,
            "axis2_rp": axis2_rp, "axis2_pct": axis2_pct,
        })

    print(f"  解析対象レース数: {len(cands)}")
    train = [c for c in cands if TRAIN_FROM <= c["race_date"] <= TRAIN_TO]
    test = [c for c in cands if TEST_FROM <= c["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def summarize(data):
        n = len(data)
        hits = sum(c["hit"] for c in data)
        bet = sum(c["bet"] for c in data)
        pay = sum(c["payout"] for c in data)
        roi = pay / bet * 100 if bet else 0.0
        hitrate = hits / n * 100 if n else 0.0
        return n, hits, hitrate, bet, pay, roi

    print("\n" + "=" * 78)
    print("1. ゲートなしベースラインROI（軸1=◎◯の強い方 + 軸2=非マーク最強手 + 5車ボックス）")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n, hits, hitrate, bet, pay, roi = summarize(data)
        print(f"  [{label}] n={n} hit={hitrate:.1f}% bet={bet:,} pay={pay:,} ROI={roi:.1f}%")

    print("\n" + "=" * 78)
    print("2. mark_sum(both除外) x gap1 x gap2 ゲートのグリッドサーチ（TRAIN）")
    print("=" * 78)
    grid = []
    for mark_th in (999, 150, 140, 130, 120):
        for g1_th in (0, 5, 10, 20):
            for g2_th in (0, 5, 10):
                sub = [c for c in train
                       if c["mark_sum"] <= mark_th and c["gap1"] >= g1_th and c["gap2"] >= g2_th]
                n, hits, hitrate, bet, pay, roi = summarize(sub)
                if n < 200:
                    continue
                grid.append((mark_th, g1_th, g2_th, n, hitrate, roi))
    grid.sort(key=lambda r: -r[5])
    print(f"{'mark<=':>8}{'gap1>=':>8}{'gap2>=':>8}{'n':>8}{'hit%':>8}{'ROI':>9}")
    for mark_th, g1_th, g2_th, n, hitrate, roi in grid[:15]:
        mark = " ★" if roi > 100 else ""
        print(f"{mark_th:>8}{g1_th:>8}{g2_th:>8}{n:>8}{hitrate:>7.1f}%{roi:>8.1f}%{mark}")

    best = grid[0]
    mark_th, g1_th, g2_th, n_tr, hr_tr, roi_tr = best
    print(f"\n[選定] TRAIN最良: mark_sum<={mark_th} & gap1>={g1_th} & gap2>={g2_th} "
          f"(n={n_tr}, ROI={roi_tr:.1f}%)")
    sub_test = [c for c in test
                if c["mark_sum"] <= mark_th and c["gap1"] >= g1_th and c["gap2"] >= g2_th]
    n, hits, hitrate, bet, pay, roi = summarize(sub_test)
    print(f"[TEST検証] n={n} hit={hitrate:.1f}% ROI={roi:.1f}%")

    print("\n" + "=" * 78)
    print("3. race_point(競争得点)とpctの乖離仮説の検証")
    print("   軸1・軸2それぞれについて、race_pointの低いグループとpctの関係、"
          "\n   および実際の的中配当との関係を確認")
    print("=" * 78)
    for axis_label, rp_key, pct_key in (("軸1", "axis1_rp", "axis1_pct"),
                                          ("軸2", "axis2_rp", "axis2_pct")):
        print(f"\n--- {axis_label} ---")
        valid = [c for c in train if c[rp_key] is not None]
        if not valid:
            continue
        rps = sorted(float(c[rp_key]) for c in valid)
        med_rp = rps[len(rps) // 2]
        low_rp = [c for c in valid if float(c[rp_key]) < med_rp]
        high_rp = [c for c in valid if float(c[rp_key]) >= med_rp]
        print(f"  race_point中央値: {med_rp:.1f}")
        for grp_label, grp in (("race_point低群", low_rp), ("race_point高群", high_rp)):
            n, hits, hitrate, bet, pay, roi = summarize(grp)
            avg_pct = sum(c[pct_key] for c in grp) / len(grp)
            print(f"    {grp_label}: n={n} 平均pct={avg_pct:.1f} 的中率={hitrate:.1f}% ROI={roi:.1f}%")


if __name__ == "__main__":
    main()
