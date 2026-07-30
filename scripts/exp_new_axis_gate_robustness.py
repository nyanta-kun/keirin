"""新設計ゲート(mark_sum<=120 & gap1>=20 & gap2>=10)の頑健性チェック
（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

前段でTRAIN選定→TEST検証によりROI139.1%(TRAIN,n=2035)/235.2%(TEST,n=244)という
これまでで最も有望な結果を得た。本スクリプトは以下の頑健性を確認する:

1. TEST的中の内訳（一部の大穴的中がROIを支配していないか、集中リスクの確認）
2. TRAIN/TESTを月次に分割した際のROI推移（安定して黒字か、特定の月だけの偶然か）
3. 閾値を少し動かした近傍（mark_sum/gap1/gap2）でROIが急激に崩れないか
   （崖のような単一点でのみ機能する多重比較ノイズでないかの確認）
4. 1日あたりの該当件数分布（実運用上の頻度感）
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

SEL_MARK_TH, SEL_GAP1_TH, SEL_GAP2_TH = 120, 20, 10


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


def build_candidates():
    print("データ読み込み中(2024-01-01〜)...")
    rows = load_entries()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)

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

        axis1 = honmei["frame_no"] if h_win >= t_win else taikou["frame_no"]

        top3_ranked = sorted(others, key=lambda e: -float(e["pred_top3_pct"]))
        axis2 = top3_ranked[0]["frame_no"]
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
            "odds": odds if hit else None,
        })
    print(f"  解析対象レース数: {len(cands)}")
    return cands


def summarize(data):
    n = len(data)
    hits = sum(c["hit"] for c in data)
    bet = sum(c["bet"] for c in data)
    pay = sum(c["payout"] for c in data)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


def main():
    cands = build_candidates()
    train = [c for c in cands if TRAIN_FROM <= c["race_date"] <= TRAIN_TO]
    test = [c for c in cands if TEST_FROM <= c["race_date"] <= TEST_TO]

    def sel(data):
        return [c for c in data
                if c["mark_sum"] <= SEL_MARK_TH and c["gap1"] >= SEL_GAP1_TH
                and c["gap2"] >= SEL_GAP2_TH]

    sel_train, sel_test = sel(train), sel(test)

    print("\n" + "=" * 78)
    print(f"選定ゲート: mark_sum<={SEL_MARK_TH} & gap1>={SEL_GAP1_TH} & gap2>={SEL_GAP2_TH}")
    print("=" * 78)
    for label, data in (("TRAIN", sel_train), ("TEST", sel_test)):
        n, hits, hitrate, bet, pay, roi = summarize(data)
        print(f"  [{label}] n={n} hit={hitrate:.1f}% ROI={roi:.1f}%")

    print("\n" + "=" * 78)
    print("1. TEST的中の内訳（配当降順・集中リスク確認）")
    print("=" * 78)
    hits_test = sorted([c for c in sel_test if c["hit"]], key=lambda c: -c["odds"])
    total_pay = sum(c["payout"] for c in sel_test)
    print(f"{'race_key':<20}{'odds':>8}{'payout':>10}{'累積%':>8}")
    cum = 0
    for c in hits_test:
        cum += c["payout"]
        print(f"{c['race_key']:<20}{c['odds']:>7.1f}倍{c['payout']:>10,}{cum/total_pay*100:>7.1f}%")
    print(f"的中件数: {len(hits_test)} / 総payout: {total_pay:,}")

    print("\n" + "=" * 78)
    print("2. 月次ROI推移（TRAIN・TEST通し）")
    print("=" * 78)
    by_month = defaultdict(list)
    for c in sel_train + sel_test:
        ym = c["race_date"][:7]
        by_month[ym].append(c)
    print(f"{'年月':<10}{'n':>6}{'hit%':>8}{'ROI':>9}")
    for ym in sorted(by_month.keys()):
        data = by_month[ym]
        n, hits, hitrate, bet, pay, roi = summarize(data)
        mark = " ★" if roi > 100 else ("  " if n == 0 else " ×")
        print(f"{ym:<10}{n:>6}{hitrate:>7.1f}%{roi:>8.1f}%{mark}")

    print("\n" + "=" * 78)
    print("3. 閾値の近傍でのROI感応度（崖状の脆弱性がないか）")
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

    print("\n" + "=" * 78)
    print("4. 1日あたりの該当件数分布")
    print("=" * 78)
    by_day = defaultdict(int)
    for c in sel_train + sel_test:
        by_day[c["race_date"]] += 1
    counts = sorted(by_day.values())
    n_days_total = (46359 // 1)  # placeholder, will compute actual span below
    all_dates_train = sorted({c["race_date"] for c in train})
    all_dates_test = sorted({c["race_date"] for c in test})
    all_dates = set(all_dates_train) | set(all_dates_test)
    n_days_with_data = len(all_dates)
    n_days_with_hit = len(by_day)
    print(f"データが存在する日数: {n_days_with_data}")
    print(f"該当あり日数: {n_days_with_hit} ({n_days_with_hit/n_days_with_data*100:.1f}%)")
    print(f"平均件数/日(該当日のみ): {sum(counts)/len(counts):.2f}")
    print(f"平均件数/日(全日数ベース): {sum(counts)/n_days_with_data:.2f}")
    from collections import Counter
    dist = Counter(counts)
    print("件数分布(該当日のみ):")
    for k in sorted(dist.keys()):
        print(f"  {k}件/日: {dist[k]}日")


if __name__ == "__main__":
    main()
