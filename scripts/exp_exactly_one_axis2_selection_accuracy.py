"""exactly_one(◎◯どちらか一方のみ3着内)における「軸2=◎◯以外の非マーク馬」
選定精度検証（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

新設計の軸1選定(pred_win_pct/pred_top3_pctが高い方を◎◯から選ぶ、TRAIN精度69%・
TEST精度67%で検証済み)に続き、軸2（◎◯以外の非マーク5車から選ぶ）の選定精度を
検証する。

exactly_oneのレースでは、的中側マーク(◎or◯どちらか)は必ず実際の3着以内に入り、
外れた側のマークは必ず3着圏外（定義上）。よって残る2枠は必ず「非マーク5車」の
中から出る（外れたマークは対象外なので実質は非マーク5車のみが候補）。

軸2選定基準: 非マーク5車のうちpred_top3_pct最上位の1車を選んだ場合、それが
実際の残り2枠(3着以内)のどちらかに一致する確率を検証する。
ランダム選択のベースライン(5車から2枠が正解=2/5=40%)と比較する。
"""
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"


def load_data():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.pred_win_pct, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.n_entries = 7 AND e.pred_top3_pct IS NOT NULL "
            "AND r.race_date >= :from_date",
            {"from_date": TRAIN_FROM}).fetchall()
    return rows


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    print("データ読み込み中(2024-01-01〜)...")
    rows = load_data()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)

    races = []
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
        if any(e["pred_top3_pct"] is None for e in others):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = honmei["frame_no"] in winners
        t_hit = taikou["frame_no"] in winners
        if h_hit == t_hit:
            continue  # exactly_oneのみ対象

        # 非マーク5車のうち、実際に3着以内に入っている車(必ず2車のはず)
        other_hits = frozenset(e["frame_no"] for e in others if e["frame_no"] in winners)

        top3_ranked = sorted(others, key=lambda e: -float(e["pred_top3_pct"]))
        win_ranked = sorted(
            others,
            key=lambda e: -(float(e["pred_win_pct"]) if e["pred_win_pct"] is not None else -1))

        races.append({
            "race_key": rk, "race_date": race_date,
            "other_hits": other_hits,
            "top3_pick": top3_ranked[0]["frame_no"],
            "top3_pick2": top3_ranked[1]["frame_no"] if len(top3_ranked) > 1 else None,
            "win_pick": win_ranked[0]["frame_no"],
        })

    print(f"  exactly_one対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def eval_pick(data, key, label):
        n = len(data)
        correct = sum(1 for r in data if r[key] in r["other_hits"])
        acc = correct / n * 100 if n else 0.0
        lo, hi = wilson_ci(correct, n)
        print(f"  [{label}] n={n} 精度={acc:.1f}% (95%CI {lo*100:.1f}%-{hi*100:.1f}%) "
              f"※ランダム選択ベースライン=40.0%")

    print("\n" + "=" * 70)
    print("軸2選定: 非マーク5車のうちpred_top3_pct最上位1車を選んだ場合の一致率")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eval_pick(data, "top3_pick", label)

    print("\n" + "=" * 70)
    print("軸2選定: 非マーク5車のうちpred_win_pct最上位1車を選んだ場合の一致率")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eval_pick(data, "win_pick", label)

    print("\n" + "=" * 70)
    print("参考: pred_top3_pct上位2車のうち少なくとも1車が一致する率(2択なら精度が上がるはず)")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n = len(data)
        correct = sum(1 for r in data
                      if (r["top3_pick"] in r["other_hits"])
                      or (r["top3_pick2"] is not None and r["top3_pick2"] in r["other_hits"]))
        acc = correct / n * 100 if n else 0.0
        print(f"  [{label}] n={n} 精度={acc:.1f}% ※ランダム(2/5選択で少なくとも1つ一致)基準=約77.1%")


if __name__ == "__main__":
    main()
