"""exactly_one(◎◯どちらか一方のみ3着内)における「軸1=どちらを選ぶか」の
事前判定精度検証（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

ユーザー提案の新設計: 軸1=◎◯のいずれか(1着を狙える方=win確率が高い方を優先)、
軸2=◎◯以外、bothは除外。本スクリプトはその最初の検証項目:
「exactly_oneが発生したレースにおいて、◎◯のうち事前のpred_win_pct
(またはpred_top3_pct)が高い方を軸1として選んだ場合、それが実際に的中した
側と一致する確率」を確認する。コイントス(50%)を明確に上回らなければ
軸1選定基準として優位性がない。

対象: n_entries=7・pred_win_pct/pred_top3_pct格納済み(2024-01-01〜)・
exactly_one(◎◯のうち片方のみ実際に3着内)のレースのみ。
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
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
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
            continue  # both/neitherは対象外。exactly_oneのみ扱う
        hit_is_honmei = h_hit  # True=◎が的中側, False=◯が的中側

        h_top3 = float(honmei["pred_top3_pct"])
        t_top3 = float(taikou["pred_top3_pct"])
        h_win = float(honmei["pred_win_pct"]) if honmei["pred_win_pct"] is not None else None
        t_win = float(taikou["pred_win_pct"]) if taikou["pred_win_pct"] is not None else None

        races.append({
            "race_key": rk, "race_date": race_date,
            "hit_is_honmei": hit_is_honmei,
            "h_top3": h_top3, "t_top3": t_top3, "h_win": h_win, "t_win": t_win,
        })

    print(f"  exactly_one対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    def eval_rule(data, key_h, key_t, label):
        n = tie = correct = 0
        for r in data:
            vh, vt = r[key_h], r[key_t]
            if vh is None or vt is None:
                continue
            n += 1
            if vh == vt:
                tie += 1
                continue
            predicted_honmei = vh > vt  # True予測=◎が的中すると予測
            if predicted_honmei == r["hit_is_honmei"]:
                correct += 1
        decided = n - tie
        acc = correct / decided * 100 if decided else 0.0
        lo, hi = wilson_ci(correct, decided)
        print(f"  [{label}] n={n} (同値={tie}) 精度={acc:.1f}% "
              f"(95%CI {lo*100:.1f}%-{hi*100:.1f}%)  ※コイントス=50%")
        return acc

    print("\n" + "=" * 70)
    print("軸1選定基準: pred_top3_pctが高い方を的中側と予測")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eval_rule(data, "h_top3", "t_top3", label)

    print("\n" + "=" * 70)
    print("軸1選定基準: pred_win_pctが高い方(1着を狙える方)を的中側と予測")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eval_rule(data, "h_win", "t_win", label)

    # 参考: 常に◎を選ぶ / 常に◯を選ぶ の精度も比較(基準線)
    print("\n" + "=" * 70)
    print("参考: 固定ルール（常に◎を軸1にする／常に◯を軸1にする）の精度")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n = len(data)
        honmei_acc = sum(1 for r in data if r["hit_is_honmei"]) / n * 100
        taikou_acc = 100 - honmei_acc
        print(f"  [{label}] 常に◎: {honmei_acc:.1f}%  常に◯: {taikou_acc:.1f}%")


if __name__ == "__main__":
    main()
