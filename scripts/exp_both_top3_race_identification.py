"""◎◯両方3着内レースの三連複配当詳細 + 追加特徴での識別精度検証（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]の続き。既に以下は確定済み:
  - ◎◯両方3着内レース(n=30,173・全7車立ての50.1%)の三連複配当は中央値4.2倍・
    平均7.1倍・5倍未満(損益分岐未達)が57.6%（`exp_wt_honmei_taikou_both_top3_payout_dist.py`）
  - mark_sum(=pred_top3_pct(◎)+pred_top3_pct(◯))はboth_top3率の強い事前予測シグナル
    （TRAIN/TESTでほぼ完全再現、100-110区間24.8%→170-200区間80.7%の単調な用量反応。
    `exp_honmei_taikou_both_top3_predict.py`）

本スクリプトは「mark_sum単体でどこまでレースを特定できるか（識別精度）」と
「mark_sumに他の発走前情報（win_sum・同ライン・フィールド全体のentropy・グレード）
を組み合わせるとさらに識別力が上がるか」を honest TRAIN(2024-01-01〜2025-12-31)/
TEST(2026-01-01〜)分割で検証する。

追加特徴:
  - win_sum: pred_win_pct(◎)+pred_win_pct(◯)（単勝側の合算、複勝側mark_sumと別角度）
  - same_line: ◎◯が同じline_groupか
  - field_entropy: 出走7車全体のpred_top3_pctから計算したエントロピー（拮抗度）
  - grade: レースグレード（S級/A級等）
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"


def load_data():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, r.grade, e.frame_no, e.prediction_mark, "
            "e.pred_top3_pct, e.pred_win_pct, e.line_group, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.n_entries = 7 AND e.pred_top3_pct IS NOT NULL "
            "AND r.race_date >= :from_date",
            {"from_date": TRAIN_FROM}).fetchall()
    return rows


def load_trio_odds(race_keys):
    import re
    out = {}
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
                    out.setdefault(rk, {})[parts] = fv
    return out


def field_entropy(pcts):
    total = sum(pcts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in pcts:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def pctile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
    return sorted_vals[idx]


def main():
    print("データ読み込み中(2024-01-01〜・pred_top3_pct格納済みのみ)...")
    rows = load_data()
    print(f"  entries行数: {len(rows)}")

    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)

    race_keys = list(by_race.keys())
    trio_odds = load_trio_odds(race_keys)

    races = []
    for rk, ents in by_race.items():
        if len(ents) != 7:
            continue
        race_date = str(ents[0]["race_date"])
        grade = ents[0]["grade"]
        honmei = next((e for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((e for e in ents if e["prediction_mark"] == 2), None)
        if honmei is None or taikou is None:
            continue
        if honmei["pred_top3_pct"] is None or taikou["pred_top3_pct"] is None:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        both_top3 = int((honmei["frame_no"] in winners) and (taikou["frame_no"] in winners))
        trio = trio_odds.get(rk)
        odds = trio.get(winners) if trio else None

        # mark_sum/win_sum は pred_top3_pct/pred_win_pct のパーセント表記(0-100)の
        # まま合算する（0-200レンジ）。exp_honmei_taikou_both_top3_predict.py の
        # 閾値グリッド(100-200)と単位を揃えるため、ここでは/100しない
        # （前段のexp_s7_axis_mark_divergence.pyで axis_sum(0-1確率)とmark_sumを
        # 比較する際は/100が必要だったが、本スクリプトは単独のpct同士の合算なので不要）。
        mark_sum = float(honmei["pred_top3_pct"]) + float(taikou["pred_top3_pct"])
        win_sum = None
        if honmei["pred_win_pct"] is not None and taikou["pred_win_pct"] is not None:
            win_sum = float(honmei["pred_win_pct"]) + float(taikou["pred_win_pct"])
        same_line = bool(honmei["line_group"] is not None
                          and honmei["line_group"] == taikou["line_group"])
        pcts = [float(e["pred_top3_pct"]) for e in ents]
        ent = field_entropy(pcts)

        races.append({
            "race_key": rk, "race_date": race_date, "grade": grade,
            "mark_sum": mark_sum, "win_sum": win_sum, "same_line": same_line,
            "entropy": ent, "both_top3": both_top3, "trio_odds": odds,
        })

    print(f"  解析対象レース数: {len(races)}")
    train = [r for r in races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    # ===== 1. ◎◯両方3着内レースの配当統計（再掲・詳細版） =====
    print("\n" + "=" * 70)
    print("1. ◎◯両方3着内レースの三連複配当統計")
    print("=" * 70)
    for label, data in (("全期間", races), ("TRAIN", train), ("TEST", test)):
        both = [r for r in data if r["both_top3"] and r["trio_odds"] is not None]
        vals = sorted(r["trio_odds"] for r in both)
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        print(f"\n[{label}] n={n}")
        print(f"  平均={mean:.2f}倍  中央値={pctile(vals,50):.2f}倍  "
              f"p25={pctile(vals,25):.2f}倍  p75={pctile(vals,75):.2f}倍  "
              f"p90={pctile(vals,90):.2f}倍  p95={pctile(vals,95):.2f}倍")
        breakeven = sum(1 for v in vals if v >= 5)
        print(f"  5倍未満(損益分岐未達)率: {(n-breakeven)/n*100:.1f}%")

    # ===== 2. mark_sum単体の識別精度（precision/recall） =====
    print("\n" + "=" * 70)
    print("2. mark_sum閾値ごとの識別精度（'両方3着内→除外対象'として）")
    print("=" * 70)
    for label, data in (("TRAIN", train), ("TEST", test)):
        print(f"\n[{label}]")
        total_both = sum(r["both_top3"] for r in data)
        print(f"{'閾値(以上を除外)':<18}{'除外数':>8}{'除外的中率(precision)':>14}"
              f"{'再現率(recall)':>14}{'除外側平均配当':>14}{'残存側平均配当':>14}")
        for th in [110, 120, 130, 140, 150, 160]:
            flagged = [r for r in data if r["mark_sum"] >= th]
            kept = [r for r in data if r["mark_sum"] < th]
            n_f = len(flagged)
            if n_f == 0:
                continue
            prec = sum(r["both_top3"] for r in flagged) / n_f * 100
            recall = sum(r["both_top3"] for r in flagged) / total_both * 100 if total_both else 0
            f_odds = [r["trio_odds"] for r in flagged if r["trio_odds"] is not None]
            k_odds = [r["trio_odds"] for r in kept if r["trio_odds"] is not None]
            f_mean = sum(f_odds) / len(f_odds) if f_odds else 0
            k_mean = sum(k_odds) / len(k_odds) if k_odds else 0
            print(f">={th:<16}{n_f:>8}{prec:>13.1f}%{recall:>13.1f}%{f_mean:>13.1f}倍{k_mean:>13.1f}倍")

    # ===== 3. mark_sumに他特徴を組み合わせた場合の追加識別力 =====
    print("\n" + "=" * 70)
    print("3. mark_sum固定帯の中で追加特徴が識別力を持つか（TRAINのみで探索）")
    print("=" * 70)

    def bucket_report(data, cond_true, cond_false, label_t, label_f, mark_lo, mark_hi):
        band = [r for r in data if mark_lo <= r["mark_sum"] < mark_hi]
        t = [r for r in band if cond_true(r)]
        f = [r for r in band if cond_false(r)]
        for lab, sub in ((label_t, t), (label_f, f)):
            n = len(sub)
            if n == 0:
                continue
            bt3 = sum(r["both_top3"] for r in sub) / n * 100
            print(f"    mark_sum[{mark_lo}-{mark_hi}) {lab:<12} n={n:>6} both_top3率={bt3:>5.1f}%")

    print("\n[TRAIN] same_line（◎◯が同一ラインか）で細分化:")
    for lo, hi in [(100, 130), (130, 150), (150, 170), (170, 200)]:
        bucket_report(train, lambda r: r["same_line"], lambda r: not r["same_line"],
                      "同一ライン", "別ライン", lo, hi)

    print("\n[TRAIN] win_sum（単勝側合算）で細分化（中央値で2分割・帯ごと）:")
    for lo, hi in [(100, 130), (130, 150), (150, 170), (170, 200)]:
        band = [r for r in train if lo <= r["mark_sum"] < hi and r["win_sum"] is not None]
        if len(band) < 20:
            continue
        ws = sorted(r["win_sum"] for r in band)
        med = pctile(ws, 50)
        bucket_report(band, lambda r, m=med: r["win_sum"] >= m,
                      lambda r, m=med: r["win_sum"] < m,
                      "win_sum高", "win_sum低", lo, hi)

    print("\n[TRAIN] entropy（フィールド全体拮抗度）で細分化（中央値で2分割・帯ごと）:")
    for lo, hi in [(100, 130), (130, 150), (150, 170), (170, 200)]:
        band = [r for r in train if lo <= r["mark_sum"] < hi]
        if len(band) < 20:
            continue
        es = sorted(r["entropy"] for r in band)
        med = pctile(es, 50)
        bucket_report(band, lambda r, m=med: r["entropy"] >= m,
                      lambda r, m=med: r["entropy"] < m,
                      "entropy高", "entropy低", lo, hi)

    print("\n[TRAIN] grade（S級/A級）で細分化:")
    for lo, hi in [(100, 130), (130, 150), (150, 170), (170, 200)]:
        bucket_report(train, lambda r: r["grade"] == "S級", lambda r: r["grade"] == "A級",
                      "S級", "A級", lo, hi)

    # ===== 4. 複合スコア(mark_sum + win_sum)の識別力確認 =====
    print("\n" + "=" * 70)
    print("4. 複合スコア combo = mark_sum + win_sum の識別精度（honest TRAIN選定→TEST検証）")
    print("=" * 70)
    train_c = [r for r in train if r["win_sum"] is not None]
    test_c = [r for r in test if r["win_sum"] is not None]
    for r in train_c + test_c:
        r["combo"] = r["mark_sum"] + r["win_sum"]

    def spearman(data, key):
        n = len(data)
        ranks = {i: v for i, v in enumerate(sorted(data, key=lambda r: r[key]))}
        rank_of = {id(v): i for i, v in ranks.items()}
        xs = [rank_of[id(r)] for r in data]
        ys = [r["both_top3"] for r in data]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        return cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else 0.0

    print(f"順位相関(TRAIN): mark_sum={spearman(train_c,'mark_sum'):.3f} "
          f"win_sum={spearman(train_c,'win_sum'):.3f} combo={spearman(train_c,'combo'):.3f}")
    print(f"順位相関(TEST):  mark_sum={spearman(test_c,'mark_sum'):.3f} "
          f"win_sum={spearman(test_c,'win_sum'):.3f} combo={spearman(test_c,'combo'):.3f}")


if __name__ == "__main__":
    main()
