"""7車立てレースにおける「崖(gap)パターン」分析。

wt_entries.pred_win_pct(1着率%) / pred_top3_pct(3着内率%) を降順ソートし、
隣接差 g12=w1-w2, g23=w2-w3, g34=w3-w4 のうちどこが最大かで4パターンに分類する:
  - 1車突出: g12が最大
  - 2車突出: g23が最大
  - 3車突出: g34が最大
  - 全体拮抗: max(g12,g23,g34) が全レース分布の下位25%点未満（上記3分類より先に判定）

分析1: パターン別の三連複配当分布（実際の勝ち組み合わせのodds_value）
分析2: 突出馬が実際に3着内へ来るか（軸選定の妥当性）
分析3: パターン別の「3着内3頭」の予測難易度（pred_top3_pct順位）

DB格納値(wt_entries.pred_win_pct/pred_top3_pct, 2026-07-30凍結vintage再計算済み)
をそのまま使う高速版。モデル再ロードは行わない。

対象: wt_races.n_entries=7 かつ cancel=0。
TRAIN: 2024-01-01〜2025-12-31 / TEST: 2026-01-01〜2026-07-30

読み取り専用（SELECTのみ）。DBへの書き込みは一切行わない。
"""
import re
import statistics
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20260730"


def load_races(date_from, date_to):
    """対象期間の7車立てレースを読み込み、各レースについて
    win/top3 の降順配列・player_id対応・finish_order・race_dateを返す。
    """
    with get_connection() as c:
        race_rows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 "
            "AND race_date BETWEEN ? AND ?",
            (date_from, date_to),
        ).fetchall()
        race_keys = [r["race_key"] for r in race_rows]
        race_date_map = {r["race_key"]: r["race_date"] for r in race_rows}

        entries_by_race = defaultdict(list)
        CHUNK = 900
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            q = (
                "SELECT race_key, frame_no, player_id, pred_win_pct, pred_top3_pct, "
                "finish_order FROM wt_entries WHERE race_key IN (%s)"
                % ",".join("?" * len(chunk))
            )
            for row in c.execute(q, chunk):
                entries_by_race[row["race_key"]].append(row)

    races = []
    for rk in race_keys:
        ents = entries_by_race.get(rk, [])
        if len(ents) != 7:
            continue
        if any(e["pred_win_pct"] is None or e["pred_top3_pct"] is None for e in ents):
            continue
        races.append({
            "race_key": rk,
            "race_date": race_date_map[rk],
            "entries": ents,
        })
    return races


def load_trio_odds(race_keys):
    """race_key -> 実際の勝ち組み合わせ(finish_order 1-3のframe_no集合)に対応する
    三連複odds_valueを返す。finish_orderは呼び出し側のracesデータから求める必要は
    なく、ここではwt_entriesから再取得して突合する。
    """
    fins_by_race = {}
    with get_connection() as c:
        CHUNK = 900
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            q = (
                "SELECT race_key, frame_no, finish_order FROM wt_entries "
                "WHERE race_key IN (%s)" % ",".join("?" * len(chunk))
            )
            for row in c.execute(q, chunk):
                fo = row["finish_order"]
                if fo is not None and fo >= 1:
                    fins_by_race.setdefault(row["race_key"], []).append(
                        (fo, int(row["frame_no"]))
                    )

        winners_by_race = {}
        for rk, fin in fins_by_race.items():
            if len(fin) < 3:
                continue
            top3 = sorted(fin)[:3]
            winners_by_race[rk] = frozenset(fno for _, fno in top3)

        odds_map = {}
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            q = (
                "SELECT race_key, combination, odds_value FROM wt_odds "
                "WHERE bet_type = 'trio' AND race_key IN (%s)"
                % ",".join("?" * len(chunk))
            )
            trio_by_race = defaultdict(dict)
            for row in c.execute(q, chunk):
                od = row["odds_value"]
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(row["combination"])))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio_by_race[row["race_key"]][parts] = fv
            for rk in chunk:
                winners = winners_by_race.get(rk)
                if winners is None:
                    continue
                odds = trio_by_race.get(rk, {}).get(winners)
                if odds is not None:
                    odds_map[rk] = odds
    return odds_map, winners_by_race


def analyze_race(race):
    ents = race["entries"]
    # win降順 (player_id込みでソート)
    win_sorted = sorted(ents, key=lambda e: -float(e["pred_win_pct"]))
    top3_sorted = sorted(ents, key=lambda e: -float(e["pred_top3_pct"]))

    w = [float(e["pred_win_pct"]) for e in win_sorted]
    g12 = w[0] - w[1]
    g23 = w[1] - w[2]
    g34 = w[2] - w[3]

    finish_by_pid = {e["player_id"]: e["finish_order"] for e in ents}

    def in_top3(pid):
        fo = finish_by_pid.get(pid)
        return fo is not None and 1 <= fo <= 3

    # top3側の順位 (1-indexed) を player_id -> rank にマップ
    top3_rank = {e["player_id"]: i + 1 for i, e in enumerate(top3_sorted)}
    # 実際の3着内3頭のtop3_rankリスト
    actual_top3_pids = [e["player_id"] for e in ents if in_top3(e["player_id"])]
    actual_top3_ranks = sorted(top3_rank[pid] for pid in actual_top3_pids) if len(actual_top3_pids) == 3 else None

    return {
        "race_key": race["race_key"],
        "g12": g12, "g23": g23, "g34": g34,
        "win_sorted_pids": [e["player_id"] for e in win_sorted],
        "in_top3": in_top3,
        "actual_top3_ranks_by_top3pct": actual_top3_ranks,
        "n_actual_top3_settled": len(actual_top3_pids),
    }


def classify(analyzed, threshold):
    max_gap = max(analyzed["g12"], analyzed["g23"], analyzed["g34"])
    if max_gap < threshold:
        return "全体拮抗"
    if analyzed["g12"] >= analyzed["g23"] and analyzed["g12"] >= analyzed["g34"]:
        return "1車突出"
    if analyzed["g23"] >= analyzed["g34"]:
        return "2車突出"
    return "3車突出"


def pct(x, n):
    return 100.0 * x / n if n else float("nan")


def dist_stats(vals):
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def quant(p):
        if n == 1:
            return vals_sorted[0]
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    return {
        "n": n,
        "mean": statistics.mean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "p25": quant(0.25),
        "p75": quant(0.75),
        "p90": quant(0.90),
        "under5_rate": pct(sum(1 for v in vals_sorted if v < 5), n),
        "over30_rate": pct(sum(1 for v in vals_sorted if v >= 30), n),
    }


def build_dataset(date_from, date_to, label):
    races = load_races(date_from, date_to)
    print(f"[{label}] n_entries=7 & cancel=0 レース数: {len(races)}", flush=True)
    analyzed = [analyze_race(r) for r in races]

    # 拮抗判定の閾値: max(g12,g23,g34) の全レース分布の下位25%点
    all_max_gaps = sorted(max(a["g12"], a["g23"], a["g34"]) for a in analyzed)
    if all_max_gaps:
        idx = 0.25 * (len(all_max_gaps) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(all_max_gaps) - 1)
        frac = idx - lo
        threshold = all_max_gaps[lo] * (1 - frac) + all_max_gaps[hi] * frac
    else:
        threshold = 0.0

    race_keys = [a["race_key"] for a in analyzed]
    odds_map, winners_by_race = load_trio_odds(race_keys)

    records = []
    for a in analyzed:
        rk = a["race_key"]
        odds = odds_map.get(rk)
        pattern = classify(a, threshold)
        records.append({
            **a,
            "pattern": pattern,
            "trio_odds": odds,
            "winners": winners_by_race.get(rk),
        })
    return records, threshold


PATTERNS = ["1車突出", "2車突出", "3車突出", "全体拮抗"]


def analysis1(records, label):
    print(f"\n{'='*100}\n[分析1: {label}] パターン別 三連複配当分布\n{'='*100}")
    total_settled = sum(1 for r in records if r["trio_odds"] is not None)
    header = f"{'パターン':<10}{'n':>7}{'構成比%':>9}{'配当n':>8}{'平均':>10}{'中央値':>10}{'p25':>9}{'p75':>9}{'p90':>10}{'5倍未満%':>10}{'30倍以上%':>11}"
    print(header)
    for p in PATTERNS:
        sub = [r for r in records if r["pattern"] == p]
        n = len(sub)
        odds_vals = [r["trio_odds"] for r in sub if r["trio_odds"] is not None]
        st = dist_stats(odds_vals)
        share = pct(n, len(records))
        if st is None:
            print(f"{p:<10}{n:>7}{share:>8.1f}%{'--':>8}")
            continue
        print(f"{p:<10}{n:>7}{share:>8.1f}%{st['n']:>8}{st['mean']:>10.2f}{st['median']:>10.2f}"
              f"{st['p25']:>9.2f}{st['p75']:>9.2f}{st['p90']:>10.2f}{st['under5_rate']:>9.1f}%{st['over30_rate']:>10.1f}%")
    print(f"(配当が判明したレース: {total_settled}/{len(records)})")


def analysis2(records, label):
    print(f"\n{'='*100}\n[分析2: {label}] 突出馬の3着内率（軸選定の妥当性）\n{'='*100}")

    # 1車突出: w1の3着内率
    sub1 = [r for r in records if r["pattern"] == "1車突出" and r["n_actual_top3_settled"] == 3]
    if sub1:
        n = len(sub1)
        w1_hit = sum(1 for r in sub1 if r["in_top3"](r["win_sorted_pids"][0]))
        print(f"\n[1車突出] n={n} (確定レースのみ)")
        print(f"  w1(pred_win_pct最大)の3着内率: {pct(w1_hit, n):.1f}% ({w1_hit}/{n})")
    else:
        print("\n[1車突出] 確定レースなし")

    # 2車突出: w1,w2の3分類
    sub2 = [r for r in records if r["pattern"] == "2車突出" and r["n_actual_top3_settled"] == 3]
    if sub2:
        n = len(sub2)
        both, only_w1, only_w2, neither = [], [], [], []
        for r in sub2:
            pid1, pid2 = r["win_sorted_pids"][0], r["win_sorted_pids"][1]
            h1, h2 = r["in_top3"](pid1), r["in_top3"](pid2)
            if h1 and h2:
                both.append(r)
            elif h1 and not h2:
                only_w1.append(r)
            elif h2 and not h1:
                only_w2.append(r)
            else:
                neither.append(r)
        print(f"\n[2車突出] n={n} (確定レースのみ) — w1/w2の共倒れ検証")

        def summarize(name, lst):
            odds_vals = [r["trio_odds"] for r in lst if r["trio_odds"] is not None]
            st = dist_stats(odds_vals)
            if st:
                print(f"  {name:<20} n={len(lst):>4} ({pct(len(lst), n):>5.1f}%)  "
                      f"配当n={st['n']:>4}  平均={st['mean']:>8.2f}  中央値={st['median']:>8.2f}")
            else:
                print(f"  {name:<20} n={len(lst):>4} ({pct(len(lst), n):>5.1f}%)  配当データなし")

        summarize("両方3着内", both)
        summarize("w1のみ3着内", only_w1)
        summarize("w2のみ3着内", only_w2)
        summarize("両方圏外", neither)
        one_only = len(only_w1) + len(only_w2)
        print(f"  --> 片方のみ3着内(共倒れ)率: {pct(one_only, n):.1f}% ({one_only}/{n})")
    else:
        print("\n[2車突出] 確定レースなし")

    # 3車突出: w1,w2,w3のうち何頭来たか
    sub3 = [r for r in records if r["pattern"] == "3車突出" and r["n_actual_top3_settled"] == 3]
    if sub3:
        n = len(sub3)
        buckets = defaultdict(list)
        for r in sub3:
            cnt = sum(1 for pid in r["win_sorted_pids"][:3] if r["in_top3"](pid))
            buckets[cnt].append(r)
        print(f"\n[3車突出] n={n} (確定レースのみ) — w1/w2/w3のうち3着内に入った頭数の分布")
        for cnt in (3, 2, 1, 0):
            lst = buckets.get(cnt, [])
            odds_vals = [r["trio_odds"] for r in lst if r["trio_odds"] is not None]
            st = dist_stats(odds_vals)
            if st:
                print(f"  {cnt}頭一致  n={len(lst):>4} ({pct(len(lst), n):>5.1f}%)  "
                      f"配当n={st['n']:>4}  平均={st['mean']:>8.2f}  中央値={st['median']:>8.2f}")
            else:
                print(f"  {cnt}頭一致  n={len(lst):>4} ({pct(len(lst), n):>5.1f}%)  配当データなし")
    else:
        print("\n[3車突出] 確定レースなし")

    # 全体拮抗: w1の3着内率 + w1〜w3のうち何頭来たか
    sub4 = [r for r in records if r["pattern"] == "全体拮抗" and r["n_actual_top3_settled"] == 3]
    if sub4:
        n = len(sub4)
        w1_hit = sum(1 for r in sub4 if r["in_top3"](r["win_sorted_pids"][0]))
        print(f"\n[全体拮抗] n={n} (確定レースのみ)")
        print(f"  w1の3着内率: {pct(w1_hit, n):.1f}% ({w1_hit}/{n})")
        buckets = defaultdict(list)
        for r in sub4:
            cnt = sum(1 for pid in r["win_sorted_pids"][:3] if r["in_top3"](pid))
            buckets[cnt].append(r)
        for cnt in (3, 2, 1, 0):
            lst = buckets.get(cnt, [])
            print(f"  w1〜w3のうち{cnt}頭一致: n={len(lst):>4} ({pct(len(lst), n):>5.1f}%)")
    else:
        print("\n[全体拮抗] 確定レースなし")


def analysis3(records, label):
    print(f"\n{'='*100}\n[分析3: {label}] パターン別「3着内3頭」の予測難易度（pred_top3_pct順位）\n{'='*100}")
    header = f"{'パターン':<10}{'n':>7}{'上位3が的中%':>13}{'上位2+4位以下%':>16}{'上位1+2頭が4位以下%':>20}{'上位3全滅%':>12}"
    print(header)
    for p in PATTERNS:
        sub = [r for r in records if r["pattern"] == p and r["actual_top3_ranks_by_top3pct"] is not None]
        n = len(sub)
        if n == 0:
            print(f"{p:<10}{n:>7}  (確定データなし)")
            continue
        exact = 0       # ranks == [1,2,3]
        two_top = 0     # 上位2頭(rank<=2)が2頭含まれ、残り1頭がrank>=4
        one_top = 0      # 上位1頭のみ(rank==1)含まれ、残り2頭がrank>=4（かつtwo_topでない）
        none_top = 0    # 上位3頭のうちrank<=3が0
        for r in sub:
            ranks = r["actual_top3_ranks_by_top3pct"]
            top3_in_actual = sum(1 for x in ranks if x <= 3)
            top2_in_actual = sum(1 for x in ranks if x <= 2)
            top1_in_actual = sum(1 for x in ranks if x <= 1)
            if ranks == [1, 2, 3]:
                exact += 1
            elif top2_in_actual == 2 and top3_in_actual == 2:
                # 上位2頭がともに含まれ、3頭目がrank>=4
                two_top += 1
            elif top1_in_actual == 1 and top3_in_actual == 1:
                # 上位1頭のみ含まれ、残り2頭がrank>=4
                one_top += 1
            elif top3_in_actual == 0:
                none_top += 1
        other = n - exact - two_top - one_top - none_top
        print(f"{p:<10}{n:>7}{pct(exact, n):>12.1f}%{pct(two_top, n):>15.1f}%"
              f"{pct(one_top, n):>19.1f}%{pct(none_top, n):>11.1f}%"
              + (f"  (その他{other}件)" if other else ""))


def main():
    train_records, train_threshold = build_dataset(TRAIN_FROM, TRAIN_TO, "TRAIN")
    test_records, test_threshold = build_dataset(TEST_FROM, TEST_TO, "TEST")

    print(f"\n拮抗判定閾値(max gapの下位25%点): TRAIN={train_threshold:.3f}pt / TEST={test_threshold:.3f}pt")

    print(f"\n{'='*100}\nパターン別 件数構成（確認用）\n{'='*100}")
    for label, recs in (("TRAIN", train_records), ("TEST", test_records)):
        print(f"\n[{label}] 総n={len(recs)}")
        for p in PATTERNS:
            n = sum(1 for r in recs if r["pattern"] == p)
            print(f"  {p}: {n} ({pct(n, len(recs)):.1f}%)")

    for label, recs in (("TRAIN", train_records), ("TEST", test_records)):
        analysis1(recs, label)
    for label, recs in (("TRAIN", train_records), ("TEST", test_records)):
        analysis2(recs, label)
    for label, recs in (("TRAIN", train_records), ("TEST", test_records)):
        analysis3(recs, label)

    print(f"\n{'='*100}\n注意: n<100の集計は信頼性に注意、n<30は判断不能として扱うこと\n{'='*100}")


if __name__ == "__main__":
    main()
