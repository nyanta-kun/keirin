"""【三連単の組み合わせ絞り込み構造の比較】1着固定で点数を半減できるか
（2026-07-30・[[keirin_dominance_pattern_verification_2026_07_30]]の続き）。

## 制約（ユーザー方針・ブランド「二軸探偵」）
- 軸2車の選定は必須。券種は三連複・三連単のみ。
- 三連単の「2軸マルチ+3着流し」は三連複と同じ的中条件（軸2車が1-2着）だが、
  **1着固定にすれば点数が半分**になる。この構造差を検証する。

## 検証する三連単の構造（軸2車 a1,a2・a1 = pred_win_pctが高い方）

| # | 構造 | 点数(k=3列目候補数) | 的中条件 |
|---|---|---|---|
| V1 | 2軸マルチ・3着流し | 2k | a1,a2が1-2着(順不同) ∧ 3着が候補内 |
| V2 | **1着a1固定・2着a2・3着流し** | **k** | a1が1着 ∧ a2が2着 ∧ 3着が候補内 |
| V3 | 1着a1固定・2-3着{a2,x}マルチ | 2k | a1が1着 ∧ 2-3着が{a2,候補}(順不同) |
| V4 | 1着a2固定・2着a1・3着流し(逆固定) | k | a2が1着 ∧ a1が2着 ∧ 3着が候補内 |
| R  | 三連複2軸流し(比較基準) | k | a1,a2が3着内 ∧ 3着が候補内 |

k=1..5 で振る（k=5が3列目総流し＝現行相当）。

V2 vs V4 の比較は「a1(pred_win_pct上位)が本当にa2より1着になりやすいか」の
直接検証でもある。V2がV4を明確に上回れば1着固定の根拠になる。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
STAKE = 100


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: str(r["race_date"]) for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, prediction_mark, "
                 "       line_group, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] odds (trio/trifecta) ...", flush=True)
    odds = defaultdict(lambda: defaultdict(dict))
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    nums = [int(x) for x in re.split(r"[-=→]", str(comb)) if x != ""]
                except ValueError:
                    continue
                key = tuple(nums) if bt == "trifecta" else frozenset(nums)
                odds[rk][bt][key] = fv
            if (i // 600) % 25 == 0:
                print(f"[load]   odds progress: {i}/{len(keys)}", flush=True)
    print(f"[load]   odds races: {len(odds)}", flush=True)
    return races, by_race, odds


def build(races, entries_by_race, odds):
    print("[build] ...", flush=True)
    out = []
    for rk, rdate in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_win_pct"] is None or e["pred_top3_pct"] is None for e in ents):
            continue
        od = odds.get(rk)
        if not od or "trio" not in od or "trifecta" not in od:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        order = [fno for _, fno in fin[:3]]
        by_frame = {int(e["frame_no"]): e for e in ents}
        frames = list(by_frame.keys())
        tsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_top3_pct"]))
        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)
        out.append({
            "race_key": rk, "race_date": rdate, "frames": frames, "by_frame": by_frame,
            "odds": od, "order": order, "top3": frozenset(order),
            "tsorted": tsorted, "honmei": honmei, "taikou": taikou,
        })
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def axis_t1_t2(r):
    """複勝確率上位2頭。a1 = そのうちpred_win_pctが高い方（1着固定候補）"""
    a, b = r["tsorted"][0], r["tsorted"][1]
    bf = r["by_frame"]
    if float(bf[a]["pred_win_pct"]) >= float(bf[b]["pred_win_pct"]):
        return a, b
    return b, a


def axis_s7_new(r):
    h, t = r["honmei"], r["taikou"]
    if h is None or t is None:
        return None
    bf = r["by_frame"]
    a1 = h if float(bf[h]["pred_win_pct"]) >= float(bf[t]["pred_win_pct"]) else t
    a2 = None
    for f in r["tsorted"]:
        if f not in (h, t):
            a2 = f
            break
    if a2 is None:
        return None
    # a1/a2 のうち pred_win_pct 高い方を1着固定候補に
    if float(bf[a1]["pred_win_pct"]) >= float(bf[a2]["pred_win_pct"]):
        return a1, a2
    return a2, a1


AXES = [("t1+t2", axis_t1_t2), ("S7新設計", axis_s7_new)]


def others_ranked(r, a1, a2):
    return [f for f in r["tsorted"] if f not in (a1, a2)]


# ---------- 構造 ----------
def st_trio(r, a1, a2, k):
    board = r["odds"]["trio"]
    ks = []
    for x in others_ranked(r, a1, a2)[:k]:
        key = frozenset({a1, a2, x})
        if key in board:
            ks.append(key)
    if not ks:
        return None
    hit = r["top3"] in ks
    return len(ks) * STAKE, hit, (int(board[r["top3"]] * STAKE) if hit else 0)


def st_tri_multi(r, a1, a2, k):
    """V1: 2軸マルチ・3着流し = 2k点"""
    board = r["odds"]["trifecta"]
    ks = []
    for x in others_ranked(r, a1, a2)[:k]:
        for t in ((a1, a2, x), (a2, a1, x)):
            if t in board:
                ks.append(t)
    if not ks:
        return None
    actual = tuple(r["order"])
    hit = actual in ks
    return len(ks) * STAKE, hit, (int(board[actual] * STAKE) if hit else 0)


def st_tri_fix1(r, a1, a2, k):
    """V2: 1着a1固定・2着a2・3着流し = k点"""
    board = r["odds"]["trifecta"]
    ks = [(a1, a2, x) for x in others_ranked(r, a1, a2)[:k]]
    ks = [t for t in ks if t in board]
    if not ks:
        return None
    actual = tuple(r["order"])
    hit = actual in ks
    return len(ks) * STAKE, hit, (int(board[actual] * STAKE) if hit else 0)


def st_tri_fix1_multi23(r, a1, a2, k):
    """V3: 1着a1固定・2-3着{a2,x}マルチ = 2k点"""
    board = r["odds"]["trifecta"]
    ks = []
    for x in others_ranked(r, a1, a2)[:k]:
        for t in ((a1, a2, x), (a1, x, a2)):
            if t in board:
                ks.append(t)
    if not ks:
        return None
    actual = tuple(r["order"])
    hit = actual in ks
    return len(ks) * STAKE, hit, (int(board[actual] * STAKE) if hit else 0)


def st_tri_fix1_rev(r, a1, a2, k):
    """V4: 1着a2固定・2着a1・3着流し = k点（逆固定・対照）"""
    board = r["odds"]["trifecta"]
    ks = [(a2, a1, x) for x in others_ranked(r, a1, a2)[:k]]
    ks = [t for t in ks if t in board]
    if not ks:
        return None
    actual = tuple(r["order"])
    hit = actual in ks
    return len(ks) * STAKE, hit, (int(board[actual] * STAKE) if hit else 0)


STRUCTS = [
    ("R:三連複2軸流し", st_trio, 1),
    ("V1:三単2軸マルチ", st_tri_multi, 2),
    ("V2:三単1着a1固定", st_tri_fix1, 1),
    ("V3:三単1着固定23マルチ", st_tri_fix1_multi23, 2),
    ("V4:三単1着a2固定(逆)", st_tri_fix1_rev, 1),
]


def evaluate(rows, axis_fn, st_fn, k):
    n = hits = 0
    bet_total = pay_total = 0
    for r in rows:
        sel = axis_fn(r)
        if sel is None:
            continue
        a1, a2 = sel
        if a1 == a2:
            continue
        res = st_fn(r, a1, a2, k)
        if res is None:
            continue
        bet, hit, pay = res
        n += 1
        bet_total += bet
        pay_total += pay
        hits += 1 if hit else 0
    hr = hits / n * 100 if n else 0.0
    roi = pay_total / bet_total * 100 if bet_total else 0.0
    return n, hr, roi


def main():
    races, entries_by_race, odds = load_all()
    rows = build(races, entries_by_race, odds)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}\n")

    # 参考: 軸2車がともに3着内のとき、a1(pred_win上位)が1着だった割合
    print("=" * 96)
    print("参考: 軸2車がともに1-2着だったとき a1(pred_win上位)が1着だった割合")
    print("  （1着固定の根拠になるか。50%付近なら固定に意味がない）")
    print("=" * 96)
    for axis_name, axis_fn in AXES:
        for label, data in (("TRAIN", train), ("TEST", test)):
            n = a1_first = 0
            for r in data:
                sel = axis_fn(r)
                if sel is None:
                    continue
                a1, a2 = sel
                if frozenset({a1, a2}) != frozenset(r["order"][:2]):
                    continue
                n += 1
                if r["order"][0] == a1:
                    a1_first += 1
            if n:
                print(f"  {axis_name:<12}[{label}] n={n:>6} a1が1着={a1_first/n*100:.1f}%")

    for axis_name, axis_fn in AXES:
        print("\n" + "=" * 96)
        print(f"三連単構造 × 3列目点数  軸選定={axis_name}")
        print("=" * 96)
        for st_name, st_fn, mult in STRUCTS:
            print(f"\n  --- {st_name} ---")
            print(f"    {'3列目k':<10}{'実点数':>7}{'TRAIN n':>9}{'的中%':>8}{'ROI%':>9}"
                  f"{'TEST n':>9}{'的中%':>8}{'ROI%':>9}")
            for k in (1, 2, 3, 4, 5):
                n1, h1, r1 = evaluate(train, axis_fn, st_fn, k)
                n2, h2, r2 = evaluate(test, axis_fn, st_fn, k)
                flag = ""
                if r1 >= 100 and r2 >= 100:
                    flag = " ★★両窓100%超"
                elif r2 >= 100:
                    flag = " ★TEST100%超"
                elif r1 >= 100:
                    flag = " ★TRAIN100%超"
                print(f"    k={k:<8}{k*mult:>7}{n1:>9}{h1:>7.1f}%{r1:>8.1f}%"
                      f"{n2:>9}{h2:>7.1f}%{r2:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
